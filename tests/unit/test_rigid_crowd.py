"""Rigid mesh humans: saved graph joint rotations, the Warp ray target and the plaza term.

The rotation tests use an FK-consistent synthetic library. The plaza tests need the
default graph, its rotations artifact and ARDY's Core skin, and skip without them;
with CUDA they also run on the GPU (captured sense graph), checked against CPU MuJoCo.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from crowdmotion.graph.config import GraphConfig
from crowdmotion.graph.rotations import GraphRotations, bone_direction_error, build
from crowdmotion.graph.tables import MotionGraph
from crowdmotion.sim import body as B
from synthetic import synthetic_library

PARENTS = [-1, 0, 1, 1, 0]


def skew(v: torch.Tensor) -> torch.Tensor:
  zero = torch.zeros_like(v[..., 0])
  return torch.stack((zero, -v[..., 2], v[..., 1], v[..., 2], zero, -v[..., 0], -v[..., 1], v[..., 0], zero),
                     -1).unflatten(-1, (3, 3))


def fk_library(seed: int = 0) -> dict:
  """`synthetic_library` streams whose joints are FK of stored local rotations."""
  lib = synthetic_library(seed)
  g = torch.Generator().manual_seed(seed + 1)
  n, t, J = lib["posed_joints"].shape[:3]
  offsets = torch.randn(J, 3, generator=g) * 0.3
  offsets[0] = 0
  phase = torch.randn(n, 1, J, 3, generator=g)
  local = torch.linalg.matrix_exp(skew(0.4 * torch.sin(torch.arange(t)[None, :, None, None] * 0.5 + phase)))
  local[:, :, 0] = B.quat_to_matrix(B.quat_yaw(lib["heading"])) @ local[:, :, 0]
  rot, pos = [], []
  for j, p in enumerate(PARENTS):
    if p < 0:
      rot.append(local[:, :, 0])
      pos.append(lib["root_pos"])
    else:
      rot.append(rot[p] @ local[:, :, j])
      pos.append(pos[p] + (rot[p] @ offsets[j][:, None])[..., 0])
  lib["posed_joints"] = torch.stack(pos, 2)
  lib["local_rot"] = local.half()
  lib["meta"]["parents"] = PARENTS
  return lib


def test_rotation_tables_follow_graph_frames(tmp_path):
  lib = fk_library()
  graph = MotionGraph(lib, GraphConfig(neighbors=3), device="cpu")
  blob = build(lib, graph.L)
  K, L = graph.K, graph.L
  glob = B.joint_transforms(lib["local_rot"], lib["posed_joints"], PARENTS)[..., :3, :3]
  local = B.quat_to_matrix(B.quat_yaw(-lib["heading"]))[..., None, :, :] @ glob  # body-local rotations
  for n in range(graph.streams):
    for j in range(K - 1):
      for frame in (0, L - 1):
        got = B.quat_to_matrix(blob["edge_rot"][n * K + j, frame].float())
        assert torch.allclose(got, local[n, (j + 1) * L + frame], atol=3e-3)
    for k in range(K):
      got = B.quat_to_matrix(blob["pred_rot"][n * K + k].float())
      assert torch.allclose(got, local[n, (k + 1) * L - 1], atol=3e-3)
  # Parent rotations carry rest offsets onto the graph's (body-local, float) bone vectors.
  assert float(bone_direction_error(graph, blob["edge_rot"], blob["offsets"], PARENTS, samples=64).max()) < 1.0
  path = tmp_path / "rotations.pt"
  torch.save(blob, path)
  GraphRotations.load(path, graph)
  shifted = dict(blob, edge_rot=torch.roll(blob["edge_rot"], 1, dims=1))
  torch.save(shifted, path)
  with pytest.raises(ValueError, match="does not match the graph"):
    GraphRotations.load(path, graph)


def test_swing_to_children_spans_posed_joints():
  gen = torch.Generator().manual_seed(0)
  offsets = torch.randn(5, 3, generator=gen) * 0.3
  assert B.single_children(PARENTS).tolist() == [-1, -1, -1, -1, -1]  # branching joints and leaves
  chain = [-1, 0, 1, 2, 3]
  child = B.single_children(chain)
  assert child.tolist() == [1, 2, 3, 4, -1]
  rot = torch.linalg.matrix_exp(skew(torch.randn(8, 5, 3, generator=gen)))
  joints = torch.randn(8, 5, 3, generator=gen)
  out, turn = B.swing_to_children(rot, joints, offsets, child)
  for j in range(4):
    u = (out[:, j] @ offsets[j + 1][:, None])[..., 0]
    v = joints[:, j + 1] - joints[:, j]
    assert torch.allclose(u / u.norm(dim=-1, keepdim=True), v / v.norm(dim=-1, keepdim=True), atol=1e-4)
  assert torch.allclose(out @ out.transpose(-1, -2), torch.eye(3).expand(8, 5, 3, 3), atol=1e-5)
  assert torch.equal(out[:, 4], rot[:, 4]) and turn.shape == (8, 4)


@pytest.mark.skipif(not B.CORE_SKIN.exists(), reason="needs third_party/ardy (Core skin)")
def test_ray_boxes_hold_posed_pieces():
  from crowdmotion.sim.rays import RigidRays

  pieces = B.core_pieces()
  gen = torch.Generator().manual_seed(1)
  rays = RigidRays(pieces, worlds=2, humans=3, geom_ids=np.zeros((3, len(pieces))), device="cpu", max_range=20)
  rot = torch.linalg.matrix_exp(skew(torch.randn(6, len(pieces), 3, generator=gen) * 2))
  pos = torch.randn(6, len(pieces), 3, generator=gen)
  rays.update(rot, pos)
  for h in range(6):
    verts = torch.cat([(rot[h, p] @ torch.as_tensor(pc.vertices).T).T + pos[h, p] for p, pc in enumerate(pieces)])
    assert (verts >= rays.lo[h] - 1e-5).all() and (verts <= rays.hi[h] + 1e-5).all()
  # A partial update touches only its rows.
  before = rays.inv.clone()
  rays.update(rot[:1], pos[:1] + 1, torch.tensor([4]))
  assert torch.equal(rays.inv[:4], before[:4]) and not torch.equal(rays.inv[4], before[4])


def _artifacts():
  try:
    from crowdmotion.core.paths import default_graph, default_graph_rotations

    return default_graph().exists() and default_graph_rotations().exists() and B.CORE_SKIN.exists()
  except (FileNotFoundError, KeyError):
    return False


@pytest.fixture(scope="module", params=[
  "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
])
def plaza(request):
  if not _artifacts():
    pytest.skip("needs artifacts graph-default and graph-rotations, and the Core skin")
  from mjlab.envs import ManagerBasedRlEnv

  from crowdmotion.core.paths import default_graph
  from crowdmotion.sim.mjlab.plaza import make_plaza_env_cfg

  torch.set_num_threads(2)
  cfg = make_plaza_env_cfg(str(default_graph()), humans=4, num_envs=2, seed=5, width=64, height=48,
                           lidar_azimuth=180, lidar_elevation=8, human_geometry=("rigid", "capsules"))
  env = ManagerBasedRlEnv(cfg, device=request.param)
  env.reset()
  yield env
  env.close()


def _step(env, steps: int) -> None:
  action = torch.zeros(env.action_space.shape, device=env.device)
  for _ in range(steps):
    env.step(action)


def _kinds(env, ids: torch.Tensor) -> set[str]:
  model = env.sim.mj_model
  names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in ids.unique() if g >= 0}
  return {("mesh" if "_mesh_" in n else "capsule") for n in names if n.startswith("crowd/")}


def test_rigid_lidar_matches_mujoco_rays_on_the_visual_pieces(plaza):
  env = plaza
  term = env.command_manager.get_term("crowd")
  assert term.geometry == "rigid"
  _step(env, 7)
  lidar = env.scene["lidar"]
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # LiDAR groups 0 and 2, plus the pieces' group 1
  scanner = model.body("scanner/scanner").id
  hits = 0
  for w in range(env.num_envs):
    data.mocap_pos[:] = env.sim.data.mocap_pos[w].cpu().numpy()
    data.mocap_quat[:] = env.sim.data.mocap_quat[w].cpu().numpy()
    mujoco.mj_kinematics(model, data)
    # Private caches: the sensor's world-frame ray origins and directions of this step.
    origins = lidar._cached_world_origins[w].double().cpu().numpy()
    dirs = lidar._cached_world_rays[w].double().cpu().numpy()
    geom = np.zeros(1, dtype=np.int32)
    ref_d, ref_g = [], []
    for o, d in zip(origins, dirs):
      dist = mujoco.mj_ray(model, data, o, d, groups, 1, scanner, geom)
      ok = lidar.cfg.min_distance <= dist <= lidar.cfg.max_distance
      ref_d.append(dist if ok else -1.0)
      ref_g.append(int(geom[0]) if ok else -1)
    ref_d, ref_g = torch.tensor(ref_d, dtype=torch.float64), torch.tensor(ref_g)
    got_d, got_g = lidar.data.distances[w].double().cpu(), lidar.data.geom_ids[w].cpu()
    human = (got_g != ref_g) | torch.tensor([
      "_mesh_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "") for g in ref_g.tolist()])
    hits += int(human.sum())
    assert torch.equal(got_g, ref_g)
    assert torch.allclose(got_d, ref_d, atol=1e-3)
  assert hits > 20  # the crowd is actually in view
  assert _kinds(env, lidar.data.geom_ids) == {"mesh"}


class _Handle:
  def __init__(self, value=None):
    self.value = value

  def on_update(self, callback):
    self.callback = callback
    return callback


class _Gui:
  """Headless stand-in for Viser's GUI API: records every handle."""

  def __init__(self):
    self.handles = {}

  def add_folder(self, name):
    return nullcontext()

  def add_markdown(self, text):
    return SimpleNamespace(content=text)

  def add_number(self, label, *, initial_value, **kwargs):
    self.handles[label] = _Handle(initial_value)
    return self.handles[label]

  add_checkbox = add_number

  def add_dropdown(self, label, *, options, initial_value):
    assert initial_value in options
    return self.add_number(label, initial_value=initial_value)


def test_toggle_switches_lidar_and_rotations_follow_positions(plaza):
  env = plaza
  term = env.command_manager.get_term("crowd")
  lidar = env.scene["lidar"]
  gui = _Gui()
  term.create_gui("crowd", SimpleNamespace(gui=gui), lambda: 0)
  dropdown = gui.handles["Geometry"]
  assert dropdown.value == "Rigid mesh"
  dropdown.value = "Capsules"
  dropdown.callback(None)  # queued: applied on the next step, on the simulation thread
  assert term.geometry == "rigid"
  _step(env, 3)
  assert term.geometry == "capsules" and _kinds(env, lidar.data.geom_ids) == {"capsule"}
  pieces = env.sim.data.mocap_pos[:, term.piece_mocap]
  assert (pieces[..., 2] <= -5).all()  # parked below the floor
  term.set_geometry("rigid")
  _step(env, 3)
  assert term.geometry == "rigid" and _kinds(env, lidar.data.geom_ids) == {"mesh"}
  assert (env.sim.data.mocap_pos[:, term.mocap][..., 2] <= -5).all()
  # Without inertialization, saved rotations already span the played joints: the swing
  # correction stays near zero once edges have turned over.
  term.graph.cfg.inertialize = False
  turns = []
  for _ in range(60):
    _step(env, 1)
    turns.append(term.swing_deg.flatten())
  turns = torch.cat(turns[25:])
  assert float(turns.median()) < 0.5 and float(turns.quantile(0.99)) < 5.0
  with pytest.raises(ValueError):
    term.set_geometry("skinned")


def test_fused_pose_matches_torch_reference(plaza):
  """The Warp pose pass equals blending, swinging and converting with `sim.body` in torch."""
  from crowdmotion.core.geometry import motion_to_mujoco

  env = plaza
  term = env.command_manager.get_term("crowd")
  term.set_geometry("rigid")
  _step(env, 5)
  E, M, L = env.num_envs, term.M, term.L
  f = (term.edge_t * term.fps).clamp(0, L)
  lo = f.floor().long().clamp(max=L - 1)
  a = (f - lo)[:, None, None, None]
  rows = torch.arange(E, device=env.device)
  joints = term.frames[rows, :, lo] * (1 - a) + term.frames[rows, :, lo + 1] * a
  rot = B.quat_to_matrix(B.quat_nlerp(term.rots[rows, :, lo], term.rots[rows, :, lo + 1], a))
  rot, turn = B.swing_to_children(rot, joints, term.rotations.offsets, B.single_children(term.graph.meta["parents"]))
  bone = term.body.bone
  frame = torch.as_tensor(B.YUP_TO_MUJOCO, dtype=torch.float32, device=env.device)
  want_rot = frame @ rot[:, :, bone]
  want_pos = motion_to_mujoco(joints[:, :, bone])
  P = len(bone)
  got_pos = env.sim.data.mocap_pos[:, term.piece_mocap].view(E, M, P, 3)
  got_rot = B.quat_to_matrix(env.sim.data.mocap_quat[:, term.piece_mocap].view(E, M, P, 4))
  assert torch.allclose(got_pos, want_pos, atol=1e-5)
  assert torch.allclose(got_rot, want_rot, atol=1e-4)
  inv = term.rays.inv.view(E, M, P, 4, 4)
  assert torch.allclose(inv[..., :3, :3], want_rot.transpose(-1, -2), atol=1e-4)
  assert torch.allclose((inv[..., :3, :3] @ want_pos[..., None])[..., 0] + inv[..., :3, 3],
                        torch.zeros_like(want_pos), atol=1e-4)


def test_sensors_tracing_the_visual_pieces_are_rejected():
  """A sensor on mjlab's default groups (0, 1, 2) would trace the piece meshes natively."""
  if not _artifacts():
    pytest.skip("needs artifacts graph-default and graph-rotations, and the Core skin")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

  from crowdmotion.core.paths import default_graph
  from crowdmotion.sim.mjlab.plaza import make_plaza_env_cfg

  cfg = make_plaza_env_cfg(str(default_graph()), humans=2, num_envs=1, width=64, height=48,
                           lidar_azimuth=36, lidar_elevation=2)
  probe = RayCastSensorCfg(name="probe", frame=ObjRef(type="site", name="lidar", entity="scanner"),
                           pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1))
  cfg.scene.sensors = (*cfg.scene.sensors, probe)
  with pytest.raises(ValueError, match="visual group"):
    ManagerBasedRlEnv(cfg, device="cpu")
