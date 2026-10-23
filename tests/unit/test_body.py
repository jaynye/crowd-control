"""Mesh humans from the Core skin: skinning, decimation, rigid pieces and MuJoCo posing.

Needs ARDY's `cskel27/skin_standard.npz` (the `third_party/ardy` clone); skips without it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from crowdmotion.sim import body as B

pytestmark = pytest.mark.skipif(not B.CORE_SKIN.exists(), reason="needs third_party/ardy (Core skin)")

PARENTS = [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0, 23, 24, 25]


@pytest.fixture(scope="module")
def skin():
  return B.load_core_skin()


def random_rotations(n: int, gen: torch.Generator) -> torch.Tensor:
  q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=gen))
  q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
  return q * torch.det(q)[:, None, None]


def random_pose(skin: B.Skin, gen: torch.Generator, scale: float = 0.4) -> torch.Tensor:
  """Joint transforms `[J, 4, 4]` (Y-up) for small random local rotations of the bind skeleton."""
  axis = torch.randn(27, 3, generator=gen) * scale
  local = torch.linalg.matrix_exp(torch.cross(
    axis[:, :, None].expand(27, 3, 3), torch.eye(3).expand(27, 3, 3), dim=1))
  bind = torch.as_tensor(skin.bind, dtype=torch.float32)
  # Bind-relative local rotations, so FK of identity reproduces the bind pose.
  bind_local = [bind[j, :3, :3] if p < 0 else bind[p, :3, :3].T @ bind[j, :3, :3] for j, p in enumerate(PARENTS)]
  rot = [None] * 27
  pos = [None] * 27
  for j, p in enumerate(PARENTS):
    r = bind_local[j] @ local[j]
    if p < 0:
      rot[j], pos[j] = r, bind[j, :3, 3]
    else:
      rot[j] = rot[p] @ r
      offset = bind[p, :3, :3].T @ (bind[j, :3, 3] - bind[p, :3, 3])
      pos[j] = pos[p] + rot[p] @ offset
  out = torch.eye(4).repeat(27, 1, 1)
  out[:, :3, :3] = torch.stack(rot)
  out[:, :3, 3] = torch.stack(pos)
  return out


def test_quat_from_matrix_matches_mujoco_including_half_turns():
  gen = torch.Generator().manual_seed(0)
  rots = torch.cat((random_rotations(500, gen), torch.diag(torch.tensor([1.0, -1.0, -1.0]))[None],
                    torch.diag(torch.tensor([-1.0, -1.0, 1.0]))[None]))
  q = B.quat_from_matrix(rots).double().numpy()
  for qi, r in zip(q, rots.double().numpy()):
    ref = np.zeros(4)
    mujoco.mju_mat2Quat(ref, r.ravel())
    assert abs(abs(np.dot(qi, ref)) - 1) < 1e-6


def test_joint_transforms_chain_and_orthonormalization():
  gen = torch.Generator().manual_seed(1)
  local = random_rotations(3, gen)
  joints = torch.randn(3, 3, generator=gen)
  t = B.joint_transforms(local.half(), joints, [-1, 0, 1])
  r = t[:, :3, :3]
  assert torch.allclose(r @ r.transpose(-1, -2), torch.eye(3).expand(3, 3, 3), atol=1e-5)
  assert torch.allclose(r[2], local[0] @ local[1] @ local[2], atol=3e-3)
  assert torch.equal(t[:, :3, 3], joints)


def test_lbs_at_bind_reproduces_bind_vertices(skin):
  verts = B.lbs(skin, torch.as_tensor(skin.bind, dtype=torch.float32))
  assert torch.allclose(verts, torch.as_tensor(skin.vertices), atol=1e-5)


@pytest.mark.parametrize("faces", [18152, 1200])
def test_rigid_pieces_closed_and_cover_the_skin(skin, faces):
  s = B.decimate(skin, faces)
  assert abs(len(s.faces) - faces) <= 0.02 * faces
  assert np.allclose(s.weights.sum(1), 1, atol=1e-5)
  assert s.volume() > 0.95 * skin.volume()
  pieces = B.rigid_pieces(s)
  assert len(pieces) == 22  # bones carrying skin weight
  assert sum(len(p.faces) for p in pieces) >= len(s.faces)
  for p in pieces:
    e = np.concatenate((p.faces[:, [0, 1]], p.faces[:, [1, 2]], p.faces[:, [2, 0]]))
    directed = set(map(tuple, e.tolist()))
    assert all((b, a) in directed for a, b in e.tolist()), p.name


def test_rigid_pieces_follow_lbs_for_single_bone_vertices(skin):
  """A vertex fully weighted to one bone lands where LBS puts it, through MuJoCo mocap."""
  s = B.decimate(skin, 4500)
  body = B.RigidBody(B.rigid_pieces(s))
  pose = random_pose(skin, torch.Generator().manual_seed(2))
  pose_mj = B.to_mujoco(pose)
  ref = B.lbs(s, pose_mj)  # [V, 3] MuJoCo frame

  spec = mujoco.MjSpec()
  body.add_to_spec(spec, 1, [(1, 1, 1, 1)])
  model = spec.compile()
  data = mujoco.MjData(model)
  pos, quat = body.poses(pose_mj[None])
  data.mocap_pos[:] = pos[0].numpy()
  data.mocap_quat[:] = quat[0].numpy()
  mujoco.mj_kinematics(model, data)

  dominant = s.dominant
  rigid = s.weights.max(1) > 0.999
  for g, piece in enumerate(body.pieces):
    mesh = model.geom_dataid[g]
    a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    world = model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    sel = np.nonzero(rigid & (dominant == piece.bone))[0]
    if len(sel) == 0:
      continue
    dist = np.linalg.norm(ref[sel].numpy()[:, None] - world[None], axis=-1).min(1)
    assert dist.max() < 1e-4, (piece.name, dist.max())


@pytest.mark.parametrize("geometry,faces,variants", [("skinned", 1200, 1), ("skinned", 600, 32), ("pieces", 1200, 32)])
def test_human_boxes_hold_posed_surfaces(skin, geometry, faces, variants):
  """The benchmark's per-human ray boxes contain the posed surface for any pose and variant scale."""
  from crowdmotion.bench.body_geometry import box_bounds

  s = B.decimate(skin, faces)
  scales = np.linspace(0.85, 1.15, variants) if variants > 1 else (1.0,)
  pairs, radius = box_bounds(s, PARENTS, geometry, scales)
  gen = torch.Generator().manual_seed(3)
  poses = torch.stack([random_pose(skin, gen, scale=0.8) for _ in range(6)])
  for sc in (min(scales), max(scales)):
    if geometry == "skinned":
      bind = s.bind.copy()
      bind[:, :3, 3] *= sc
      scaled = B.Skin((s.vertices * sc).astype(np.float32), s.faces, s.bones, s.weights, bind, s.joint_names)
      verts = B.lbs(scaled, poses)
    else:
      verts = torch.cat([
        (poses[:, p.bone, :3, :3] @ torch.as_tensor(p.vertices * sc).float().T).transpose(1, 2)
        + poses[:, p.bone, None, :3, 3] for p in B.rigid_pieces(s)], 1)
    joints = poses[:, pairs, :3, 3]
    r = torch.tensor(radius)[:, None]
    lo, hi = (joints - r).amin(1), (joints + r).amax(1)
    assert ((verts >= lo[:, None] - 1e-5) & (verts <= hi[:, None] + 1e-5)).all()
