"""`GraphCrowd`: an mjlab command term whose humans are driven by a saved motion graph.

Each environment step it advances every environment's clock and, when an edge
(0.4 s) runs out, asks the `CrowdController` for the next edge: the behavior design
gives a command (`design`: plaza intent and local steering, or the encounter
design's conditioning, bubble and decisions), the planner a root path, and the
graph motion source the frames. The term then interpolates those frames to the current time,
converts them to MuJoCo Z-up, and poses the humans' bodies.

Two body geometries exist (`human_geometry`; the first listed is active):

- `rigid`: the full-resolution Core skin cut into one rigid piece per bone
  (`crowdmotion.sim.body`). Bone rotations come from the graph's saved joint rotations
  (`crowd graph rotations`), turned with the edge's heading and swung onto the posed
  joints. Each piece is a visual mesh geom on a mocap body (geom group 1, which the
  plaza LiDAR does not trace); ray-cast sensors that include `lidar_group` hit the
  pieces through a Warp ray target (`crowdmotion.sim.rays`). That target reads the
  poses written this step, while MuJoCo-traced geometry such as capsules is seen as
  of the last `forward()`, one step earlier.
- `capsules`: one capsule per bone (`crowdmotion.sim.capsules`) in `lidar_group`,
  traced by MuJoCo.

With both built, `set_geometry` (or the Viser panel) switches the active one on the
simulation thread: the other is parked below the floor and the rigid ray target is
turned off or on.

Because the term lives inside the environment, any mjlab viewer (Viser, native,
offscreen `render()`) shows the crowd without extra plumbing. The graph comes
from `crowd graph build` (`MotionGraph.save`), so it loads on CPU without the
library or the kNN build.
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from crowdmotion.behavior.controller import CrowdController
from crowdmotion.behavior.designs import DESIGNS, make_behavior
from crowdmotion.behavior.encounter import EncounterCfg
from crowdmotion.behavior.kinematics import BufferVelocity
from crowdmotion.behavior.plaza import PlazaBehavior, PlazaSteeringCfg
from crowdmotion.behavior.sources import GraphMotionSource
from crowdmotion.core.geometry import motion_to_mujoco
from crowdmotion.core.rng import human_seeds
from crowdmotion.graph import GraphRuntime, MotionGraph
from crowdmotion.sim.body import YUP_TO_MUJOCO, quat_mul, quat_yaw, single_children
from crowdmotion.sim.capsules import CapsuleSkeleton
from crowdmotion.sim.scene import COLORS
from crowdmotion.sim.mjlab.tuning import LiveCrowdTuning

GEOMETRIES = ("rigid", "capsules")
GEOMETRY_LABELS = {"rigid": "Rigid mesh", "capsules": "Capsules"}
MESH_TAG = "mesh_"  # rigid piece bodies are `h<i>_mesh_<bone>`, beside capsule bodies `h<i>_<bone>`

_GRAPHS: dict = {}
_ROTATIONS: dict = {}


def load_graph(path: str, device, overrides: dict | None = None) -> MotionGraph:
  """Share cached immutable tables, but give each crowd its own live configuration."""
  key = (str(path), str(device), tuple(sorted((overrides or {}).items())))
  if key not in _GRAPHS:
    _GRAPHS[key] = MotionGraph.load(path, device=device, cfg_overrides=overrides)
  graph = copy(_GRAPHS[key])
  graph.cfg = replace(graph.cfg)
  return graph


def load_rotations(path, graph: MotionGraph):
  """Joint rotations for `graph`, checked against it once and shared like its tables."""
  from crowdmotion.graph.rotations import GraphRotations

  key = (str(path), str(graph.device), id(graph.edge_pose))
  if key not in _ROTATIONS:
    if not Path(path).exists():
      raise FileNotFoundError(
        f"rigid humans need joint rotations for the graph at {path}: fetch artifact graph-rotations "
        "(scripts/fetch_artifacts.sh) or build them with `crowd graph rotations`"
      )
    _ROTATIONS[key] = GraphRotations.load(path, graph)
  return _ROTATIONS[key]


@dataclass(kw_only=True)
class GraphCrowdCfg(CommandTermCfg):
  graph_path: str
  humans: int
  entity_name: str = "crowd"
  steering: PlazaSteeringCfg = field(default_factory=PlazaSteeringCfg)
  design: str = "plaza"
  """Behavior design (`behavior.designs`): `plaza` uses `steering`, `encounter` uses `encounter`."""
  encounter: EncounterCfg = field(default_factory=EncounterCfg)
  seed: int = 0
  graph_overrides: dict = field(default_factory=dict)
  resampling_time_range: tuple[float, float] = (1e9, 1e9)  # humans persist until reset
  human_geometry: tuple[str, ...] = ("capsules",)
  """Body geometries in the scene (from `GEOMETRIES`); the first is active."""
  rotations_path: str | None = None
  """Rigid: the graph's joint rotations (default: `$CROWD_GRAPH_ROTATIONS` or artifact
  graph-rotations)."""
  lidar_group: int = 2
  """The geom group that stands for the humans in ray casts: capsules sit in it, and
  rigid humans register with every ray-cast sensor that includes it."""

  def build(self, env) -> "GraphCrowd":
    return GraphCrowd(self, env)


class GraphCrowd(CommandTerm):
  cfg: GraphCrowdCfg

  def __init__(self, cfg: GraphCrowdCfg, env):
    if cfg.design not in DESIGNS:
      raise ValueError(f"design must be one of {DESIGNS}, got {cfg.design!r}")
    cfg = replace(cfg, steering=replace(cfg.steering), encounter=replace(cfg.encounter),
                  graph_overrides=dict(cfg.graph_overrides))
    super().__init__(cfg, env)
    self.graph = g = load_graph(cfg.graph_path, self.device, cfg.graph_overrides)
    encounter = cfg.encounter if cfg.design == "encounter" else None
    self.live_tuning = LiveCrowdTuning(g, cfg.steering, encounter=encounter)
    geometries = tuple(cfg.human_geometry)
    if not geometries or len(set(geometries)) < len(geometries) or not set(geometries) <= set(GEOMETRIES):
      raise ValueError(f"human_geometry must list distinct values from {GEOMETRIES}, got {geometries}")
    self.geometries, self.geometry = geometries, geometries[0]
    self._pending_geometry: SimpleQueue = SimpleQueue()
    E, M, L, J = self.num_envs, cfg.humans, g.L, g.joints_n
    self.M, self.L, self.fps, self.edge_s = M, L, g.fps, L / g.fps
    self.mocap = self.rotations = self.rays = None
    if "capsules" in geometries:
      self.skeleton = CapsuleSkeleton.from_rest(g.meta, g.rest_joints, self.device)
      self.mocap = self._mocap_ids(env, self.skeleton.names, "", "capsule")
    if "rigid" in geometries:
      self._init_rigid(env)
    self.source = GraphMotionSource(GraphRuntime(g))
    if cfg.design == "plaza":
      self.behavior = PlazaBehavior(E, M, cfg.steering, cfg.seed, self.device)
      self.controller = CrowdController(self.behavior, self.source, BufferVelocity(g.fps), group=M)
    else:
      self.behavior, planner = make_behavior(cfg.design, cfg.encounter, E, M, cfg.seed, self.device)
      self.controller = CrowdController(self.behavior, self.source, BufferVelocity(g.fps), planner=planner,
                                        group=M)
    self.controller.allocate(E * M, self.device)
    dev = self.device
    self.frames = torch.zeros(E, M, L + 1, J, 3, device=dev)  # frame 0 = start of the edge
    self.edge_t = torch.zeros(E, device=dev)  # seconds into the current edge
    self.clock = torch.zeros(E, device=dev)  # seconds since reset
    self.resets = torch.zeros(E, dtype=torch.long, device=dev)
    self._command = torch.zeros(E, M, 2, device=dev)  # (speed, heading)
    self.metrics["natural_successor"] = torch.zeros(E, device=dev)

  def _mocap_ids(self, env, names, tag: str, kind: str) -> torch.Tensor:
    """Mocap ids of bodies `<entity>/h<i>_<tag><name>`, rows `i * len(names) + n`."""
    model = env.sim.mj_model
    ids = []
    for i in range(self.M):
      for name in names:
        full = f"{self.cfg.entity_name}/h{i:02d}_{tag}{name}"
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full)
        if body < 0:
          raise ValueError(f"missing {kind} body {full}")
        ids.append(int(model.body_mocapid[body]))
    return torch.tensor(ids, device=self.device)

  def _init_rigid(self, env) -> None:
    """Rotation tables, piece bodies and the ray target for rigid mesh humans."""
    from mjlab.sensor import RayCastSensor

    from crowdmotion.core.paths import default_graph_rotations
    from crowdmotion.sim.body import RigidBody, core_pieces
    from crowdmotion.sim.rays import RigidRays

    g, dev, E, M = self.graph, self.device, self.num_envs, self.M
    self.rotations = load_rotations(self.cfg.rotations_path or default_graph_rotations(), g)
    pieces = core_pieces()
    self.body = RigidBody(list(pieces), dev)
    self.piece_mocap = self._mocap_ids(env, self.body.names, MESH_TAG, "rigid piece")
    model = env.sim.mj_model
    geom_ids = np.array([[int(model.body_geomadr[mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, f"{self.cfg.entity_name}/h{i:02d}_{MESH_TAG}{name}")])
      for name in self.body.names] for i in range(M)])
    self.rots = torch.zeros(E, M, self.L + 1, g.joints_n, 4, device=dev)  # world (Y-up) wxyz, like `frames`
    self.rots[..., 0] = 1
    self.swing_deg = torch.zeros(0, device=dev)  # last pose write: turn applied per single-child joint
    # The pieces are visual geoms: a sensor that traced their group natively would
    # pull every piece mesh into mujoco_warp's BVH (the slow mesh path) and see the
    # humans twice.
    visual = {int(model.geom_group[g]) for g in geom_ids.ravel()}
    for name, s in env.scene.sensors.items():
      groups = s.include_geom_groups if isinstance(s, RayCastSensor) else ()
      if groups is None or visual & set(groups):
        raise ValueError(
          f"ray-cast sensor {name!r} traces geom groups {groups}, which include the rigid humans' visual "
          f"group(s) {sorted(visual)}; set include_geom_groups without them (the humans reach sensors "
          f"that include lidar_group {self.cfg.lidar_group} through a ray target)"
        )
    sensors = [s for s in env.scene.sensors.values() if isinstance(s, RayCastSensor)
               and self.cfg.lidar_group in s.include_geom_groups]
    max_range = max((s.cfg.max_distance for s in sensors), default=100.0)
    self.rays = RigidRays(pieces, E, M, geom_ids, dev, max_range)
    self.rays.set_skeleton(self.body.bone, single_children(g.meta["parents"]), self.rotations.offsets,
                           self.piece_mocap.view(M, -1), YUP_TO_MUJOCO)
    self.rays.set_enabled(self.geometry == "rigid")
    for sensor in sensors:
      sensor.add_ray_target(self.rays)
    env.sim.create_graph()  # the sense graph was captured before command terms existed

  # Geometry switch -----------------------------------------------------------
  def set_geometry(self, name: str) -> None:
    """Queue a switch of the active body geometry (any thread; applied on the next step)."""
    if name not in self.geometries:
      raise ValueError(f"geometry {name!r} is not built; built: {self.geometries}")
    self._pending_geometry.put(name)

  def _apply_geometry(self) -> None:
    while True:
      try:
        name = self._pending_geometry.get_nowait()
      except Empty:
        return
      if name == self.geometry:
        continue
      self.geometry = name
      parked = self.mocap if name == "rigid" else self.piece_mocap
      self._park(parked, len(parked) // self.M)
      if self.rays is not None:
        self.rays.set_enabled(name == "rigid")
      self._write_poses(torch.arange(self.num_envs, device=self.device))

  def _park(self, mocap: torch.Tensor, per_human: int) -> None:
    """Move mocap bodies below the floor, human `i` at depth `5 + i` (their spec pose)."""
    data = self._env.sim.data
    depth = 5.0 + torch.arange(self.M, device=self.device).repeat_interleave(per_human)
    pos = torch.zeros(len(mocap), 3, device=self.device)
    pos[:, 2] = -depth
    quat = torch.zeros(len(mocap), 4, device=self.device)
    quat[:, 0] = 1
    data.mocap_pos[:, mocap] = pos
    data.mocap_quat[:, mocap] = quat

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def state(self):
    """Graph state of every human, rows `env * M + i`."""
    return self.controller.state

  @property
  def vel(self) -> torch.Tensor:
    """`[E, M, 2]` root velocity estimate used for planning and steering."""
    return self.controller.kin.vel.view(self.num_envs, self.M, 2)

  # Reset ---------------------------------------------------------------------
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.live_tuning.apply_pending()
    env_ids = env_ids.to(self.device)
    e, M = len(env_ids), self.M
    xz, heading = self.behavior.reset(env_ids)
    rows = self.controller.rows(env_ids)
    seeds = human_seeds(self.cfg.seed, rows, "graphcrowd", episode=self.resets[env_ids].repeat_interleave(M))
    sub = self.controller.reset(seeds, xz.reshape(-1, 2), heading.reshape(-1), groups=env_ids)
    self.frames[env_ids] = self.source.current_joints(sub).view(e, M, -1, 3)[:, :, None]
    if self.rotations is not None:
      rot = quat_mul(quat_yaw(sub.heading)[:, None], self.rotations.pred_rot[sub.boundary].float())
      self.rots[env_ids] = rot.view(e, M, 1, -1, 4)
    self.clock[env_ids] = 0.0
    self.resets[env_ids] += 1
    self._advance(env_ids)
    self.edge_t[env_ids] = 0.0

  # Stepping ------------------------------------------------------------------
  def _advance(self, env_ids: torch.Tensor) -> None:
    """Play the next edge for every human in `env_ids`."""
    self.live_tuning.apply_pending()
    e, M, L = len(env_ids), self.M, self.L
    step = self.controller.advance(self.clock[env_ids], groups=env_ids)
    self.frames[env_ids, :, 0] = self.frames[env_ids, :, -1]
    self.frames[env_ids, :, 1:] = step.motion.joints.view(e, M, L, -1, 3)
    if self.rotations is not None:
      # World rotation = edge heading turn * saved body-local rotation; a held human
      # (edge -1) keeps its last rotation, as the runtime keeps its pose.
      edge = step.motion.extras["edge"]
      rot = quat_mul(quat_yaw(step.motion.heading)[..., None, :],
                     self.rotations.edge_rot[edge.clamp(min=0)].float())  # [e*M, L, J, 4]
      last = self.rots[env_ids, :, -1].reshape(e * M, 1, -1, 4)
      rot = torch.where((edge < 0)[:, None, None, None], last, rot)
      self.rots[env_ids, :, 0] = self.rots[env_ids, :, -1]
      self.rots[env_ids, :, 1:] = rot.view(e, M, L, -1, 4)
    self._command[env_ids] = torch.stack((step.speed.view(e, M), step.heading.view(e, M)), -1)
    self.metrics["natural_successor"][env_ids] = step.motion.extras["natural"].view(e, M).float().mean(1)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    self._apply_geometry()
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
      dt = self._env.step_dt
      self.clock += dt
      self.edge_t += dt
    due = env_ids[self.edge_t[env_ids] >= self.edge_s - 1e-6]
    while len(due):
      self._advance(due)
      self.edge_t[due] -= self.edge_s
      due = due[self.edge_t[due] >= self.edge_s - 1e-6]
    # Commands run after the env's forward(), so sensors and viewers see these
    # poses one step (20 ms) late. That lag is invisible here, and skipping an
    # extra forward() matters on CPU, where each call costs tens of ms.
    self._write_poses(env_ids)

  def _write_poses(self, env_ids: torch.Tensor) -> None:
    f = (self.edge_t[env_ids] * self.fps).clamp(0, self.L)
    lo = f.floor().long().clamp(max=self.L - 1)
    a = (f - lo)[:, None, None, None]
    rows = torch.arange(len(env_ids), device=self.device)
    frames = self.frames[env_ids]
    joints = frames[rows, :, lo] * (1 - a) + frames[rows, :, lo + 1] * a  # [e, M, J, 3]
    e = len(env_ids)
    data = self._env.sim.data
    if self.geometry == "rigid":
      # One fused Warp pass: blended rotations swung onto the joints, mocap poses, and
      # the ray target's transforms and boxes (`sim.rays.RigidRays.pose`).
      turn = self.rays.pose(joints, self.rots, env_ids, lo, f - lo, data.mocap_pos.wp_array,
                            data.mocap_quat.wp_array)
      self.swing_deg = torch.rad2deg(turn)
      return
    pos, quat = self.skeleton.poses(motion_to_mujoco(joints.flatten(0, 1)))
    data.mocap_pos[env_ids[:, None], self.mocap[None]] = pos.view(e, -1, 3)
    data.mocap_quat[env_ids[:, None], self.mocap[None]] = quat.view(e, -1, 4)

  def _update_metrics(self) -> None:
    pass

  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
    if len(self.geometries) > 1:
      with server.gui.add_folder(f"{name.capitalize()} body"):
        server.gui.add_markdown("Body geometry for display and LiDAR; applies on the next step.")
        labels = {GEOMETRY_LABELS[g]: g for g in self.geometries}
        dropdown = server.gui.add_dropdown("Geometry", options=tuple(labels),
                                           initial_value=GEOMETRY_LABELS[self.geometry])

        @dropdown.on_update
        def _(_event) -> None:
          self.set_geometry(labels[dropdown.value])

    self.live_tuning.create_gui(server, name)

  def _debug_vis_impl(self, visualizer) -> None:
    """Goal markers and commanded velocity arrows (MuJoCo frame)."""
    M = self.M
    for env in visualizer.get_env_indices(self.num_envs):
      root = self.state.root_xz.view(-1, M, 2)[env].cpu()
      goal = self.behavior.goal[env].cpu()
      cmd = self._command[env].cpu()
      for i in range(M):
        color = COLORS[i % len(COLORS)]
        gx, gy = float(goal[i, 0]), -float(goal[i, 1])
        visualizer.add_cylinder(
          start=np.array([gx, gy, 0.0]), end=np.array([gx, gy, 0.02]), radius=0.12,
          color=(color[0], color[1], color[2], 0.6),
        )
        speed, head = float(cmd[i, 0]), float(cmd[i, 1])
        if speed > 0.05:
          x, y = float(root[i, 0]), -float(root[i, 1])
          dx, dy = speed * math.sin(head), -speed * math.cos(head)
          visualizer.add_arrow(
            start=np.array([x, y, 2.0]), end=np.array([x + dx, y + dy, 2.0]), color=color, width=0.02
          )
