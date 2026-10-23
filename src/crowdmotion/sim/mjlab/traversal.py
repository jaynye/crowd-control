"""Crowd traversal: a G1 walks to goals through graph-driven humans that it sees by LiDAR.

The training environment for the robot policy. It extends mjlab's flat G1 velocity
task (`Mjlab-Velocity-Flat-Unitree-G1`) with:

- the encounter design's walled plaza and `humans` rigid mesh humans (`GraphCrowd`,
  configured from a behavior config such as `configs/behavior/encounter_turns.json`);
- `CrowdGoalCommand`: the velocity command (`twist`: vx, vy, wz in the base frame)
  toward random goals in the plaza, recomputed every step from the robot's pose. The
  velocity task's rewards apply unchanged, and a detour around a human shows up as
  tracking error that the crowd costs must outweigh. It also places the robot, after
  the crowd resets, at the candidate start farthest from every human;
- a head LiDAR (`lidar` site on `torso_link`; provisional inverted MID-360 field). The
  policy sees its ranges as per-azimuth-bin minima over a short history
  (`LidarScan`), with floor returns removed. Humans reach it through the crowd's ray
  target (`GraphCrowdCfg.lidar_group`);
- robot-human clearance between robot spheres (`ROBOT_SPHERES`) and human joint spheres
  (`sim.capsules` radii). Rigid pieces have no MuJoCo contacts, so proximity and contact
  costs and the collision termination use these distances;
- privileged observations: the nearest humans' states and the robot's plaza pose.

Observation groups: `actor` is the deployable LiDAR policy's input; `teacher` replaces
the LiDAR with the privileged observations, for a teacher trained first and distilled
into the LiDAR policy; `critic` has both. Three tasks share the env: `TASK_ID` trains
the LiDAR policy directly with PPO, `TEACHER_TASK_ID` trains the privileged teacher with
PPO, and `STUDENT_TASK_ID` distills a teacher checkpoint into the LiDAR policy
(RSL-RL `Distillation`: the student acts, and its actions are regressed onto the
teacher's).

Humans do not perceive the robot: toward it, every human is stubborn.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg
from mjlab.sensor import LidarPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, wrap_to_pi
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from crowdmotion.behavior.encounter import EncounterCfg
from crowdmotion.core.geometry import motion_to_mujoco
from crowdmotion.sim.capsules import BONE_RADIUS, HEAD_RADIUS
from crowdmotion.sim.mjlab.command import MESH_TAG, GraphCrowd, GraphCrowdCfg
from crowdmotion.sim.scene import COLORS

TASK_ID = "Crowd-Traversal-Flat-Unitree-G1"
TEACHER_TASK_ID = "Crowd-Traversal-Teacher-Flat-Unitree-G1"
STUDENT_TASK_ID = "Crowd-Traversal-Student-Flat-Unitree-G1"
POLICIES = {"lidar": TASK_ID, "teacher": TEACHER_TASK_ID, "student": STUDENT_TASK_ID}
HUMANS_GROUP = 4  # ray-target group of the humans: only the LiDAR includes it
# Provisional LiDAR mount on torso_link: about head height, the MID-360's asymmetric
# (-7, 52) degree field mounted inverted so that it looks down at people and the floor.
# Replace with the hardware calibration.
LIDAR_POS = (0.05, 0.0, 0.42)
LIDAR_ELEVATION = (-52.0, 7.0)
# (body, offset in the body frame, radius): spheres covering the G1 for clearance.
ROBOT_SPHERES = (
  ("pelvis", (0.0, 0.0, 0.0), 0.14),
  ("torso_link", (0.0, 0.0, 0.15), 0.15),
  ("torso_link", (0.0, 0.0, 0.42), 0.11),
  ("left_shoulder_roll_link", (0.0, 0.0, 0.0), 0.07),
  ("right_shoulder_roll_link", (0.0, 0.0, 0.0), 0.07),
  ("left_elbow_link", (0.0, 0.0, 0.0), 0.06),
  ("right_elbow_link", (0.0, 0.0, 0.0), 0.06),
  ("left_wrist_yaw_link", (0.06, 0.0, 0.0), 0.05),
  ("right_wrist_yaw_link", (0.06, 0.0, 0.0), 0.05),
  ("left_knee_link", (0.0, 0.0, 0.0), 0.07),
  ("right_knee_link", (0.0, 0.0, 0.0), 0.07),
  ("left_ankle_roll_link", (0.03, 0.0, -0.02), 0.06),
  ("right_ankle_roll_link", (0.03, 0.0, -0.02), 0.06),
)


def human_joint_radii(names) -> torch.Tensor:
  """Capsule radius at each human joint: the bone ending there (as the evaluator's
  joint-sphere clearance); Hips takes the Spine radius, unlisted joints 3 cm."""
  return torch.tensor([HEAD_RADIUS if n == "Head" else BONE_RADIUS.get("Spine" if n == "Hips" else n, 0.03)
                       for n in names])


# Crowd geometry --------------------------------------------------------------------
def human_joints(crowd: GraphCrowd) -> torch.Tensor:
  """Every human's joints now, `[E, M, J, 3]` in MuJoCo world coordinates (the poses
  `GraphCrowd` last wrote)."""
  f = (crowd.edge_t * crowd.fps).clamp(0, crowd.L)
  lo = f.floor().long().clamp(max=crowd.L - 1)
  a = (f - lo)[:, None, None, None]
  rows = torch.arange(crowd.num_envs, device=crowd.device)
  joints = crowd.frames[rows, :, lo] * (1 - a) + crowd.frames[rows, :, lo + 1] * a
  return motion_to_mujoco(joints)


class _Clearance:
  """Robot-sphere to human-joint-sphere clearance, computed at most once per env step."""

  def __init__(self, env, robot_name: str, crowd_name: str):
    self.robot = env.scene[robot_name]
    self.crowd: GraphCrowd = env.command_manager.get_term(crowd_name)
    names = [b for b, _, _ in ROBOT_SPHERES]
    unique = list(dict.fromkeys(names))
    ids, found = self.robot.find_bodies(unique, preserve_order=True)
    if list(found) != unique:
      raise ValueError(f"robot bodies {unique} not found: {found}")
    index = dict(zip(found, ids, strict=True))
    dev = env.device
    self.body_ids = torch.tensor([index[n] for n in names], device=dev)
    self.offsets = torch.tensor([o for _, o, _ in ROBOT_SPHERES], device=dev)
    self.robot_r = torch.tensor([r for _, _, r in ROBOT_SPHERES], device=dev)
    self.human_r = human_joint_radii(self.crowd.graph.meta["joint_names"]).to(dev)
    self.key, self.value = None, None

  def __call__(self, env) -> tuple[torch.Tensor, torch.Tensor]:
    """`(min clearance [E], per-human clearance [E, M])` in meters; negative = overlap."""
    key = (env.common_step_counter, int(self.crowd.resets.sum()))
    if key != self.key:
      data = self.robot.data
      quat = data.body_link_quat_w[:, self.body_ids]
      points = data.body_link_pos_w[:, self.body_ids] + quat_apply(quat, self.offsets.expand_as(quat[..., :3]))
      joints = human_joints(self.crowd)  # [E, M, J, 3]
      E, M, J, _ = joints.shape
      d = torch.cdist(points, joints.reshape(E, M * J, 3)).view(E, -1, M, J)
      clear = d - self.robot_r[None, :, None, None] - self.human_r[None, None, None, :]
      per_human = clear.amin((1, 3))
      self.key, self.value = key, (per_human.amin(-1), per_human)
    return self.value


def _clearance(env, robot_name: str = "robot", crowd_name: str = "crowd") -> _Clearance:
  cache = getattr(env, "_crowd_clearance", None)
  if cache is None:
    cache = env._crowd_clearance = _Clearance(env, robot_name, crowd_name)
  return cache


# Rewards and terminations ------------------------------------------------------------
def crowd_proximity(env, margin_m: float = 0.4) -> torch.Tensor:
  """Sum over humans of how far each is inside `margin_m` of the robot, as a fraction."""
  _, per_human = _clearance(env)(env)
  return ((margin_m - per_human) / margin_m).clamp(0.0, 1.0).sum(-1)


def crowd_contact(env) -> torch.Tensor:
  """1 while any robot sphere overlaps any human joint sphere."""
  closest, _ = _clearance(env)(env)
  return (closest < 0).float()


def crowd_collision(env, penetration_m: float = 0.05) -> torch.Tensor:
  """Termination: the robot overlaps a human by more than `penetration_m`."""
  closest, _ = _clearance(env)(env)
  return closest < -penetration_m


def wall_proximity(env, half_x: float, half_y: float, margin_m: float = 0.5,
                   asset_name: str = "robot") -> torch.Tensor:
  """How far the robot's root is inside `margin_m` of the plaza walls, as a fraction."""
  pos = env.scene[asset_name].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  gap = torch.minimum(half_x - pos[:, 0].abs(), half_y - pos[:, 1].abs())
  return ((margin_m - gap) / margin_m).clamp(0.0, 1.0)


# Observations --------------------------------------------------------------------------
class LidarScan(ManagerTermBase):
  """LiDAR ranges for the policy: per azimuth bin, the nearest horizontal range of any
  return above `floor_m` (floor returns and misses read as `max_range_m`), over
  `frames` scans `stride` env steps apart, normalized to [0, 1]. Bins are in the
  sensor frame, so they turn with the body."""

  def __init__(self, cfg: ObservationTermCfg, env):
    super().__init__(env)
    p = cfg.params
    pattern = env.scene[p["sensor_name"]].cfg.pattern
    assert isinstance(pattern, LidarPatternCfg)
    self.az, self.el = pattern.azimuth_samples, pattern.elevation_samples
    if self.az % p["bins"]:
      raise ValueError(f"azimuth samples {self.az} must divide into {p['bins']} bins")
    self.history = torch.ones(env.num_envs, (p["frames"] - 1) * p["stride"] + 1, p["bins"], device=env.device)
    self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None) -> None:
    self.fresh[slice(None) if env_ids is None else env_ids] = True

  def __call__(self, env, sensor_name: str, bins: int, frames: int, stride: int, max_range_m: float,
               floor_m: float) -> torch.Tensor:
    data = env.scene[sensor_name].data
    E = env.num_envs
    hit = (data.distances > 0) & (data.hit_pos_w[..., 2] > floor_m)
    planar = (data.hit_pos_w[..., :2] - data.pos_w[:, None, :2]).norm(dim=-1)
    ranges = torch.where(hit, planar, max_range_m).clamp(max=max_range_m)
    scan = ranges.view(E, self.el, self.az).amin(1).view(E, bins, -1).amin(-1) / max_range_m
    self.history = torch.where(self.fresh[:, None, None], scan[:, None], self.history.roll(1, dims=1))
    self.history[:, 0] = scan
    self.fresh[:] = False
    return self.history[:, ::stride].reshape(E, -1)


def crowd_nearest(env, k: int = 4, max_range_m: float = 6.0, crowd_name: str = "crowd",
                  asset_name: str = "robot") -> torch.Tensor:
  """Privileged (critic): the `k` nearest humans' hip positions and velocities in the
  robot's yaw frame, and their clearance, ordered by clearance; far humans read as
  `max_range_m` ahead and still."""
  crowd: GraphCrowd = env.command_manager.get_term(crowd_name)
  robot = env.scene[asset_name]
  hips = human_joints(crowd)[:, :, 0, :2]  # [E, M, 2]
  vel_xz = crowd.vel  # library XZ
  vel = torch.stack((vel_xz[..., 0], -vel_xz[..., 1]), -1)
  rel = hips - robot.data.root_link_pos_w[:, None, :2]
  rel_v = vel - robot.data.root_link_lin_vel_w[:, None, :2]
  h = robot.data.heading_w[:, None]
  c, s = torch.cos(h), torch.sin(h)
  rotate = lambda v: torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]), -1)  # noqa: E731
  _, per_human = _clearance(env)(env)
  order = per_human.argsort(-1)[:, :k]
  take = lambda x: x.gather(1, order[..., None].expand(-1, -1, x.shape[-1]))  # noqa: E731
  feats = torch.cat((take(rotate(rel)), take(rotate(rel_v)), per_human.gather(1, order)[..., None]), -1)
  far = feats[..., 4:5] > max_range_m
  default = torch.tensor([max_range_m, 0.0, 0.0, 0.0, max_range_m], device=env.device)
  feats = torch.where(far, default, feats)
  if feats.shape[1] < k:
    feats = torch.cat((feats, default.expand(len(feats), k - feats.shape[1], 5)), 1)
  return feats.reshape(len(feats), -1)


def plaza_pose(env, half_x: float, half_y: float, asset_name: str = "robot") -> torch.Tensor:
  """Privileged: the robot's position in the plaza as a fraction of its half sizes, and
  the sine and cosine of its heading (the walls follow from these)."""
  robot = env.scene[asset_name]
  pos = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  heading = robot.data.heading_w
  return torch.stack((pos[:, 0] / half_x, pos[:, 1] / half_y, torch.sin(heading), torch.cos(heading)), -1)


# Command -------------------------------------------------------------------------------
class CrowdGoalCommand(UniformVelocityCommand):
  """Velocity toward random goals in the plaza, recomputed each step (see the module
  docstring). Standing envs (`rel_standing_envs`) command zero and must still keep
  clear of the humans."""

  cfg: "CrowdGoalCommandCfg"

  def __init__(self, cfg: "CrowdGoalCommandCfg", env):
    super().__init__(cfg, env)
    self.goal_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.speed = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goals_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)

  def _box(self, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    hi = torch.tensor([self.cfg.half_x - margin, self.cfg.half_y - margin], device=self.device)
    return -hi, hi

  def _origins(self, env_ids) -> torch.Tensor:
    return self._env.scene.env_origins[env_ids, :2]

  def _sample_goals(self, env_ids: torch.Tensor) -> None:
    n, C = len(env_ids), 16
    lo, hi = self._box(self.cfg.goal_margin_m)
    cand = lo + (hi - lo) * torch.rand(n, C, 2, device=self.device)
    pos = self.robot.data.root_link_pos_w[env_ids, :2] - self._origins(env_ids)
    dist = (cand - pos[:, None]).norm(dim=-1)
    pick = torch.where((dist >= self.cfg.min_goal_m).any(1), (dist >= self.cfg.min_goal_m).float().argmax(1),
                       dist.argmax(1))
    self.goal_w[env_ids] = cand[torch.arange(n, device=self.device), pick] + self._origins(env_ids)
    self.speed[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.speed_range)

  def reset(self, env_ids):
    """Place the reset robots at the start farthest from the (already reset) humans."""
    if isinstance(env_ids, torch.Tensor) and len(env_ids):
      crowd: GraphCrowd = self._env.command_manager.get_term(self.cfg.crowd_name)
      n, C = len(env_ids), self.cfg.spawn_candidates
      lo, hi = self._box(self.cfg.spawn_margin_m)
      cand = lo + (hi - lo) * torch.rand(n, C, 2, device=self.device)
      hips = human_joints(crowd)[env_ids, :, 0, :2] - self._origins(env_ids)[:, None]
      gap = (cand[:, :, None] - hips[:, None]).norm(dim=-1).amin(-1)
      xy = cand[torch.arange(n, device=self.device), gap.argmax(1)] + self._origins(env_ids)
      yaw = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
      pose = torch.zeros(n, 7, device=self.device)
      pose[:, :2] = xy
      pose[:, 2] = self.robot.data.default_root_state[env_ids, 2]
      zero = torch.zeros_like(yaw)
      pose[:, 3:] = quat_from_euler_xyz(zero, zero, yaw)
      self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._sample_goals(env_ids)
    r = torch.empty(len(env_ids), device=self.device)
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.is_heading_env[env_ids] = True
    self.is_world_env[env_ids] = True
    self.is_forward_env[env_ids] = False

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    pos = self.robot.data.root_link_pos_w[:, :2]
    to_goal = self.goal_w - pos
    dist = to_goal.norm(dim=-1)
    arrived = (dist < self.cfg.arrive_m) & ~self.is_standing_env
    if arrived.any():
      ids = arrived.nonzero(as_tuple=False).flatten()
      self.metrics["goals_reached"][ids] += 1.0
      self._sample_goals(ids)
      to_goal = self.goal_w - pos
      dist = to_goal.norm(dim=-1)
    direction = to_goal / dist[:, None].clamp(min=1e-6)
    speed = self.speed * (dist / self.cfg.slow_m).clamp(max=1.0)
    self.vel_command_w[:, :2] = direction * speed[:, None]
    self.heading_target = torch.atan2(direction[:, 1], direction[:, 0])
    heading = self.robot.data.heading_w
    self.heading_error = wrap_to_pi(self.heading_target - heading)
    c, s = torch.cos(heading), torch.sin(heading)
    vx, vy = self.vel_command_w[:, 0], self.vel_command_w[:, 1]
    self.vel_command_b[:, 0] = c * vx + s * vy
    self.vel_command_b[:, 1] = -s * vx + c * vy
    self.vel_command_b[:, 2] = (self.cfg.heading_control_stiffness * self.heading_error).clamp(
      *self.cfg.ranges.ang_vel_z)
    standing = self.is_standing_env
    self.vel_command_b[standing] = 0.0
    self.vel_command_w[standing] = 0.0

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["goal_distance"] = (self.goal_w - self.robot.data.root_link_pos_w[:, :2]).norm(dim=-1)


@dataclass(kw_only=True)
class CrowdGoalCommandCfg(UniformVelocityCommandCfg):
  crowd_name: str = "crowd"
  half_x: float = 7.0  # plaza half sizes in MuJoCo x and y (library x and z)
  half_y: float = 5.0
  goal_margin_m: float = 1.5
  min_goal_m: float = 3.0
  arrive_m: float = 0.5
  slow_m: float = 1.0  # speed ramps down within this distance of the goal
  speed_range: tuple[float, float] = (0.3, 1.2)
  spawn_candidates: int = 32
  spawn_margin_m: float = 1.2

  def build(self, env) -> CrowdGoalCommand:
    return CrowdGoalCommand(self, env)


# Scene -------------------------------------------------------------------------------------
def _walls_spec(half_x: float, half_y: float):
  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="walls")
    t, h = 0.1, 1.0
    for name, pos, size in (
      ("north", (0, half_y + t, h), (half_x + 2 * t, t, h)),
      ("south", (0, -half_y - t, h), (half_x + 2 * t, t, h)),
      ("east", (half_x + t, 0, h), (t, half_y, h)),
      ("west", (-half_x - t, 0, h), (t, half_y, h)),
    ):
      body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, group=0,
                    rgba=(0.55, 0.55, 0.6, 1.0))
    return spec
  return spec_fn


def _crowd_spec(humans: int):
  def spec_fn() -> mujoco.MjSpec:
    from crowdmotion.sim.body import RigidBody, core_pieces

    spec = mujoco.MjSpec()
    RigidBody(list(core_pieces())).add_to_spec(spec, humans, COLORS, group=1, prefix="core", tag=MESH_TAG)
    return spec
  return spec_fn


def _robot_with_lidar(robot: EntityCfg) -> EntityCfg:
  inner = robot.spec_fn

  def spec_fn() -> mujoco.MjSpec:
    spec = inner()
    spec.body("torso_link").add_site(name="lidar", pos=LIDAR_POS, size=(0.02, 0, 0))
    return spec
  return replace(robot, spec_fn=spec_fn)


def load_behavior(path: str | Path) -> tuple[dict, EncounterCfg]:
  """`(graph_overrides, encounter)` from an encounter-design behavior config."""
  raw = json.loads(Path(path).read_text())
  if raw.get("design") != "encounter":
    raise ValueError(f"{path}: the traversal env uses the encounter design")
  return dict(raw.get("graph_overrides", {})), EncounterCfg.from_dict(raw.get("encounter", {}))


def make_traversal_env_cfg(
  graph_path: str,
  behavior_path: str,
  rotations_path: str | None = None,
  humans: int = 8,
  seed: int = 0,
  play: bool = False,
  lidar_azimuth: int = 144,
  lidar_elevation: int = 12,
  lidar_bins: int = 72,
  lidar_frames: int = 4,
  lidar_stride: int = 5,
  lidar_range_m: float = 6.0,
  episode_s: float = 20.0,
  teacher_neighbors: int = 6,
) -> ManagerBasedRlEnvCfg:
  """The crowd traversal env (module docstring). `behavior_path`: an encounter-design
  behavior config (its `graph_overrides` and `encounter` sections; the scene size comes
  from the encounter section)."""
  from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

  overrides, encounter = load_behavior(behavior_path)
  encounter = replace(encounter, obstacles=())  # the env draws no steering obstacles
  hx, hy = encounter.half_x, encounter.half_z
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.seed = seed
  cfg.episode_length_s = 1e9 if play else episode_s
  cfg.scene.env_spacing = 0.0  # every world holds its own plaza at the origin
  cfg.scene.entities["robot"] = _robot_with_lidar(cfg.scene.entities["robot"])
  cfg.scene.entities["walls"] = EntityCfg(spec_fn=_walls_spec(hx, hy))
  cfg.scene.entities["crowd"] = EntityCfg(spec_fn=_crowd_spec(humans))
  lidar = RayCastSensorCfg(
    name="lidar",
    frame=ObjRef(type="site", name="lidar", entity="robot"),
    pattern=LidarPatternCfg(azimuth_samples=lidar_azimuth, elevation_samples=lidar_elevation,
                            elevation_range=LIDAR_ELEVATION),
    ray_alignment="base",
    min_distance=0.1,
    max_distance=lidar_range_m + 2.0,
    exclude_parent_body=True,
    include_geom_groups=(0, HUMANS_GROUP),  # floor, walls and (through the ray target) humans
    debug_vis=play,
  )
  cfg.scene.sensors = (*(cfg.scene.sensors or ()), lidar)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  cfg.commands = {
    "crowd": GraphCrowdCfg(
      graph_path=str(graph_path), humans=humans, design="encounter", encounter=encounter, seed=seed,
      graph_overrides=overrides, human_geometry=("rigid",), rotations_path=rotations_path,
      lidar_group=HUMANS_GROUP, debug_vis=False,
    ),
    "twist": CrowdGoalCommandCfg(
      entity_name="robot", resampling_time_range=(10.0, 15.0), rel_standing_envs=0.1,
      heading_command=True, heading_control_stiffness=1.0, debug_vis=True, viz=twist.viz,
      ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.2, 1.2), lin_vel_y=(-1.2, 1.2),
                                              ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)),
      half_x=hx, half_y=hy,
    ),
  }
  cfg.curriculum.pop("command_vel", None)
  cfg.events["reset_base"].params["pose_range"] = {"z": (0.01, 0.05)}  # x, y, yaw: CrowdGoalCommand.reset
  if play:
    cfg.events.pop("push_robot", None)

  scan = dict(sensor_name="lidar", bins=lidar_bins, frames=lidar_frames, stride=lidar_stride,
              max_range_m=lidar_range_m, floor_m=0.1)
  actor = cfg.observations["actor"]
  privileged = {
    "crowd_nearest": ObservationTermCfg(func=crowd_nearest, params={"k": min(humans, teacher_neighbors)}),
    "plaza_pose": ObservationTermCfg(func=plaza_pose, params={"half_x": hx, "half_y": hy}),
  }
  cfg.observations["teacher"] = ObservationGroupCfg(
    terms={**actor.terms, **privileged}, concatenate_terms=True, enable_corruption=actor.enable_corruption)
  for group in ("actor", "critic"):
    cfg.observations[group].terms["lidar"] = ObservationTermCfg(
      func=LidarScan, params=scan, noise=Unoise(n_min=-0.01, n_max=0.01) if group == "actor" else None)
  cfg.observations["critic"].terms.update(privileged)

  cfg.rewards["crowd_proximity"] = RewardTermCfg(func=crowd_proximity, weight=-1.0, params={"margin_m": 0.4})
  cfg.rewards["crowd_contact"] = RewardTermCfg(func=crowd_contact, weight=-10.0)
  cfg.rewards["wall_proximity"] = RewardTermCfg(func=wall_proximity, weight=-1.0,
                                                params={"half_x": hx, "half_y": hy})
  cfg.rewards["track_angular_velocity"].weight = 1.0  # turning away from people costs less than detours
  if not play:
    cfg.terminations["crowd_collision"] = TerminationTermCfg(func=crowd_collision, params={"penetration_m": 0.05})
  return cfg


def traversal_rl_cfg(max_iterations: int = 10_000, teacher: bool = False):
  """PPO settings of the G1 velocity task; the teacher's actor reads the `teacher` group."""
  from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

  rl = unitree_g1_ppo_runner_cfg()
  rl.experiment_name = "g1_crowd_traversal_teacher" if teacher else "g1_crowd_traversal"
  if teacher:
    rl.obs_groups = {"actor": ("teacher",), "critic": ("critic",)}
  rl.logger = "tensorboard"
  rl.save_interval = 100
  rl.max_iterations = max_iterations
  return rl


@dataclass
class DistillationAlgorithmCfg:
  class_name: str = "Distillation"
  num_learning_epochs: int = 2
  gradient_length: int = 1
  learning_rate: float = 5e-4
  max_grad_norm: float | None = 1.0
  loss_type: str = "mse"


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  """RSL-RL `DistillationRunner` settings. The teacher must match the PPO actor it is
  loaded from; the student acts with a small fixed Gaussian noise (DAgger-style
  rollouts) and learns the teacher's mean actions."""

  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(
    hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"}))
  teacher: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(
    hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}))
  algorithm: DistillationAlgorithmCfg = field(default_factory=DistillationAlgorithmCfg)


def traversal_distill_cfg(max_iterations: int = 3_000) -> DistillationRunnerCfg:
  return DistillationRunnerCfg(
    obs_groups={"student": ("actor",), "teacher": ("teacher",)},
    experiment_name="g1_crowd_traversal_student", logger="tensorboard", save_interval=100,
    max_iterations=max_iterations, num_steps_per_env=24,
  )


def _distillation_runner():
  from rsl_rl.runners import DistillationRunner

  class TraversalDistillationRunner(DistillationRunner):
    """`DistillationRunner` with mjlab's cleanup of unset model options."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
      for key in ("student", "teacher"):
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
      super().__init__(env, train_cfg, log_dir, device)

  return TraversalDistillationRunner


def register(graph_path: str, behavior_path: str, rotations_path: str | None = None, humans: int = 8,
             policy: str = "lidar") -> str:
  """Register a traversal task with mjlab's registry (training and play configs) and
  return its id. `policy` (`POLICIES`): `lidar` (PPO on LiDAR), `teacher` (PPO on
  privileged observations) or `student` (distillation of a teacher into LiDAR)."""
  from mjlab.tasks.registry import list_tasks, register_mjlab_task
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  task_id = POLICIES[policy]
  if task_id not in list_tasks():
    student = policy == "student"
    register_mjlab_task(
      task_id=task_id,
      env_cfg=make_traversal_env_cfg(graph_path, behavior_path, rotations_path, humans),
      play_env_cfg=make_traversal_env_cfg(graph_path, behavior_path, rotations_path, humans, play=True),
      rl_cfg=traversal_distill_cfg() if student else traversal_rl_cfg(teacher=policy == "teacher"),
      runner_cls=_distillation_runner() if student else VelocityOnPolicyRunner,
    )
  return task_id
