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

Observation groups: `actor` is the deployable LiDAR policy's input; `teacher` adds the
privileged observations to it (`teacher_lidar`, the default; without it the teacher has
no LiDAR and learned to collide about twice as often), for a teacher trained first and
distilled into the LiDAR policy; `critic` has both. Three tasks share the env: `TASK_ID` trains
the LiDAR policy directly with PPO, `TEACHER_TASK_ID` trains the privileged teacher with
PPO, and `STUDENT_TASK_ID` distills a teacher checkpoint into the LiDAR policy
(RSL-RL `Distillation`: the student acts, and its actions are regressed onto the
teacher's).

Terminal penalties `collision_end` and `fall_end` (weight 0, off by default) price an
episode that ends in a collision or a fall; `penalty_ramp` brings them in gradually.
`no_avoidance` builds b0, the non-avoidant baseline, which learns nothing about the
humans. `CrowdGoalCommandCfg.paired_seed` pairs goals across policies for evaluation
(`crowd eval traversal`).

With `perceive_robot` (the default) the humans perceive the robot as an external agent
(`GraphCrowdCfg.perceive_entity`): each is oblivious to it, stubborn toward it (stops,
never gives way) or cooperative, by the behavior config's `robot_*` encounter settings.
Without it every human is oblivious, so the robot must avoid all of them alone.
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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
from crowdmotion.core.rng import hash_uniform, human_seeds
from crowdmotion.sim.capsules import BONE_RADIUS, HEAD_RADIUS
from crowdmotion.sim.mjlab.command import MESH_TAG, GraphCrowd, GraphCrowdCfg
from crowdmotion.sim.mjlab.navigation import NavigationCfg
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


def goal_heading(env, command_name: str = "twist") -> torch.Tensor:
  """(1 + cos(heading error)) / 2 toward the goal command's heading target: 1 facing the
  goal, 0 facing away, with a gradient through ±180° where the turn command flips.
  Zero in standing envs."""
  term: CrowdGoalCommand = env.command_manager.get_term(command_name)
  return torch.where(term.is_standing_env, 0.0, (1.0 + torch.cos(term.heading_error)) / 2)


def crowd_collision(env, penetration_m: float = 0.05) -> torch.Tensor:
  """Termination: the robot overlaps a human by more than `penetration_m`."""
  closest, _ = _clearance(env)(env)
  return closest < -penetration_m


def collision_termination(penetration_m: float = 0.05) -> TerminationTermCfg:
  """The `crowd_collision` termination, as training, play and evaluation use it."""
  return TerminationTermCfg(func=crowd_collision, params={"penetration_m": penetration_m})


def termination_end(env, term_name: str = "crowd_collision") -> torch.Tensor:
  """1 on the step a `term_name` termination ends the episode: a terminal penalty with
  a negative weight (rewards are scaled by the step duration, so weight w costs w * dt).
  `collision_end` uses `crowd_collision`, `fall_end` mjlab's `fell_over`."""
  return env.termination_manager.get_term(term_name).float()


class penalty_ramp_term:
  """Curriculum: scale reward terms' configured weights from 0 to their full value over
  `steps` env steps (`env.common_step_counter`, which a resumed run restores), so a
  terminal penalty cannot trap early training. Terms with weight 0 stay off."""

  def __init__(self, cfg: CurriculumTermCfg, env):
    self._terms = {n: env.reward_manager.get_term_cfg(n) for n in cfg.params["reward_names"]}
    self._targets = {n: t.weight for n, t in self._terms.items()}

  def __call__(self, env, env_ids, reward_names, steps: int) -> dict[str, torch.Tensor]:
    del env_ids, reward_names
    scale = min(1.0, env.common_step_counter / max(steps, 1))
    for name, term in self._terms.items():
      term.weight = self._targets[name] * scale
    return {name: torch.tensor(term.weight) for name, term in self._terms.items()}


class crowd_difficulty_term:
  """Curriculum: how robot-blind the crowd is. The level d in [0, 1] is the chance that a
  human is drawn from the robot-blind crowd instead of the behavior config's: the
  robot-oblivious share becomes o + d (1 - o) and the robot-stubborn share s (1 - d), so
  d = 0 is the configured crowd and d = 1 the robot-blind one. Once per `interval` env
  steps (one PPO iteration) the level rises by `step` while the training collision rate
  (`crowd_collision` ends per robot-minute since the last update) is at most `target`,
  and falls by `step` above `band` times `target`. It edits the crowd's own encounter
  config, which each human reads when it resets. A checkpoint keeps the level and the
  last measurements (`state_dict`); a resumed run measures afresh from its first step.
  `blind_share` is read back from the humans at each update (the configured share until
  the first one: their traits are drawn at their first reset, after this term runs)."""

  def __init__(self, cfg: CurriculumTermCfg, env):
    del cfg
    self._crowd = env.command_manager.get_term("crowd")
    self._encounter = self._crowd.live_tuning.encounter  # the config the behavior reads
    self._base = (self._encounter.robot_oblivious_fraction, self._encounter.robot_stubborn_fraction)
    self._collisions = torch.zeros((), device=env.device)
    self._since: int | None = env.common_step_counter
    self.level, self.rate, self.blind_share = 0.0, 0.0, self._base[0]
    self._set(0.0)

  def _blind_share(self) -> float:
    """The share of the humans now in the plaza that do not perceive the robot."""
    return float((~self._crowd.behavior.robot_perceives).float().mean())

  def _set(self, level: float) -> None:
    level = round(min(1.0, max(0.0, level)), 6)
    oblivious, stubborn = self._base
    fields = {"robot_oblivious_fraction": min(1.0, oblivious + level * (1.0 - oblivious)),
              "robot_stubborn_fraction": stubborn * (1.0 - level)}
    replace(self._encounter, **fields)  # validates the pair before either changes
    for name, value in fields.items():
      setattr(self._encounter, name, value)
    self.level = level

  def __call__(self, env, env_ids, target: float, step: float, band: float, interval: int) -> dict[str, float]:
    if self._since is None:  # just loaded
      self._since = env.common_step_counter
      self._collisions.zero_()
    self._collisions += env.termination_manager.get_term("crowd_collision")[env_ids].sum()
    steps = env.common_step_counter - self._since
    if steps >= interval:
      self.rate = float(self._collisions) / (steps * env.num_envs * env.step_dt / 60.0)
      if self.rate <= target:
        self._set(self.level + step)
      elif self.rate > band * target:
        self._set(self.level - step)
      self.blind_share = self._blind_share()
      self._since = env.common_step_counter
      self._collisions.zero_()
    return {"level": self.level, "oblivious": self._encounter.robot_oblivious_fraction,
            "stubborn": self._encounter.robot_stubborn_fraction, "collision_rate": self.rate,
            "blind_share": self.blind_share}

  def state_dict(self) -> dict[str, float]:
    return {"level": self.level, "rate": self.rate, "blind_share": self.blind_share}

  def load_state_dict(self, state: dict[str, float]) -> None:
    self._set(state["level"])
    self.rate, self.blind_share = state.get("rate", 0.0), state.get("blind_share", self.blind_share)
    self._since = None


# The crowd curriculum's fixed settings: a step of 0.05 per iteration crosses 0 to 1 in
# 20 iterations; it holds between `target` and 1.5 x `target`.
CROWD_CURRICULUM = {"step": 0.05, "band": 1.5}


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
# Paired draws (`CrowdGoalCommandCfg.paired_seed`): counter = index * _KINDS + kind, where
# the index is the goal count for goals and speeds (arrivals and resamples both draw one)
# and the scheduled-resample count for timers and standing flags (which arrivals do not
# touch), so that neither depends on how fast a policy reaches its goals.
_GOAL, _SPEED, _STANDING, _TIMER, _SPAWN, _YAW, _KINDS = range(7)


class CrowdGoalCommand(UniformVelocityCommand):
  """Velocity toward random goals in the plaza, recomputed each step (see the module
  docstring). Standing envs (`rel_standing_envs`) command zero and must still keep
  clear of the humans.

  With `paired_seed` (evaluation) every random choice of the command (spawn candidates,
  yaw, goals, speeds, standing flags, resample times) comes from a per-env stream keyed
  by (seed, env, episode, leg), and each goal is the first candidate at least
  `min_goal_m` from the previous goal rather than from the robot. Two policies then get
  the same goal chain in the same env and episode however they walk. Training leaves it
  off and draws from the global generator as before."""

  cfg: "CrowdGoalCommandCfg"

  def __init__(self, cfg: "CrowdGoalCommandCfg", env):
    super().__init__(cfg, env)
    if cfg.paired_seed is not None and cfg.init_velocity_prob > 0:
      raise ValueError("paired draws do not cover init_velocity_prob")
    self.goal_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.speed = torch.zeros(self.num_envs, device=self.device)
    self.episodes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # resets so far
    self.legs = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # goals drawn this episode
    self.resamples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # scheduled resamples
    self.spawn = torch.zeros(self.num_envs, 2, device=self.device)  # this episode's start, plaza frame
    self.nav_start_distance = torch.zeros(self.num_envs, device=self.device)
    self.nav_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics["goals_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
    # Episode means over moving (non-standing) steps: |heading error| (deg), the share of
    # steps walking backward (body vx < -0.1 m/s), and planar speed.
    for name in ("heading_error_deg", "backward_share", "speed"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
    self._sums = torch.zeros(self.num_envs, 3, device=self.device)
    self._moving_steps = torch.zeros(self.num_envs, device=self.device)

  def _box(self, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    hi = torch.tensor([self.cfg.half_x - margin, self.cfg.half_y - margin], device=self.device)
    return -hi, hi

  def _origins(self, env_ids) -> torch.Tensor:
    return self._env.scene.env_origins[env_ids, :2]

  def _uniform(self, env_ids: torch.Tensor, shape: tuple[int, ...], kind: int) -> torch.Tensor:
    """`[n, *shape]` uniforms in [0, 1): the global generator, or with `paired_seed` the
    env's stream at (episode, index, kind) (`_KINDS`)."""
    n = len(env_ids)
    if self.cfg.paired_seed is None:
      return torch.rand(n, *shape, device=self.device)
    seeds = human_seeds(self.cfg.paired_seed, env_ids, "traversal_goals", episode=self.episodes[env_ids])
    index = self.resamples if kind in (_TIMER, _STANDING) else self.legs
    return hash_uniform(seeds, index[env_ids] * _KINDS + kind, math.prod(shape)).view(n, *shape)

  def _sample_goals(self, env_ids: torch.Tensor) -> None:
    n, C = len(env_ids), 128 if self.cfg.navigation else 16
    lo, hi = self._box(self.cfg.goal_margin_m)
    cand = lo + (hi - lo) * self._uniform(env_ids, (C, 2), _GOAL)
    pos = self.robot.data.root_link_pos_w[env_ids, :2] - self._origins(env_ids)
    if self.cfg.paired_seed is not None:  # from the previous goal, or the spawn for the first
      prev = self.goal_w[env_ids] - self._origins(env_ids)
      pos = torch.where((self.legs[env_ids] > 0)[:, None], prev, self.spawn[env_ids])
    dist = (cand - pos[:, None]).norm(dim=-1)
    if self.cfg.navigation:
      # Navigation goals are paired in training too, from the actual recorded spawn.
      pos = self.spawn[env_ids]
      dist = (cand - pos[:, None]).norm(dim=-1)
      nav = self.cfg.navigation
      valid = (dist >= nav.min_goal_m) & (dist <= nav.max_goal_m)
      if not valid.any(1).all():
        raise ValueError("navigation goal range has no sampled feasible point in this plaza")
      pick = valid.long().argmax(1)
      self.goal_w[env_ids] = cand[torch.arange(n, device=self.device), pick] + self._origins(env_ids)
      self.speed[env_ids] = nav.speed_mps
      self.nav_start_distance[env_ids] = dist[torch.arange(n, device=self.device), pick]
      self.legs[env_ids] += 1
      return
    pick = torch.where((dist >= self.cfg.min_goal_m).any(1), (dist >= self.cfg.min_goal_m).float().argmax(1),
                       dist.argmax(1))
    self.goal_w[env_ids] = cand[torch.arange(n, device=self.device), pick] + self._origins(env_ids)
    if self.cfg.paired_seed is None:
      self.speed[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.speed_range)
    else:
      a, b = self.cfg.speed_range
      self.speed[env_ids] = a + (b - a) * self._uniform(env_ids, (), _SPEED)
    self.legs[env_ids] += 1

  def _resample(self, env_ids: torch.Tensor) -> None:
    if self.cfg.paired_seed is None or not len(env_ids):
      return super()._resample(env_ids)
    a, b = self.cfg.resampling_time_range
    self.time_left[env_ids] = a + (b - a) * self._uniform(env_ids, (), _TIMER)
    self._resample_command(env_ids)
    self.command_counter[env_ids] += 1
    self.resamples[env_ids] += 1

  def reset(self, env_ids):
    """Place the reset robots at the start farthest from the (already reset) humans."""
    if isinstance(env_ids, torch.Tensor) and len(env_ids):
      self.episodes[env_ids] += 1
      self.legs[env_ids] = 0
      self.resamples[env_ids] = 0
      crowd: GraphCrowd = self._env.command_manager.get_term(self.cfg.crowd_name)
      n, C = len(env_ids), self.cfg.spawn_candidates
      lo, hi = self._box(self.cfg.spawn_margin_m)
      cand = lo + (hi - lo) * self._uniform(env_ids, (C, 2), _SPAWN)
      hips = human_joints(crowd)[env_ids, :, 0, :2] - self._origins(env_ids)[:, None]
      gap = (cand[:, :, None] - hips[:, None]).norm(dim=-1).amin(-1)
      self.spawn[env_ids] = cand[torch.arange(n, device=self.device), gap.argmax(1)]
      xy = self.spawn[env_ids] + self._origins(env_ids)
      if self.cfg.paired_seed is None:
        yaw = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
      else:
        yaw = (2 * self._uniform(env_ids, (), _YAW) - 1) * math.pi
      pose = torch.zeros(n, 7, device=self.device)
      pose[:, :2] = xy
      pose[:, 2] = self.robot.data.default_root_state[env_ids, 2]
      zero = torch.zeros_like(yaw)
      pose[:, 3:] = quat_from_euler_xyz(zero, zero, yaw)
      self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
    extras = super().reset(env_ids)
    self._sums[env_ids] = 0.0
    self._moving_steps[env_ids] = 0.0
    self.nav_contact[env_ids] = False
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._sample_goals(env_ids)
    if self.cfg.paired_seed is None:
      r = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
    else:
      r = self._uniform(env_ids, (), _STANDING)
    self.is_standing_env[env_ids] = r <= self.cfg.rel_standing_envs
    self.is_heading_env[env_ids] = True
    self.is_world_env[env_ids] = True
    self.is_forward_env[env_ids] = False

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    pos = self.robot.data.root_link_pos_w[:, :2]
    to_goal = self.goal_w - pos
    dist = to_goal.norm(dim=-1)
    arrived = (dist < self.cfg.arrive_m) & ~self.is_standing_env
    if self.cfg.navigation is None and arrived.any():
      ids = arrived.nonzero(as_tuple=False).flatten()
      self.metrics["goals_reached"][ids] += 1.0
      self._sample_goals(ids)
      to_goal = self.goal_w - pos
      dist = to_goal.norm(dim=-1)
    direction = to_goal / dist[:, None].clamp(min=1e-6)
    speed = self.speed * (dist / self.cfg.slow_m).clamp(max=1.0)
    self.heading_target = torch.atan2(direction[:, 1], direction[:, 0])
    heading = self.robot.data.heading_w
    self.heading_error = wrap_to_pi(self.heading_target - heading)
    if self.cfg.forward_first:
      speed = speed * torch.cos(self.heading_error).clamp(min=0.0)
    self.vel_command_w[:, :2] = direction * speed[:, None]
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
    moving = (~self.is_standing_env).float()
    vel = self.robot.data.root_link_lin_vel_b[:, :2]
    step = torch.stack((torch.rad2deg(self.heading_error.abs()), (vel[:, 0] < -0.1).float(), vel.norm(dim=-1)), -1)
    self._sums += step * moving[:, None]
    self._moving_steps += moving
    mean = self._sums / self._moving_steps.clamp(min=1.0)[:, None]
    for i, name in enumerate(("heading_error_deg", "backward_share", "speed")):
      self.metrics[name] = mean[:, i]


@dataclass(kw_only=True)
class CrowdGoalCommandCfg(UniformVelocityCommandCfg):
  navigation: NavigationCfg | None = None
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
  forward_first: bool = False  # scale the linear command by max(cos(heading error), 0): turn, then walk
  paired_seed: int | None = None  # evaluation: per-env streams for every random choice (CrowdGoalCommand)

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
  perceive_robot: bool = True,
  forward_first: bool = False,
  teacher_lidar: bool = True,
  no_avoidance: bool = False,
  penalty_ramp: int = 0,
  planner_rate: float = 0.0,
  crowd_curriculum: float = 0.0,
  navigation: dict | None = None,
) -> ManagerBasedRlEnvCfg:
  """The crowd traversal env (module docstring). `behavior_path`: an encounter-design
  behavior config (its `graph_overrides` and `encounter` sections; the scene size comes
  from the encounter section).

  `no_avoidance` builds b0, the non-avoidant baseline: nothing about the humans reaches
  learning. No group reads the LiDAR or crowd state, no reward reads the crowd, and no
  collision ends a training episode (the forfeited return alone would teach avoidance).
  The humans still walk, see the robot and are measured by evaluation.
  `penalty_ramp` (iterations, 0 = off) ramps `collision_end` and `fall_end` in from 0.
  `crowd_curriculum` (collisions per robot-minute, 0 = off) makes the crowd more
  robot-blind while training collisions stay at or below it (`crowd_difficulty_term`).
  `planner_rate` (Hz, 0 = off) builds the env for the planning arms (`planning.PlanningVecEnv`
  steps it under frozen SONIC): only the task rewards stay (`PLANNER_REWARDS`), since the
  tracker, not the policy, owns the gait."""
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
      lidar_group=HUMANS_GROUP, debug_vis=False, perceive_entity="robot" if perceive_robot else None,
    ),
    "twist": CrowdGoalCommandCfg(
      entity_name="robot", resampling_time_range=(10.0, 15.0), rel_standing_envs=0.1,
      heading_command=True, heading_control_stiffness=1.0, debug_vis=True, viz=twist.viz,
      ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.2, 1.2), lin_vel_y=(-1.2, 1.2),
                                              ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)),
      half_x=hx, half_y=hy, forward_first=forward_first,
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
  if not no_avoidance:
    cfg.observations["teacher"] = ObservationGroupCfg(
      terms={**actor.terms, **privileged}, concatenate_terms=True, enable_corruption=actor.enable_corruption)
    for group in ("actor", "critic", *(("teacher",) if teacher_lidar else ())):
      cfg.observations[group].terms["lidar"] = ObservationTermCfg(
        func=LidarScan, params=scan, noise=None if group == "critic" else Unoise(n_min=-0.01, n_max=0.01))
    cfg.observations["critic"].terms.update(privileged)
    cfg.rewards["crowd_proximity"] = RewardTermCfg(func=crowd_proximity, weight=-1.0, params={"margin_m": 0.4})
    cfg.rewards["crowd_contact"] = RewardTermCfg(func=crowd_contact, weight=-10.0)

  cfg.rewards["wall_proximity"] = RewardTermCfg(func=wall_proximity, weight=-1.0,
                                                params={"half_x": hx, "half_y": hy})
  # Facing the goal: without it the policy walks backward toward goals with its back to
  # them, a local optimum at the turn command's ±180° flip. mjlab's turn-rate weight
  # (2.0) is kept.
  cfg.rewards["goal_heading"] = RewardTermCfg(func=goal_heading, weight=1.0)
  if not play:
    # Terminal penalties, off by default (weight 0 is skipped). Without them an episode
    # end costs only the forfeited return, a few seconds of reward at gamma 0.99, and a
    # collision penalty alone makes falling the cheaper end.
    cfg.rewards["fall_end"] = RewardTermCfg(func=termination_end, weight=0.0, params={"term_name": "fell_over"})
    if not no_avoidance:
      cfg.terminations["crowd_collision"] = collision_termination()
      cfg.rewards["collision_end"] = RewardTermCfg(func=termination_end, weight=0.0,
                                                   params={"term_name": "crowd_collision"})
    if penalty_ramp:
      from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

      names = tuple(n for n in ("collision_end", "fall_end") if n in cfg.rewards)
      steps = penalty_ramp * unitree_g1_ppo_runner_cfg().num_steps_per_env
      cfg.curriculum["penalty_ramp"] = CurriculumTermCfg(func=penalty_ramp_term,
                                                         params={"reward_names": names, "steps": steps})
    if crowd_curriculum:
      if no_avoidance or not perceive_robot:
        raise ValueError("the crowd curriculum needs collision ends and a robot-aware crowd")
      from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

      interval = unitree_g1_ppo_runner_cfg().num_steps_per_env  # one iteration; train.py sets it per run
      cfg.curriculum["crowd_difficulty"] = CurriculumTermCfg(
        func=crowd_difficulty_term, params={"target": crowd_curriculum, "interval": interval, **CROWD_CURRICULUM})
  if planner_rate:
    cfg.rewards = {k: v for k, v in cfg.rewards.items() if k in PLANNER_REWARDS}
  if navigation is not None:
    if not planner_rate or no_avoidance or crowd_curriculum or penalty_ramp:
      raise ValueError("navigation requires a planning arm without legacy reward/crowd curricula")
    from crowdmotion.sim.mjlab.navigation import (
      goal_deadline,
      goal_observation,
      goal_success,
      terminal_reward,
      tracking_baseline,
    )

    nav = NavigationCfg.from_dict(navigation)
    twist_cfg = cfg.commands["twist"]
    twist_cfg.navigation = nav
    twist_cfg.resampling_time_range = (1e9, 1e9)
    twist_cfg.rel_standing_envs = 0.
    twist_cfg.paired_seed = seed
    cfg.episode_length_s = nav.deadline_s
    cfg.is_finite_horizon = True
    cfg.terminations.pop("time_out", None)
    cfg.terminations["goal_deadline"] = TerminationTermCfg(func=goal_deadline)
    cfg.terminations["goal_success"] = TerminationTermCfg(func=goal_success)
    cfg.terminations["crowd_collision"] = collision_termination()
    # Outcome reads fell_over and clearance directly, so order among the added terms
    # does not affect failure precedence. All failures are task terminals, not truncations.
    keep = {"tracking": {"track_linear_velocity", "track_angular_velocity", "goal_heading"},
            "proximity": {"crowd_proximity", "crowd_contact", "wall_proximity"},
            "all": set(PLANNER_REWARDS) - {"fall_end", "collision_end"}}
    keep["centered"] = keep["all"]
    cfg.rewards = {k: v for k, v in cfg.rewards.items() if k in keep.get(nav.reward_mode, set())}
    cfg.rewards["goal_arrival"] = RewardTermCfg(func=terminal_reward, weight=1.)
    if nav.reward_mode == "centered":
      cfg.rewards["tracking_baseline"] = RewardTermCfg(func=tracking_baseline, weight=1.)
    for obs in cfg.observations.values():
      obs.terms["navigation_goal"] = ObservationTermCfg(func=goal_observation)
  return cfg


# The planning arms' rewards: goal following, the crowd and wall terms and the terminal
# penalties; the gait terms of mjlab's velocity task go (the frozen tracker walks).
PLANNER_REWARDS = ("track_linear_velocity", "track_angular_velocity", "goal_heading", "crowd_proximity",
                   "crowd_contact", "wall_proximity", "fall_end", "collision_end")


def traversal_rl_cfg(max_iterations: int = 10_000, teacher: bool = False, planner_rate: float = 0.0,
                     navigation: dict | None = None):
  """PPO settings of the G1 velocity task; the teacher's actor reads the `teacher` group.
  With `planner_rate`, steps are decisions: 16 s of rollout per env and iteration, and the
  discount and GAE factors per decision of a 4 s and 10 s horizon (gamma 0.75 and lambda
  0.9 per second), so that 1 Hz and 2 Hz see the same horizons."""
  from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

  rl = unitree_g1_ppo_runner_cfg()
  rl.experiment_name = "g1_crowd_traversal_teacher" if teacher else "g1_crowd_traversal"
  if teacher:
    rl.obs_groups = {"actor": ("teacher",), "critic": ("critic",)}
  if planner_rate:
    rl.experiment_name += f"_planner{planner_rate:g}hz"
    rl.num_steps_per_env = int(round(16 * planner_rate))
    rl.algorithm.gamma = 0.75 ** (1.0 / planner_rate)
    rl.algorithm.lam = 0.9 ** (1.0 / planner_rate)
    rl.actor.distribution_cfg["init_std"] = 0.5
    rl.save_interval = 5  # an iteration takes minutes
  else:
    rl.save_interval = 100
  rl.logger = "tensorboard"
  rl.max_iterations = max_iterations
  if navigation is not None:
    nav = NavigationCfg.from_dict(navigation)
    rl.algorithm.gamma = nav.gamma_per_s ** (1 / planner_rate)
    rl.algorithm.lam = nav.gae_lambda_per_s ** (1 / planner_rate)
    rl.num_steps_per_env = round(nav.rollout_s * planner_rate)
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


def traversal_distill_cfg(max_iterations: int = 3_000, planner_rate: float = 0.0) -> DistillationRunnerCfg:
  cfg = DistillationRunnerCfg(
    obs_groups={"student": ("actor",), "teacher": ("teacher",)},
    experiment_name="g1_crowd_traversal_student", logger="tensorboard", save_interval=100,
    max_iterations=max_iterations, num_steps_per_env=24,
  )
  if planner_rate:  # decisions, as `traversal_rl_cfg`
    cfg.experiment_name += f"_planner{planner_rate:g}hz"
    cfg.num_steps_per_env = int(round(16 * planner_rate))
    cfg.save_interval = 5
  return cfg


def _planning_env(env, planner_rate: float):
  """The env a runner trains on: the planning wrapper over the raw env for the planning arms."""
  from crowdmotion.sim.mjlab.planning import PlanningVecEnv

  if not planner_rate or isinstance(env, PlanningVecEnv):
    return env
  return PlanningVecEnv(env.unwrapped, planner_rate)


def _ppo_runner(planner_rate: float = 0.0):
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  from crowdmotion.sim.mjlab.wandb_log import Tracked

  class TraversalPpoRunner(Tracked, VelocityOnPolicyRunner):
    """mjlab's velocity runner that also starts from an initialization checkpoint
    (`init_only`): actor and critic weights only, a fresh optimizer and iteration 0.
    `crowd train traversal --finetune` writes one from a distilled student. Under
    `crowd train traversal` it logs through `wandb_log` (`Tracked`). For the planning arms
    it trains on `planning.PlanningVecEnv`."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
      super().__init__(_planning_env(env, planner_rate), train_cfg, log_dir, device)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location=None):
      if load_cfg is None and torch.load(path, weights_only=False, map_location="cpu").get("init_only"):
        load_cfg = {"actor": True, "critic": True}
      return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

  return TraversalPpoRunner


def _distillation_runner(planner_rate: float = 0.0):
  from rsl_rl.runners import DistillationRunner

  from crowdmotion.sim.mjlab.wandb_log import Tracked

  class TraversalDistillationRunner(Tracked, DistillationRunner):
    """`DistillationRunner` with mjlab's cleanup of unset model options; under `crowd train
    traversal` it logs through `wandb_log` (`Tracked`)."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
      for key in ("student", "teacher"):
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
      super().__init__(_planning_env(env, planner_rate), train_cfg, log_dir, device)

  return TraversalDistillationRunner


def register(graph_path: str, behavior_path: str, rotations_path: str | None = None, humans: int = 8,
             policy: str = "lidar", perceive_robot: bool = True, forward_first: bool = False,
             teacher_lidar: bool = True, no_avoidance: bool = False, penalty_ramp: int = 0,
             planner_rate: float = 0.0, crowd_curriculum: float = 0.0,
             navigation: dict | None = None) -> str:
  """Register a traversal task with mjlab's registry (training and play configs) and
  return its id. `policy` (`POLICIES`): `lidar` (PPO on LiDAR), `teacher` (PPO on
  privileged observations) or `student` (distillation of a teacher into LiDAR).
  `no_avoidance` (b0) needs the `lidar` policy. `planner_rate` (1 or 2 Hz) registers the
  planning arm of that policy (its own task id)."""
  from mjlab.tasks.registry import list_tasks, register_mjlab_task

  if no_avoidance and policy != "lidar":
    raise ValueError("no_avoidance (b0) trains the plain policy; it has no teacher or student")
  if planner_rate and no_avoidance:
    raise ValueError("the planning arms learn avoidance; no_avoidance applies to b0 only")
  task_id = POLICIES[policy] + (f"-Planner{planner_rate:g}Hz" if planner_rate else "")
  if navigation is not None:
    import hashlib
    task_id += "-Goal" + hashlib.sha256(json.dumps(navigation, sort_keys=True).encode()).hexdigest()[:10]
  if task_id not in list_tasks():
    student = policy == "student"
    options = dict(perceive_robot=perceive_robot, forward_first=forward_first, teacher_lidar=teacher_lidar,
                   no_avoidance=no_avoidance, planner_rate=planner_rate, navigation=navigation)
    register_mjlab_task(
      task_id=task_id,
      env_cfg=make_traversal_env_cfg(graph_path, behavior_path, rotations_path, humans, penalty_ramp=penalty_ramp,
                                     crowd_curriculum=crowd_curriculum, **options),
      play_env_cfg=make_traversal_env_cfg(graph_path, behavior_path, rotations_path, humans, play=True, **options),
      rl_cfg=(traversal_distill_cfg(planner_rate=planner_rate) if student
              else traversal_rl_cfg(teacher=policy == "teacher", planner_rate=planner_rate, navigation=navigation)),
      runner_cls=_distillation_runner(planner_rate) if student else _ppo_runner(planner_rate),
    )
  return task_id
