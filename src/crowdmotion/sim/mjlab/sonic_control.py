"""Frozen SONIC driving mjlab's G1: planned references in, joint-position actions out.

`SonicController` keeps, per env, the planned motion (`backends.sonic.PlanBuffer`), the
tracker's 10-frame proprioceptive history and its last action, and runs the tracker once per
control step (`act`). Plans come from a batched planner callable (the kinematic planner's
inputs -> 30 Hz qpos and frame counts), requested with `replan`.

Frames follow the release's deploy: plans live in a planner frame whose origin and zero yaw
are the robot's pose when the env was reset; `delta` (deploy `apply_delta_heading`) turns
planned orientations into the world for the tracker's anchor orientation, and
`to_planner` turns world directions into planner commands. mjlab's G1 has SONIC's joint
order, PD gains, default pose and action scale, so the tracker's raw actions are the env's
`joint_pos` actions (reordered by name).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from crowdmotion.backends import sonic

IDLE, SLOW_WALK, WALK, RUN = 0, 1, 2, 3

# `backends.sonic.SonicPlanner`, or any callable with its inputs and outputs.
Planner = Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]


def yaw_of(q: torch.Tensor) -> torch.Tensor:
  """Heading (rad) of wxyz quaternions."""
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class SonicController:
  def __init__(self, env, planner: Planner, robot: str = "robot", action: str = "joint_pos",
               tracker: sonic.SonicTracker | None = None):
    self.env = env
    self.robot = env.scene[robot]
    self.device = env.device
    self.planner = planner
    self.tracker = (tracker or sonic.SonicTracker()).to(self.device)
    n = env.num_envs
    self.plans = sonic.PlanBuffer(n, self.device)
    h = lambda d: torch.zeros(n, sonic.HISTORY, d, device=self.device)  # noqa: E731
    self.hist = {"ang_vel": h(3), "joint_pos": h(29), "joint_vel": h(29), "last_action": h(29), "gravity": h(3)}
    self.action = torch.zeros(n, 29, device=self.device)  # last raw action, IsaacLab order
    self.delta = torch.zeros(n, 4, device=self.device)
    self.delta[:, 0] = 1.0
    self.origin = torch.zeros(n, 2, device=self.device)  # world xy of the planner frame's origin
    self.seed = torch.arange(n, device=self.device) * 7919 % 10000
    self.tokens = sonic.WALK_TOKENS  # allowed plan lengths (6 + i tokens), per request
    names = list(self.robot.joint_names)
    mjcf = [names.index(j) for j in sonic_joint_names()]
    self._robot_to_isaaclab = torch.tensor(mjcf, device=self.device)[torch.tensor(sonic.TO_ISAACLAB)]
    term = env.action_manager.get_term(action)
    self._action_from_isaaclab = torch.tensor(
      [sonic.TO_ISAACLAB.index(mjcf.index(names.index(t))) for t in term.target_names], device=self.device)
    self._default = torch.tensor(sonic.DEFAULT_ANGLES, device=self.device)[torch.tensor(sonic.TO_ISAACLAB)]

  # State in the tracker's conventions (IsaacLab joint order).

  def _state(self):
    d = self.robot.data
    q = d.joint_pos[:, self._robot_to_isaaclab]
    return {"ang_vel": d.root_link_ang_vel_b, "joint_pos": q - self._default,
            "joint_vel": d.joint_vel[:, self._robot_to_isaaclab], "gravity": d.projected_gravity_b}

  def reset(self, ids: torch.Tensor) -> None:
    """Fill the history with the current state, zero the action and start each env's plan with
    the release's initialization (idle, context at the origin with zero yaw)."""
    if ids.numel() == 0:
      return
    state = self._state()
    for k, v in state.items():
      self.hist[k][ids] = v[ids, None].expand(-1, sonic.HISTORY, -1)
    self.hist["last_action"][ids] = 0.0
    self.action[ids] = 0.0
    d = self.robot.data
    self.origin[ids] = d.root_link_pos_w[ids, :2]
    qpos = torch.zeros(ids.numel(), 4, 36, device=self.device)
    qpos[..., 2] = sonic.DEFAULT_HEIGHT
    qpos[..., 3] = 1.0
    joints = d.joint_pos[ids][:, self._robot_to_isaaclab][:, torch.tensor(sonic.TO_MJCF, device=self.device)]
    qpos[..., 7:] = joints[:, None]
    face = torch.zeros(ids.numel(), 3, device=self.device)
    face[:, 0] = 1.0
    self._plan(ids, qpos, IDLE, -1.0, torch.zeros_like(face), face, first=True)
    self.delta[ids] = sonic.quat_mul(sonic.heading_quat(d.root_link_quat_w[ids]),
                                     sonic.quat_conj(sonic.heading_quat(self.plans.quat[ids, 0])))

  def to_planner(self, ids: torch.Tensor, world_xy: torch.Tensor) -> torch.Tensor:
    """World planar directions `[B, 2]` -> planner-frame directions `[B, 3]` (z = 0)."""
    yaw = yaw_of(self.delta[ids])
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y = world_xy.unbind(-1)
    return torch.stack([c * x + s * y, -s * x + c * y, torch.zeros_like(x)], -1)

  def replan(self, ids: torch.Tensor, mode: int | torch.Tensor, speed: float | torch.Tensor,
             movement: torch.Tensor, facing: torch.Tensor) -> None:
    """New plans for envs `ids` from their playing plans (planner-frame directions `[B, 3]`)."""
    if ids.numel():
      self._plan(ids, self.plans.context(ids), mode, speed, movement, facing, first=False)

  def _plan(self, ids, context, mode, speed, movement, facing, first: bool) -> None:
    b, dev = ids.numel(), self.device
    full = lambda v, dtype: torch.as_tensor(v, device=dev, dtype=dtype).expand(b).reshape(b, 1)  # noqa: E731
    inputs = {
      "context_mujoco_qpos": context.reshape(b, 1, 4, 36),
      "target_vel": full(speed, torch.float32),
      "mode": full(mode, torch.int64),
      "movement_direction": movement.reshape(b, 1, 3).float(),
      "facing_direction": facing.reshape(b, 1, 3).float(),
      "random_seed": self.seed[ids].reshape(b, 1),
      "has_specific_target": torch.zeros(b, 1, 1, dtype=torch.int64, device=dev),
      "specific_target_positions": torch.zeros(b, 1, 4, 3, device=dev),
      "specific_target_headings": torch.zeros(b, 1, 4, device=dev),
      "allowed_pred_num_tokens": torch.tensor(self.tokens, device=dev).expand(b, 1, -1).contiguous(),
      "height": torch.full((b, 1), -1.0, device=dev),
    }
    qpos, frames = self.planner(inputs)
    first_t = torch.full((b,), first, dtype=torch.bool, device=dev)
    self.plans.install(ids, qpos.reshape(b, -1, 36), frames.reshape(b).long(), first_t)

  def act(self) -> torch.Tensor:
    """One control step: log the state, run the tracker on the plan's next 0.9 s, advance the
    plan. Returns the env's `joint_pos` action."""
    state = self._state()
    state["last_action"] = self.action
    for k, v in state.items():
      self.hist[k] = torch.cat([self.hist[k][:, 1:], v[:, None]], 1)
    joint_pos, joint_vel, quat = self.plans.window()
    base = self.robot.data.root_link_quat_w
    ref = sonic.quat_mul(self.delta[:, None].expand_as(quat), quat)
    anchor = sonic.anchor_6d(base[:, None].expand_as(ref), ref)
    with torch.inference_mode():
      token = self.tracker.encode(joint_pos, joint_vel, anchor)
      action = self.tracker.decode(token, self.hist["ang_vel"], self.hist["joint_pos"], self.hist["joint_vel"],
                                   self.hist["last_action"], self.hist["gravity"])
    self.action = action.clone()
    self.plans.advance()
    return action[:, self._action_from_isaaclab]

  def reference_world(self) -> tuple[torch.Tensor, torch.Tensor]:
    """The playing plan's current root position (planar, world) and orientation (world)."""
    i = torch.arange(self.plans.frame.shape[0], device=self.device)
    p = self.plans.pos[i, self.plans.frame, :2]
    yaw = yaw_of(self.delta)
    c, s = torch.cos(yaw), torch.sin(yaw)
    world = torch.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], -1) + self.origin
    return world, sonic.quat_mul(self.delta, self.plans.quat[i, self.plans.frame])


def sonic_joint_names() -> list[str]:
  """The G1's 29 joints in MJCF order (SONIC's `g1_29dof.xml`, equal to mjlab's `g1.xml`)."""
  legs = [f"{s}_{j}_joint" for s in ("left", "right")
          for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
  waist = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
  arms = [f"{s}_{j}_joint" for s in ("left", "right")
          for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
                    "wrist_yaw")]
  return legs + waist + arms
