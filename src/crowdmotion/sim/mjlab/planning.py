"""The planning arms: a high-level policy deciding at 1 or 2 Hz over the released kinematic
planner and the frozen SONIC tracker.

`PlanningVecEnv` wraps the traversal env (built with `planner_rate`) as an RSL-RL `VecEnv`
whose step is one decision:

1. The action, 3 numbers in [-1, 1], is a residual on the goal command: a planar velocity
   offset in the robot's heading frame (up to `max_delta_v` m/s) and a facing offset from the
   direction of motion (up to `max_delta_yaw`). A zero action is plain goal following: walk
   toward the goal at the command's speed, facing the way it walks; standing envs stand.
   Below `STAND_SPEED` the planner idles.
2. Every env replans once with that command (`SonicController.replan`); then the tracker
   runs for 1 / rate seconds of control steps (50 or 25), and a plan about to run out is
   extended with the same command.
3. The reward is the env's reward summed over those steps, minus small action and
   action-change penalties. An env that terminates or times out ends its transition at that
   step; its later rewards in the option are dropped, its controller restarts idle, and it
   replans at the next decision. `time_outs` carries the truncations for bootstrapping.

Every observation group gets the plan summary appended: the playing plan's root position
0.25, 0.5 and 1 s ahead in the robot's heading frame, the plan's remaining seconds, and the
previous action.

`decide` runs the same decision one env step per `next()`, so that a viewer can draw every
step (`crowd play traversal`'s Viser viewer); `step` drains it.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from crowdmotion.backends import sonic
from crowdmotion.sim.mjlab.sonic_control import IDLE, WALK, SonicController, yaw_of

STAND_SPEED = 0.15  # m/s: slower commands idle
MAX_SPEED = 1.2
SUMMARY_S = (0.25, 0.5, 1.0)
# Walking plans long enough to outlast one decision (6 + i tokens of 4 frames at 30 Hz).
TOKENS = {1.0: (0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0), 2.0: sonic.WALK_TOKENS}


class PlanningVecEnv(VecEnv):
  def __init__(self, env, rate_hz: float, planner=None, command: str = "twist", max_delta_v: float = 1.0,
               max_delta_yaw: float = math.pi / 2, action_l2: float = 0.01, action_rate: float = 0.02):
    self.env = env
    self.rate = float(rate_hz)
    if self.rate not in TOKENS:
      raise ValueError(f"planner rate {rate_hz} Hz: supported {sorted(TOKENS)}")
    self.substeps = int(round(1.0 / (self.rate * env.step_dt)))
    self.num_envs = env.num_envs
    self.device = torch.device(env.device)
    self.num_actions = 3
    self.max_episode_length = math.ceil(env.max_episode_length / self.substeps)
    self.command = env.command_manager.get_term(command)
    self.max_delta_v, self.max_delta_yaw = max_delta_v, max_delta_yaw
    self.navigation = getattr(self.command.cfg, "navigation", None)
    if self.navigation is not None:
      regularize = self.navigation.reward_mode in ("regularized", "all", "centered")
      action_l2 = self.navigation.action_l2 if regularize else 0.
      action_rate = self.navigation.action_rate if regularize else 0.
    self.action_l2, self.action_rate = action_l2, action_rate
    self.ctl = SonicController(env, planner or sonic.SonicPlanner(device=str(self.device)))
    self.on_substep = None  # called after every env step of a decision (evaluation metrics)
    self.ctl.tokens = TOKENS[self.rate]
    n = self.num_envs
    self.prev_action = torch.zeros(n, 3, device=self.device)
    self._cmd = {"mode": torch.full((n,), IDLE, dtype=torch.long, device=self.device),
                 "speed": torch.full((n,), -1.0, device=self.device),
                 "move": torch.zeros(n, 3, device=self.device), "face": torch.zeros(n, 3, device=self.device)}
    self._cmd["face"][:, 0] = 1.0
    self.env.reset()
    self.ctl.reset(torch.arange(n, device=self.device))

  # VecEnv interface.

  @property
  def cfg(self):
    return self.env.cfg

  @property
  def unwrapped(self):
    return self.env.unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.env.episode_length_buf // self.substeps

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.env.episode_length_buf = value * self.substeps

  def seed(self, seed: int = -1) -> int:
    return self.env.seed(seed)

  def get_observations(self) -> TensorDict:
    return self._observations(self.env.observation_manager.compute())

  def reset(self) -> tuple[TensorDict, dict]:
    obs, extras = self.env.reset()
    self.ctl.reset(torch.arange(self.num_envs, device=self.device))
    self.prev_action.zero_()
    return self._observations(obs), extras

  def close(self) -> None:
    self.env.close()

  def step(self, actions: torch.Tensor):
    """One decision: `(obs, reward, done, extras)` after `substeps` env steps."""
    decision = self.decide(actions)
    while True:
      try:
        next(decision)
      except StopIteration as stop:
        return stop.value

  def decide(self, actions: torch.Tensor):
    """`step` one env step at a time, for a viewer that draws every env step: each `next()`
    takes one env step, and the last raises `StopIteration` with `step`'s result. A
    decision abandoned part way needs a `reset` before the next one."""
    a = actions.clamp(-1.0, 1.0)
    ids = torch.arange(self.num_envs, device=self.device)
    stats = getattr(self.ctl.planner, "stats", None)
    plans0, seconds0 = (stats["plans"], stats["seconds"]) if stats else (0, 0.0)
    self._set_command(a)
    self.ctl.replan(ids, self._cmd["mode"], self._cmd["speed"], self._cmd["move"], self._cmd["face"])
    total = torch.zeros(self.num_envs, device=self.device)
    done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    timeout = torch.zeros_like(done)
    live = torch.ones_like(done)
    phi_before = phi_after = None
    if self.navigation is not None:
      phi_before = (self.command.goal_w - self.ctl.robot.data.root_link_pos_w[:, :2]).norm(dim=-1)
      phi_after = phi_before.clone()
    logs: dict[str, list[tuple[float, float]]] = {}
    extras: dict = {}
    obs = None
    for i in range(self.substeps):
      if i:
        yield
      low = self.ctl.act()
      obs, rew, terminated, truncated, extras = self.env.step(low)
      if self.on_substep is not None:
        self.on_substep()
      total += torch.where(live, rew, 0.0)
      if self.navigation is not None:
        phi_after = torch.where(live, self.env._navigation_pre["distance"], phi_after)
      ended = terminated | truncated
      done |= ended & live
      timeout |= truncated & live
      live &= ~ended
      reset_ids = ended.nonzero().flatten()
      if reset_ids.numel():
        self.ctl.reset(reset_ids)
        for k in self._cmd:
          self._cmd[k][reset_ids] = 0.0
        self._cmd["mode"][reset_ids] = IDLE
        self._cmd["speed"][reset_ids] = -1.0
        self._cmd["face"][reset_ids, 0] = 1.0
        for key, value in extras.get("log", {}).items():
          logs.setdefault(key, []).append((float(value), float(reset_ids.numel())))
      ending = (self.ctl.plans.length - 1 - self.ctl.plans.frame <= 2) & live
      if ending.any():  # extend a plan about to run out with the same command
        e = ending.nonzero().flatten()
        self.ctl.replan(e, self._cmd["mode"][e], self._cmd["speed"][e], self._cmd["move"][e], self._cmd["face"][e])
    shaping = None
    if self.navigation is not None and self.navigation.reward_mode == "potential":
      from crowdmotion.sim.mjlab.navigation import potential_reward
      shaping = potential_reward(phi_before, phi_after, done,
                                 self.navigation.gamma_per_s ** (1 / self.rate),
                                 self.navigation.potential_weight)
      total += shaping
    total -= self.action_l2 * a.square().sum(-1) + self.action_rate * (a - self.prev_action).square().sum(-1)
    self.prev_action = torch.where(done[:, None], 0.0, a)
    extras = dict(extras)
    # mjlab logs episode means at each reset, except `Episode_Termination/*`, which are counts:
    # means are weighted by the resets behind them, counts summed over the decision.
    extras["log"] = {k: sum(v for v, _ in vs) if k.startswith("Episode_Termination/")
                     else sum(v * w for v, w in vs) / max(sum(w for _, w in vs), 1.0) for k, vs in logs.items()}
    if shaping is not None:
      extras["log"]["Navigation/potential_reward"] = float(shaping.mean())
    if stats:
      extras["log"]["Planner/plans_per_env"] = (stats["plans"] - plans0) / self.num_envs
      extras["log"]["Planner/seconds"] = stats["seconds"] - seconds0
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = timeout
    return self._observations(obs), total, done.long(), extras

  # Commands and observations.

  def _set_command(self, a: torch.Tensor) -> None:
    """Goal command plus the residual `a`, as planner-frame commands for every env."""
    robot = self.ctl.robot
    pos = robot.data.root_link_pos_w[:, :2]
    to_goal = self.command.goal_w - pos
    dist = to_goal.norm(dim=-1)
    goal_dir = to_goal / dist[:, None].clamp(min=1e-6)
    v_nom = self.command.speed * (dist / self.command.cfg.slow_m).clamp(max=1.0)
    v_nom = torch.where(self.command.is_standing_env, 0.0, v_nom)
    heading = robot.data.heading_w
    c, s = torch.cos(heading), torch.sin(heading)
    dv = a[:, :2] * self.max_delta_v
    v = goal_dir * v_nom[:, None] + torch.stack([c * dv[:, 0] - s * dv[:, 1], s * dv[:, 0] + c * dv[:, 1]], -1)
    speed = v.norm(dim=-1)
    moving = speed >= STAND_SPEED
    direction = torch.where(moving[:, None], v / speed[:, None].clamp(min=1e-6),
                            torch.stack([c, s], -1))
    face_yaw = torch.atan2(direction[:, 1], direction[:, 0]) + a[:, 2] * self.max_delta_yaw
    face = torch.stack([torch.cos(face_yaw), torch.sin(face_yaw)], -1)
    ids = torch.arange(self.num_envs, device=self.device)
    self._cmd["mode"] = torch.where(moving, WALK, IDLE)
    self._cmd["speed"] = torch.where(moving, speed.clamp(max=MAX_SPEED), -1.0)
    self._cmd["move"] = self.ctl.to_planner(ids, direction) * moving[:, None]
    self._cmd["face"] = self.ctl.to_planner(ids, face)

  def plan_summary(self) -> torch.Tensor:
    """`[E, 2 * len(SUMMARY_S) + 1]`: planned root positions ahead (robot heading frame) and
    the plan's remaining seconds."""
    plans, robot = self.ctl.plans, self.ctl.robot
    rows = torch.arange(self.num_envs, device=self.device)
    last = (plans.length - 1).clamp(min=0)
    yaw = yaw_of(self.ctl.delta)
    cp, sp = torch.cos(yaw), torch.sin(yaw)
    heading = robot.data.heading_w
    ch, sh = torch.cos(heading), torch.sin(heading)
    rel = []
    for t in SUMMARY_S:
      f = torch.minimum(plans.frame + int(round(t * sonic.CONTROL_FPS)), last)
      p = plans.pos[rows, f, :2]
      w = torch.stack([cp * p[:, 0] - sp * p[:, 1], sp * p[:, 0] + cp * p[:, 1]], -1) + self.ctl.origin
      d = w - robot.data.root_link_pos_w[:, :2]
      rel.append(torch.stack([ch * d[:, 0] + sh * d[:, 1], -sh * d[:, 0] + ch * d[:, 1]], -1))
    left = ((last - plans.frame).float() / sonic.CONTROL_FPS)[:, None]
    return torch.cat([*rel, left], -1)

  def _observations(self, obs: dict) -> TensorDict:
    extra = torch.cat([self.plan_summary(), self.prev_action], -1)
    return TensorDict({k: torch.cat([v, extra], -1) for k, v in obs.items()}, batch_size=[self.num_envs])
