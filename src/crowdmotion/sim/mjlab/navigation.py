"""Persistent goal navigation settings and terminal rewards over frozen SONIC.

The optional task is recorded as a complete config in each training lineage. Legacy
velocity tasks do not use it. Potential shaping is applied at planner decisions, where
its discount matches PPO; it is never summed undiscounted over physics substeps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from crowdmotion.core.config import from_dict

REWARD_MODES = ("sparse", "potential", "tracking", "proximity", "regularized", "all", "centered")


@dataclass(frozen=True)
class NavigationCfg:
  reward_mode: str = "sparse"
  gamma_per_s: float = .99
  gae_lambda_per_s: float = .95
  rollout_s: int = 16
  deadline_s: float = 30.
  min_goal_m: float = 4.
  max_goal_m: float = 8.
  speed_mps: float = .8
  success_reward: float = 10.
  time_bonus: float = 2.
  failure_reward: float = 0.
  potential_weight: float = 1.
  action_l2: float = .01
  action_rate: float = .02

  def __post_init__(self):
    if self.reward_mode not in REWARD_MODES:
      raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
    if not all(math.isfinite(v) for v in self.__dict__.values() if isinstance(v, (int, float))):
      raise ValueError("navigation values must be finite")
    if not (0 < self.gamma_per_s <= 1 and 0 < self.gae_lambda_per_s <= 1):
      raise ValueError("navigation discounts must lie in (0, 1]")
    if not (self.rollout_s >= 1 and self.deadline_s > 0 and 0 < self.min_goal_m <= self.max_goal_m
            and self.speed_mps > 0 and self.success_reward > 0 and self.time_bonus >= 0
            and self.failure_reward <= 0 and self.potential_weight >= 0
            and self.action_l2 >= 0 and self.action_rate >= 0):
      raise ValueError("invalid navigation distances, times or rewards")

  @classmethod
  def from_dict(cls, d):
    return from_dict(cls, d, what="navigation")


def outcome(env):
  """Terminal physics state, cached before auto-reset. Failure wins arrival ties."""
  cached = getattr(env, "_navigation_pre", None)
  if cached is not None and cached["step"] == env.common_step_counter:
    return cached
  from crowdmotion.sim.mjlab.traversal import crowd_collision

  term = env.command_manager.get_term("twist")
  cfg = term.cfg.navigation
  dist = (term.goal_w - term.robot.data.root_link_pos_w[:, :2]).norm(dim=-1)
  elapsed = env.episode_length_buf * env.step_dt
  collision = crowd_collision(env)
  term.nav_contact |= collision
  fell = env.termination_manager.get_term("fell_over")
  deadline = elapsed >= cfg.deadline_s - 1e-6
  success = (dist < term.cfg.arrive_m) & ~term.nav_contact & ~fell & ~deadline
  if hasattr(term, "metrics"):
    term.metrics["goals_reached"] += success.float()
  cached = dict(step=env.common_step_counter, distance=dist.clone(), success=success.clone(),
                failure=(collision | fell | deadline).clone(), deadline=deadline.clone(), elapsed=elapsed.clone())
  env._navigation_pre = cached
  return cached


def goal_success(env):
  return outcome(env)["success"]


def goal_deadline(env):
  return outcome(env)["deadline"]


def terminal_reward(env):
  """An event reward, divided by dt because mjlab multiplies reward rates by dt."""
  state = outcome(env)
  term = env.command_manager.get_term("twist")
  cfg = term.cfg.navigation
  nominal = (term.nav_start_distance / cfg.speed_mps).clamp(min=env.step_dt)
  bonus = cfg.time_bonus / (1 + state["elapsed"] / nominal)
  value = state["success"] * (cfg.success_reward + bonus) + state["failure"] * cfg.failure_reward
  return value / env.step_dt


def goal_observation(env):
  term = env.command_manager.get_term("twist")
  delta = term.goal_w - term.robot.data.root_link_pos_w[:, :2]
  heading = term.robot.data.heading_w
  c, s = heading.cos(), heading.sin()
  local = torch.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), -1)
  remaining = (1 - env.episode_length_buf * env.step_dt / term.cfg.navigation.deadline_s).clamp(min=0)
  return torch.cat((local / 10., remaining[:, None], term.nav_start_distance[:, None] / 10.), -1)


def tracking_baseline(env):
  """Remove the positive tracking ceiling, isolating reward for prolonging episodes."""
  ceiling = sum(env.reward_manager.get_term_cfg(k).weight
                for k in ("track_linear_velocity", "track_angular_velocity", "goal_heading"))
  return torch.full((env.num_envs,), -ceiling, device=env.device)


def potential_reward(before, after, done, gamma, weight=1.):
  """Discount-correct shaping, with zero potential at every task terminal state."""
  return weight * (before - gamma * torch.where(done, 0., after))
