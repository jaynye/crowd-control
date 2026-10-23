"""Plaza behavior: `GoalIntent` then `LocalSteering`, as one command source.

Coordinates are motion-library XZ in meters, heading 0 faces +Z.

Randomness: one torch generator, seeded once, drives every draw for all
environments and humans, in call order. Results therefore depend on the call
sequence and batch composition. This matches the saved baselines; per-human
streams (required for training) are a planned behavior change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from crowdmotion.behavior.intent import CommittedTurnIntent, GoalIntent
from crowdmotion.behavior.steering import LocalSteering
from crowdmotion.core.config import from_dict


@dataclass
class PlazaSteeringCfg:
  half_x: float = 7.0
  half_z: float = 5.0
  speed_range: tuple[float, float] = (0.8, 1.6)
  wait_probability: float = 0.3
  wait_s: tuple[float, float] = (1.5, 4.0)
  arrive_m: float = 0.6
  personal_m: float = 1.2
  horizon_s: float = 3.0
  avoid_gain: float = 3.0
  contact_m: float = 0.9  # hard push below this current separation
  wall_margin_m: float = 0.8
  obstacles: tuple[tuple[float, float, float], ...] = ()  # (x, z, radius)
  committed_turns: bool = False
  turn_angles_deg: tuple[float, ...] = (45.0, 60.0, 90.0, 135.0, 180.0)
  turn_weights: tuple[float, ...] = (0.15, 0.20, 0.50, 0.10, 0.05)
  turn_jitter_deg: float = 10.0
  turn_walk_s: tuple[float, float] = (4.0, 12.0)
  turn_lead_deg: float = 25.0  # preferred travel leads executed facing by at most this
  turn_speed_scale: float = 0.7  # keep walking through turns; scarce slow-turn motion
  turn_finish_deg: float = 10.0
  forward_steering: bool = False
  forward_lead_deg: float = 30.0
  forward_speed_floor: float = 0.5
  relax_orientation_for_avoidance: bool = False
  turn_before_walk: bool = False  # retain a body target, slow down, then resume forward travel
  redirect_start_deg: float = 45.0
  redirect_finish_deg: float = 15.0
  redirect_speed_mps: float = 0.20  # desired translation while turning; soft graph request

  def __post_init__(self):
    if self.turn_before_walk and (not 0 < self.redirect_finish_deg < self.redirect_start_deg <= 180
                                  or not math.isfinite(self.redirect_speed_mps) or self.redirect_speed_mps < 0):
      raise ValueError("turn-before-walk needs 0 < finish < start <= 180 degrees and nonnegative speed")
    if self.forward_steering and (not 0 < self.forward_lead_deg <= 90
                                  or not 0 < self.forward_speed_floor <= 1):
      raise ValueError("forward steering needs lead in (0, 90] and speed floor in (0, 1]")
    if not self.committed_turns:
      return
    if (not self.turn_angles_deg or len(self.turn_angles_deg) != len(self.turn_weights)
        or any(not math.isfinite(x) or not 0 < x <= 180 for x in self.turn_angles_deg)
        or any(not math.isfinite(x) or x < 0 for x in self.turn_weights) or sum(self.turn_weights) <= 0):
      raise ValueError("turn angles and nonnegative weights must have matching nonempty lengths")
    if not 0 < self.turn_finish_deg < self.turn_lead_deg <= 90:
      raise ValueError("turn angles require 0 < finish < lead <= 90 degrees")
    if not 0 < self.turn_speed_scale <= 1 or not 0 <= self.turn_jitter_deg < 45:
      raise ValueError("turn speed scale must be in (0, 1], jitter in [0, 45)")
    if len(self.turn_walk_s) != 2 or not 0 < self.turn_walk_s[0] <= self.turn_walk_s[1] < float("inf"):
      raise ValueError("turn walk dwell must be a positive finite interval")

  @classmethod
  def from_dict(cls, d: dict | None) -> "PlazaSteeringCfg":
    return from_dict(cls, d, what="steering")


class PlazaBehavior:
  """Command source for `E` plazas of `M` humans each."""

  def __init__(self, envs: int, humans: int, cfg: PlazaSteeringCfg, seed: int, device):
    self.cfg, self.E, self.M, self.device = cfg, envs, humans, torch.device(device)
    self.gen = torch.Generator(device="cpu").manual_seed(seed)
    self.intent = (CommittedTurnIntent(cfg, envs, humans, self.gen, device, seed)
                   if cfg.committed_turns else GoalIntent(cfg, envs, humans, self.gen, device))
    self.steering = LocalSteering(cfg, humans, device, envs=envs)

  @property
  def goal(self) -> torch.Tensor:
    return self.intent.goal

  @property
  def wait_until(self) -> torch.Tensor:
    return self.intent.wait_until

  def reset(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """New goals and start states `xz [e, M, 2]`, `heading [e, M]` for `env_ids`."""
    self.steering.reset(env_ids)
    return self.intent.reset(env_ids)

  @torch.no_grad()
  def plaza_command(self, t: torch.Tensor, xz, vel, heading, env_ids: torch.Tensor):
    """`(speed [e, M], heading [e, M])` for envs `env_ids` at their times `t [e]`."""
    if self.cfg.committed_turns:
      goal_vel, speed, waiting = self.intent.preferred_velocity(t, xz, env_ids, heading)
    else:
      goal_vel, speed, waiting = self.intent.preferred_velocity(t, xz, env_ids)
    return self.steering.command(xz, vel, heading, goal_vel, speed, waiting, env_ids=env_ids)

  def command(self, t: torch.Tensor, kin, groups: torch.Tensor):
    """Controller interface: flat `[e*M]` kinematics for env ids `groups`."""
    e, M = len(groups), self.M
    speed, heading = self.plaza_command(
      t, kin.root_xz.view(e, M, 2), kin.vel.view(e, M, 2), kin.heading.view(e, M), groups
    )
    return speed.reshape(-1), heading.reshape(-1)

  def motion_options(self) -> dict:
    """Per-human priority for soft orientation costs on the command just produced."""
    return self.steering.motion_options()
