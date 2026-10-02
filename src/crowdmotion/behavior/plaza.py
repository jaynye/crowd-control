"""Plaza behavior: `GoalIntent` then `LocalSteering`, as one command source.

Coordinates are motion-library XZ in meters, heading 0 faces +Z.

Randomness: one torch generator, seeded once, drives every draw for all
environments and humans, in call order. Results therefore depend on the call
sequence and batch composition. This matches the saved baselines; per-human
streams (required for training) are a planned behavior change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crowdmotion.behavior.intent import GoalIntent
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

  @classmethod
  def from_dict(cls, d: dict | None) -> "PlazaSteeringCfg":
    return from_dict(cls, d, what="steering")


class PlazaBehavior:
  """Command source for `E` plazas of `M` humans each."""

  def __init__(self, envs: int, humans: int, cfg: PlazaSteeringCfg, seed: int, device):
    self.cfg, self.E, self.M, self.device = cfg, envs, humans, torch.device(device)
    self.gen = torch.Generator(device="cpu").manual_seed(seed)
    self.intent = GoalIntent(cfg, envs, humans, self.gen, device)
    self.steering = LocalSteering(cfg, humans, device)

  @property
  def goal(self) -> torch.Tensor:
    return self.intent.goal

  @property
  def wait_until(self) -> torch.Tensor:
    return self.intent.wait_until

  def reset(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """New goals and start states `xz [e, M, 2]`, `heading [e, M]` for `env_ids`."""
    return self.intent.reset(env_ids)

  @torch.no_grad()
  def plaza_command(self, t: torch.Tensor, xz, vel, heading, env_ids: torch.Tensor):
    """`(speed [e, M], heading [e, M])` for envs `env_ids` at their times `t [e]`."""
    goal_vel, speed, waiting = self.intent.preferred_velocity(t, xz, env_ids)
    return self.steering.command(xz, vel, heading, goal_vel, speed, waiting)

  def command(self, t: torch.Tensor, kin, groups: torch.Tensor):
    """Controller interface: flat `[e*M]` kinematics for env ids `groups`."""
    e, M = len(groups), self.M
    speed, heading = self.plaza_command(
      t, kin.root_xz.view(e, M, 2), kin.vel.view(e, M, 2), kin.heading.view(e, M), groups
    )
    return speed.reshape(-1), heading.reshape(-1)
