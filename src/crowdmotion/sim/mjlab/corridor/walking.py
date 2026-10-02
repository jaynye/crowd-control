"""Seeded, persistent waypoint walking for a level corridor fixture."""

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class CorridorWalkCfg(CommandTermCfg):
  entity_name: str
  neighbor_names: tuple[str, ...] = ()
  seed: int = 0
  speed_range: tuple[float, float] = (0.5, 1.1)
  pause_probability: float = 0.15

  def __post_init__(self) -> None:
    low, high = self.speed_range
    if not 0 <= low <= high or not 0 <= self.pause_probability <= 1:
      raise ValueError(
        "Require ordered nonnegative speeds and pause probability [0, 1]"
      )
    if not 0 < self.resampling_time_range[0] <= self.resampling_time_range[1]:
      raise ValueError("Resampling times must be positive and ordered")

  def build(self, env: ManagerBasedRlEnv) -> "CorridorWalk":
    return CorridorWalk(self, env)


class CorridorWalk(CommandTerm):
  """Waypoint intent at seconds scale; finite-force actuators handle substeps.

  The repulsion here is a baseline for this empty rectangular corridor, not a
  general navigation or social behavior model. RNG state is private per person
  and environment, including command durations, so partial resets are isolated.
  """

  cfg: CorridorWalkCfg

  def __init__(self, cfg: CorridorWalkCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.human: Entity = env.scene[cfg.entity_name]
    self.neighbors: list[Entity] = [env.scene[n] for n in cfg.neighbor_names]
    self._velocity = torch.zeros(self.num_envs, 3, device=self.device)
    self.goal = torch.zeros(self.num_envs, 2, device=self.device)
    self.speed = torch.zeros(self.num_envs, device=self.device)
    self.paused = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._rng = (
      (torch.arange(self.num_envs, device=self.device) + 1) * 104729
      + (cfg.seed % 2147483646) * 1009
    ) % 2147483646 + 1
    # Decorrelate neighboring seeds before the first sample.
    ids = torch.arange(self.num_envs, device=self.device)
    for _ in range(5):
      self._uniform(ids)

  @property
  def command(self) -> torch.Tensor:
    return self._velocity

  def _uniform(self, env_ids: torch.Tensor) -> torch.Tensor:
    # Park-Miller generator: the product fits in int64 on CPU and GPU.
    state = (self._rng[env_ids] * 48271) % 2147483647
    self._rng[env_ids] = state
    return state.to(torch.float32) / 2147483647

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self._velocity[env_ids] = 0.0
    return super().reset(env_ids)

  def _resample(self, env_ids: torch.Tensor) -> None:
    low, high = self.cfg.resampling_time_range
    self.time_left[env_ids] = low + (high - low) * self._uniform(env_ids)
    self._resample_command(env_ids)
    self.command_counter[env_ids] += 1

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.goal[env_ids, 0] = torch.where(self._uniform(env_ids) < 0.5, -4.5, 4.5)
    self.goal[env_ids, 1] = 1.8 * (self._uniform(env_ids) - 0.5)
    low, high = self.cfg.speed_range
    self.speed[env_ids] = low + (high - low) * self._uniform(env_ids)
    self.paused[env_ids] = self._uniform(env_ids) < self.cfg.pause_probability

  def _update_metrics(self) -> None:
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    xy = self.human.data.joint_pos[ids, :2]
    delta = self.goal[ids] - xy
    arrived = (delta.norm(dim=-1) < 0.25) & ~self.paused[ids]
    self._resample(ids[arrived])
    delta = self.goal[ids] - xy
    distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    desired = delta / distance * self.speed[ids, None] * distance.clamp(max=1.0)
    for neighbor in self.neighbors:
      away = xy - neighbor.data.joint_pos[ids, :2]
      separation = away.norm(dim=-1, keepdim=True).clamp_min(0.05)
      desired += away / separation * (1.1 - separation).clamp_min(0.0) * 2.0
    # Soft boundary steering supplements physical wall contacts.
    bounds = torch.tensor([5.0, 1.25], device=self.device)
    desired -= xy.sign() * (xy.abs() - bounds).clamp_min(0.0) * 4.0
    desired *= (
      self.speed[ids, None] / desired.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ).clamp(max=1.0)
    desired[self.paused[ids]] = 0.0
    current = self._velocity[ids, :2]
    change = desired - current
    change *= (
      2.0 * self._env.step_dt / change.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ).clamp(max=1.0)
    self._velocity[ids, :2] = current + change
    yaw = self.human.data.joint_pos[ids, 2]
    heading = torch.atan2(desired[:, 1], desired[:, 0])
    error = torch.atan2(torch.sin(heading - yaw), torch.cos(heading - yaw))
    self._velocity[ids, 2] = (2.0 * error).clamp(-1.5, 1.5)
    self._velocity[ids[self.paused[ids]], 2] = 0.0
    self.human.set_joint_velocity_target(self._velocity[ids], env_ids=ids)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for idx in visualizer.get_env_indices(self.num_envs):
      start = self.human.data.joint_pos[idx, :2].cpu().numpy()
      velocity = self._velocity[idx, :2].cpu().numpy()
      visualizer.add_arrow(
        start=np.array([start[0], start[1], 1.9]),
        end=np.array([start[0] + velocity[0], start[1] + velocity[1], 1.9]),
        color=(1.0, 0.7, 0.0, 1.0),
        label=self.cfg.entity_name,
      )
