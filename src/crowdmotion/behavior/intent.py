"""Goal intent for walled plazas: where each human wants to go, and when it waits.

Each human walks to random goals at a per-goal preferred speed and, on arrival,
sometimes waits before choosing the next goal. Spawn points and goals avoid
circular obstacles. The output is a preferred planar velocity per human, which
`LocalSteering` turns into a command. These are controllable heuristics for
demos, not calibrated pedestrian behavior.

Tensors are `[E, M, ...]` (environments x humans). All draws come from one
generator passed in by the owner, in a fixed order (see `PlazaBehavior`).
"""

from __future__ import annotations

import math

import torch


class GoalIntent:
  def __init__(self, cfg, envs: int, humans: int, gen: torch.Generator, device):
    self.cfg, self.E, self.M, self.device = cfg, envs, humans, torch.device(device)
    self.gen = gen
    self.goal = torch.zeros(envs, humans, 2, device=self.device)
    self.speed = torch.zeros(envs, humans, device=self.device)
    self.wait_until = torch.full((envs, humans), -1.0, device=self.device)
    obs = cfg.obstacles or ((1e6, 1e6, 0.0),)
    self.obstacles = torch.tensor(obs, dtype=torch.float32, device=self.device)  # [O, 3]

  def _rand(self, *shape) -> torch.Tensor:
    return torch.rand(*shape, generator=self.gen).to(self.device)

  def _free_points(self, n: int, margin: float) -> torch.Tensor:
    """`[n, 2]` uniform points inside the walls, clear of obstacles."""
    c = self.cfg
    lo = torch.tensor([-c.half_x + margin, -c.half_z + margin], device=self.device)
    hi = -lo
    out = lo + (hi - lo) * self._rand(n, 2)
    for _ in range(100):
      d = (out[:, None] - self.obstacles[None, :, :2]).norm(dim=-1)
      bad = (d < self.obstacles[None, :, 2] + 0.8).any(1)
      if not bad.any():
        break
      out[bad] = lo + (hi - lo) * self._rand(int(bad.sum()), 2)
    return out

  def _new_goals(self, env_ids: torch.Tensor, human_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    self.goal[env_ids, human_ids] = self._free_points(n, 1.0)
    lo, hi = self.cfg.speed_range
    self.speed[env_ids, human_ids] = lo + (hi - lo) * self._rand(n)
    self.wait_until[env_ids, human_ids] = -1.0

  def reset(self, env_ids: torch.Tensor, min_gap: float = 1.2) -> tuple[torch.Tensor, torch.Tensor]:
    """New goals and non-overlapping start states `xz [e, M, 2]`, `heading [e, M]`."""
    env_ids = env_ids.to(self.device)
    starts = []
    for _ in env_ids.tolist():
      pts: list[torch.Tensor] = []
      while len(pts) < self.M:
        p = self._free_points(1, 0.8)[0]
        if all((p - q).norm() > min_gap for q in pts):
          pts.append(p)
      starts.append(torch.stack(pts))
    e = env_ids.repeat_interleave(self.M)
    h = torch.arange(self.M, device=self.device).repeat(len(env_ids))
    self._new_goals(e, h)
    heading = (self._rand(len(env_ids), self.M) * 2 - 1) * math.pi
    return torch.stack(starts), heading

  def preferred_velocity(self, t: torch.Tensor, xz: torch.Tensor, env_ids: torch.Tensor):
    """Advance arrivals and waits at times `t [e]`; return `(goal_vel [e, M, 2],
    speed [e, M], waiting [e, M])` for humans at `xz [e, M, 2]`."""
    c = self.cfg
    goal, speed, wait = self.goal[env_ids], self.speed[env_ids], self.wait_until[env_ids]
    t = t[:, None].expand_as(wait)
    dist = (goal - xz).norm(dim=-1)
    arrived = dist < c.arrive_m
    start_wait = arrived & (wait < 0) & (self._rand(*wait.shape) < c.wait_probability)
    wait = torch.where(start_wait, t + c.wait_s[0] + (c.wait_s[1] - c.wait_s[0]) * self._rand(*wait.shape), wait)
    self.wait_until[env_ids] = wait
    done = arrived & ~start_wait & (wait < t)
    ei, hi = torch.nonzero(done, as_tuple=True)
    self._new_goals(env_ids[ei], hi)
    goal, speed, wait = self.goal[env_ids], self.speed[env_ids], self.wait_until[env_ids]

    to_goal = goal - xz
    dist = to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    goal_vel = to_goal / dist * speed[..., None] * (dist / 1.0).clamp(max=1.0)
    waiting = wait >= t
    goal_vel = torch.where(waiting[..., None], torch.zeros_like(goal_vel), goal_vel)
    return goal_vel, speed, waiting
