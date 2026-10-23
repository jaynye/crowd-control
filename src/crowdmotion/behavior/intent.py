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

from crowdmotion.core.geometry import wrap
from crowdmotion.core.rng import hash_uniform, human_seeds


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


class CommittedTurnIntent(GoalIntent):
  """Optional WALK/WAIT/TURN intent with a sampled, retained body-heading target.

  Spawning uses the unchanged goal-intent generator. New decisions use independent
  per-human streams, so no turn draw consumes that legacy generator. A turn ends
  only when executed body facing reaches its target; avoidance never resamples it.
  Preferred travel advances toward the target with a bounded lead over actual
  facing. Local steering retains priority for close avoidance, walls and obstacles.
  These are uncalibrated heuristics, not a model fitted to pedestrian observations.
  """

  WALK, WAIT, TURN = 0, 1, 2

  def __init__(self, cfg, envs, humans, gen, device, seed):
    super().__init__(cfg, envs, humans, gen, device)
    shape = (envs, humans)
    self.base_seed = seed
    self.mode = torch.zeros(shape, dtype=torch.long, device=self.device)
    self.target = torch.zeros(shape, device=self.device)
    self.until = torch.zeros(shape, device=self.device)
    self.walk_s = torch.zeros(shape, device=self.device)
    self.sampled_angle = torch.zeros(shape, device=self.device)
    self.turn_count = torch.zeros(shape, dtype=torch.long, device=self.device)
    self.counters = torch.zeros_like(self.turn_count)
    self.episodes = torch.full((envs,), -1, dtype=torch.long, device=self.device)
    self.seeds = torch.zeros_like(self.turn_count)
    weights = torch.tensor(cfg.turn_weights, device=self.device)
    self.cdf = weights.cumsum(0) / weights.sum()
    self.angles = torch.tensor(cfg.turn_angles_deg, device=self.device)

  def _draw(self, e, h):
    u = hash_uniform(self.seeds[e, h], self.counters[e, h], 7)
    self.counters[e, h] += 1
    return u

  def reset(self, env_ids, min_gap=1.2):
    xz, heading = super().reset(env_ids, min_gap)
    env_ids = env_ids.to(self.device)
    self.episodes[env_ids] += 1
    e = env_ids.repeat_interleave(self.M)
    h = torch.arange(self.M, device=self.device).repeat(len(env_ids))
    self.seeds[e, h] = human_seeds(self.base_seed, e * self.M + h, "plaza_turn", episode=self.episodes[e])
    self.counters[e, h] = 0
    u = self._draw(e, h)
    self.mode[e, h] = self.WALK
    self.target[env_ids] = heading
    self.until[e, h] = self.cfg.turn_walk_s[0] + u[:, 0] * (self.cfg.turn_walk_s[1] - self.cfg.turn_walk_s[0])
    self.turn_count[e, h] = 0
    self.sampled_angle[e, h] = 0
    return xz, heading

  def preferred_velocity(self, t, xz, env_ids, heading):
    c = self.cfg
    clock = t[:, None].expand_as(heading)
    mode = self.mode[env_ids]
    target = self.target[env_ids]
    error = wrap(target - heading)
    finished = (mode == self.TURN) & (error.abs() <= math.radians(c.turn_finish_deg))
    ei, hi = torch.nonzero(finished, as_tuple=True)
    self.mode[env_ids[ei], hi] = self.WALK
    self.until[env_ids[ei], hi] = clock[ei, hi] + self.walk_s[env_ids[ei], hi]

    # Only walking humans whose dwell ended sample; active turns keep their target.
    due = (self.mode[env_ids] == self.WALK) & (clock >= self.until[env_ids])
    ei, hi = torch.nonzero(due, as_tuple=True)
    e = env_ids[ei]
    if len(e):
      u = self._draw(e, hi)
      bins = torch.searchsorted(self.cdf, u[:, 0].contiguous()).clamp(max=len(self.angles) - 1)
      magnitude = (self.angles[bins] + (2 * u[:, 1] - 1) * c.turn_jitter_deg).clamp(1.0, 180.0)
      angle = torch.where(u[:, 2] < 0.5, -magnitude, magnitude) * (math.pi / 180)
      self.target[e, hi] = heading[ei, hi] + angle
      self.sampled_angle[e, hi] = angle
      self.turn_count[e, hi] += 1
      self.walk_s[e, hi] = c.turn_walk_s[0] + u[:, 3] * (c.turn_walk_s[1] - c.turn_walk_s[0])
      self.speed[e, hi] = c.speed_range[0] + u[:, 4] * (c.speed_range[1] - c.speed_range[0])
      wait = u[:, 5] < c.wait_probability
      self.mode[e, hi] = torch.where(wait, self.WAIT, self.TURN)
      self.wait_until[e, hi] = torch.where(wait, clock[ei, hi] + c.wait_s[0]
                                          + u[:, 6] * (c.wait_s[1] - c.wait_s[0]), -1.0)
    wake = (self.mode[env_ids] == self.WAIT) & (clock > self.wait_until[env_ids])
    ei, hi = torch.nonzero(wake, as_tuple=True)
    self.mode[env_ids[ei], hi] = self.TURN

    mode, target = self.mode[env_ids], self.target[env_ids]
    error = wrap(target - heading)
    # Keep the sampled sign when the target is exactly 180 degrees away.
    error = torch.where(error.abs() > math.pi - 1e-5,
                        self.sampled_angle[env_ids].sign() * math.pi, error)
    lead = math.radians(c.turn_lead_deg)
    travel_heading = heading + error.clamp(-lead, lead)
    scale = torch.where(mode == self.TURN, c.turn_speed_scale, 1.0)
    waiting = mode == self.WAIT
    speed = self.speed[env_ids]
    direction = torch.stack((torch.sin(travel_heading), torch.cos(travel_heading)), -1)
    velocity = direction * (speed * scale)[..., None]
    velocity = velocity.masked_fill(waiting[..., None], 0.0)
    # A virtual point records heading intent for trace consumers; it is not an arrival goal.
    self.goal[env_ids] = xz + 3 * torch.stack((torch.sin(target), torch.cos(target)), -1)
    return velocity, speed, waiting
