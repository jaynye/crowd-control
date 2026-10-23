"""Local steering: a preferred velocity, neighbors, walls and obstacles to a command.

- time-to-closest-approach avoidance of neighbors in the same environment,
  which takes priority over the goal;
- a hard push at contact range;
- soft wall repulsion and circular static obstacles.

Deterministic: no random draws. Tensors are `[E, M, ...]`.
"""

from __future__ import annotations

import math

import torch

from crowdmotion.core.geometry import wrap


class LocalSteering:
  def __init__(self, cfg, humans: int, device, envs: int | None = None):
    self.cfg, self.M, self.device = cfg, humans, torch.device(device)
    obs = cfg.obstacles or ((1e6, 1e6, 0.0),)
    self.obstacles = torch.tensor(obs, dtype=torch.float32, device=self.device)  # [O, 3]
    self.envs = envs
    self._turning = self._turn_target = None

  def reset(self, env_ids):
    if self._turning is not None:
      self._turning[env_ids] = False

  def motion_options(self):
    options = {}
    if self.cfg.relax_orientation_for_avoidance:
      options["orientation_weight"] = (~self.evasive).float().reshape(-1)
    if self.cfg.turn_before_walk:
      options.update(turn_target=self.turn_target.reshape(-1), turn_active=self.turn_active.reshape(-1))
    return options

  @torch.no_grad()
  def command(self, xz, vel, heading, goal_vel, speed, waiting, *, env_ids=None):
    """`(speed [e, M], heading [e, M])` toward `goal_vel` while avoiding others."""
    c = self.cfg
    # Neighbors: repel along the separation at the time of closest approach.
    eye = torch.eye(self.M, dtype=torch.bool, device=self.device)
    r = xz[:, :, None] - xz[:, None]  # [e, i, j, 2], i minus j
    rv = vel[:, :, None] - vel[:, None]
    tau = (-(r * rv).sum(-1) / (rv * rv).sum(-1).clamp(min=1e-6)).clamp(0.0, c.horizon_s)
    sep = r + rv * tau[..., None]
    d = sep.norm(dim=-1).masked_fill(eye, float("inf"))
    push = (c.personal_m - d).clamp(min=0.0) / (1.0 + tau)
    avoid = c.avoid_gain * (sep / d.clamp(min=1e-6)[..., None] * push[..., None]).sum(2)
    now = r.norm(dim=-1).masked_fill(eye, float("inf"))
    close = (c.contact_m - now).clamp(min=0.0)
    avoid = avoid + 6.0 * (r / now.clamp(min=1e-6)[..., None] * close[..., None]).sum(2)
    # Avoidance takes priority: the goal term yields as avoidance grows.
    yield_ = (1.0 - avoid.norm(dim=-1) / speed.clamp(min=0.3)).clamp(0.0, 1.0)
    desired = goal_vel * yield_[..., None] + avoid

    # Walls and static obstacles.
    lim = torch.tensor([c.half_x, c.half_z], device=self.device) - c.wall_margin_m
    desired = desired - torch.sign(xz) * (xz.abs() - lim).clamp(min=0.0) * 3.0
    away = xz[:, :, None] - self.obstacles[None, None, :, :2]
    dd = away.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    desired = desired + (away / dd * (self.obstacles[None, None, :, 2:3] + 0.9 - dd).clamp(min=0.0) * 3.0).sum(2)

    cap = torch.where(waiting, torch.full_like(speed, 0.6), speed * 1.15)
    norm = desired.norm(dim=-1)
    desired = desired * (cap / norm.clamp(min=1e-6)).clamp(max=1.0)[..., None]
    out_speed = desired.norm(dim=-1)
    out_heading = torch.atan2(desired[..., 0], desired[..., 1])
    if c.forward_steering or c.relax_orientation_for_avoidance or c.turn_before_walk:
      # Ordinary redirects are realized as forward arcs through body turning.
      # Imminent conflicts and close walls/obstacles retain the full avoidance
      # vector, including deliberate lateral or backward escape commands.
      imminent = ((tau < 0.8) & (d < c.contact_m)).any(-1)
      near_wall = (xz.abs() > lim).any(-1)
      near_obstacle = (dd[..., 0] < self.obstacles[None, None, :, 2] + 0.6).any(-1)
      self.evasive = (now.min(-1).values < c.contact_m) | imminent | near_wall | near_obstacle
    if c.turn_before_walk:
      if self._turning is None:
        envs = self.envs or xz.shape[0]
        self._turning = torch.zeros(envs, self.M, dtype=torch.bool, device=self.device)
        self._turn_target = torch.zeros(envs, self.M, device=self.device)
      ids = torch.arange(xz.shape[0], device=self.device) if env_ids is None else env_ids
      turning, target = self._turning[ids], self._turn_target[ids]
      ordinary = ~self.evasive & ~waiting & (out_speed >= .15)
      done = wrap(target - heading).abs() <= math.radians(c.redirect_finish_deg)
      turning = turning & ordinary & ~done
      begin = ordinary & ~turning & (wrap(out_heading - heading).abs() >= math.radians(c.redirect_start_deg))
      target = torch.where(begin, out_heading, target)
      turning = turning | begin
      self._turning[ids], self._turn_target[ids] = turning, target
      self.turn_active, self.turn_target = turning, target
      # Separate orientation intent from the slow translating path. A stationary
      # request still retains a body target via motion_options and TurnCost.
      out_heading = torch.where(turning, heading, out_heading)
      out_speed = torch.where(turning, out_speed.clamp(max=c.redirect_speed_mps), out_speed)
    elif self._turning is not None:
      ids = torch.arange(xz.shape[0], device=self.device) if env_ids is None else env_ids
      self._turning[ids] = False
    if c.forward_steering:
      error = wrap(out_heading - heading)
      lead = math.radians(c.forward_lead_deg)
      ordinary_heading = heading + error.clamp(-lead, lead)
      ordinary_speed = out_speed * torch.cos(error).clamp(min=c.forward_speed_floor)
      out_heading = torch.where(self.evasive, out_heading, ordinary_heading)
      out_speed = torch.where(self.evasive, out_speed, ordinary_speed)
    still = out_speed < 0.15
    return out_speed.masked_fill(still, 0.0), torch.where(still, heading, out_heading)
