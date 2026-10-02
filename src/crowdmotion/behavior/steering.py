"""Local steering: a preferred velocity, neighbors, walls and obstacles to a command.

- time-to-closest-approach avoidance of neighbors in the same environment,
  which takes priority over the goal;
- a hard push at contact range;
- soft wall repulsion and circular static obstacles.

Deterministic: no random draws. Tensors are `[E, M, ...]`.
"""

from __future__ import annotations

import torch


class LocalSteering:
  def __init__(self, cfg, humans: int, device):
    self.cfg, self.M, self.device = cfg, humans, torch.device(device)
    obs = cfg.obstacles or ((1e6, 1e6, 0.0),)
    self.obstacles = torch.tensor(obs, dtype=torch.float32, device=self.device)  # [O, 3]

  @torch.no_grad()
  def command(self, xz, vel, heading, goal_vel, speed, waiting):
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
    still = out_speed < 0.15
    return out_speed.masked_fill(still, 0.0), torch.where(still, heading, out_heading)
