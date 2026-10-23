"""Root-path planners: from a human's executed root state to a world XZ path.

Paths are `[B, frames, 2]` world positions for frames 1..frames after the current
one, at the motion frame rate. Heading 0 faces +Z; positive turns toward +X.
"""

from __future__ import annotations

import math

import torch

from crowdmotion.core.rng import row_generator_seed


def plan_path(xz, vel, speed, heading, frames: int, fps: float, blend_s: float = 0.5):
  """Path from the executed state toward a `(speed, heading)` command (ARDY demo style).

  Velocity blends linearly from the current velocity to the commanded one over
  `blend_s`, then holds; positions integrate.
  """
  target = torch.stack((speed * torch.sin(heading), speed * torch.cos(heading)), -1)
  steps = torch.arange(1, frames + 1, device=xz.device, dtype=torch.float32)
  alpha = (steps / max(1.0, blend_s * fps)).clamp(max=1.0)[None, :, None]
  v = (1 - alpha) * vel[:, None] + alpha * target[:, None]
  return xz[:, None] + torch.cumsum(v, 1) / fps


def plan_arc(xz, vel, speed, heading, yaw_rate, frames: int, fps: float, blend_s: float = 0.5,
             max_turn: torch.Tensor | None = None):
  """Path and per-frame heading for a command whose heading turns at `yaw_rate`.

  The commanded heading starts at `heading` and turns at `yaw_rate` (rad/s, positive
  toward +X), stopping after `max_turn` radians when given. Velocity blends from the
  current velocity to the commanded one over `blend_s`, as in `plan_path`, so a
  constant speed `v` traces an arc of radius `v / |yaw_rate|` and a zero speed turns
  in place. Returns `(path [B, frames, 2], heading [B, frames])`.
  """
  t = torch.arange(1, frames + 1, device=xz.device, dtype=torch.float32) / fps
  turn = yaw_rate[:, None] * t
  if max_turn is not None:
    turn = torch.maximum(torch.minimum(turn, max_turn[:, None]), -max_turn[:, None])
  h = heading[:, None] + turn
  target = speed[:, None, None] * torch.stack((torch.sin(h), torch.cos(h)), -1)
  alpha = (torch.arange(1, frames + 1, device=xz.device, dtype=torch.float32)
           / max(1.0, blend_s * fps)).clamp(max=1.0)[None, :, None]
  v = (1 - alpha) * vel[:, None] + alpha * target
  return xz[:, None] + torch.cumsum(v, 1) / fps, h


def random_arc_paths(
  start_xz: torch.Tensor,
  start_heading: torch.Tensor,
  frames: int,
  fps: float,
  seeds: torch.Tensor,
  counters: torch.Tensor,
  speed_range=(0.5, 1.6),
  max_turn_rate=math.radians(60.0),
) -> torch.Tensor:
  """Constant-speed, constant-turn-rate arcs from each human's current root.

  Speed and turn rate are drawn per human from (seed, counter), so a path does
  not depend on batch composition.
  """
  device = start_xz.device
  gen = torch.Generator()
  params = []
  for seed, counter in zip(seeds.tolist(), counters.tolist()):
    gen.manual_seed(row_generator_seed(seed, counter))
    params.append(torch.rand(2, generator=gen))
  u = torch.stack(params).to(device)
  speed = speed_range[0] + (speed_range[1] - speed_range[0]) * u[:, 0]
  turn = (2 * u[:, 1] - 1) * max_turn_rate
  t = torch.arange(1, frames + 1, device=device, dtype=torch.float32) / fps
  heading = start_heading[:, None] + turn[:, None] * t
  # Integrate exactly for a constant-rate arc; straight line when turn ~ 0.
  small = turn.abs() < 1e-4
  safe = torch.where(small, torch.ones_like(turn), turn)
  dx = torch.where(
    small[:, None],
    speed[:, None] * t * torch.sin(start_heading)[:, None],
    speed[:, None] / safe[:, None] * (torch.cos(start_heading[:, None]) - torch.cos(heading)),
  )
  dz = torch.where(
    small[:, None],
    speed[:, None] * t * torch.cos(start_heading)[:, None],
    speed[:, None] / safe[:, None] * (torch.sin(heading) - torch.sin(start_heading[:, None])),
  )
  return torch.stack((start_xz[:, None, 0] + dx, start_xz[:, None, 1] + dz), -1)


def straight_paths(root_xz: torch.Tensor, heading: torch.Tensor, speed: torch.Tensor, frames: int,
                   fps: float) -> torch.Tensor:
  """Straight lines along each human's heading at `speed`."""
  t = torch.arange(1, frames + 1, device=speed.device, dtype=torch.float32) / fps
  direction = torch.stack((torch.sin(heading), torch.cos(heading)), -1)
  return root_xz[:, None] + (speed[:, None] * t)[..., None] * direction[:, None]
