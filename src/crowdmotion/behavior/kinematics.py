"""Executed root state of each human, and how its velocity is estimated.

Two estimators exist because two code paths grew them; both are kept bit-exact
so saved results reproduce. They differ on the first edges after a reset.

- `WindowVelocity`: displacement over min(k, n-1) frames of the *executed* roots,
  zero until two frames exist (quality benchmark, library generation).
- `BufferVelocity`: a fixed buffer of the last `frames` roots of each *full* step,
  seeded with the start position, divided by `frames - 1` (behavior evaluator,
  mjlab GraphCrowd).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Kinematics:
  """Per-human planar state `[H, ...]` the command source and planner read."""

  root_xz: torch.Tensor  # last executed root
  heading: torch.Tensor  # last executed body heading
  vel: torch.Tensor  # velocity estimate

  def rows(self, rows: torch.Tensor | None) -> "Kinematics":
    if rows is None:
      return self
    return Kinematics(self.root_xz[rows], self.heading[rows], self.vel[rows])


class WindowVelocity:
  grouped = False  # its history grows, so rows cannot be stepped in groups

  def __init__(self, fps: float, k: int = 5):
    self.fps, self.k = fps, k

  def reset(self, root_xz: torch.Tensor) -> dict:
    n = root_xz.shape[0]
    return {"recent": root_xz.new_zeros(n, 0, 2), "vel": root_xz.new_zeros(n, 2)}

  def update(self, buf: dict, step_root_xz: torch.Tensor, take: int) -> None:
    """`step_root_xz [H, F, 2]`: the step's roots; the first `take` were executed."""
    recent = torch.cat((buf["recent"], step_root_xz[:, :take]), 1)[:, -(self.k + 1):]
    buf["recent"] = recent
    if recent.shape[1] >= 2:
      n = min(self.k, recent.shape[1] - 1)
      buf["vel"] = (recent[:, -1] - recent[:, -1 - n]) * self.fps / n


class BufferVelocity:
  grouped = True

  def __init__(self, fps: float, frames: int = 6):
    self.fps, self.frames = fps, frames

  def reset(self, root_xz: torch.Tensor) -> dict:
    return {
      "recent": root_xz[:, None].expand(-1, self.frames, -1).clone(),
      "vel": root_xz.new_zeros(root_xz.shape[0], 2),
    }

  def update(self, buf: dict, step_root_xz: torch.Tensor, take: int) -> None:
    recent = torch.cat((buf["recent"], step_root_xz), 1)[:, -self.frames:]
    buf["recent"] = recent
    buf["vel"] = (recent[:, -1] - recent[:, 0]) * self.fps / (self.frames - 1)
