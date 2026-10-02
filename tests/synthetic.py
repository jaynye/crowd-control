"""Synthetic motion libraries for CPU tests (no checkpoints needed)."""

from __future__ import annotations

import torch

N, T, L, J = 6, 32, 4, 5


def synthetic_library(seed: int = 0) -> dict:
  """`N` smooth curving streams of `T` frames in the `save_library` layout (edge L = 4)."""
  g = torch.Generator().manual_seed(seed)
  fps = 20.0
  rate = (torch.rand(N, 1, generator=g) - 0.5) * 2.0
  heading = rate * torch.arange(T)[None] / fps + torch.rand(N, 1, generator=g) * 6
  speed = 0.5 + torch.rand(N, 1, generator=g)
  step = torch.stack((torch.sin(heading), torch.cos(heading)), -1) * speed[..., None] / fps
  xz = torch.cumsum(step, 1)
  root = torch.stack((xz[..., 0], torch.full_like(xz[..., 0], 0.9), xz[..., 1]), -1)
  offsets = torch.randn(J, 3, generator=g) * 0.3
  offsets[0] = 0
  wobble = 0.05 * torch.sin(torch.arange(T)[None, :, None, None] * 0.7 + torch.randn(N, 1, J, 3, generator=g))
  local = offsets[None, None] + wobble
  c, s = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
  wx = local[..., 0] * c + local[..., 2] * s
  wz = -local[..., 0] * s + local[..., 2] * c
  joints = torch.stack((wx + root[..., None, 0], local[..., 1] + 0.9, wz + root[..., None, 2]), -1)
  return {
    "posed_joints": joints,
    "root_pos": root,
    "heading": heading,
    "contacts": torch.rand(N, T, 2, generator=g) > 0.5,
    "meta": {
      "fps": fps, "edge_frames": L, "history": 4, "joint_names": [f"j{i}" for i in range(J)],
      "parents": [-1] + [0] * (J - 1), "foot_idx": [1, 2], "head_idx": 3,
      "neutral_head_height": 1.0,
    },
  }
