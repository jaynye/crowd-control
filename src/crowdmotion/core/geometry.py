"""Planar frames and coordinate conventions.

Motion coordinates follow ARDY: Y-up, meters, heading 0 faces +Z, positive yaw
turns toward +X. "Local" planar vectors are (lateral, forward) in the frame
facing a heading. MuJoCo is Z-up: `(x, y, z)_mj = (x, -z, y)_motion`.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def to_local(xz: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
  """World XZ vectors to (lateral, forward) in the frame facing `heading`."""
  c, s = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
  x, z = xz[..., :1], xz[..., 1:]
  return torch.cat((x * c - z * s, x * s + z * c), -1)


def to_world(lf: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
  """(lateral, forward) in the frame facing `heading` to world XZ."""
  c, s = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
  lat, fwd = lf[..., :1], lf[..., 1:]
  return torch.cat((lat * c + fwd * s, -lat * s + fwd * c), -1)


def wrap(a: torch.Tensor) -> torch.Tensor:
  """Angles to [-pi, pi)."""
  return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def wrap_np(angle):
  """NumPy `wrap`."""
  return (angle + np.pi) % (2 * np.pi) - np.pi


def body_local(joints: torch.Tensor, root_xz: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
  """Joints `[..., J, 3]` relative to the root XZ and heading; Y stays absolute."""
  rel = to_local(joints[..., [0, 2]] - root_xz[..., None, :], heading[..., None])
  return torch.stack((rel[..., 0], joints[..., 1], rel[..., 1]), -1)


def body_world(local: torch.Tensor, root_xz: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
  """Inverse of `body_local`."""
  xz = to_world(local[..., [0, 2]], heading[..., None]) + root_xz[..., None, :]
  return torch.stack((xz[..., 0], local[..., 1], xz[..., 1]), -1)


def motion_to_mujoco(p: torch.Tensor) -> torch.Tensor:
  """Motion-library Y-up points to MuJoCo Z-up (a proper rotation)."""
  return torch.stack((p[..., 0], -p[..., 2], p[..., 1]), -1)
