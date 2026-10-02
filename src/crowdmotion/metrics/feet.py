"""Foot sliding: horizontal speed of foot joints while they are planted.

Three definitions are in use and are kept separate:
- `contact_skate` (torch): planted = the model's contact label on two consecutive frames;
- `label_free_skate` (torch): planted = within a margin of the joint's own low height;
- `planted_foot_speed` (NumPy): contact-labeled, as in the behavior evaluator.
"""

from __future__ import annotations

import numpy as np
import torch


def _pct(x: torch.Tensor, q: float) -> float:
  return float(torch.quantile(x.float().flatten(), q)) if x.numel() else float("nan")


def contact_skate(fpos: torch.Tensor, contacts: torch.Tensor, fps: float) -> dict:
  """`fpos [N, T, F, 3]` foot joints, `contacts [N, T, F]`."""
  fvel = (fpos[:, 1:, :, [0, 2]] - fpos[:, :-1, :, [0, 2]]).norm(dim=-1) * fps
  both = contacts[:, 1:] & contacts[:, :-1]
  skate = fvel[both]
  return {
    "contact_fraction": float(contacts.float().mean()),
    "foot_skate_mean_mps": float(skate.mean()) if skate.numel() else float("nan"),
    "foot_skate_p95_mps": _pct(skate, 0.95),
    "foot_skate_frac_gt_0p1": float((skate > 0.1).float().mean()) if skate.numel() else float("nan"),
  }


def label_free_skate(fpos: torch.Tensor, fps: float, margin: float = 0.025) -> dict:
  """Foot sliding gated by joint height instead of the model's contact labels.

  A foot joint counts as planted when it is within `margin` of its own 5th
  percentile height for that human; skating is its horizontal speed on frames
  planted in two consecutive frames.
  """
  h = fpos[..., 1]  # [N, T, J]
  base = torch.quantile(h.float(), 0.05, dim=1, keepdim=True)
  planted = h < base + margin
  both = planted[:, 1:] & planted[:, :-1]
  vel = (fpos[:, 1:, :, [0, 2]] - fpos[:, :-1, :, [0, 2]]).norm(dim=-1) * fps
  skate = vel[both]
  return {
    "hgate_planted_fraction": float(planted.float().mean()),
    "hgate_skate_mean_mps": float(skate.mean()) if skate.numel() else float("nan"),
    "hgate_skate_p95_mps": _pct(skate, 0.95),
    "hgate_skate_frac_gt_0p1": float((skate > 0.1).float().mean()) if skate.numel() else float("nan"),
  }


def planted_foot_speed(joints: np.ndarray, foot_idx, contacts: np.ndarray, fps: float) -> np.ndarray:
  """Horizontal foot speeds (flat) on frames labeled in contact twice in a row."""
  feet = joints[:, :, foot_idx][:, :, :, [0, 2]]
  foot_speed = np.linalg.norm(np.diff(feet, axis=1), axis=-1) * fps
  planted = contacts[:, 1:] & contacts[:, :-1]
  return foot_speed[planted]
