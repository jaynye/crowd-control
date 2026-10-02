"""Posture and ground contact of executed frames."""

from __future__ import annotations

import torch

from crowdmotion.core.skeleton import HUNCH_RATIO


def _pct(x: torch.Tensor, q: float) -> float:
  return float(torch.quantile(x.float().flatten(), q)) if x.numel() else float("nan")


def hunch_stats(joints: torch.Tensor, head_idx: int, neutral_head_height: float) -> dict:
  head = joints[:, :, head_idx, 1]
  return {
    "frac_frames_hunched": float((head < HUNCH_RATIO * neutral_head_height).float().mean()),
    "head_height_ratio_median": float(head.median()) / neutral_head_height,
  }


def ground_stats(fpos: torch.Tensor, contacts: torch.Tensor) -> dict:
  """`fpos [N, T, 4, 3]`: toe joints are the second of each foot chain; walking has
  no flight phase."""
  toes = fpos[:, :, [1, 3], 1]
  lowest = toes.min(dim=-1).values
  lowest_foot = fpos[..., 1].min(dim=-1).values
  return {
    "toe_min_height_p05_m": _pct(lowest, 0.05),
    "frac_frames_penetration_gt_2cm": float((lowest < -0.02).float().mean()),
    "frac_frames_no_contact": float((~contacts.any(dim=-1)).float().mean()),
    "frac_frames_lowest_foot_gt_15cm": float((lowest_foot > 0.15).float().mean()),
  }
