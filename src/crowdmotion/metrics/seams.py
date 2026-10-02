"""Seams: motion at generation or edge boundaries compared with the interior."""

from __future__ import annotations

import torch


def _pct(x: torch.Tensor, q: float) -> float:
  return float(torch.quantile(x.float().flatten(), q)) if x.numel() else float("nan")


def seam_stats(joints: torch.Tensor, vel: torch.Tensor, root_xz: torch.Tensor, boundaries: list[int],
               fps: float) -> dict:
  """Joint displacement, root acceleration and root step change at boundaries vs elsewhere.

  `boundaries`: frame index where each call's or edge's frames start.
  """
  out: dict = {}
  frames = root_xz.shape[1]
  step = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1).mean(dim=-1)  # [N, T-1]
  acc = (vel[:, 1:] - vel[:, :-1]).norm(dim=-1) * fps
  seam_idx = [b - 1 for b in boundaries if b > 0]
  mask = torch.zeros(frames - 1, dtype=torch.bool)
  mask[seam_idx] = True
  if seam_idx:
    out["seam_joint_step_mean_m"] = float(step[:, mask].mean())
    out["seam_joint_step_p95_m"] = _pct(step[:, mask], 0.95)
    out["interior_joint_step_mean_m"] = float(step[:, ~mask].mean())
    out["seam_to_interior_step_ratio"] = out["seam_joint_step_mean_m"] / max(1e-9, out["interior_joint_step_mean_m"])
    out["seam_root_acc_p95_mps2"] = _pct(acc[:, mask], 0.95)
    out["interior_root_acc_p95_mps2"] = _pct(acc[:, ~mask], 0.95)
    rstep = (root_xz[:, 1:] - root_xz[:, :-1]).norm(dim=-1)
    rjerk = (rstep[:, 1:] - rstep[:, :-1]).abs()  # change in per-frame root travel
    jmask = mask[1:]
    out["seam_root_step_change_p95_m"] = _pct(rjerk[:, jmask], 0.95)
    out["interior_root_step_change_p95_m"] = _pct(rjerk[:, ~jmask], 0.95)
  return out
