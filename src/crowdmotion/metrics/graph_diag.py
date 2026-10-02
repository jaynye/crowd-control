"""Motion-graph transition diagnostics from the per-step outputs of a graph rollout."""

from __future__ import annotations

import math

import torch

GRAPH_DIAG = (
  "traj_err_m", "best_traj_err_m", "end_vel_err_mps", "best_end_vel_err_mps",
  "desc_dist_m", "pose_offset_m", "vel_jump_mps",
)
DIAG_KEYS = (*GRAPH_DIAG, "natural", "edge", "yaw_corr_rad", "speed_scale")


def _pct(x: torch.Tensor, q: float) -> float:
  return float(torch.quantile(x.float().flatten(), q)) if x.numel() else float("nan")


def graph_metrics(diag: dict, stats: dict) -> dict:
  """Transition diagnostics `[H, steps]`; the first step (from the start pose) is excluded."""
  d = {k: v[:, 1:] for k, v in diag.items()}
  out = {}
  for key in GRAPH_DIAG:
    out[f"g_{key}_mean"] = float(d[key].mean())
    out[f"g_{key}_p95"] = _pct(d[key], 0.95)
  jumps = ~d["natural"]
  out["g_natural_frac"] = float(d["natural"].float().mean())
  out["g_pose_offset_jump_mean_m"] = float(d["pose_offset_m"][jumps].mean()) if jumps.any() else 0.0
  out["g_traj_err_frac_gt_5cm"] = float((d["traj_err_m"] > 0.05).float().mean())
  out["g_best_traj_err_frac_gt_5cm"] = float((d["best_traj_err_m"] > 0.05).float().mean())
  out["g_unique_edges_frac"] = d["edge"].unique().numel() / d["edge"].numel()
  out["g_yaw_corr_abs_mean_deg"] = math.degrees(float(d["yaw_corr_rad"].abs().mean()))
  out["g_speed_scale_dev_mean"] = float((d["speed_scale"] - 1).abs().mean())
  out.update({f"g_{k}": v for k, v in stats.items()})
  return out


def graph_window_split(diag: dict, boundaries: list[int], fps: float, start: torch.Tensor,
                       response: torch.Tensor, window_s: float = 1.2) -> dict:
  """Graph diagnostics over the first `window_s` after each human's command change,
  split by humans who respond within the scenario window and humans who never do.

  Separates coverage (even the best candidate is far from the request) from
  selection (a close candidate exists but loses the cost).
  """
  b = torch.tensor(boundaries)[None]
  s = start.cpu()[:, None]
  win = ((b >= s) & (b < s + round(window_s * fps))).float()
  never = response.isnan()
  out = {"gw_never_count": int(never.sum()), "gw_responder_count": int((~never).sum())}
  keys = ("best_end_vel_err_mps", "end_vel_err_mps", "best_traj_err_m", "traj_err_m", "natural")
  for key in keys:
    per_human = (diag[key].float().cpu() * win).sum(1) / win.sum(1).clamp(min=1)
    for tag, sel in (("responders", ~never), ("never", never)):
      out[f"gw_{key}_{tag}"] = float(per_human[sel].mean()) if sel.any() else float("nan")
  return out
