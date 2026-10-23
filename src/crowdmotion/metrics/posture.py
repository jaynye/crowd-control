"""Executed ground heights and torso posture, including transition persistence."""

from __future__ import annotations

import numpy as np
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


def _quantiles(x):
  x = np.asarray(x)
  return {str(p): float(np.percentile(x, p)) for p in (0, 50, 95, 99, 100)} if x.size else None


def vertical_metrics(trace, fps: float, edge_frames: int, hips: int = 0) -> dict:
  """Absolute Y of the executed joints, including inertialization and reset pose."""
  joints = trace["joints"]
  pelvis = joints[..., hips, 1]
  low_foot = joints[..., trace["foot_idx"], 1].min(-1)
  ranges = [np.ptp(pelvis[:, k:min(k + edge_frames + 1, pelvis.shape[1])], axis=1)
            for k in range(0, pelvis.shape[1] - 1, edge_frames)]
  delta = np.abs(np.diff(pelvis, axis=1))
  return {
    "pelvis_height_m": _quantiles(pelvis), "pelvis_range_per_human_m": _quantiles(np.ptp(pelvis, axis=1)),
    "pelvis_range_per_executed_edge_m": _quantiles(np.concatenate(ranges)),
    "pelvis_frame_delta_m": _quantiles(delta), "pelvis_vertical_speed_mps": _quantiles(delta * fps),
    "lowest_foot_height_m": _quantiles(low_foot),
    "lowest_foot_above012_percent": 100 * float((low_foot > .12).mean()),
    "lowest_foot_above018_percent": 100 * float((low_foot > .18).mean()),
    "vertical_fallback_steps": int(trace.get("vertical_fallback", np.zeros(0)).sum()),
    "definition": "Y absolute above flat ground; lower of all named contact-foot joints. "
                  "Edge ranges include the predecessor. Includes reset frame and inertialized execution.",
  }


def posture_metrics(trace, fps: float, edge_frames: int, names: list[str], cutoff_deg: float = 15.) -> dict:
  hips, neck = names.index("Hips"), names.index("Neck")
  trunk = trace["joints"][..., neck, :] - trace["joints"][..., hips, :]
  heading = trace["heading"]
  forward = trunk[..., 0] * np.sin(heading) + trunk[..., 2] * np.cos(heading)
  pitch = np.degrees(np.arctan2(forward, trunk[..., 1]))
  by_edge = np.stack([pitch[:, k:min(k + edge_frames, pitch.shape[1])].mean(1)
                      for k in range(1, pitch.shape[1], edge_frames)], 1)
  actual = by_edge >= cutoff_deg
  switches = np.count_nonzero(actual[:, 1:] != actual[:, :-1], axis=1)
  result = {
    "pitch_deg": _quantiles(pitch), "class_cutoff_deg": cutoff_deg,
    "hunched_frame_percent": 100 * float((pitch >= cutoff_deg).mean()),
    "hunched_edge_percent": 100 * float(actual.mean()),
    "observed_class_switches": int(switches.sum()), "observed_switches_per_human": switches.tolist(),
    "observed_switches_per_human_minute": float(switches.sum()) / (len(pitch) * (pitch.shape[1] - 1) / fps / 60),
    "initial_hunched_humans": np.flatnonzero(pitch[:, 0] >= cutoff_deg).tolist(),
    "per_human_hunched_percent": (100 * actual.mean(1)).tolist(),
    "definition": "Signed sagittal Hips-to-Neck pitch from executed joints/body heading. "
                  "Class uses mean pitch per edge, so threshold crossings are not latent switches; cutoff is a diagnostic heuristic.",
  }
  if "posture_hunched" in trace:
    latent = trace["posture_hunched"].astype(bool)
    changed = latent[:, 1:] != latent[:, :-1]
    result["latent"] = {
      "initial_hunched_percent": 100 * float(latent[:, 0].mean()),
      "initial_hunched_humans": np.flatnonzero(latent[:, 0]).tolist(),
      "hunched_edge_percent": 100 * float(latent.mean()),
      "switches": int(changed.sum()), "switches_per_human": changed.sum(1).tolist(),
      "observed_class_agreement_percent": 100 * float((actual == latent).mean()),
      "observed_hunched_when_upright_percent": 100 * float(actual[~latent].mean()) if (~latent).any() else None,
      "observed_hunched_when_hunched_percent": 100 * float(actual[latent].mean()) if latent.any() else None,
      **{f"{style}_to_other_per_second": float((changed & (latent[:, :-1] == value)).sum()) /
         max(1e-12, float((latent[:, :-1] == value).sum()) * edge_frames / fps)
         for style, value in (("upright", False), ("hunched", True))},
    }
  return result
