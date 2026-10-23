"""Encounter-design diagnostics: decisions, interventions, stops and body clearance.

Bodies are the design's vertical cylinders (`body_radius_m`) unless joint radii are
given. Cutoffs are diagnostics, not calibrated acceptance limits.
"""

from __future__ import annotations

import numpy as np

from crowdmotion.metrics.spacing import root_gaps

MODES = ("cruise", "sidestep", "veer", "pivot", "yield")


def _percentiles(values):
  values = np.asarray(values, dtype=float)
  return np.percentile(values, [50, 95]).tolist() if values.size else None


def cylinder_clearance(root: np.ndarray, body_radius_m: float, tolerance_m: float, fps: float) -> dict:
  """Root-gap clearance between `body_radius_m` cylinders, over every frame and pair."""
  gaps = root_gaps(root)[1:]  # exclude the reset frame
  clear = gaps - 2 * body_radius_m
  frames = clear.min((1, 2))
  pairs = np.triu(np.ones(clear.shape[1:], dtype=bool), 1)
  touching = (clear < 0)[:, pairs]  # [T, pairs]
  entries = int((touching[1:] & ~touching[:-1]).sum() + touching[:1].sum()) if len(touching) else 0
  return {
    "min_clearance_m": float(frames.min()) if frames.size else None,
    "contact_frame_percent": 100 * float((frames < 0).mean()) if frames.size else None,
    "within_tolerance_frame_percent": 100 * float((frames < tolerance_m).mean()) if frames.size else None,
    "contact_pair_events": entries,
    "contact_pair_seconds": float(touching.sum()) / fps,
  }


def joint_clearance(joints: np.ndarray, radii: np.ndarray, stride: int = 1) -> dict:
  """Minimum distance between any two humans' joint spheres (`radii [J]`, meters).

  Joint spheres approximate the capsule bodies at their joints only (not along bones).
  """
  frames = joints[:, 1::stride]  # [H, T, J, 3]
  H = len(frames)
  worst = np.full(frames.shape[1], np.inf)
  for i in range(H):
    for j in range(i + 1, H):
      d = np.linalg.norm(frames[i][:, :, None] - frames[j][:, None], axis=-1)  # [T, J, J]
      d = d - radii[None, :, None] - radii[None, None, :]
      worst = np.minimum(worst, d.min((1, 2)))
  return {"min_joint_sphere_clearance_m": float(worst.min()) if worst.size else None,
          "joint_sphere_contact_frame_percent": 100 * float((worst < 0).mean()) if worst.size else None,
          "frame_stride": stride}


def encounter_metrics(trace: dict, fps: float, edge_frames: int, cfg: dict) -> dict:
  """Decision statistics from an encounter-design trace (`trace` holds `encounter_*`)."""
  mode, switch = trace["encounter_mode"], trace["encounter_switch"]
  threat, urgency = trace["encounter_threat"], trace["encounter_urgency"]
  stopped, assertive = trace["encounter_stopped"], trace["encounter_assertiveness"][:, 0]
  steps = mode.shape[1]
  step_s = edge_frames / fps
  minutes = len(mode) * steps * step_s / 60
  chosen = mode[switch]
  # Empirical per-step switch probability given a threat, binned by urgency and assertiveness.
  at_risk = trace["encounter_eligible"]
  urgency_bins = [0, .25, .5, .75, 1.0001]
  by_urgency = {}
  for lo, hi in zip(urgency_bins[:-1], urgency_bins[1:]):
    sel = at_risk & (urgency >= lo) & (urgency < hi)
    by_urgency[f"{lo:g}-{min(hi, 1):g}"] = {"steps": int(sel.sum()),
                                           "switch_percent": 100 * float(switch[sel].mean()) if sel.any() else None}
  by_assertiveness = {}
  a = np.broadcast_to(assertive[:, None], mode.shape)
  for lo, hi in ((0, .3), (.3, .6), (.6, 1.0001)):
    sel = at_risk & (a >= lo) & (a < hi)
    by_assertiveness[f"{lo:g}-{min(hi, 1):g}"] = {"steps": int(sel.sum()),
                                                 "switch_percent": 100 * float(switch[sel].mean()) if sel.any() else None}
  # Stops: consecutive stopped steps per human.
  runs = []
  for row in stopped:
    edges = np.diff(np.r_[0, row.astype(int), 0])
    runs += list((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) * step_s)
  # Executed ordinary turns: in place versus arcs, from the turn radius at onset.
  turning = trace["encounter_ordinary_turn"]
  onset = turning & ~np.concatenate((np.zeros((len(turning), 1), dtype=bool), turning[:, :-1]), 1)
  radius = trace["encounter_ordinary_turn_radius"][onset]
  return {
    "human_minutes": minutes,
    "mode_step_percent": {name: 100 * float((mode == k).mean()) for k, name in enumerate(MODES)},
    "threat_step_percent": 100 * float(threat.mean()),
    "switches": int(switch.sum()),
    "switches_per_human_minute": float(switch.sum()) / max(minutes, 1e-9),
    "strategy_counts": {name: int((chosen == k).sum()) for k, name in enumerate(MODES) if k},
    "switch_percent_given_threat_by_urgency": by_urgency,
    "switch_percent_given_threat_by_assertiveness": by_assertiveness,
    "stopped_step_percent": 100 * float(stopped.mean()),
    "separating_step_percent": 100 * float(trace["encounter_separating"].mean()),
    "stop_episodes_per_human_minute": len(runs) / max(minutes, 1e-9),
    "stop_episode_s_p50_p95": _percentiles(runs),
    "ordinary_turns": int(onset.sum()),
    "ordinary_turns_in_place": int((radius == 0).sum()),
    "ordinary_turn_radius_m_p50_p95": _percentiles(radius[radius > 0]),
    "body_clearance": cylinder_clearance(trace["root"], cfg["body_radius_m"], cfg["stop_clearance_m"], fps),
    "definition": "Switch percentages are per command step while eligible (cruising, not waiting, with a "
                  "threat in the bubble). Clearance uses body_radius_m cylinders around roots; contact means "
                  "negative clearance. Diagnostics, not limits.",
  }
