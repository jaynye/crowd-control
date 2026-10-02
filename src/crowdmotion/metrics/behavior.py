"""Per-rollout behavior summary: body-relative direction, turning, spacing, feet.

Metrics over executed frames, excluding the initial pose and reset seams.
Analysis cutoffs only; no empirical human calibration or acceptance thresholds.
"""

from __future__ import annotations

import numpy as np

from crowdmotion.core.geometry import wrap_np
from crowdmotion.metrics.feet import planted_foot_speed
from crowdmotion.metrics.locomotion import STILL_MPS, classify, composition, motion_features
from crowdmotion.metrics.spacing import root_gaps

PAIRED_METRICS = (
  "moving_forward_percent",
  "moving_sideways_percent",
  "moving_backward_percent",
  "natural_successor_percent",
  "graph_path_error_m_mean",
  "min_pair_root_gap_m",
  "root_gap_below_0_6m_frame_percent",
  "facing_travel_error_deg_p50_p95",
  "body_turn_deg_s_p50_p95",
  "path_turn_deg_s_p50_p95",
  "acceleration_mps2_p50_p95",
  "planted_foot_speed_mps_p50_p95",
)


def summarize_trace(trace, fps: float) -> dict:
  """`trace`: a `Trace` or a dict with root `[H,T,2]`, heading, and optionally
  joints, contacts and foot_idx."""
  if not isinstance(trace, dict):
    trace = trace.arrays()
  speed, angle, turn = motion_features(trace["root"], trace["heading"], fps)
  labels = classify(speed, angle)
  moving = speed >= STILL_MPS
  vel = np.diff(trace["root"], axis=1) * fps
  travel = np.arctan2(vel[..., 0], vel[..., 1])
  path_turn = np.degrees(wrap_np(np.diff(travel, axis=1))) * fps
  both_moving = moving[:, 1:] & moving[:, :-1]
  acceleration = np.linalg.norm(np.diff(vel, axis=1), axis=-1) * fps
  gaps = root_gaps(trace["root"])
  finite_gaps = gaps[np.isfinite(gaps)]
  slide = []
  if "contacts" in trace and "foot_idx" in trace:
    slide = planted_foot_speed(trace["joints"], trace["foot_idx"], trace["contacts"], fps)
  return {
    "frame_composition": composition(labels),
    "moving_forward_percent": 100
    * float(((labels == 1) & moving).sum())
    / max(1, int(moving.sum())),
    "moving_sideways_percent": 100
    * float(((labels == 3) & moving).sum())
    / max(1, int(moving.sum())),
    "moving_backward_percent": 100
    * float(((labels == 4) & moving).sum())
    / max(1, int(moving.sum())),
    "speed_mps_p50_p95": np.percentile(speed, [50, 95]).tolist(),
    "facing_travel_error_deg_p50_p95": np.percentile(
      np.abs(angle[moving]), [50, 95]
    ).tolist()
    if moving.any()
    else None,
    "body_turn_deg_s_p50_p95": np.percentile(np.abs(turn[moving]), [50, 95]).tolist()
    if moving.any()
    else None,
    "path_turn_deg_s_p50_p95": np.percentile(
      np.abs(path_turn[both_moving]), [50, 95]
    ).tolist()
    if both_moving.any()
    else None,
    "acceleration_mps2_p50_p95": np.percentile(acceleration, [50, 95]).tolist()
    if acceleration.size
    else None,
    "min_pair_root_gap_m": float(finite_gaps.min()) if finite_gaps.size else None,
    "root_gap_below_0_6m_frame_percent": 100 * float((gaps.min((1, 2)) < 0.6).mean())
    if finite_gaps.size
    else None,
    "planted_foot_speed_mps_p50_p95": np.percentile(slide, [50, 95]).tolist()
    if len(slide)
    else None,
    "note": "Analysis cutoffs only; no empirical human calibration or acceptance thresholds.",
  }
