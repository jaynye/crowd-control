"""`GraphConfig`: how a motion graph is built and how its runtime scores and plays edges.

`neighbors`, `vel_weight` and `hunch_ratio` are baked into the saved tables; the
other fields may be overridden when a saved graph is loaded. Cost weights are
read by the terms in `crowdmotion.graph.costs` (each term names its field).
"""

from __future__ import annotations

from dataclasses import dataclass

BAKED = ("neighbors", "vel_weight", "hunch_ratio")


@dataclass
class GraphConfig:
  neighbors: int = 32
  vel_weight: float = 0.1  # seconds: 1 m/s of joint velocity counts as 10 cm
  pose_weight: float = 1.0  # cost per meter of descriptor RMS distance
  temperature: float = 0.01  # meters of cost per e-fold in probability; 0 = greedy
  inertialize: bool = True
  correct_yaw_deg: float = 0.0  # max root-path yaw correction per edge; 0 = off
  correct_speed: float = 0.0  # max fractional root-speed correction per edge; 0 = off
  hunch_ratio: float = 0.0  # drop edges whose mean head height < ratio * neutral
  end_vel_weight: float = 0.0  # seconds: cost per m/s of end-of-edge root velocity error
  lookahead: bool = False  # also score each candidate's own successor (needs a 2L path)
  facing_weight: float = 0.0  # seconds: cost per m/s of body-relative non-forward travel
  heading_weight: float = 0.0  # seconds: cost per (m/s * rad) of requested body-heading error
  heading_deadband_deg: float = 0.0  # ignore small heading errors; 0 preserves the original cost
  vertical_mask: bool = False  # opt-in flat-ground eligibility from saved absolute heights
  vertical_pelvis_range_m: float = 0.15  # includes edge predecessor
  vertical_foot_min_m: float = 0.12  # lowest foot must reach this height at least once
  vertical_foot_max_m: float = 0.60  # reject extreme foot elevations
  vertical_lowest_foot_max_m: float = 0.0  # optional peak lower-foot limit; 0 disables this extra test
  twist_mask: bool = False  # reject foot/body yaw mismatch while every foot-contact label is on
  twist_max_foot_yaw_deg: float = 21.4  # edge-mean absolute yaw; heuristic from base-graph p99
  posture_enabled: bool = False  # sticky per-human style and class-matched starts
  posture_weight: float = 0.0  # soft squared pitch error cost; no hard candidate filter
  posture_cutoff_deg: float = 15.0  # diagnostic class split, not a calibrated label
  posture_target_margin_deg: float = 10.0  # targets cutoff +/- margin (upright/hunched)
  posture_initial_hunched: float = 0.10  # sampled at reset only
  posture_upright_to_hunched_rate: float = 0.005  # continuous-time rate per second
  posture_hunched_to_upright_rate: float = 0.020  # continuous-time rate per second
  turn_weight: float = 0.0  # cost per radian of retained body-target error during an ordinary redirect
  turn_preserve_yaw: bool = False  # do not straighten native yaw toward the slow path during a body turn
  start_vel_weight: float = 0.0  # seconds: first-frame root velocity error against last executed velocity
  clearance_weight: float = 0.0  # cost per meter a candidate closes on a neighbor inside the clearance

  @classmethod
  def from_dict(cls, d: dict | None) -> "GraphConfig":
    """Lenient: unknown keys are ignored (saved blobs and older configs keep loading)."""
    return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})
