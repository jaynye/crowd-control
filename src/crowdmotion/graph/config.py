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

  @classmethod
  def from_dict(cls, d: dict | None) -> "GraphConfig":
    """Lenient: unknown keys are ignored (saved blobs and older configs keep loading)."""
    return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})
