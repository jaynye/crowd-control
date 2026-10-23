"""Independent sticky posture latent, observed at graph edge boundaries."""

from __future__ import annotations

import math

import torch

from crowdmotion.core.rng import hash_uniform, human_seeds


def reset_posture(motion_seeds: torch.Tensor, probability: float):
  if not 0 <= probability <= 1:
    raise ValueError("posture_initial_hunched must be in [0, 1]")
  seeds = human_seeds(0, motion_seeds, "graph_posture")
  u = hash_uniform(seeds, torch.full_like(seeds, -1), 1)[:, 0]
  return u < probability, seeds


def switch_posture(hunched, seeds, counters, cfg, dt: float):
  """Exact two-state CTMC transition probabilities over dt; first edge keeps reset.

  Rates are read on every call. The initial distribution is deliberately absent.
  Multiple continuous-time switches within dt are marginalized; diagnostics count
  the changes observable at boundaries, not unobserved intermediate transitions.
  """
  a, b = cfg.posture_upright_to_hunched_rate, cfg.posture_hunched_to_upright_rate
  if not all(math.isfinite(x) and x >= 0 for x in (a, b)):
    raise ValueError("Posture switching rates must be finite and nonnegative")
  if a + b == 0:
    return hunched
  scale = -math.expm1(-(a + b) * dt) / (a + b)
  probability = torch.where(hunched, b * scale, a * scale)
  u = hash_uniform(seeds, counters, 1)[:, 0]
  return hunched ^ ((u < probability) & (counters > 0))
