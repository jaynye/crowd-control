"""Posture state persists per human across batching, live rate edits and resets."""

import dataclasses
import math

import pytest
import torch
from test_edge_annotations import annotated_graph

from crowdmotion.behavior.sources import GraphMotionSource
from crowdmotion.graph import GraphConfig, GraphRuntime
from crowdmotion.graph.posture import reset_posture, switch_posture


def two_style_graph():
  g = annotated_graph()
  halfway = len(g.valid) // 2
  g.edge_pose[halfway:, :, 4, 2] = .5 * math.tan(math.radians(25))
  g.pred_pose[halfway:, 4, 2] = .5 * math.tan(math.radians(25))
  g.cfg.posture_enabled = True
  g.cfg.posture_weight = .2
  return g


def test_style_matched_starts_and_independent_streams():
  g = two_style_graph()
  runtime = GraphRuntime(g)
  seeds = torch.arange(128)
  full = runtime.reset(seeds, torch.zeros(128, 2), torch.zeros(128))
  order = torch.tensor([37, 4, 120, 1])
  sub = runtime.reset(seeds[order], torch.zeros(4, 2), torch.zeros(4))
  assert torch.equal(full.posture_hunched[order], sub.posture_hunched)
  assert torch.equal(full.boundary[order], sub.boundary)
  assert torch.equal(g.annotations.start_pitch_deg[full.boundary] >= 15, full.posture_hunched)
  assert torch.equal(g.annotations.trunk_pitch_deg[full.boundary] >= 15, full.posture_hunched)
  g.cfg.posture_initial_hunched = 1.
  reset = runtime.reset(seeds, torch.zeros(128, 2), torch.zeros(128))
  assert reset.posture_hunched.all()
  assert not torch.equal(full.posture_hunched, reset.posture_hunched)


def test_rates_are_live_and_initial_probability_does_not_redraw_existing_latent():
  cfg = GraphConfig(posture_upright_to_hunched_rate=0., posture_hunched_to_upright_rate=0.)
  state, seeds = reset_posture(torch.arange(1000), .1)
  cfg.posture_initial_hunched = 1.
  counters = torch.ones(1000, dtype=torch.long)
  assert torch.equal(switch_posture(state, seeds, counters, cfg, .4), state)
  cfg.posture_upright_to_hunched_rate = 100.
  assert switch_posture(state, seeds, counters, cfg, .4).all()
  # The first edge retains the sampled reset style even at huge switching rates.
  assert torch.equal(switch_posture(state, seeds, counters * 0, cfg, .4), state)
  cfg.posture_upright_to_hunched_rate = 0.
  cfg.posture_hunched_to_upright_rate = 100.
  assert not switch_posture(state, seeds, counters, cfg, .4).any()


def test_ctmc_probabilities_and_batch_order():
  cfg = GraphConfig(posture_upright_to_hunched_rate=.2, posture_hunched_to_upright_rate=.4)
  n = 50_000
  state, seeds = reset_posture(torch.arange(n), .3)
  counters = torch.full_like(seeds, 5)
  out = switch_posture(state, seeds, counters, cfg, .4)
  order = torch.tensor([5, 2, 432, 900, 2])
  assert torch.equal(out[order], switch_posture(state[order], seeds[order], counters[order], cfg, .4))
  scale = -math.expm1(-.6 * .4) / .6
  # Sampling error, not a behavioral acceptance threshold.
  assert float(out[~state].float().mean()) == pytest.approx(.2 * scale, abs=.005)
  assert float((~out[state]).float().mean()) == pytest.approx(.4 * scale, abs=.01)


def test_grouped_reset_and_state_scatter_preserve_other_humans():
  g = two_style_graph()
  source = GraphMotionSource(GraphRuntime(g))
  full = source.allocate(8)
  start = source.reset(torch.arange(8), torch.zeros(8, 2), torch.zeros(8))
  source.put(full, torch.arange(8), start)
  old = source.select(full, torch.tensor([0, 1, 2, 3]))
  fresh = source.reset(torch.arange(4) + 900, torch.zeros(4, 2), torch.zeros(4))
  source.put(full, torch.tensor([4, 5, 6, 7]), fresh)
  for f in dataclasses.fields(full):
    assert torch.equal(getattr(full, f.name)[:4], getattr(old, f.name))


def test_posture_cost_is_soft_and_live():
  from crowdmotion.graph.costs import PostureCost, candidate_context
  g = two_style_graph()
  rt = GraphRuntime(g)
  state = rt.reset(torch.arange(2), torch.zeros(2, 2), torch.zeros(2))
  state.posture_hunched[:] = torch.tensor([False, True])
  ctx = candidate_context(g, state, torch.zeros(2, g.L, 2))
  ctx.cand = torch.tensor([[0, len(g.valid) // 2], [0, len(g.valid) // 2]])
  cost = PostureCost()(ctx)
  assert torch.isfinite(cost).all()
  assert cost[0, 0] < cost[0, 1] and cost[1, 1] < cost[1, 0]
