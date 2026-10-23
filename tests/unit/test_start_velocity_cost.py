"""Entry velocity continuity uses executed world velocity, including corrections."""

import math

import torch
from test_edge_annotations import annotated_graph

from crowdmotion.core.geometry import to_world
from crowdmotion.graph import GraphRuntime
from crowdmotion.graph.costs import StartVelocityCost, candidate_context


def test_entry_cost_uses_world_velocity_and_resets_at_rest():
  g = annotated_graph()
  rt = GraphRuntime(g)
  state = rt.reset(torch.tensor([1]), torch.zeros(1, 2), torch.tensor([math.pi / 2]))
  assert not state.root_velocity.any()
  ctx = candidate_context(g, state, torch.zeros(1, g.L, 2))
  velocity = to_world(g.edge_root[ctx.cand, 0] * g.fps, ctx.head[..., 0])
  state.root_velocity = velocity[:, 0].clone()
  cost = StartVelocityCost()(ctx)
  assert cost[0, 0] == 0.
  assert torch.allclose(cost, (velocity - state.root_velocity[:, None]).norm(dim=-1))


def test_step_retains_actual_velocity_and_masked_hold_clears_it():
  g = annotated_graph()
  g.cfg.start_vel_weight = .2
  g.cfg.correct_yaw_deg, g.cfg.correct_speed = 5., .1
  rt = GraphRuntime(g)
  state = rt.reset(torch.tensor([1]), torch.zeros(1, 2), torch.zeros(1))
  out = rt.step(state, torch.ones(1, g.L, 2))
  expected = (out["root"][:, -1, [0, 2]] - out["root"][:, -2, [0, 2]]) * g.fps
  assert torch.equal(state.root_velocity, expected)
  g.cfg.vertical_mask, g.cfg.vertical_foot_max_m = True, 0.
  out = rt.step(state, torch.ones(1, g.L, 2))
  assert out["vertical_fallback"].all()
  assert not state.root_velocity.any()
