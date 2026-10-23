"""Retained ordinary body targets, live controls, and immediate evasive release."""

import math
from types import SimpleNamespace

import torch

from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.behavior.steering import LocalSteering
from crowdmotion.graph.costs import TurnCost
from crowdmotion.iterate.evaluate import resolved_config


def command(steering, heading, xz=None, env_ids=None):
  shape = heading.shape
  xz = torch.zeros(*shape, 2) if xz is None else xz
  return steering.command(xz, torch.zeros_like(xz), heading,
                          torch.tensor([1., 0.]).expand_as(xz), torch.ones(shape),
                          torch.zeros(shape, dtype=torch.bool), env_ids=env_ids)


def test_retains_body_target_while_translation_slows_then_releases():
  cfg = PlazaSteeringCfg(turn_before_walk=True, forward_steering=True, relax_orientation_for_avoidance=True)
  s = LocalSteering(cfg, 1, "cpu")
  speed, heading = command(s, torch.zeros(1, 1))
  assert speed.item() == torch.tensor(.2).item() and heading.item() == 0
  assert s.turn_active.item()
  assert s.motion_options()["turn_target"].item() == torch.tensor(math.pi / 2).item()
  # Below the start angle still keeps the turn phase until the finish angle.
  speed, _ = command(s, torch.tensor([[math.radians(60)]]))
  assert s.turn_active.item() and speed.item() <= .2 + 1e-6
  speed, _ = command(s, torch.tensor([[math.radians(80)]]))
  assert not s.turn_active.item() and speed.item() > .9


def test_evasion_releases_retained_target_and_reset_is_grouped():
  cfg = PlazaSteeringCfg(turn_before_walk=True, relax_orientation_for_avoidance=True)
  s = LocalSteering(cfg, 2, "cpu", envs=2)
  far = torch.tensor([[[0., 0.], [0., 3.]], [[0., 0.], [0., 3.]]])
  command(s, torch.zeros(2, 2), far)
  assert s._turning.all()
  s.reset(torch.tensor([1]))
  assert s._turning[0].all() and not s._turning[1].any()
  close = torch.tensor([[[0., 0.], [.3, 0.]]])
  command(s, torch.zeros(1, 2), close, torch.tensor([0]))
  assert s.evasive.all() and not s.turn_active.any()
  assert not s.motion_options()["orientation_weight"].any()


def test_parameters_are_read_live_and_config_default_serialization_is_preserved():
  assert "turn_before_walk" not in resolved_config({})["steering"]
  assert resolved_config({"steering": {"turn_before_walk": True}})["steering"]["turn_before_walk"]
  cfg = PlazaSteeringCfg(turn_before_walk=True)
  s = LocalSteering(cfg, 1, "cpu")
  command(s, torch.zeros(1, 1))
  cfg.redirect_speed_mps = .4
  assert command(s, torch.zeros(1, 1))[0].item() == torch.tensor(.4).item()
  cfg.turn_before_walk = False
  assert command(s, torch.zeros(1, 1))[0].item() == 1.
  assert s.motion_options() == {}


def test_turn_cost_has_target_even_when_path_is_stationary():
  ctx = SimpleNamespace(
    graph=SimpleNamespace(edge_yaw=torch.tensor([[0., 0.], [.1, .2]])),
    cand=torch.tensor([[0, 1], [0, 1]]), state=SimpleNamespace(heading=torch.zeros(2)),
    turn_target=torch.tensor([math.pi/2, math.pi/2]), turn_active=torch.tensor([True, False]),
    ddist=torch.zeros(2, 2),
  )
  cost = TurnCost()(ctx)
  assert cost[0, 1] < cost[0, 0] and torch.equal(cost[1], torch.zeros(2))


def test_optional_native_yaw_preservation_applies_only_during_turn_phase():
  from test_edge_annotations import annotated_graph

  from crowdmotion.graph import GraphRuntime
  g = annotated_graph()
  g.cfg.correct_yaw_deg = 5.
  rt = GraphRuntime(g)
  state = rt.reset(torch.tensor([0, 1]), torch.zeros(2, 2), torch.zeros(2))
  edge = torch.tensor([0, 1])
  t = torch.arange(1, g.L + 1) / g.fps
  g.edge_root[edge] = torch.stack((t, t), -1)
  g.edge_yaw[edge] = .1
  path = torch.stack((t * 0, t), -1)[None].repeat(2, 1, 1)
  _, yaw_before, phi_before, _ = rt.correct(state, path, edge)
  assert (phi_before < 0).all()
  g.cfg.turn_preserve_yaw = True
  _, yaw, phi, _ = rt.correct(state, path, edge, torch.tensor([True, False]))
  assert phi[0] == 0 and phi[1] == phi_before[1]
  assert torch.equal(yaw[0], g.edge_yaw[0]) and torch.equal(yaw[1], yaw_before[1])
