"""Controlled response requests use shared steering and preserve isolated humans."""

import math
from types import SimpleNamespace

import pytest
import torch

from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.iterate.probe import SteeredScenario


def test_controlled_turn_uses_same_forward_limit_and_orientation_priority():
  def commands(t):
    return torch.ones(2), torch.full((2,), math.pi / 2)
  cfg = PlazaSteeringCfg(forward_steering=True, relax_orientation_for_avoidance=True)
  scenario = SteeredScenario(commands, cfg)
  # Identical coordinates are in different empty plazas, so they must not repel.
  kin = SimpleNamespace(root_xz=torch.zeros(2, 2), heading=torch.zeros(2), vel=torch.zeros(2, 2))
  speed, heading = scenario.command(5., kin)
  assert speed.tolist() == pytest.approx([.5, .5])
  assert heading.tolist() == pytest.approx([math.pi / 6, math.pi / 6])
  assert torch.equal(scenario.motion_options()["orientation_weight"], torch.ones(2))
