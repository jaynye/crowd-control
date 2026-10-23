"""Ordinary steering turns with the body; urgent avoidance retains lateral escape."""

import math

import pytest
import torch

from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.behavior.steering import LocalSteering


def _command(xz, goal, enabled, **cfg):
  xz = torch.tensor([xz], dtype=torch.float32)
  goal = torch.tensor([goal], dtype=torch.float32)
  M = xz.shape[1]
  steer = LocalSteering(PlazaSteeringCfg(forward_steering=enabled, **cfg), M, "cpu")
  speed, heading = steer.command(xz, torch.zeros_like(xz), torch.zeros(1, M), goal,
                                 torch.ones(1, M), torch.zeros(1, M, dtype=torch.bool))
  return speed, heading, steer


def test_ordinary_side_request_is_a_slowed_forward_arc():
  speed, heading, steer = _command([[0, 0]], [[1, 0]], True)
  assert float(heading[0, 0]) == pytest.approx(math.pi / 6)
  assert float(speed[0, 0]) == pytest.approx(.5)
  assert not steer.evasive.any()


@pytest.mark.parametrize("xz,goal,cfg", [
  ([[-.3, 0], [.3, 0]], [[0, 1], [0, 1]], {}),
  ([[6.5, 0]], [[-1, 0]], {}),
  ([[.5, 0]], [[1, 0]], {"obstacles": ((0., 0., .3),)}),
])
def test_close_humans_walls_and_obstacles_keep_the_original_escape_command(xz, goal, cfg):
  speed, heading, steer = _command(xz, goal, True, **cfg)
  base_speed, base_heading, _ = _command(xz, goal, False, **cfg)
  assert steer.evasive.all()
  assert torch.equal(speed, base_speed)
  assert torch.equal(heading, base_heading)
