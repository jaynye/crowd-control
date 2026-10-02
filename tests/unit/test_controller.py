"""CrowdController: grouped and ungrouped stepping agree; kinematics follow executed frames."""

from __future__ import annotations

import pytest
import torch

from crowdmotion.behavior.controller import CrowdController
from crowdmotion.behavior.kinematics import BufferVelocity, WindowVelocity
from crowdmotion.behavior.plaza import PlazaBehavior, PlazaSteeringCfg
from crowdmotion.behavior.sources import GraphMotionSource
from crowdmotion.graph import GraphConfig, GraphRuntime, MotionGraph
from synthetic import synthetic_library


class _Straight:
  """Constant command: speed 1 m/s along +Z."""

  def command(self, t, kin, groups=None):
    n = kin.root_xz.shape[0]
    return torch.ones(n), torch.zeros(n)


def _graph():
  return MotionGraph(synthetic_library(), GraphConfig(neighbors=3, temperature=0.02), device="cpu")


def test_window_velocity_uses_executed_frames_and_partial_takes():
  graph = _graph()
  ctrl = CrowdController(_Straight(), GraphMotionSource(GraphRuntime(graph)), WindowVelocity(graph.fps))
  xz = torch.zeros(3, 2)
  ctrl.reset(torch.arange(3), xz, torch.zeros(3))
  step = ctrl.advance(0.0, limit=2)
  assert step.take == 2
  root = step.motion.root[:, :2, [0, 2]]
  assert torch.equal(ctrl.kin.root_xz, root[:, 1])
  assert torch.allclose(ctrl.kin.vel, (root[:, 1] - root[:, 0]) * graph.fps)
  assert torch.equal(xz, torch.zeros(3, 2))  # inputs are not aliased


def test_grouped_advance_matches_one_group_per_controller():
  graph = _graph()
  cfg = PlazaSteeringCfg()

  def make(envs):
    behavior = PlazaBehavior(envs, 2, cfg, seed=5, device="cpu")
    ctrl = CrowdController(behavior, GraphMotionSource(GraphRuntime(graph)), BufferVelocity(graph.fps), group=2)
    ctrl.allocate(envs * 2, "cpu")
    return behavior, ctrl

  behavior, ctrl = make(1)
  env = torch.tensor([0])
  xz, heading = behavior.reset(env)
  ctrl.reset(torch.arange(2), xz.reshape(-1, 2), heading.reshape(-1), groups=env)
  steps = [ctrl.advance(torch.tensor([0.4 * k]), groups=env) for k in range(3)]

  behavior2, ctrl2 = make(1)
  xz2, heading2 = behavior2.reset(env)
  ctrl2.reset(torch.arange(2), xz2[0], heading2[0])
  steps2 = [ctrl2.advance(torch.tensor([0.4 * k]), groups=env) for k in range(3)]
  for a, b in zip(steps, steps2, strict=True):
    assert torch.equal(a.motion.joints, b.motion.joints)
    assert torch.equal(a.path, b.path)


def test_unsupported_combinations_fail_loudly():
  graph = _graph()
  source = GraphMotionSource(GraphRuntime(graph))
  with pytest.raises(ValueError, match="BufferVelocity"):
    CrowdController(_Straight(), source, WindowVelocity(graph.fps), group=2)
  ctrl = CrowdController(_Straight(), source, WindowVelocity(graph.fps))
  ctrl.reset(torch.arange(2), torch.zeros(2, 2), torch.zeros(2))
  ctrl.advance(0.0, limit=1)  # a graph edge cannot resume from a partial step
  with pytest.raises(RuntimeError, match="partly executed"):
    ctrl.advance(0.05)
