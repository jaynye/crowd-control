"""Goal-state capture and option termination with real SONIC and the graph crowd."""
from argparse import Namespace
from dataclasses import asdict

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
  pytest.skip("needs CUDA", allow_module_level=True)

from crowdmotion.metrics.traversal import EPISODE_ENDS, TraversalEvalCfg, TraversalMetrics  # noqa: E402
from crowdmotion.sim.mjlab.evaluate import command_state, make_env  # noqa: E402
from crowdmotion.sim.mjlab.navigation import NavigationCfg  # noqa: E402
from crowdmotion.sim.mjlab.train import DEFAULT_BEHAVIOR  # noqa: E402


def test_success_survives_auto_reset_and_is_counted_once():
  import mjlab.tasks  # noqa: F401
  options = dict(graph=None, rotations=None, behavior=str(DEFAULT_BEHAVIOR), humans=4,
                 forward_first=False, teacher_lidar=True, no_avoidance=False, planner_rate=1.,
                 navigation=asdict(NavigationCfg(reward_mode="potential")))
  args = Namespace(behavior=None, humans=None, seed=9, envs=16, seconds=2., episode_s=None,
                   paired=True, cfg=TraversalEvalCfg())
  raw, wrapped = make_env(options, "blind", "end", args, None, "cuda:0")
  try:
    wrapped.reset()
    twist = raw.command_manager.get_term("twist")
    twist.goal_w[:] = raw.scene["robot"].data.root_link_pos_w[:, :2]
    m = TraversalMetrics(16, 4, raw.step_dt, 50, twist.cfg.slow_m, twist.cfg.arrive_m,
                         args.cfg, raw.device, seed=9)
    m.start(command_state(raw))
    wrapped.on_substep = lambda: m.step(raw._eval_pre, command_state(raw))
    _, reward, done, _ = wrapped.step(torch.zeros(16, 3, device=raw.device))
    metrics, tables = m.finish()
    assert done.bool().all() and reward.min() >= 10.
    assert metrics["command/arrivals"] == 16
    assert (tables["episodes"]["end"] == EPISODE_ENDS.index("success")).sum() == 16
    assert (tables["legs"]["arrived"] > 0).sum() == 16
    assert not raw.cfg.terminations["goal_deadline"].time_out
    assert wrapped.action_l2 == wrapped.action_rate == 0.
  finally:
    raw.close()
