"""`crowd eval traversal` on the GPU: terminal-state capture, paired goals and the metrics.

  source scripts/env.sh && .venv/bin/python -m pytest tests/gpu/test_traversal_eval.py -q
"""

from argparse import Namespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
  pytest.skip("needs CUDA", allow_module_level=True)

from crowdmotion.metrics.traversal import EPISODE_ENDS, TraversalEvalCfg  # noqa: E402
from crowdmotion.sim.mjlab.evaluate import make_env, rollout  # noqa: E402
from crowdmotion.sim.mjlab.train import DEFAULT_BEHAVIOR  # noqa: E402

ENV = {"graph": None, "rotations": None, "behavior": str(DEFAULT_BEHAVIOR), "humans": 4, "forward_first": False,
       "teacher_lidar": True, "no_avoidance": False}


def first_legs(legs):
  out = {}
  for i in np.argsort(legs["t0"], kind="stable"):
    key = (int(legs["env"][i]), int(legs["episode"][i]))
    out.setdefault(key, (legs["x0"][i], legs["y0"][i], legs["goal_x"][i], legs["goal_y"][i]))
  return out


def test_two_policies_get_the_same_goals_and_their_ends_are_captured():
  args = Namespace(behavior=None, humans=None, seed=3, envs=16, seconds=6.0, episode_s=None, paired=True,
                   cfg=TraversalEvalCfg())
  import mjlab.tasks  # noqa: F401

  env, wrapped = make_env(ENV, "blind", "end", args, None, "cuda:0")
  try:
    dim = env.action_manager.total_action_dim
    still = lambda obs: torch.zeros(env.num_envs, dim, device=env.device)  # noqa: E731
    gen = torch.Generator(device=env.device).manual_seed(0)
    flail = lambda obs: 2.0 * torch.randn(env.num_envs, dim, device=env.device, generator=gen)  # noqa: E731
    steps = round(args.seconds / env.step_dt)
    m_still, t_still = rollout(env, wrapped, still, steps, args.cfg, args.seed)
    m_flail, t_flail = rollout(env, wrapped, flail, steps, args.cfg, args.seed)
  finally:
    env.close()
  assert m_still["exposure/robot_min"] == pytest.approx(16 * 6.0 / 60)
  a, b = first_legs(t_still["legs"]), first_legs(t_flail["legs"])
  shared = [k for k in a if k[1] == 1 and k in b]
  assert len(shared) >= 8  # moving envs' first legs (standing envs have none)
  for k in shared:
    assert a[k] == pytest.approx(b[k], abs=1e-5)
  # Flailing falls; each fall ends its episode early and is attributed.
  falls, episodes = t_flail["falls"], t_flail["episodes"]
  assert m_flail["fall/count"] > 0 and len(falls["env"]) == m_flail["fall/count"]
  ended = episodes["end"] == EPISODE_ENDS.index("fall")
  assert ended.sum() == m_flail["fall/count"] and (episodes["duration"][ended] < 6.0).all()
  assert m_flail["first_fall/events"] == m_flail["fall/count"]
  for key in ("command/success", "collision/per_robot_min", "first_collision/rmst_s", "encounter/per_robot_min"):
    assert key in m_still
