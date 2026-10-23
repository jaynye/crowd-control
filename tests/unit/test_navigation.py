from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch

from crowdmotion.sim.mjlab.navigation import NavigationCfg, outcome, potential_reward, terminal_reward
from crowdmotion.sim.mjlab.train import DEFAULT_BEHAVIOR
from crowdmotion.sim.mjlab.traversal import make_traversal_env_cfg, traversal_rl_cfg


def test_goal_rewards_and_credit_are_independent_and_legacy_is_unchanged():
  for mode, expected in (("sparse", set()), ("potential", set()),
                         ("tracking", {"track_linear_velocity", "track_angular_velocity", "goal_heading"}),
                         ("proximity", {"crowd_proximity", "crowd_contact", "wall_proximity"})):
    nav = asdict(NavigationCfg(reward_mode=mode, gamma_per_s=.95))
    cfg = make_traversal_env_cfg("g.pt", str(DEFAULT_BEHAVIOR), planner_rate=1, navigation=nav)
    assert set(cfg.rewards) == expected | {"goal_arrival"}
    assert not cfg.terminations["goal_deadline"].time_out
    assert "time_out" not in cfg.terminations
    assert cfg.commands["twist"].resampling_time_range == (1e9, 1e9)
    assert cfg.commands["twist"].rel_standing_envs == 0
    assert "navigation_goal" in cfg.observations["actor"].terms
    assert cfg.commands["twist"].navigation.gamma_per_s == .95
    rl = traversal_rl_cfg(planner_rate=2, navigation=nav)
    assert rl.algorithm.gamma**2 == pytest.approx(.95)
    assert rl.num_steps_per_env == 32
  cfg = make_traversal_env_cfg("g.pt", str(DEFAULT_BEHAVIOR), planner_rate=1)
  assert cfg.commands["twist"].navigation is None
  assert "navigation_goal" not in cfg.observations["actor"].terms


def test_sparse_reward_is_an_event_not_a_dt_scaled_accident(monkeypatch):
  from crowdmotion.sim.mjlab import traversal
  nav = NavigationCfg(time_bonus=2.)
  term = NS(cfg=NS(navigation=nav, arrive_m=.5), goal_w=torch.zeros(4, 2),
            robot=NS(data=NS(root_link_pos_w=torch.zeros(4, 3))),
            nav_contact=torch.zeros(4, dtype=torch.bool), nav_start_distance=torch.full((4,), 4.))
  collision = torch.tensor([False, True, False, False])
  fell = torch.tensor([False, False, True, False])
  monkeypatch.setattr(traversal, "crowd_collision", lambda env: collision)
  for dt in (.02, .1):
    term.nav_contact.zero_()
    env = NS(command_manager=NS(get_term=lambda name: term), common_step_counter=1, step_dt=dt,
             episode_length_buf=torch.tensor([5/dt, 5/dt, 5/dt, 30/dt]),
             termination_manager=NS(get_term=lambda name: fell))
    state = outcome(env)
    assert state["success"].tolist() == [True, False, False, False]
    assert state["deadline"].tolist() == [False, False, False, True]
    assert (terminal_reward(env) * dt).tolist() == pytest.approx([11., 0., 0., 0.])


def test_potential_telescopes_for_detours_and_every_terminal_outcome():
  gamma = .95
  # Same start/terminal and length, one path moves temporarily away. Both shape
  # returns are exactly the same constant; no bonus for oscillation or failure.
  def total(distances):
    return sum(gamma**i * potential_reward(torch.tensor(a), torch.tensor(b), torch.tensor(i == 2), gamma)
               for i, (a, b) in enumerate(zip(distances, distances[1:])))
  assert total([5., 4., 2., 0.]).item() == pytest.approx(5.)
  assert total([5., 6., 4., 8.]).item() == pytest.approx(5.)
  assert potential_reward(torch.tensor(0.), torch.tensor(9.), torch.tensor(True), gamma).item() == 0


def test_navigation_rejects_unknown_and_nonfinite_settings():
  with pytest.raises(ValueError):
    NavigationCfg.from_dict({"typo": 1})
  for values in ({"gamma_per_s": 0}, {"deadline_s": float("nan")}, {"min_goal_m": 9, "max_goal_m": 8}):
    with pytest.raises(ValueError):
      NavigationCfg(**values)
