"""The crowd curriculum (`traversal.crowd_difficulty_term`): the robot-blind share of the
humans rises while training collisions stay at or below the target."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from crowdmotion.behavior.encounter import EncounterBehavior, EncounterCfg
from crowdmotion.sim.mjlab.traversal import CROWD_CURRICULUM, crowd_difficulty_term

ENVS, HUMANS, DT = 64, 8, 0.02
PARAMS = {"target": 1.0, "interval": 100, **CROWD_CURRICULUM}


def fake_env():
  """The parts of the env the term reads: the crowd's live encounter config and
  behavior, the `crowd_collision` termination and the step counter."""
  encounter = EncounterCfg(robot_oblivious_fraction=0.05, robot_stubborn_fraction=0.1)
  behavior = EncounterBehavior(ENVS, HUMANS, encounter, seed=3, device="cpu")
  behavior.reset(torch.arange(ENVS))
  crowd = SimpleNamespace(live_tuning=SimpleNamespace(encounter=encounter), behavior=behavior)
  env = SimpleNamespace(num_envs=ENVS, step_dt=DT, device="cpu", common_step_counter=0,
                        command_manager=SimpleNamespace(get_term=lambda name: crowd),
                        collided=torch.zeros(ENVS, dtype=torch.bool))
  env.termination_manager = SimpleNamespace(get_term=lambda name: env.collided)
  return env, encounter, behavior


def iteration(term, env, collisions: int):
  """One update interval with `collisions` collision ends on its last step."""
  env.common_step_counter += PARAMS["interval"]
  env.collided[:] = False
  env.collided[:collisions] = True
  return term(env, torch.arange(ENVS), **PARAMS)


def per_minute(rate: float) -> int:
  """Collision ends in one interval at `rate` per robot-minute."""
  return round(rate * PARAMS["interval"] * ENVS * DT / 60.0)


def test_the_level_follows_the_training_collision_rate():
  env, encounter, _ = fake_env()
  term = crowd_difficulty_term(SimpleNamespace(params=PARAMS), env)
  assert term.level == 0.0 and encounter.robot_oblivious_fraction == 0.05
  low, mid, high = per_minute(0.5), per_minute(1.2), per_minute(2.0)
  for _ in range(3):
    out = iteration(term, env, low)
  assert term.level == pytest.approx(0.15) and out["collision_rate"] == pytest.approx(0.5, rel=0.1)
  assert iteration(term, env, mid)["level"] == pytest.approx(0.15)  # inside the band: held
  assert iteration(term, env, high)["level"] == pytest.approx(0.10)
  # A call between updates only counts its collisions; they enter the next update.
  env.common_step_counter += 1
  env.collided[:] = True
  assert term(env, torch.arange(ENVS), **PARAMS)["level"] == pytest.approx(0.10)
  assert iteration(term, env, 0)["level"] == pytest.approx(0.05)
  for _ in range(30):
    iteration(term, env, 0)
  assert term.level == 1.0 and encounter.robot_stubborn_fraction == 0.0


def test_the_fractions_move_together_and_the_humans_draw_them_at_reset():
  env, encounter, behavior = fake_env()
  term = crowd_difficulty_term(SimpleNamespace(params=PARAMS), env)
  term._set(0.5)
  assert encounter.robot_oblivious_fraction == pytest.approx(0.525)
  assert encounter.robot_stubborn_fraction == pytest.approx(0.05)
  term._set(1.0)  # the pair is validated together: never above 1 in between
  assert encounter.robot_oblivious_fraction == 1.0 and encounter.robot_stubborn_fraction == 0.0
  assert term._blind_share() < 0.2  # traits change only when the humans reset
  behavior.reset(torch.arange(ENVS))
  assert term._blind_share() == 1.0
  term._set(0.0)
  assert (encounter.robot_oblivious_fraction, encounter.robot_stubborn_fraction) == (0.05, 0.1)


def test_a_loaded_level_measures_afresh_from_the_restored_step():
  env, encounter, _ = fake_env()
  term = crowd_difficulty_term(SimpleNamespace(params=PARAMS), env)
  assert term.blind_share == 0.05  # before the humans' first reset: the configured share
  for _ in range(4):
    iteration(term, env, per_minute(0.5))
  state = term.state_dict()
  assert state["level"] == pytest.approx(0.2) and state["rate"] == pytest.approx(0.5, rel=0.1)
  env2, encounter2, _ = fake_env()
  resumed = crowd_difficulty_term(SimpleNamespace(params=PARAMS), env2)
  env2.common_step_counter = 52_800  # mjlab restores the counter, then the curriculum
  resumed.load_state_dict(state)
  assert resumed.level == pytest.approx(0.2) and encounter2.robot_oblivious_fraction == pytest.approx(0.24)
  out = resumed(env2, torch.arange(ENVS), **PARAMS)  # the jump in the counter is not an interval
  assert out["level"] == pytest.approx(0.2) and out["collision_rate"] == state["rate"]
  assert out["blind_share"] == state["blind_share"]
  assert iteration(resumed, env2, per_minute(2.0))["level"] == pytest.approx(0.15)
