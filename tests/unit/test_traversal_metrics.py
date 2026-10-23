"""`crowdmotion.metrics.traversal`: legs, collisions, time to contact, encounters, reactions,
falls and survival on synthetic trajectories with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from crowdmotion.metrics.traversal import (
  LEG_ENDS,
  CommandState,
  StepState,
  TraversalMetrics,
  cfg_from_dict,
  ideal_time,
  kaplan_meier,
  segment_distance,
)

DT = 0.02


def state(E=1, M=1, **kw) -> StepState:
  zeros = lambda *s: torch.zeros(*s)  # noqa: E731
  base = dict(pos=zeros(E, 2), vel=zeros(E, 2), heading=zeros(E), vel_b=zeros(E, 3), cmd_w=zeros(E, 2),
              cmd_b=zeros(E, 3), standing=torch.zeros(E, dtype=torch.bool), clearance=torch.full((E, M), 5.0),
              human_pos=torch.full((E, M, 2), 50.0), human_vel=zeros(E, M, 2),
              trait=torch.zeros(E, M, dtype=torch.long), fell=torch.zeros(E, dtype=torch.bool),
              timed_out=torch.zeros(E, dtype=torch.bool), collided=torch.zeros(E, dtype=torch.bool),
              done=torch.zeros(E, dtype=torch.bool))
  return StepState(**{**base, **kw})


def command(E=1, pos=(0.0, 0.0), goal=(5.0, 0.0), speed=1.0, standing=False, arrivals=0.0) -> CommandState:
  return CommandState(pos=torch.tensor([pos] * E), goal=torch.tensor([goal] * E), speed=torch.full((E,), speed),
                      standing=torch.full((E,), standing), arrivals=torch.full((E,), arrivals))


def metrics(E=1, M=1, steps=1000, **cfg) -> TraversalMetrics:
  return TraversalMetrics(E, M, DT, steps, slow_m=1.0, arrive_m=0.5, cfg=cfg_from_dict(cfg))


def test_geometry_helpers():
  assert float(ideal_time(torch.tensor(5.0), torch.tensor(1.0), 1.0, 0.5)) == pytest.approx(4 + math.log(2))
  assert float(ideal_time(torch.tensor(0.8), torch.tensor(1.0), 1.0, 0.5)) == pytest.approx(math.log(1.6))
  d, u = segment_distance(torch.tensor([[1.0, 2.0], [7.0, 0.0]]), torch.zeros(2, 2), torch.tensor([[5.0, 0.0]] * 2))
  assert d.tolist() == pytest.approx([2.0, 2.0]) and u.tolist() == pytest.approx([0.2, 1.0])


def test_kaplan_meier_censors_and_restricts():
  km = kaplan_meier(np.array([1, 2, 3, 4.0]), np.array([1, 0, 1, 0]), horizon=4.0, at=(2.0, 3.5, 9.0))
  assert km["median_s"] == 3.0 and km["events"] == 2
  assert km["rmst_s"] == pytest.approx(1 + 2 * 0.75 + 0.375)
  assert km["survival_2s"] == pytest.approx(0.75) and km["survival_3.5s"] == pytest.approx(0.375)
  assert km["survival_9s"] is None  # beyond the last observed time
  assert kaplan_meier(np.array([5.0]), np.array([0]), horizon=5.0)["median_s"] is None


def test_a_straight_walk_to_the_goal_tracks_perfectly():
  m = metrics()
  m.start(command())
  pos, arrived_at = torch.zeros(1, 2), None
  for i in range(400):
    pos = pos + torch.tensor([[1.0, 0.0]]) * DT
    dist = 5.0 - float(pos[0, 0])
    arrived = dist < 0.5
    pre = state(pos=pos.clone(), vel=torch.tensor([[1.0, 0.0]]), cmd_w=torch.tensor([[1.0, 0.0]]),
                vel_b=torch.tensor([[1.0, 0.0, 0.0]]), cmd_b=torch.tensor([[1.0, 0.0, 0.0]]))
    post = command(pos=tuple(pos[0].tolist()), goal=(5.0, 4.0) if arrived else (5.0, 0.0), arrivals=float(arrived))
    m.step(pre, post)
    if arrived:
      arrived_at = (i + 1) * DT
      break
  out, tables = m.finish()
  legs = tables["legs"]
  first = legs["end"] == LEG_ENDS.index("arrived")
  assert first.sum() == 1 and arrived_at == pytest.approx(4.5, abs=2 * DT)
  assert legs["dev_max_m"][first][0] == pytest.approx(0.0, abs=1e-6)
  assert legs["straight_m"][first][0] == pytest.approx(4.5) and legs["detour"][first][0] == pytest.approx(1.0, abs=0.01)
  # It walks at full speed into the ramp-down zone, so it beats the command's ideal time.
  assert legs["lag_s"][first][0] == pytest.approx(4.5 - (4 + math.log(2)), abs=2 * DT)
  assert out["command/success"] == 1.0 and len(legs["env"]) == 1  # the next leg had no steps: not recorded
  assert len(tables["episodes"]["env"]) == 1
  assert out["command/tracking_score"] == pytest.approx(1.0, abs=1e-3)
  assert out["command/vel_error_xy"] == pytest.approx(0.0)
  assert out["collision/count"] == 0 and out["encounter/count"] == 0
  assert out["command/drift_1s_mean_m"] == pytest.approx(0.0, abs=0.01)


def test_a_human_walking_into_a_standing_robot():
  m = metrics(M=2)
  m.start(command(standing=True))
  far = torch.tensor([9.0])
  for i in range(200):
    c = 2.0 - i * DT  # the first human closes at 1 m/s and stops 0.5 m into the robot, then leaves
    c = c if i < 125 else -0.5 + (i - 125) * 0.05
    pre = state(M=2, standing=torch.tensor([True]), clearance=torch.cat((torch.tensor([c]), far))[None],
                human_pos=torch.tensor([[[c + 0.3, 0.0], [9.0, 0.0]]]), human_vel=torch.tensor([[[-1.0, 0.0], [0, 0]]]),
                trait=torch.tensor([[0, 2]]))
    m.step(pre, command(standing=True))
  out, tables = m.finish()
  col, enc = tables["collisions"], tables["encounters"]
  assert len(col["env"]) == 1 and col["human"][0] == 0 and col["trait"][0] == 0 and col["standing"][0] == 1
  assert col["human_toward"][0] == pytest.approx(1.0) and col["robot_toward"][0] == pytest.approx(0.0)
  assert out["collision/human_led_share"] == 1.0 and out["collision/oblivious_per_robot_min"] > 0
  assert len(enc["env"]) == 1 and enc["collided"][0] == 1 and enc["contact"][0] == 1
  assert enc["min_ttc_s"][0] == 0.0 and enc["min_clear_m"][0] == pytest.approx(-0.5)
  assert enc["threat"][0] == 0  # a standing robot is not asked to react
  assert out["command/legs"] == 0  # standing envs have no legs
  assert tables["episodes"]["collisions"][0] == 1


def test_ttc_is_clearance_over_the_line_of_sight_closing_speed():
  m = metrics(M=2)
  m.start(command())
  # Robot at the origin walking +x at 0.6 m/s; human 0 ahead walking toward it at 0.4 m/s,
  # human 1 beside it walking away.
  pre = state(M=2, vel=torch.tensor([[0.6, 0.0]]), clearance=torch.tensor([[2.0, 1.5]]),
              human_pos=torch.tensor([[[2.5, 0.0], [0.0, 2.0]]]), human_vel=torch.tensor([[[-0.4, 0.0], [0.0, 1.0]]]))
  ttc = m._ttc(pre.clearance, m._relative(pre))
  assert float(ttc[0, 0]) == pytest.approx(2.0 / 1.0) and math.isinf(float(ttc[0, 1]))
  assert float(m._ttc(torch.tensor([[-0.01, 1.5]]), m._relative(pre))[0, 0]) == 0.0


def test_reaction_latency_from_the_threat_to_slowing_down():
  cfg = cfg_from_dict({})
  alpha = DT / cfg.reaction_smooth_s
  speeds = [1.0 if i < 60 else 0.8 for i in range(150)]  # slows by 0.2 m/s at step 60
  smooth, expected = None, None
  for i, v in enumerate(speeds):
    smooth = v if smooth is None else smooth + alpha * (v - smooth)
    if expected is None and smooth <= 1.0 - cfg.reaction_speed_drop:
      expected = (i + 1) * DT - DT  # the threat starts at the first step: TTC 1.8 s
  m = metrics()
  m.start(command(goal=(9.0, 0.0)))
  pos = torch.zeros(1, 2)
  for i, v in enumerate(speeds):
    pos = pos + torch.tensor([[v, 0.0]]) * DT
    clear = 1.8 - 0.5 * i * DT
    pre = state(pos=pos.clone(), vel=torch.tensor([[v, 0.0]]), cmd_w=torch.tensor([[1.0, 0.0]]),
                clearance=torch.tensor([[clear]]), human_pos=pos[:, None] + torch.tensor([[[clear + 0.3, 0.0]]]))
    m.step(pre, command(goal=(9.0, 0.0)))
  out, tables = m.finish()
  enc = tables["encounters"]
  assert len(enc["env"]) == 1 and enc["threat"][0] == 1 and enc["responded"][0] == 1
  assert enc["latency_s"][0] == pytest.approx(expected, abs=1e-4)
  assert expected > 59 * DT + 0.1  # the smoothing puts the response well after the drop
  assert out["reaction/response_share"] == 1.0


def test_falls_are_attributed_to_the_nearest_human_before_them():
  m = metrics(E=2)
  m.start(command(E=2))
  for i in range(150):
    fell = torch.tensor([i == 149, i == 149])
    clear = torch.tensor([[0.3 if i > 100 else 3.0], [4.0]])
    m.step(state(E=2, clearance=clear, fell=fell, done=fell), command(E=2))
  out, tables = m.finish()
  falls = tables["falls"]
  assert len(falls["env"]) == 2 and falls["near"].tolist() == [1.0, 0.0]
  assert falls["min_clear_m_1s"].tolist() == pytest.approx([0.3, 4.0])
  assert out["fall/near_share"] == 0.5 and out["fall/count"] == 2
  ep = tables["episodes"]
  assert (ep["end"][ep["duration"] > 2.9] == 1).all()  # both first episodes ended by the fall
  assert out["first_fall/events"] == 2


def test_the_config_rejects_unknown_keys():
  assert cfg_from_dict({"ttc_cutoffs_s": [0.5]}).ttc_cutoffs_s == (0.5,)
  with pytest.raises(ValueError, match="Unknown"):
    cfg_from_dict({"sigma": 0.3})


def test_a_continuous_contact_counts_once_until_the_pair_separates():
  m = metrics()
  m.start(command(standing=True))
  for c in (0.5, -0.06, -0.04, -0.06, -0.02, 0.1, -0.06):
    m.step(state(standing=torch.tensor([True]), clearance=torch.tensor([[c]])), command(standing=True))
  out, tables = m.finish()
  assert out["collision/count"] == 2  # the dip to -0.04 is still contact; 0.1 is a separation


def test_a_fall_tied_with_a_collision_still_counts_as_a_fall():
  m = metrics()
  m.start(command())
  m.step(state(), command())
  both = torch.tensor([True])
  m.step(state(clearance=torch.tensor([[-0.1]]), fell=both, collided=both, done=both), command())
  out, tables = m.finish()
  assert tables["episodes"]["end"][0] == 0 and tables["episodes"]["fell"][0] == 1  # cause: collision
  assert out["fall/count"] == 1 and out["first_fall/events"] == 1 and out["first_collision/events"] == 1


def test_a_new_goal_is_not_a_reaction():
  m = metrics()
  m.start(command(goal=(9.0, 0.0)))
  for i in range(60):  # constant velocity; the goal (and command direction) changes at step 20
    goal = (9.0, 0.0) if i < 20 else (0.0, 9.0)
    cmd = torch.tensor([[1.0, 0.0]]) if i <= 20 else torch.tensor([[0.0, 1.0]])
    pre = state(vel=torch.tensor([[1.0, 0.0]]), cmd_w=cmd, clearance=torch.tensor([[1.5]]),
                human_pos=torch.tensor([[[1.8, 0.0]]]))
    m.step(pre, command(goal=goal))
  out, tables = m.finish()
  assert tables["encounters"]["threat"][0] == 1 and tables["encounters"]["responded"][0] == 0


def test_large_deviations_are_exact_and_ttc_medians_keep_infinity():
  m = metrics()
  m.start(command(goal=(10.0, 0.0)))
  for _ in range(5):
    m.step(state(pos=torch.tensor([[2.0, 8.0]])), command(goal=(10.0, 0.0)))
  out, _ = m.finish()
  assert out["command/cross_track_mean_m"] == pytest.approx(8.0) and out["command/cross_track_rms_m"] == pytest.approx(8.0)
  m = metrics(M=3)
  m.start(command(standing=True))
  # One human closing at 1 m/s from 0.9 m, two standing still 0.5 m away: minima 0.9 s, inf, inf.
  pre = state(M=3, standing=torch.tensor([True]), clearance=torch.tensor([[0.9, 0.5, 0.5]]),
              human_pos=torch.tensor([[[1.2, 0.0], [0.0, 0.8], [0.0, -0.8]]]),
              human_vel=torch.tensor([[[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]))
  m.step(pre, command(standing=True))
  out, _ = m.finish()
  assert out["encounter/count"] == 3 and math.isinf(out["encounter/min_ttc_median_s"])


def test_paired_timers_and_standing_flags_ignore_arrivals():
  from types import SimpleNamespace

  from crowdmotion.sim.mjlab.traversal import _GOAL, _STANDING, _TIMER, CrowdGoalCommand

  class Probe(CrowdGoalCommand):  # the draws alone, without an env
    device = "cpu"

  cmd = object.__new__(Probe)
  zeros = lambda: torch.zeros(2, dtype=torch.long)  # noqa: E731
  cmd.cfg = SimpleNamespace(paired_seed=5)
  cmd.episodes, cmd.legs, cmd.resamples = torch.tensor([1, 1]), zeros(), zeros()
  envs = torch.tensor([0, 1])
  before = {k: cmd._uniform(envs, (), k) for k in (_TIMER, _STANDING, _GOAL)}
  cmd.legs = torch.tensor([3, 0])  # env 0 reached three goals
  after = {k: cmd._uniform(envs, (), k) for k in (_TIMER, _STANDING, _GOAL)}
  assert torch.equal(before[_TIMER], after[_TIMER]) and torch.equal(before[_STANDING], after[_STANDING])
  assert after[_GOAL][0] != before[_GOAL][0] and after[_GOAL][1] == before[_GOAL][1]
