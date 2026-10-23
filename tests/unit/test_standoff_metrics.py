"""Standoff diagnostics distinguish idle stalls, maneuvers and censored threats."""

import numpy as np
import pytest

from crowdmotion.metrics.encounters import standoff_metrics


def stopped_pair(steps=10):
  root = np.zeros((2, steps + 1, 2))
  root[1, :, 0] = .8
  threats = np.zeros((2, steps, 2), dtype=bool)
  threats[0, :2, 1] = True
  trace = {"root": root, "heading": np.zeros((2, steps + 1)),
           "command": np.zeros((2, steps, 2)), "waiting": np.zeros((2, steps), dtype=bool),
           "goal": np.full((2, steps, 2), 5.), "encounter_mode": np.zeros((2, steps), dtype=int),
           "encounter_pair_threat": threats}
  return trace


def test_idle_standoff_and_censoring_have_explicit_denominators():
  result = standoff_metrics(stopped_pair(), 2.5, 1, {"encounter": {}})
  assert result["pair_standoffs"]["count"] == 1
  assert result["pair_standoffs"]["duration_s_max"] == pytest.approx(4.)
  assert result["pending_goal_stops"]["count"] == 2
  assert result["stopped_pending_goal_s_per_human_minute"] == pytest.approx(60.)
  assert result["threat_encounters"]["censored_at_end"] == 1
  assert result["threat_encounters"]["resolution_s_p50_p95"] is None
  assert result["threat_encounters"]["neither_deviates_percent"] == 100


def test_stalled_maneuver_is_reported_separately_and_resolution_is_measured():
  trace = stopped_pair()
  trace["encounter_mode"][0, 1:5] = 3
  trace["root"][1, 6:, 0] = np.arange(5) * .2 + 1.0
  result = standoff_metrics(trace, 2.5, 1, {"encounter": {}})
  assert result["pair_stalls_including_maneuvers"]["count"] == 1
  assert result["pair_standoffs"]["count"] == 0
  assert result["threat_encounters"]["resolved"] == 1
  assert result["threat_encounters"]["either_deviates_percent"] == 100
  assert result["threat_encounters"]["resolution_s_p50_p95"] == pytest.approx([2.8, 2.8])


def test_waiting_pairs_and_single_humans_do_not_form_standoffs():
  trace = stopped_pair()
  trace["waiting"][:] = True
  result = standoff_metrics(trace, 2.5, 1, {"encounter": {}})
  assert result["pair_standoffs"]["count"] == 0
  assert result["pending_goal_stops"]["count"] == 0
  trace = {k: v[:1] for k, v in trace.items()}
  assert standoff_metrics(trace, 2.5, 1, {})["threat_encounters"]["count"] == 0


def test_fixed_route_does_not_count_an_intermediate_arrival_as_the_final_one():
  trace = stopped_pair()
  trace["goal"][:] = 0.
  trace["goal"][:, 3:, 0] = 5.
  cfg = {"encounter": {"conditioning": "waypoints", "loop_waypoints": False,
                        "waypoints": [[[0., 0.], [5., 0.]], [[0., 0.], [5., 0.]]]}}
  result = standoff_metrics(trace, 2.5, 1, cfg)
  assert result["goal_arrivals"] == 2  # One goal change per human, final goal unreached.
