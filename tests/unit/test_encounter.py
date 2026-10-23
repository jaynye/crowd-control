"""Encounter design: config checks, switching hazard, stop rule, sides, turns, streams."""

from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_live_crowd_tuning import Gui
from test_posture_style import two_style_graph

from crowdmotion.behavior.designs import context_radii, make_behavior, scene_section
from crowdmotion.behavior.encounter import CRUISE, PIVOT, SIDESTEP, EncounterBehavior, EncounterCfg
from crowdmotion.behavior.planner import plan_arc
from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.graph import GraphConfig, MotionGraph
from crowdmotion.iterate import diagnose
from crowdmotion.iterate import evaluate as evaluation
from crowdmotion.sim.mjlab.tuning import CONTROLS, ENCOUNTER_CONTROLS, LiveCrowdTuning
from synthetic import synthetic_library


def head_on(envs: int, gap: float, assertive: float, speed: float = 1.0, **kw):
  """`envs` copies of two humans walking straight at each other, `gap` apart."""
  cfg = EncounterCfg(conditioning="velocity", starts=((-gap / 2, 0, 90), (gap / 2, 0, -90)),
                     velocities=((speed, 90), (speed, -90)), assertiveness_range=(assertive, assertive), **kw)
  b = EncounterBehavior(envs, 2, cfg, seed=3, device="cpu")
  ids = torch.arange(envs)
  xz, heading = b.reset(ids)
  vel = torch.stack((torch.sin(heading), torch.cos(heading)), -1) * speed
  return b, ids, xz, vel, heading


def test_config_is_strict_and_validated():
  with pytest.raises(ValueError, match="Unknown encounter"):
    EncounterCfg.from_dict({"bubble_radius": 2.0})
  with pytest.raises(ValueError, match="route per human"):
    EncounterCfg(conditioning="waypoints")
  with pytest.raises(ValueError, match="bubble"):
    EncounterCfg(bubble_radius_m=0.4)
  with pytest.raises(ValueError, match="weights"):
    EncounterCfg(strategy_far=(0, 0, 0, 0))
  cfg = EncounterCfg.from_dict({"waypoints": [[[1, 2], [3, 4]]], "conditioning": "waypoints"})
  assert cfg.waypoints == (((1.0, 2.0), (3.0, 4.0)),)
  with pytest.raises(ValueError, match="crowd has 2"):
    EncounterBehavior(1, 2, cfg, 0, "cpu")


def test_evaluation_config_selects_the_design_and_keeps_plaza_unchanged():
  plaza = evaluation.resolved_config({"steering": {"contact_m": 1.0}})
  assert "design" not in plaza and "encounter" not in plaza
  enc = evaluation.resolved_config({"design": "encounter", "encounter": {"bubble_radius_m": 3.0}})
  assert enc["encounter"]["bubble_radius_m"] == 3.0 and "steering" not in enc
  assert scene_section(enc)["half_x"] == 7.0
  assert context_radii(enc) == pytest.approx((2 * 0.35 + 0.3, 3.0))
  with pytest.raises(ValueError, match="not `steering`"):
    evaluation.resolved_config({"design": "encounter", "steering": {}})
  with pytest.raises(ValueError, match="needs"):
    evaluation.resolved_config({"encounter": {}})
  with pytest.raises(ValueError, match="one entry per human"):
    evaluation.resolved_config({"design": "encounter", "humans": 3,
                                "encounter": {"conditioning": "velocity", "velocities": [[1, 0]]}})
  behavior, planner = make_behavior("plaza", PlazaSteeringCfg(), 1, 2, 0, "cpu")
  assert planner.__name__ == "plan_path"


@pytest.mark.parametrize("gap,assertive", [(1.5, 0.5), (2.0, 0.0), (1.0, 0.25)])
def test_switch_probability_follows_assertiveness_and_closest_distance(gap, assertive):
  b, ids, xz, vel, heading = head_on(4000, gap, assertive, switch_rate_hz=2.0, stop_horizon_s=0.1)
  b.encounter_command(torch.zeros(4000), xz, vel, heading, ids)  # first step: dt = 0
  assert not b.switched.any()
  b.encounter_command(torch.full((4000,), 0.4), xz, vel, heading, ids)
  assert b.threat.all() and b.eligible.all()
  urgency = (2.5 - gap) / (2.5 - 0.7)
  expected = 1 - math.exp(-2.0 * (1 - assertive) * urgency * 0.4)
  assert torch.allclose(b.urgency, torch.tensor(urgency), atol=1e-5)
  assert abs(b.switched.float().mean().item() - expected) < 0.025


def test_fully_assertive_humans_stop_and_then_lose_patience():
  b, ids, xz, vel, heading = head_on(256, 1.2, 1.0, stop_horizon_s=1.0, switch_rate_hz=4.0)
  speed, _ = b.encounter_command(torch.zeros(256), xz, vel, heading, ids)
  assert not b.switched.any() and (b.mode == CRUISE).all()
  assert b.stopped.all() and (speed == 0).all()  # both stop: nothing else breaks the tie
  assert (b.blocked_since == 0).all()
  b.assertiveness[:, 1] = 1.0
  b.blocked_since[:, 1] = -1.0  # not yet blocked: full assertiveness, no switching
  b.encounter_command(torch.full((256,), 0.4), xz, vel, heading, ids)
  assert not b.switched[:, 1].any()
  switched = b.switched[:, 0].clone()
  for t in (0.8, 1.2, 1.6):
    b.encounter_command(torch.full((256,), t), xz, vel, heading, ids)
    switched |= b.switched[:, 0]
  assert 0.1 < switched.float().mean() < 0.9  # blocked humans gradually give way


def test_stop_rule_ignores_separating_and_parallel_neighbors():
  cfg = EncounterCfg(conditioning="velocity", starts=((0, 0, 90), (0.6, 0, 90)), velocities=((1, 90), (1, 90)),
                     assertiveness_range=(1, 1))
  b = EncounterBehavior(1, 2, cfg, 0, "cpu")
  xz, heading = b.reset(torch.tensor([0]))
  vel = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])  # in line, same velocity
  speed, _ = b.encounter_command(torch.zeros(1), xz, vel, heading, torch.tensor([0]))
  assert not b.stopped.any() and (speed > 0).all()
  cfg = EncounterCfg(conditioning="velocity", starts=((-0.3, 0, -90), (0.3, 0, 90)), velocities=((1, -90), (1, 90)))
  b = EncounterBehavior(1, 2, cfg, 0, "cpu")
  xz, heading = b.reset(torch.tensor([0]))  # already touching, but walking apart
  speed, _ = b.encounter_command(torch.zeros(1), xz, -vel * torch.tensor([[[1.0], [-1.0]]]), heading,
                                 torch.tensor([0]))
  assert not b.stopped.any() and (speed > 0).all()


def test_constrained_veer_keeps_turn_target_when_opted_in():
  from crowdmotion.behavior.encounter import VEER

  b, ids, xz, vel, heading = head_on(1, .9, 0., speed=.5, blocked_turn_first=True)
  b.mode[:] = VEER
  b.target[:] = heading + .5
  b.radius[:] = 1.
  b.until[:] = 10.
  speed, _ = b.encounter_command(torch.tensor([.4]), xz, torch.zeros_like(vel), heading, ids)
  assert b.resolution_turn.all() and b.turn_active.all()
  assert torch.equal(b.turn_target, b.target)
  assert (speed == 0).all()


def test_minimal_heading_is_tangent_and_prefers_shorter_goal_detour():
  b = EncounterBehavior(1, 1, EncounterCfg(minimal_avoidance=True), 0, "cpu")
  zero = torch.zeros(1, 1)

  def target(relative, head_on):
    return b._minimal_heading(torch.zeros(1, 1, 2), zero, torch.ones(1, 1),
                             torch.tensor([[relative]]), torch.zeros(1, 1, 2), torch.ones(1, 1),
                             torch.full((1, 1), head_on), zero)[0].item()

  assert math.degrees(target([0., 2.], True)) == pytest.approx(30., abs=.01)
  # Already inside the passing margin: a tangent (90 degrees) does not close further.
  assert math.degrees(target([0., .8], True)) == pytest.approx(90., abs=.01)
  assert math.degrees(target([.6, 2.], False)) == pytest.approx(-12., abs=.01)


def test_minimal_heading_checks_second_neighbor_and_is_order_independent():
  b = EncounterBehavior(1, 1, EncounterCfg(minimal_avoidance=True), 0, "cpu")
  zero = torch.zeros(1, 1)
  relative = torch.tensor([[[[0., 2.], [.8, 1.2]]]])
  def choose(r):
    return b._minimal_heading(torch.zeros(1, 1, 2), zero, torch.ones(1, 1), r,
                             torch.zeros_like(r), torch.ones(1, 1),
                             torch.zeros(1, 1, dtype=torch.bool), zero)[0]
  target = choose(relative)
  # The nearest-only positive tangent would enter the second body. The least safe
  # detour goes left, clearing both rather than choosing a deterministic human first.
  assert math.degrees(target.item()) == pytest.approx(-30., abs=.01)
  assert torch.equal(target, choose(relative.flip(-2)))


def test_two_stubborn_humans_release_without_pair_precedence():
  b, ids, xz, vel, heading = head_on(1, .8, .2, stubborn_fraction=1., stubborn_patience_s=(1.2, 1.2),
                                   switch_rate_hz=0.)
  vel.zero_()
  for t in (0., .4, .8):
    b.encounter_command(torch.tensor([t]), xz, vel, heading, ids)
    assert not b.switched.any() and b.stubborn.all() and not b.stubborn_released.any()
  b.encounter_command(torch.tensor([1.2]), xz, vel, heading, ids)
  assert b.stubborn_released.all() and b.switched.all() and (b.mode == PIVOT).all()
  assert not b.trace_records()["encounter_stubborn_active"].any()


def test_mutual_stop_recovery_uses_execution_and_keeps_stubborn_humans():
  from crowdmotion.behavior.encounter import YIELD

  b, ids, xz, vel, heading = head_on(1, .85, 0., stall_recovery_s=.8, switch_rate_hz=0.)
  b.mode[:] = YIELD
  b.scale[:] = .2
  b.until[:] = 10.
  b.threat[:] = True
  b.stubborn[0, 1] = True
  b.stubborn_patience[:] = 10.
  for t in (0., .4, .8):
    b.encounter_command(torch.tensor([t]), xz, torch.zeros_like(vel), heading, ids)
  assert b.mode[0, 0] == PIVOT and b.mode[0, 1] == YIELD
  assert b.trace_records()["encounter_stall_recovery"].tolist() == [True, False]


def test_geometric_pass_completion_waits_for_clear_goal_path_and_has_a_timeout():
  b, ids, xz, vel, heading = head_on(1, .85, 0., pass_commitment_s=3.)
  b.mode[:] = PIVOT
  b.phase[:] = 1
  b.until[:] = .4  # The legacy straight hold has expired.
  b.pass_deadline[:] = 3.
  b.target[:] = 0.
  def advance(t, positions):
    b._advance_maneuvers(ids, torch.full((1, 2), t), positions, heading, b.threat[ids],
                        vel=torch.zeros_like(vel), pref_speed=torch.ones(1, 2), pref_heading=heading)
  advance(1., xz)
  assert (b.mode == PIVOT).all()  # Direct resumption would re-form the conflict.
  separated = xz.clone()
  separated[0, 1, 1] += 2.
  advance(1.4, separated)
  assert (b.mode == CRUISE).all()
  b.mode[:] = PIVOT
  b.phase[:] = 1
  advance(3., xz)
  assert (b.mode == CRUISE).all()  # The commitment cannot persist forever.


def test_velocity_projection_does_not_reverse_for_duplicate_neighbors():
  from crowdmotion.behavior.encounter import project_velocity

  v = torch.tensor([[1., .5]])
  normals = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
  bounds = torch.tensor([[.2, .2, .3]])
  result = project_velocity(v, normals, bounds)
  assert torch.allclose(result, torch.tensor([[.2, .3]]), atol=1e-6)
  assert torch.allclose(result, project_velocity(v, normals[:, [2, 0, 1]], bounds[:, [2, 0, 1]]))
  assert torch.equal(project_velocity(torch.zeros_like(v), normals, bounds), torch.zeros_like(v))


def test_bypass_duration_is_independent_and_bounded_without_a_new_random_slot():
  durations = []
  for jitter in (0., .75):
    b, ids, xz, vel, heading = head_on(1, .85, 0., pass_commitment_s=3.,
                                     pass_commitment_jitter=jitter, stall_recovery_s=.8, switch_rate_hz=0.)
    for t in (0., .4, .8):
      b.encounter_command(torch.tensor([t]), xz, torch.zeros_like(vel), heading, ids)
    assert b.switched.all()  # Both initiate; neither is assigned precedence.
    durations.append(b.pass_duration.clone())
    assert (b.counters == 3).all()
  assert (durations[0] == 3.).all()
  assert ((durations[1] >= .75) & (durations[1] <= 3.)).all()
  assert durations[1][0, 0] != durations[1][0, 1]


def test_duration_jitter_only_shortens_failed_parallel_bypasses():
  b, ids, xz, vel, heading = head_on(1, .85, 0., pass_commitment_s=3., pass_commitment_jitter=.75)
  b.mode[:] = PIVOT
  b.phase[:] = 1
  b.hold[:] = 1.
  b.until[:] = 1.
  b.pass_deadline[:] = 3.
  b.pass_duration[:] = torch.tensor([[1., 2.]])
  vel[:] = torch.tensor([0., .45])  # Both walk north, neither toward its east/west goal.
  b._advance_maneuvers(ids, torch.full((1, 2), 1.4), xz, heading, b.threat[ids],
                      vel=vel, pref_speed=torch.ones(1, 2), pref_heading=heading)
  assert b.pass_deadline.tolist() == [[1., 2.]]
  assert b.mode.tolist() == [[CRUISE, PIVOT]]
  # With useful forward goal progress, the same sampled duration has no effect.
  b.mode[:] = PIVOT
  b.phase[:] = 1
  b.pass_deadline[:] = 3.
  vel[:] = torch.tensor([.45, 0.])
  b._advance_maneuvers(ids, torch.full((1, 2), 1.4), xz, heading, b.threat[ids],
                      vel=vel, pref_speed=torch.ones(1, 2), pref_heading=heading)
  assert b.pass_deadline[0, 0] == 3.


def test_braking_forecast_keeps_executed_motion_and_adds_slowdown_scenario():
  b, ids, xz, vel, heading = head_on(1, .85, 0., anticipate_braking=True, switch_rate_hz=0.)
  b.encounter_command(torch.zeros(1), xz, vel, heading, ids)
  points, forecasts, _ = b._neighbors
  assert points.shape == (2, 4, 2)  # One person, two walls, one braking scenario.
  assert torch.equal(points[:, 0], points[:, -1])
  assert torch.equal(forecasts[:, 0], vel[0].flip(0))
  assert (forecasts[:, -1].norm(dim=-1) < forecasts[:, 0].norm(dim=-1)).all()
  distances = (xz[:, None] - xz[:, :, None]).norm(dim=-1)
  distances[:, torch.arange(2), torch.arange(2)] = torch.inf
  unneeded, _, _ = b._nearest(xz, vel, distances, .71, braked=vel)
  assert (unneeded[:, -1] == 1e6).all()  # Identical forecasts are not charged twice.


def test_stubborn_traits_are_independent_of_batch_order_and_composition():
  cfg = EncounterCfg(stubborn_fraction=.15)
  a = EncounterBehavior(5, 16, cfg, 17, "cpu")
  b = EncounterBehavior(5, 16, cfg, 17, "cpu")
  a.reset(torch.arange(5))
  b.reset(torch.tensor([3, 1]))
  for name in ("stubborn", "stubborn_patience", "assertiveness", "counters"):
    assert torch.equal(getattr(a, name)[[3, 1]], getattr(b, name)[[3, 1]])


def test_stubborn_live_controls_queue_and_validate_ranges():
  cfg = EncounterCfg()
  tuning = LiveCrowdTuning(two_style_graph(), PlazaSteeringCfg(), encounter=cfg)
  tuning.create_gui(SimpleNamespace(gui=Gui()))
  tuning.handles[("encounter", "stubborn_fraction")].change(.07)
  tuning.handles[("encounter", "nonstubborn_assertiveness_range", 1)].change(.25)
  tuning.handles[("encounter", "stubborn_patience_s", 0)].change(10.)
  assert cfg.stubborn_fraction == 0  # callbacks enqueue, without immediate mutation
  assert tuning.apply_pending() == 3
  assert cfg.stubborn_fraction == .07 and cfg.nonstubborn_assertiveness_range == (0., .25)
  tuning.handles[("encounter", "stubborn_patience_s", 1)].change(5.)
  assert tuning.apply_pending() == 0 and cfg.stubborn_patience_s == (10., 14.)


def test_sidestep_goes_away_from_the_threat_and_respects_walls():
  # Human 1 approaches slightly to human 0's right (+X is right when facing +Z).
  def run(start_x):
    cfg = EncounterCfg(conditioning="velocity", starts=((start_x, 0, 0), (start_x + 0.3, 2.0, 180)),
                       velocities=((1, 0), (1, 180)), assertiveness_range=(0, 0), switch_rate_hz=1e6,
                       strategy_far=(1, 0, 0, 0), strategy_near=(1, 0, 0, 0), side_noise=0, stop_horizon_s=0.1)
    b = EncounterBehavior(1, 2, cfg, 0, "cpu")
    xz, heading = b.reset(torch.tensor([0]))
    vel = torch.stack((torch.sin(heading), torch.cos(heading)), -1)
    for t in (0.0, 0.4):
      b.encounter_command(torch.tensor([t]), xz, vel, heading, torch.tensor([0]))
    return b
  b = run(0.0)
  assert b.mode[0, 0] == SIDESTEP and b.side[0, 0] == -1  # left, away from the threat
  assert b.turn_active[0, 0] and b.orientation[0, 0] == 0  # body held; no forward preference
  b = run(-6.0)  # the wall is on the left
  assert b.mode[0, 0] == SIDESTEP and b.side[0, 0] == 1


def test_ordinary_turns_are_in_place_or_arcs():
  def turn(probability):
    cfg = EncounterCfg(conditioning="velocity", starts=((0, 0, 0),), velocities=((1.2, 120),),
                       inplace_probability=probability, turn_radius_m=(1.0, 1.0))
    b = EncounterBehavior(1, 1, cfg, 0, "cpu")
    xz, heading = b.reset(torch.tensor([0]))
    heading = torch.zeros_like(heading)  # facing +Z, asked to travel at 120 degrees
    speed, travel = b.encounter_command(torch.zeros(1), xz, torch.zeros(1, 1, 2), heading, torch.tensor([0]))
    return b, speed, travel
  b, speed, _ = turn(1.0)
  assert speed.item() == 0 and b.turn_active[0, 0]
  assert math.isclose(b.turn_target[0, 0].item(), math.radians(120), abs_tol=1e-5)
  b, speed, travel = turn(0.0)
  assert math.isclose(speed.item(), 1.2, rel_tol=1e-5) and travel.item() == 0  # the arc starts from facing
  yaw_rate, max_turn = b._plan
  assert math.isclose(yaw_rate.item(), 1.2 / 1.0, rel_tol=1e-5)  # radius 1 m
  assert math.isclose(max_turn.item(), math.radians(120), rel_tol=1e-5)


def test_plan_arc_radius_and_in_place_sweep():
  v, w = 1.2, 0.6
  path, heading = plan_arc(torch.zeros(1, 2), torch.tensor([[0.0, v]]), torch.tensor([v]), torch.zeros(1),
                           torch.tensor([w]), 60, 20.0, blend_s=0.0)
  steps = torch.diff(path[0], dim=0)
  turn = torch.diff(torch.atan2(steps[:, 0], steps[:, 1]))
  radius = steps.norm(dim=-1)[1:] / turn  # arc length per radian of travel turn
  assert torch.allclose(radius, torch.full_like(radius, v / w), rtol=1e-3)
  path, heading = plan_arc(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1), torch.zeros(1),
                           torch.tensor([-2.0]), 40, 20.0, max_turn=torch.tensor([1.0]))
  assert torch.equal(path, torch.zeros_like(path)) and math.isclose(heading[0, -1].item(), -1.0)


def test_streams_do_not_depend_on_batch_composition():
  cfg = EncounterCfg()
  a = EncounterBehavior(3, 6, cfg, 5, "cpu")
  b = EncounterBehavior(3, 6, cfg, 5, "cpu")
  xa, ha = a.reset(torch.arange(3))
  xb, hb = b.reset(torch.tensor([1]))
  assert torch.equal(xa[1], xb[0]) and torch.equal(ha[1], hb[0])
  va, vb = torch.zeros(3, 6, 2), torch.zeros(1, 6, 2)
  for k in range(30):
    t = torch.full((3,), 0.4 * k)
    sa, da = a.encounter_command(t, xa, va, ha, torch.arange(3))
    sb, db = b.encounter_command(t[:1], xb, vb, hb, torch.tensor([1]))
    assert torch.equal(sa[1], sb[0]) and torch.equal(da[1], db[0])
    va = torch.stack((torch.sin(da), torch.cos(da)), -1) * sa[..., None]
    vb = torch.stack((torch.sin(db), torch.cos(db)), -1) * sb[..., None]
    xa, xb = xa + 0.4 * va, xb + 0.4 * vb
    ha, hb = torch.where(sa > 0, da, ha), torch.where(sb > 0, db, hb)


def test_random_waypoints_are_directed_and_clear_of_obstacles():
  cfg = EncounterCfg(obstacles=((0.0, 0.0, 1.0),), wait_probability=0.0)
  b = EncounterBehavior(8, 8, cfg, 1, "cpu")
  xz, heading = b.reset(torch.arange(8))
  leg = (b.goal - xz).norm(dim=-1)
  assert (leg >= 3.0).float().mean() > 0.95
  assert ((b.goal.norm(dim=-1) >= 1.8) & (b.goal.abs() <= torch.tensor([6.0, 4.0]) + 1e-6).all(-1)).all()
  gaps = (xz[:, :, None] - xz[:, None]).norm(dim=-1) + torch.eye(8) * 9
  assert (gaps > 1.2).all()
  to_goal = b.goal - xz  # spawned facing the first waypoint
  assert torch.allclose(heading, torch.atan2(to_goal[..., 0], to_goal[..., 1]))


def test_live_encounter_controls_replace_plaza_steering_controls():
  cfg = EncounterCfg()
  tuning = LiveCrowdTuning(two_style_graph(), PlazaSteeringCfg(), encounter=cfg)
  tuning.create_gui(SimpleNamespace(gui=Gui()))
  owners = {key[0] for key in tuning.handles}
  assert owners == {"graph", "encounter"}
  assert len(tuning.handles) == len([c for c in CONTROLS if c.owner == "graph"]) + len(ENCOUNTER_CONTROLS)
  tuning.handles[("encounter", "strategy_near", 2)].change(4.0)
  tuning.handles[("encounter", "bubble_radius_m")].change(3.5)
  tuning.handles[("encounter", "body_radius_m")].change(0.5)  # contact 1.0 m: still inside a 3.5 m bubble
  assert tuning.apply_pending() == 3
  assert cfg.strategy_near[2] == 4.0 and cfg.bubble_radius_m == 3.5
  tuning.submit("encounter", "bubble_radius_m", 0.8)  # smaller than contact
  assert tuning.apply_pending() == 0 and cfg.bubble_radius_m == 3.5
  assert tuning.handles[("encounter", "bubble_radius_m")].value == 3.5


def test_encounter_evaluation_saves_decisions_and_diagnoses(tmp_path, monkeypatch):
  monkeypatch.setattr("crowdmotion.core.rundir.environment", lambda: {"host": "toy-test", "gpu": None})
  lib = synthetic_library()
  graph = MotionGraph(lib, GraphConfig(neighbors=3), device="cpu")
  graph_path = tmp_path / "toy.pt"
  graph.save(graph_path, lib["posed_joints"][0, 0])
  config = tmp_path / "toy.json"
  config.write_text(json.dumps({"design": "encounter", "humans": 4, "seconds": 4.0, "seeds": [0],
                                "graph_overrides": {"turn_weight": 1.0}}))
  run = tmp_path / "run"
  evaluation.evaluate(argparse.Namespace(config=config, out=run, graph=graph_path, device="cpu", threads=1))
  with np.load(run / "seed_0.npz") as trace:
    steps = trace["edge"].shape[1]
    for key in ("encounter_mode", "encounter_switch", "encounter_urgency", "steering_evasive", "turn_active"):
      assert trace[key].shape == (4, steps)
    assert np.isfinite(trace["root"]).all()
  summary = json.loads((run / "seed_0.json").read_text())
  assert summary["encounter"]["body_clearance"]["contact_pair_events"] >= 0
  diagnose.main(["--run", str(run), "--out", str(tmp_path / "diag")])
  row = json.loads((tmp_path / "diag" / "seed_0.json").read_text())
  assert row["ordinary_context_source"].startswith("saved encounter")
  assert "encounter" in row and "ordinary_redirects" in row
  assert PIVOT > SIDESTEP > CRUISE


def test_clearance_cost_penalizes_only_closing_inside_the_clearance():
  from crowdmotion.graph.costs import CandidateContext, ClearanceCost

  frames = torch.arange(1, 9, dtype=torch.float32)[:, None] / 8
  stand = torch.zeros(8, 2)
  toward = frames * torch.tensor([0.0, 0.3])
  away = -frames * torch.tensor([0.0, 0.3])

  def cost(neighbor_z):
    pred = torch.stack((stand, toward, away))[None]  # [1, 3, 8, 2]
    ctx = CandidateContext(graph=SimpleNamespace(fps=20.0), state=SimpleNamespace(root_xz=torch.zeros(1, 2)),
                           path=None, cand=None, ddist=torch.zeros(1, 3), head=None, local=None, pred=pred)
    ctx.neighbors = (torch.tensor([[[0.0, neighbor_z]]]), torch.zeros(1, 1, 2), torch.tensor([0.51]))
    return ClearanceCost()(ctx)[0]

  assert torch.allclose(cost(0.8), torch.tensor([0.0, 0.01, 0.0]), atol=1e-6)  # 0.8 - 0.3 = 0.5 < 0.51
  assert torch.allclose(cost(0.3), torch.tensor([0.0, 0.3, 0.0]), atol=1e-6)  # already too close: only closing
  assert torch.equal(cost(5.0), torch.zeros(3))  # out of reach


def test_clearance_cost_counts_each_neighbor_separately():
  from crowdmotion.graph.costs import CandidateContext, ClearanceCost

  frames = torch.arange(1, 9, dtype=torch.float32)[:, None] / 8
  toward = frames * torch.tensor([0.0, 0.3])
  pred = torch.stack((torch.zeros(8, 2), toward))[None]
  ctx = CandidateContext(graph=SimpleNamespace(fps=20.0), state=SimpleNamespace(root_xz=torch.zeros(1, 2)),
                         path=None, cand=None, ddist=torch.zeros(1, 2), head=None, local=None, pred=pred)
  # A wall mirror already inside the clearance behind, and a person ahead: closing on
  # the person is still penalized.
  ctx.neighbors = (torch.tensor([[[0.0, -0.3], [0.0, 0.8]]]), torch.zeros(1, 2, 2), torch.tensor([0.51]))
  assert torch.allclose(ClearanceCost()(ctx)[0], torch.tensor([0.0, 0.01]), atol=1e-6)
