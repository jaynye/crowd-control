"""`crowd play traversal`'s live viewer without a simulator: runs on offer, crowds, refusals,
switch planning, the live panel's counters, queued crowd edits and stepping a planning
decision one env step at a time."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crowdmotion.behavior.encounter import EncounterCfg
from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.metrics.traversal import CommandState, StepState, TraversalMetrics, cfg_from_dict
from crowdmotion.sim.mjlab.ablate_navigation import BASE_BEHAVIOR, condition_behavior, navigation_conditions
from crowdmotion.sim.mjlab.live import (
  TEACHER_NEIGHBORS,
  TRAINED,
  CrowdChoice,
  PolicyRun,
  Target,
  checkpoint_inputs,
  crowd_choices,
  discover_runs,
  incompatibility,
  panel_rows,
  plan_switch,
  resolve_checkpoint,
  resolve_crowd,
  run_labels,
)
from crowdmotion.sim.mjlab.navigation import NavigationCfg
from crowdmotion.sim.mjlab.planning import PlanningVecEnv
from crowdmotion.sim.mjlab.tuning import LiveCrowdTuning

DT = 0.02


# Runs and checkpoints ----------------------------------------------------------------------
def fake_run(root: Path, name: str, checkpoints=(5, 10), **record) -> Path:
  path = root / name
  folder = path / "logs" / "g1_crowd_traversal" / "2026-10-21_00-00-00_x"
  folder.mkdir(parents=True)
  for i in checkpoints:
    (folder / f"model_{i}.pt").write_bytes(b"")
  (path / "run.json").write_text(json.dumps({"policy": "lidar", "humans": 8, "perceive_robot": True,
                                             "behavior": str(BASE_BEHAVIOR), **record}))
  return path


def test_discover_runs_newest_first_and_skips_unplayable(tmp_path):
  fake_run(tmp_path, "20261021T000000Z_train_traversal_a")
  fake_run(tmp_path, "20261022T000000Z_train_traversal_b", planner_rate=1.0)
  fake_run(tmp_path, "20261023T000000Z_train_traversal_empty", checkpoints=())
  fake_run(tmp_path, "20261024T000000Z_train_traversal_odd", policy="mystery")
  fake_run(tmp_path, "20261025T000000Z_eval_traversal_b")
  runs = discover_runs(tmp_path)
  assert [r.name for r in runs] == ["b", "a"]
  assert [c.name for c in runs[0].checkpoints] == ["model_5.pt", "model_10.pt"]
  assert "planner 1 Hz" in runs[0].describe() and "velocity goals" in runs[0].describe()
  assert resolve_checkpoint(runs[0], None).name == "model_10.pt"
  assert resolve_checkpoint(runs[0], "model_5.pt").name == "model_5.pt"
  with pytest.raises(FileNotFoundError):
    resolve_checkpoint(runs[0], "model_7.pt")


def test_run_labels_disambiguate_repeated_names(tmp_path):
  fake_run(tmp_path, "20261021T000000Z_train_traversal_check")
  fake_run(tmp_path, "20261022T000000Z_train_traversal_check")
  fake_run(tmp_path, "20261022T000000Z_train_traversal_solo")
  labels = run_labels(discover_runs(tmp_path))
  assert set(labels) == {"solo", "check (20261022T000000Z)", "check (20261021T000000Z)"}


def test_checkpoint_inputs_reads_the_actor_width(tmp_path):
  layer = {"mlp.0.weight": torch.zeros(4, 7)}
  torch.save({"actor_state_dict": layer}, tmp_path / "ppo.pt")
  torch.save({"student_state_dict": {"mlp.0.weight": torch.zeros(4, 9)}, "teacher_state_dict": layer},
             tmp_path / "student.pt")
  torch.save({"iter": 3}, tmp_path / "none.pt")
  assert checkpoint_inputs(tmp_path / "ppo.pt") == 7
  assert checkpoint_inputs(tmp_path / "student.pt") == 9
  assert checkpoint_inputs(tmp_path / "none.pt") is None


# Crowds -------------------------------------------------------------------------------------
def test_crowds_are_the_campaign_conditions():
  choices = crowd_choices()
  conditions = navigation_conditions()
  assert [c.name for c in choices] == [TRAINED] + [c["name"] for c in conditions]
  base = json.loads(BASE_BEHAVIOR.read_text())
  for choice, condition in zip(choices[1:], conditions):
    assert choice.behavior == condition_behavior(base, condition)
    assert (choice.humans, choice.perception) == (condition["humans"], condition["crowd"])
  aware = next(c for c in choices if c.name == "reactive_aware_n8")
  enc = base["encounter"]
  assert f"{enc['robot_oblivious_fraction']:.0%} oblivious" in aware.description
  assert f"{enc['robot_stubborn_fraction']:.0%} stubborn" in aware.description


def test_frozen_campaign_conditions_are_used_verbatim(tmp_path):
  base = json.loads(BASE_BEHAVIOR.read_text())
  conditions = []
  for c in navigation_conditions()[:2]:
    path = tmp_path / f"{c['name']}.json"
    path.write_text(json.dumps(condition_behavior(base, c)))
    conditions.append({**c, "behavior": str(path)})
  (tmp_path / "run.json").write_text(json.dumps({"conditions": conditions}))
  choices = crowd_choices(tmp_path)
  assert [c.name for c in choices] == [TRAINED, "independent_n4", "independent_n8"]
  assert choices[1].behavior == conditions[0]["behavior"]


def policy(name="p", plays_as="lidar", humans=8, perceive=True, navigation=None, rate=1.0, checkpoints=("m1.pt",)):
  env = {"graph": None, "rotations": None, "behavior": str(BASE_BEHAVIOR), "humans": humans,
         "perceive_robot": perceive, "forward_first": False, "teacher_lidar": True, "no_avoidance": False,
         "planner_rate": rate, "navigation": navigation}
  return PolicyRun(Path(f"/runs/20261021T000000Z_train_traversal_{name}"), plays_as, plays_as, env,
                   tuple(Path(c) for c in checkpoints))


def crowd(name, pol, scratch) -> object:
  choice = next(c for c in crowd_choices() if c.name == name)
  return resolve_crowd(choice, pol, scratch)


def test_resolve_crowd_trained_and_conditions(tmp_path):
  p = policy(perceive=False, humans=16)
  own = resolve_crowd(CrowdChoice(TRAINED, ""), p, tmp_path)
  assert (own.humans, own.perception, own.behavior) == (16, "blind", str(BASE_BEHAVIOR))
  assert own.encounter.obstacles == ()  # as the env uses it
  four = crowd("independent_n4", p, tmp_path)
  assert (four.humans, four.perception) == (4, "blind")
  assert four.encounter.reactive is False and json.loads(Path(four.behavior).read_text())["encounter"][
    "reactive"] is False
  aware = crowd("reactive_aware_n8", p, tmp_path)
  assert (aware.humans, aware.perception, aware.encounter.reactive) == (8, "aware", True)


def test_teacher_refused_where_its_neighbor_count_changes():
  teacher = policy(plays_as="teacher", humans=8)
  assert TEACHER_NEIGHBORS == 6
  refusal = incompatibility(teacher, 4)
  assert refusal and "6 nearest humans" in refusal and "gives it 4" in refusal and "10 observation values" in refusal
  assert incompatibility(teacher, 16) is None and incompatibility(teacher, 6) is None
  assert incompatibility(policy(plays_as="lidar"), 4) is None
  assert "exactly 4" in incompatibility(policy(plays_as="teacher", humans=4), 8)


# Switches -----------------------------------------------------------------------------------
def test_switch_plans(tmp_path):
  nav = asdict(NavigationCfg())
  a = policy("a", navigation=nav, checkpoints=("m1.pt", "m2.pt"))
  t = lambda p, name, ckpt=None: Target(p, ckpt or p.checkpoints[-1], crowd(name, p, tmp_path))  # noqa: E731

  same = plan_switch(t(a, "independent_n8"), t(a, "independent_n8", a.checkpoints[0]))
  assert (same.rebuild, same.load, same.fields) == (False, True, {})

  flow = plan_switch(t(a, "independent_n8"), t(a, "coherent90_n8"))
  assert (flow.rebuild, flow.load, flow.fields) == (False, False, {"flow_fraction": 0.9})
  react = plan_switch(t(a, "coherent90_n8"), t(a, "reactive_blind_n8"))
  assert not react.rebuild and react.fields == {"flow_fraction": 0.0, "reactive": True}

  blind = plan_switch(t(a, "reactive_aware_n8"), t(a, "reactive_blind_n8"))
  assert blind.rebuild and any("robot perception aware → blind" in c for c in blind.changes)
  more = plan_switch(t(a, "independent_n8"), t(a, "independent_n16"))
  assert more.rebuild and any("humans 8 → 16" in c for c in more.changes)

  # Navigation arms that differ only in their rewards share an env; the task itself does not.
  b = policy("b", navigation={**nav, "reward_mode": "potential", "gamma_per_s": 0.75})
  assert not plan_switch(t(a, "independent_n8"), t(b, "independent_n8")).rebuild
  c = policy("c", navigation={**nav, "deadline_s": 20.0})
  assert plan_switch(t(a, "independent_n8"), t(c, "independent_n8")).rebuild
  velocity = policy("v")
  task = plan_switch(t(a, "independent_n8"), t(velocity, "independent_n8"))
  assert task.rebuild and any(x.startswith("task") for x in task.changes)
  teacher = policy("t", plays_as="teacher")
  assert any("policy observations" in x for x in plan_switch(t(velocity, "independent_n8"),
                                                              t(teacher, "independent_n8")).changes)


# Live counters ------------------------------------------------------------------------------
def state(E=1, M=1, **kw) -> StepState:
  zeros = lambda *s: torch.zeros(*s)  # noqa: E731
  false = torch.zeros(E, dtype=torch.bool)
  base = dict(pos=zeros(E, 2), vel=zeros(E, 2), heading=zeros(E), vel_b=zeros(E, 3), cmd_w=zeros(E, 2),
              cmd_b=zeros(E, 3), standing=false, clearance=torch.full((E, M), 5.0),
              human_pos=torch.full((E, M, 2), 50.0), human_vel=zeros(E, M, 2), trait=torch.zeros(E, M, dtype=torch.long),
              fell=false, timed_out=false, collided=false, done=false, succeeded=false, deadline=false)
  return StepState(**{**base, **kw})


def command(E=1) -> CommandState:
  return CommandState(pos=torch.zeros(E, 2), goal=torch.tensor([[5.0, 0.0]] * E), speed=torch.ones(E),
                      standing=torch.zeros(E, dtype=torch.bool), arrivals=torch.zeros(E))


def episodes(m: TraversalMetrics, snapshots: bool) -> list:
  """Three episodes: a collision at 1 s, a success at 2 s, then one still running."""
  out = []
  yes = torch.ones(1, dtype=torch.bool)
  m.start(command())
  for i in range(150):
    kw = {}
    if i == 49:
      kw = dict(clearance=torch.full((1, 1), -0.1), collided=yes, done=yes)
    elif i == 99:
      kw = dict(succeeded=yes, done=yes)
    m.step(state(**kw), command())
    if snapshots and i % 10 == 0:
      out.append(m.snapshot())
  return out


def test_snapshot_leaves_the_rollout_unchanged_and_feeds_the_panel():
  cfg = cfg_from_dict({"fall_controls": 0})
  plain, watched = TraversalMetrics(1, 1, DT, 1, 1.0, 0.5, cfg), TraversalMetrics(1, 1, DT, 1, 1.0, 0.5, cfg)
  episodes(plain, False)
  seen = episodes(watched, True)
  assert seen[0][0]["episode/count"] == 0 and seen[-1][0]["episode/count"] == 2  # ended episodes only
  metrics, tables = watched.snapshot()
  assert metrics["collision/count"] == 1 and metrics["exposure/robot_min"] == pytest.approx(150 * DT / 60)
  rows = dict(panel_rows(metrics, tables, navigation=True, seconds=watched.t, envs=1, realtime=0.9))
  assert rows["Collisions"].startswith("1 (")
  assert rows["Goals reached"].startswith("1 of 2 episodes (50%)")
  assert rows["Outcomes"] == "success 1 · collision 1 · fall 0 · deadline 0"
  assert rows["Real-time factor"] == "0.90×" and rows["Simulated"].startswith("3.0 s × 1 env")
  velocity = dict(panel_rows(metrics, tables, navigation=False, seconds=watched.t, envs=2, realtime=None))
  assert velocity["Episode ends"].startswith("collision 1") and velocity["Real-time factor"] == "–"
  a, ta = plain.finish()
  b, tb = watched.finish()
  assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
  assert all(np.array_equal(ta[k][c], tb[k][c], equal_nan=True) for k in ta for c in ta[k])


# Queued crowd edits ---------------------------------------------------------------------------
def test_encounter_edit_applies_all_fields_or_none():
  enc = EncounterCfg(robot_oblivious_fraction=0.05, robot_stubborn_fraction=0.1)
  tuning = LiveCrowdTuning(SimpleNamespace(cfg=None), PlazaSteeringCfg(), encounter=enc)
  # One field at a time, oblivious = 1 would first meet stubborn = 0.1 (sum above 1).
  tuning.submit_encounter({"robot_oblivious_fraction": 1.0, "robot_stubborn_fraction": 0.0, "reactive": False})
  assert (enc.robot_oblivious_fraction, enc.robot_stubborn_fraction, enc.reactive) == (0.05, 0.1, True)  # queued
  assert tuning.apply_pending() == 1
  assert (enc.robot_oblivious_fraction, enc.robot_stubborn_fraction, enc.reactive) == (1.0, 0.0, False)
  tuning.submit_encounter({"robot_oblivious_fraction": 0.8, "robot_stubborn_fraction": 0.5, "flow_fraction": 0.9})
  assert tuning.apply_pending() == 0 and tuning.last_result.startswith("Rejected")
  assert (enc.robot_oblivious_fraction, enc.flow_fraction) == (1.0, 0.0)  # nothing changed
  with pytest.raises(ValueError):
    tuning.submit_encounter({"no_such_field": 1})


def test_encounter_edit_updates_shown_controls_without_resubmitting():
  enc = EncounterCfg()
  tuning = LiveCrowdTuning(SimpleNamespace(cfg=None), PlazaSteeringCfg(), encounter=enc)
  handle = SimpleNamespace(value=enc.robot_oblivious_fraction)
  tuning.handles[("encounter", "robot_oblivious_fraction")] = handle
  tuning.submit_encounter({"robot_oblivious_fraction": 0.5})
  tuning.apply_pending()
  assert handle.value == 0.5 and tuning.pending.empty()


# Planning decisions, one env step at a time -----------------------------------------------------
class FakeEnv:
  """Rewards 1 per step; env 0 terminates at the step numbered `end`."""

  def __init__(self, n=2, end=None):
    self.n, self.end, self.steps = n, end, 0
    self.cfg = SimpleNamespace(is_finite_horizon=False)

  def step(self, low):
    self.steps += 1
    term = torch.zeros(self.n, dtype=torch.bool)
    if self.steps == self.end:
      term[0] = True
    obs = {"actor": torch.full((self.n, 2), float(self.steps))}
    return obs, torch.ones(self.n), term, torch.zeros(self.n, dtype=torch.bool), {"log": {"x": 1.0}}


def planning(env, substeps=4) -> PlanningVecEnv:
  p = object.__new__(PlanningVecEnv)
  n = env.n
  p.env, p.substeps, p.num_envs, p.device, p.rate = env, substeps, n, torch.device("cpu"), 1.0
  p.navigation, p.action_l2, p.action_rate, p.on_substep = None, 0.01, 0.02, None
  p.prev_action = torch.zeros(n, 3)
  p._cmd = {"mode": torch.zeros(n, dtype=torch.long), "speed": torch.zeros(n), "move": torch.zeros(n, 3),
            "face": torch.zeros(n, 3)}
  plans = SimpleNamespace(length=torch.full((n,), 100), frame=torch.zeros(n, dtype=torch.long))
  p.ctl = SimpleNamespace(act=lambda: torch.zeros(n, 29), replan=lambda *a: None, planner=object(), plans=plans,
                          reset=lambda ids: p.resets.append(ids.tolist()))
  p.resets = []
  p._set_command = lambda a: None
  p._observations = lambda obs: obs
  return p


def test_decide_takes_one_env_step_per_next_and_returns_the_step():
  a = torch.full((2, 3), 0.5)
  whole_env, stepped_env = FakeEnv(end=2), FakeEnv(end=2)
  whole, stepped = planning(whole_env), planning(stepped_env)
  counted = []
  stepped.on_substep = lambda: counted.append(stepped_env.steps)
  expected = whole.step(a)
  decision = stepped.decide(a)
  for k in range(1, 4):
    next(decision)
    assert stepped_env.steps == k and counted == list(range(1, k + 1))
  with pytest.raises(StopIteration) as stop:
    next(decision)
  got = stop.value.value
  assert stepped_env.steps == 4 and counted == [1, 2, 3, 4]
  assert torch.equal(got[0]["actor"], expected[0]["actor"]) and torch.equal(got[1], expected[1])
  assert torch.equal(got[2], expected[2]) and got[3]["log"] == expected[3]["log"]
  # env 0 ended at its second step: two rewards, then nothing; env 1 took all four.
  penalty = 0.01 * 0.75 + 0.02 * 0.75
  assert got[1].tolist() == pytest.approx([2 - penalty, 4 - penalty])
  assert got[2].tolist() == [1, 0] and stepped.resets == [[0]] and whole.resets == [[0]]


# Rebuilding the viewer's GUI on one server -----------------------------------------------------
def test_gui_with_tab_groups_clears_for_the_next_viewer():
  viser = pytest.importorskip("viser")
  from crowdmotion.sim.mjlab.live_viewer import _remove_tabs

  server = viser.ViserServer(host="127.0.0.1", port=8299, verbose=False)
  try:
    for _ in range(2):  # mjlab's viewer puts its tabs at the top level; tabs may nest
      with server.gui.add_folder("Live play"):
        server.gui.add_markdown("panel")
      tabs = server.gui.add_tab_group()
      with tabs.add_tab("Controls"):
        inner = server.gui.add_tab_group()
        with inner.add_tab("Inner"):
          server.gui.add_button("Apply")
      _remove_tabs(server.gui._container_handle_from_uuid["root"])
      server.gui.reset()
      assert not server.gui._container_handle_from_uuid["root"]._children
  finally:
    server.stop()


def test_crowd_option_is_for_the_viser_viewer(tmp_path, capsys):
  from crowdmotion.sim.mjlab.play import main

  run = fake_run(tmp_path, "20261021T000000Z_train_traversal_a")
  for extra in (["--video"], ["--viewer", "native"]):
    with pytest.raises(SystemExit) as stop:
      main([str(run), "--crowd", "independent_n4", "--device", "cpu", *extra])
    assert stop.value.code == 2 and "--crowd applies to the Viser viewer" in capsys.readouterr().err
