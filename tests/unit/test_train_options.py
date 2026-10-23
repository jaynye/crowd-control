"""`crowd train traversal`: resumed and distilled runs share the earlier run's env."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from crowdmotion.sim.mjlab.train import DEFAULT_BEHAVIOR, env_options, env_overrides, newest_checkpoint


def args(**kw):
  base = dict(graph=None, rotations=None, behavior=None, humans=None, robot_blind=None, forward_first=None,
              teacher_lidar=None, no_avoidance=None, penalty_ramp=None, planner_rate=None,
              crowd_curriculum=None)
  return Namespace(**{**base, **kw})


def earlier_run(tmp_path, **record):
  run = tmp_path / "20261012T000000Z_teacher"
  run.mkdir()
  (tmp_path / "configs").mkdir()
  (tmp_path / "configs/crowd.json").write_text("{}")
  (run / "run.json").write_text(json.dumps({"source_root": str(tmp_path), **record}))
  return run


def test_a_fresh_run_uses_the_defaults_and_its_flags():
  env = env_options(args(), None)
  assert env == {"graph": None, "rotations": None, "behavior": str(DEFAULT_BEHAVIOR), "humans": 8,
                 "perceive_robot": True, "forward_first": False, "teacher_lidar": True, "no_avoidance": False,
                 "penalty_ramp": 0, "planner_rate": 0.0, "crowd_curriculum": 0.0, "navigation": None}
  env = env_options(args(robot_blind=True, forward_first=True, humans=4, teacher_lidar=False), None)
  assert not env["perceive_robot"] and env["forward_first"] and env["humans"] == 4 and not env["teacher_lidar"]
  env = env_options(args(no_avoidance=True, penalty_ramp=500), None)
  assert env["no_avoidance"] and env["penalty_ramp"] == 500


def test_distillation_inherits_the_teachers_env_and_rejects_contradictions(tmp_path):
  run = earlier_run(tmp_path, graph="g.pt", rotations="r.pt", behavior="configs/crowd.json", humans=6,
                    perceive_robot=True, forward_first=True, teacher_lidar=True)
  env = env_options(args(), run)
  assert env == {"graph": "g.pt", "rotations": "r.pt", "behavior": str(tmp_path / "configs/crowd.json"),
                 "humans": 6, "perceive_robot": True, "forward_first": True, "teacher_lidar": True,
                 "no_avoidance": False, "penalty_ramp": 0, "planner_rate": 0.0, "crowd_curriculum": 0.0,
                 "navigation": None}
  assert env_options(args(forward_first=True, behavior=str(tmp_path / "configs/crowd.json")), run) == env
  for wrong in (dict(humans=8), dict(robot_blind=True), dict(behavior=str(DEFAULT_BEHAVIOR)), dict(graph="h.pt"),
                dict(teacher_lidar=False)):
    with pytest.raises(ValueError, match="differs"):
      env_options(args(**wrong), run)


def test_runs_recorded_before_robot_perception_were_robot_blind(tmp_path):
  run = earlier_run(tmp_path, graph="g.pt", rotations="r.pt", behavior="configs/crowd.json", humans=8)
  env = env_options(args(), run)
  assert env["perceive_robot"] is False and env["forward_first"] is False and env["teacher_lidar"] is False


def test_resume_repeats_only_the_env_overrides():
  argv = ["t5", "--teacher", "--", "--env.scene.num-envs", "4096", "--agent.max-iterations", "4000",
          "--env.rewards.goal-heading.weight", "1.0", "--agent.run-name", "t5"]
  assert env_overrides(argv) == ["--env.scene.num-envs", "4096", "--env.rewards.goal-heading.weight", "1.0"]
  assert env_overrides(["t0", "--teacher"]) == []


def test_newest_checkpoint_skips_linked_runs(tmp_path):
  logs = tmp_path / "logs" / "g1_crowd_traversal_student"
  for folder, iters in (("2026-10-12_student", (0, 100, 200)), ("teacher_t5", (1000,)), ("init_s", (1500,))):
    (logs / folder).mkdir(parents=True)
    for i in iters:
      (logs / folder / f"model_{i}.pt").write_text("")
  assert newest_checkpoint(tmp_path) == logs / "2026-10-12_student" / "model_200.pt"


def test_the_lidar_teacher_reads_everything_the_student_reads():
  from crowdmotion.sim.mjlab.traversal import make_traversal_env_cfg

  for teacher_lidar in (False, True):
    obs = make_traversal_env_cfg("graph.pt", str(DEFAULT_BEHAVIOR), teacher_lidar=teacher_lidar).observations
    teacher, student = list(obs["teacher"].terms), list(obs["actor"].terms)
    assert {"crowd_nearest", "plaza_pose"} <= set(teacher)
    assert ("lidar" in teacher) == teacher_lidar
    assert set(student) - {"lidar"} <= set(teacher)


def test_the_terminal_penalties_are_off_by_default_and_absent_in_play():
  from crowdmotion.sim.mjlab.traversal import make_traversal_env_cfg

  train = make_traversal_env_cfg("graph.pt", str(DEFAULT_BEHAVIOR))
  assert train.rewards["collision_end"].weight == 0.0 and "crowd_collision" in train.terminations
  assert train.rewards["fall_end"].weight == 0.0 and train.rewards["fall_end"].params["term_name"] == "fell_over"
  assert "penalty_ramp" not in train.curriculum
  play = make_traversal_env_cfg("graph.pt", str(DEFAULT_BEHAVIOR), play=True)
  assert "collision_end" not in play.rewards and "fall_end" not in play.rewards
  ramped = make_traversal_env_cfg("graph.pt", str(DEFAULT_BEHAVIOR), penalty_ramp=500)
  params = ramped.curriculum["penalty_ramp"].params
  assert params["reward_names"] == ("collision_end", "fall_end") and params["steps"] == 500 * 24


def test_the_penalty_ramp_scales_the_configured_weights():
  from types import SimpleNamespace

  from crowdmotion.sim.mjlab.traversal import penalty_ramp_term

  terms = {"collision_end": SimpleNamespace(weight=-1000.0), "fall_end": SimpleNamespace(weight=0.0)}
  env = SimpleNamespace(reward_manager=SimpleNamespace(get_term_cfg=terms.__getitem__), common_step_counter=0)
  cfg = SimpleNamespace(params={"reward_names": ("collision_end", "fall_end"), "steps": 100})
  ramp = penalty_ramp_term(cfg, env)
  for step, weight in ((0, 0.0), (25, -250.0), (100, -1000.0), (400, -1000.0)):
    env.common_step_counter = step
    ramp(env, None, **cfg.params)
    assert terms["collision_end"].weight == weight and terms["fall_end"].weight == 0.0


def test_b0_learns_nothing_about_the_humans():
  from crowdmotion.sim.mjlab.traversal import make_traversal_env_cfg, register

  b0 = make_traversal_env_cfg("graph.pt", str(DEFAULT_BEHAVIOR), no_avoidance=True)
  assert "teacher" not in b0.observations
  for group in ("actor", "critic"):
    assert not {"lidar", "crowd_nearest", "plaza_pose"} & set(b0.observations[group].terms)
  assert not {"crowd_proximity", "crowd_contact", "collision_end"} & set(b0.rewards)
  assert "crowd_collision" not in b0.terminations and "fall_end" in b0.rewards
  assert b0.commands["crowd"].perceive_entity == "robot"  # the humans still see the robot
  with pytest.raises(ValueError, match="no teacher"):
    register("graph.pt", str(DEFAULT_BEHAVIOR), policy="teacher", no_avoidance=True)


def test_a_planning_arm_keeps_only_the_task_rewards_and_counts_in_decisions():
  from crowdmotion.sim.mjlab.traversal import PLANNER_REWARDS, make_traversal_env_cfg, traversal_rl_cfg

  plain = make_traversal_env_cfg("graph.pt", str(DEFAULT_BEHAVIOR))
  planned = make_traversal_env_cfg("graph.pt", str(DEFAULT_BEHAVIOR), planner_rate=2.0)
  assert set(planned.rewards) == set(PLANNER_REWARDS) & set(plain.rewards)
  assert "collision_end" in planned.rewards and "air_time" in plain.rewards
  assert set(planned.observations) == set(plain.observations)
  one, two = traversal_rl_cfg(planner_rate=1.0), traversal_rl_cfg(planner_rate=2.0)
  assert (one.num_steps_per_env, two.num_steps_per_env) == (16, 32)  # 16 s of rollout either way
  assert two.algorithm.gamma ** 2 == pytest.approx(one.algorithm.gamma) == 0.75
  assert traversal_rl_cfg().num_steps_per_env == 24 and traversal_rl_cfg().algorithm.gamma == 0.99
  assert env_options(args(planner_rate=2.0), None)["planner_rate"] == 2.0


def test_play_takes_the_policy_and_env_from_the_run(tmp_path):
  from crowdmotion.sim.mjlab.play import label, run_policy

  run = earlier_run(tmp_path, graph="g.pt", rotations="r.pt", behavior="configs/crowd.json", humans=6,
                    perceive_robot=True, forward_first=False, teacher_lidar=True, policy="student")
  trained, plays_as, env = run_policy(run)
  assert (trained, plays_as) == ("student", "lidar")
  assert env["humans"] == 6 and env["teacher_lidar"] and env["behavior"] == str(tmp_path / "configs/crowd.json")
  record = json.loads((run / "run.json").read_text())
  for policy, plays in (("teacher", "teacher"), ("lidar", "lidar")):
    (run / "run.json").write_text(json.dumps({**record, "policy": policy}))
    assert run_policy(run)[1] == plays
  (run / "run.json").write_text(json.dumps({k: v for k, v in record.items() if k != "teacher_lidar"}))
  assert run_policy(run)[2]["teacher_lidar"] is False  # a legacy teacher had no LiDAR
  assert label(tmp_path / "20260101T000000Z_train_traversal_teacher_t9") == "teacher_t9"


def test_a_students_weights_load_as_the_lidar_actor(tmp_path):
  import torch

  from crowdmotion.sim.mjlab.train import actor_checkpoint

  ppo = tmp_path / "model_10.pt"
  torch.save({"actor_state_dict": {"w": torch.ones(1)}, "iter": 10, "infos": None}, ppo)
  assert actor_checkpoint(ppo) == ppo
  student = tmp_path / "s" / "model_20.pt"
  student.parent.mkdir()
  torch.save({"student_state_dict": {"w": torch.zeros(2)}, "teacher_state_dict": {}, "iter": 20}, student)
  out = actor_checkpoint(student)
  assert out != student and out.name == "model_20.pt"
  state = torch.load(out, weights_only=False)
  assert torch.equal(state["actor_state_dict"]["w"], torch.zeros(2)) and state["iter"] == 20


def test_an_attempt_repeats_the_lineages_arguments():
  from crowdmotion.sim.mjlab.train import attempt_argv, lineage_flags

  first = ["a3", "--run", "runs/x", "--humans", "8", "--wandb", "online", "--", "--agent.max-iterations", "4000"]
  assert lineage_flags(first) == (["a3", "--humans", "8", "--", "--agent.max-iterations", "4000"],
                                  ["--run", "runs/x", "--wandb", "online"])
  record = {"argv": first}
  assert lineage_flags(attempt_argv(first, record)) == lineage_flags(first)  # a requeue: the same command line
  assert attempt_argv(["--run", "runs/x"], record) == ["a3", "--humans", "8", "--run", "runs/x", "--",
                                                     "--agent.max-iterations", "4000"]
  assert attempt_argv(["--run=runs/x", "--wandb", "disabled"], record)[3:5] == ["--run=runs/x", "--wandb"]
  with pytest.raises(ValueError, match="repeats the lineage's arguments"):
    attempt_argv(["--run", "runs/x", "--humans", "4"], record)
  legacy = {"argv": ["t7", "--teacher", "--", "--agent.max-iterations", "4000"]}  # recorded before --run
  assert attempt_argv(["--run", "runs/t7"], legacy) == ["t7", "--teacher", "--run", "runs/t7", "--",
                                                        "--agent.max-iterations", "4000"]


def test_the_target_and_the_wandb_mode():
  from crowdmotion.sim.mjlab.train import lineage_target, wandb_mode
  from crowdmotion.tracking.lineage import DEFAULT_MODE

  assert lineage_target({"target_iterations": 4000, "train_config": {"max_iterations": 1}}) == 4000
  assert lineage_target({"train_config": {"max_iterations": 4000}}) == 4000  # recorded before the key
  assert lineage_target({"resume_from": "runs/a/logs/x/y/model_3700.pt", "train_config": {"max_iterations": 500}}) \
    == 4200
  assert DEFAULT_MODE == "online"
  assert wandb_mode(None, None, check=False) == DEFAULT_MODE
  assert wandb_mode(None, {"mode": "online"}, check=False) == "online"
  assert wandb_mode("disabled", {"mode": "online"}, check=False) == "disabled"
  assert wandb_mode("online", None, check=True) == "disabled"  # verification runs stay off W&B


@pytest.fixture
def lineage_env(tmp_path, monkeypatch):
  """`crowd train traversal` with mjlab's launch replaced by RSL-RL's logging loop
  (`fake_wandb.MiniRunner`) and a fake W&B SDK."""
  import signal
  from dataclasses import dataclass, field

  import mjlab.scripts.train
  import torch
  from mjlab.utils.os import get_checkpoint_path

  from crowdmotion.core import provenance, rundir
  from crowdmotion.sim.mjlab import wandb_log
  from crowdmotion.tracking import lineage
  from fake_wandb import FakeSDK, MiniRunner

  for module in (rundir, provenance):
    monkeypatch.setattr(module, "environment", lambda: {"host": "test"})
    monkeypatch.setattr(module, "git_diff", lambda: "")
  monkeypatch.setattr(wandb_log, "SESSION", None)
  monkeypatch.setenv("SLURM_JOB_ID", "123")
  sdk = FakeSDK()
  monkeypatch.setattr(lineage, "SDK", sdk)

  @dataclass
  class Env:
    sdk: FakeSDK
    calls: list = field(default_factory=list)
    loaded: list = field(default_factory=list)
    hook: object = None
    fail: bool = False

  env = Env(sdk)

  def launch(task, cfg):
    env.calls.append(cfg)
    exp = Path(cfg.log_root) / cfg.agent.experiment_name
    start = 0
    if cfg.agent.resume:
      ckpt = get_checkpoint_path(exp, cfg.agent.load_run, cfg.agent.load_checkpoint)
      env.loaded.append(ckpt)
      start = torch.load(ckpt)["iter"]
    if env.fail:
      raise RuntimeError("crashed before the first checkpoint")
    runner = type("Runner", (wandb_log.Tracked, MiniRunner), {})(
      {"start": start, "hook": env.hook},
      {"logger": "tensorboard", "algorithm": {}, "num_steps_per_env": 1, "save_interval": cfg.agent.save_interval},
      str(exp / f"2026-10-18_00-00-{len(env.calls):02d}"))
    runner.learn(cfg.agent.max_iterations)

  monkeypatch.setattr(mjlab.scripts.train, "launch_training", launch)
  previous = signal.getsignal(signal.SIGTERM)
  yield env
  signal.signal(signal.SIGTERM, previous)


def common(run, *more):
  return ["train_traversal_l1", "--graph", "g.pt", "--rotations", "r.pt", "--run", str(run), *more]


def test_a_lineage_checkpoints_at_sigterm_and_resumes_in_place_to_its_target(tmp_path, lineage_env):
  import os
  import re
  import signal

  from crowdmotion.sim.mjlab.train import EXIT_CHECKPOINTED, main
  from crowdmotion.tracking.lineage import WriterLock

  run = tmp_path / "20261018T000000Z_train_traversal_l1"
  argv = common(run, "--wandb", "online", "--", "--agent.max-iterations", "150", "--agent.save-interval", "50")
  lineage_env.hook = lambda it, runner: it == 70 and os.kill(os.getpid(), signal.SIGTERM)
  assert main(argv) == EXIT_CHECKPOINTED
  record = json.loads((run / "run.json").read_text())
  (first,) = record["attempts"]
  assert record["status"] == "checkpointed" and record["target_iterations"] == 150
  assert first["outcome"] == "checkpointed" and first["checkpoint_saved"] == 70 and first["slurm_job"] == "123"
  assert record["wandb"]["published_through"] == 70 and record["wandb"]["status"] == "preempted"
  assert lineage_env.sdk.rows() == list(range(71)) and WriterLock(run).free()

  lineage_env.hook = None
  assert main(argv) == 0  # the requeue: the same command line
  cfg = lineage_env.calls[-1]
  assert cfg.agent.load_checkpoint == re.escape("model_70.pt") and cfg.agent.max_iterations == 80
  assert lineage_env.loaded[-1].parent.name == first["log_dir"].split("/")[-1]
  record = json.loads((run / "run.json").read_text())
  assert [a["outcome"] for a in record["attempts"]] == ["checkpointed", "done"]
  assert record["attempts"][1]["start_iteration"] == 70 and record["status"] == "done"
  assert lineage_env.sdk.rows() == list(range(150))  # each iteration once
  assert {r.kwargs["id"] for r in lineage_env.sdk.runs} == {run.name}

  assert main(["--run", str(run)]) == 0 and len(lineage_env.calls) == 2  # reached its target
  with pytest.raises(SystemExit):
    main(common(run, "--humans", "4"))


def test_a_check_run_stays_off_wandb(tmp_path, lineage_env):
  from crowdmotion.sim.mjlab.train import main

  run = tmp_path / "20261018T000000Z_train_traversal_check"
  assert main(common(run, "--wandb", "online", "--", "--agent.max-iterations", "5")) == 0
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert block["mode"] == "disabled" and block["status"].startswith("check") and lineage_env.sdk.runs == []


def test_a_branch_records_its_parent_and_survives_a_repeated_first_attempt(tmp_path, lineage_env):
  import re

  import torch

  from crowdmotion.sim.mjlab.train import main

  parent = tmp_path / "20261017T000000Z_train_traversal_a2"
  logs = parent / "logs/g1_crowd_traversal/2026-10-17_a2"
  logs.mkdir(parents=True)
  torch.save({"iter": 300}, logs / "model_300.pt")
  (parent / "run.json").write_text(json.dumps({
    "kind": "train_traversal", "policy": "lidar", "argv": ["a2", "--", "--env.rewards.pose.weight", "2.0"],
    "graph": "g.pt", "rotations": "r.pt", "behavior": str(DEFAULT_BEHAVIOR), "humans": 8, "perceive_robot": True,
    "teacher_lidar": False}))
  run = tmp_path / "20261018T000000Z_train_traversal_a2p"
  argv = common(run, "--resume", str(parent), "--", "--agent.max-iterations", "100", "--agent.save-interval", "50")
  lineage_env.fail = True
  with pytest.raises(RuntimeError):
    main(argv)
  record = json.loads((run / "run.json").read_text())
  assert record["attempts"][0]["outcome"] == "failed" and record["status"] == "failed"
  record["attempts"][0].update(outcome="running", end=None)  # as a SIGKILL would leave it
  (run / "run.json").write_text(json.dumps(record))

  lineage_env.fail = False
  assert main(argv) == 0  # the resume_ link from the first attempt is reused
  cfg = lineage_env.calls[-1]
  assert cfg.agent.load_run == re.escape("resume_2026-10-17_a2") and cfg.agent.max_iterations == 100
  assert cfg.env.rewards["pose"].weight == 2.0  # the parent's --env.* overrides, repeated
  record = json.loads((run / "run.json").read_text())
  assert [a["outcome"] for a in record["attempts"]] == ["lost", "done"]
  assert record["target_iterations"] == 400 and record["wandb"]["parent"] == parent.name
  assert record["wandb"]["mode"] == "online"  # the default
  assert {r.kwargs["id"] for r in lineage_env.sdk.runs} == {run.name}


def test_an_attempt_keeps_the_lineages_recorded_env(tmp_path, lineage_env, monkeypatch):
  import os
  import signal

  import torch

  from crowdmotion.sim.mjlab import train, traversal

  seen = []
  register = traversal.register

  def recording(*args, **kwargs):
    seen.append((args, kwargs))
    return register(*args, **kwargs)

  monkeypatch.setattr(traversal, "register", recording)
  monkeypatch.setattr(train, "default_graph", lambda: "graph-A.pt")
  monkeypatch.setattr(train, "default_graph_rotations", lambda: "rotations-A.pt")
  run = tmp_path / "20261018T000000Z_train_traversal_env"
  argv = ["train_traversal_env", "--run", str(run), "--", "--agent.max-iterations", "60", "--agent.save-interval", "20"]
  lineage_env.hook = lambda it, runner: it == 30 and os.kill(os.getpid(), signal.SIGTERM)
  assert train.main(argv) == train.EXIT_CHECKPOINTED
  monkeypatch.setattr(train, "default_graph", lambda: "graph-B.pt")  # $CROWD_GRAPH changed since
  monkeypatch.setattr(train, "default_graph_rotations", lambda: "rotations-B.pt")
  lineage_env.hook = None
  seen.clear()
  assert train.main(["--run", str(run)]) == 0
  assert {(a[0], a[2]) for a, _ in seen} == {("graph-A.pt", "rotations-A.pt")}

  # A run recorded before the robot options resumes in place with them off, as it trained.
  legacy = tmp_path / "20261012T000000Z_train_traversal_teacher_t9"
  logs = legacy / "logs/g1_crowd_traversal_teacher/2026-10-12_t9"
  logs.mkdir(parents=True)
  torch.save({"iter": 100}, logs / "model_100.pt")
  (legacy / "run.json").write_text(json.dumps({
    "kind": "train_traversal", "policy": "teacher", "status": "interrupted", "graph": "g.pt", "rotations": "r.pt",
    "behavior": str(DEFAULT_BEHAVIOR), "humans": 8, "train_config": {"max_iterations": 150},
    "argv": ["t9", "--teacher", "--", "--agent.max-iterations", "150", "--agent.save-interval", "50"],
    "environment": {"slurm_job": "4059000", "host": "n1"}}))
  seen.clear()
  assert train.main(["--run", str(legacy)]) == 0
  (args, kwargs), *_ = seen
  assert args[0] == "g.pt" and kwargs["perceive_robot"] is False and kwargs["teacher_lidar"] is False
  record = json.loads((legacy / "run.json").read_text())
  assert [a["outcome"] for a in record["attempts"]] == ["interrupted", "done"]
  assert record["attempts"][0]["slurm_job"] == "4059000" and record["attempts"][1]["start_iteration"] == 100
  assert record["target_iterations"] == 150 and record["wandb"]["legacy"] and record["wandb"]["mode"] == "online"
