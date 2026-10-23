"""`crowd train traversal`: train the crowd traversal policy with mjlab's PPO runner.

A run directory `runs/<UTC>_<name>/` is one training **lineage**: `run.json` and mjlab's
log directories `logs/<experiment>/<time>_<run name>/` (checkpoints `model_<iteration>.pt`,
`params/`, TensorBoard events), one per attempt. Arguments after `--` go to mjlab's
`TrainConfig` (tyro), for example `-- --env.scene.num-envs 4096 --agent.max-iterations 3000`.

`--run runs/<dir>` names the lineage's directory: absent, it is created; present, the
attempt resumes in place from the lineage's newest checkpoint and trains to its
`target_iterations` (the first attempt's start plus `--agent.max-iterations`). It
repeats the recorded arguments: give the same ones (a Slurm requeue reruns the same
command line) or only `--run` (and `--wandb`). `run.json`'s `attempts` lists each attempt
(Slurm job, host, start, end, checkpoint loaded, outcome). A lineage that reached its
target exits at once. Without `--run` (or with `--out`) the directory is fresh.

`--teacher` trains the privileged teacher (task `Crowd-Traversal-Teacher-Flat-Unitree-G1`,
experiment `g1_crowd_traversal_teacher`) instead of the LiDAR policy. `--distill <teacher
run>` distills that run's newest teacher checkpoint into the LiDAR policy (experiment
`g1_crowd_traversal_student`); mjlab's agent options after `--` that both runners share
(iterations, seed, run name, logger, save interval) carry over.

`--finetune <student run>` is RL plus imitation: PPO on the LiDAR policy, starting from
the distilled student as the actor and its teacher's critic (both read the same
`critic` observations) with a fresh optimizer.

`--resume <run>` branches: a new lineage (a new directory and W&B run, the earlier one
recorded as its parent) that continues from the newest checkpoint of an earlier
traversal run, for example to train it on with another reward. `--resume` and
`--distill` take the earlier run's env options (`ENV_OPTIONS`) from its `run.json` and
reject an explicit option that differs. `--resume` also repeats the earlier run's
`--env.*` overrides (rewards, for example) before this run's own, which take precedence.
The env is `crowdmotion.sim.mjlab.traversal`; `--robot-blind` turns off the humans' perception
of the robot. `--no-avoidance` trains b0, the non-avoidant baseline (no LiDAR, crowd rewards or
collision termination); `--penalty-ramp N` ramps the terminal penalties (`collision_end`,
`fall_end`, set with `-- --env.rewards.<name>.weight`) in over N iterations.
`--crowd-curriculum RATE` makes the crowd more robot-blind, one step per iteration, while
training collisions stay at or below RATE per robot-minute (`traversal.crowd_difficulty_term`;
its level is saved in checkpoints).

`--wandb online|disabled` mirrors the lineage to W&B (`crowdmotion.tracking`; default: the
lineage's recorded mode, for a new lineage `DEFAULT_MODE`). Rows reach W&B only through
the last saved checkpoint. A lineage of at most `CHECK_ITERATIONS` iterations is a
verification run and stays off W&B. On SIGTERM (Slurm's signal before the time limit)
training stops after the current iteration, saves `model_<it>.pt` and exits with
`EXIT_CHECKPOINTED`, for the job to requeue.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import sys
from dataclasses import asdict, replace
from pathlib import Path

from crowdmotion.core.paths import ROOT, default_graph, default_graph_rotations
from crowdmotion.core.rundir import RunDir
from crowdmotion.tracking.lineage import (
  CHECK_ITERATIONS,
  DEFAULT_MODE,
  LINKED,
  MODES,
  WriterLock,
  checkpoint_iteration,
  env_overrides,
  first_start,
  lineage_id,
  lineage_target,
  new_block,
  own_checkpoints,
  utc_now,
)

__all__ = ["LINKED", "env_overrides"]  # imported from here by older callers

EXIT_CHECKPOINTED = 75  # EX_TEMPFAIL: the attempt stopped at a checkpoint; requeue the job
LOCK_WAIT_S = 600.0  # a sync or an evaluation attachment holds the lineage's lock briefly
LINEAGE_FLAGS = ("--run", "--out", "--wandb")  # the arguments an attempt may change (--out becomes --run)
STARTS = {"resume": "resume_from", "distill": "teacher", "finetune": "finetune_from"}  # option -> run.json key


def newest_checkpoint(run: Path) -> Path:
  """The highest-iteration `model_<i>.pt` the earlier run itself wrote, skipping the
  directories it links to (`LINKED`: the run it resumed, its teacher, its init)."""
  found = own_checkpoints(run)
  if not found:
    raise FileNotFoundError(f"no logs/<experiment>/*/model_<i>.pt under {run}")
  return found[-1][1]


def actor_checkpoint(path: Path) -> Path:
  """A checkpoint whose actor the PPO runner loads (`load_cfg={"actor": True}`). A
  distilled student's `student_state_dict` is rewritten, to a temporary file, as the
  LiDAR policy's `actor_state_dict`; other checkpoints are returned unchanged."""
  import tempfile

  import torch

  state = torch.load(path, weights_only=False, map_location="cpu")
  if "student_state_dict" not in state:
    return Path(path)
  out = Path(tempfile.mkdtemp(prefix="crowd_student_")) / Path(path).name
  torch.save({"actor_state_dict": state["student_state_dict"], "iter": state.get("iter", 0), "infos": None}, out)
  return out


DEFAULT_BEHAVIOR = ROOT / "configs/behavior/encounter_decisive_robot.json"
# Env options a resumed or distilled run must share with the earlier run, and their
# values for runs recorded before the option existed (humans then ignored the robot,
# and the teacher had no LiDAR). Fresh runs default to FRESH_TRUE options on.
ENV_OPTIONS = {"graph": None, "rotations": None, "behavior": str(DEFAULT_BEHAVIOR), "humans": 8,
               "perceive_robot": False, "forward_first": False, "teacher_lidar": False, "no_avoidance": False,
               "penalty_ramp": 0, "planner_rate": 0.0, "crowd_curriculum": 0.0, "navigation": None}
FRESH_TRUE = ("perceive_robot", "teacher_lidar")


def env_options(args, earlier: Path | None) -> dict:
  """The env options of this run: explicit arguments, else the earlier run's, else the
  defaults. Raises ValueError when an explicit option contradicts the earlier run."""
  given = {"graph": args.graph, "rotations": args.rotations, "behavior": args.behavior, "humans": args.humans,
           "perceive_robot": None if args.robot_blind is None else not args.robot_blind,
           "forward_first": args.forward_first, "teacher_lidar": args.teacher_lidar,
           "no_avoidance": args.no_avoidance, "penalty_ramp": args.penalty_ramp, "planner_rate": args.planner_rate,
           "crowd_curriculum": args.crowd_curriculum, "navigation": None}
  nav_path = getattr(args, "navigation", None)
  if nav_path is not None:
    from crowdmotion.sim.mjlab.navigation import NavigationCfg
    given["navigation"] = asdict(NavigationCfg.from_dict(json.loads(Path(nav_path).read_text())))
  if earlier is None:
    return {k: v if v is not None else (k in FRESH_TRUE or ENV_OPTIONS[k]) for k, v in given.items()}
  record = json.loads((earlier / "run.json").read_text())
  root = Path(record.get("source_root", ROOT))
  out = {}
  for key, value in given.items():
    before = record.get(key, ENV_OPTIONS[key])
    if key == "behavior" and before is not None:
      before = str(Path(before) if Path(before).is_absolute() else root / before)
      same = value is None or Path(value).resolve() == Path(before).resolve()
    else:
      same = value is None or value == before
    if not same:
      raise ValueError(f"{key}={value!r} differs from {earlier.name}'s {before!r}")
    out[key] = before if value is None else value
  return out


def split(argv: list[str]) -> tuple[list[str], list[str]]:
  """(our arguments, mjlab's arguments after `--`)."""
  return (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])


def lineage_flags(argv: list[str]) -> tuple[list[str], list[str]]:
  """(the arguments without `--run X`, `--out X` and `--wandb X`, those flags), before
  `--` only."""
  ours, theirs = split(argv)
  kept, flags, i = [], [], 0
  while i < len(ours):
    token = ours[i]
    if token in LINEAGE_FLAGS:
      flags += ours[i:i + 2]
      i += 2
      continue
    (flags if token.split("=", 1)[0] in LINEAGE_FLAGS else kept).append(token)
    i += 1
  return kept + (["--", *theirs] if "--" in argv else []), flags


def attempt_argv(argv: list[str], record: dict) -> list[str]:
  """The command line of a further attempt in an existing lineage: the recorded one, with
  this invocation's `--run` and `--wandb` (a recorded `--out`, which created the
  directory, is dropped). Raises ValueError when other arguments are given and differ
  from the recorded ones."""
  given, flags = lineage_flags(argv)
  recorded, _ = lineage_flags(record.get("argv", []))
  if given and given != recorded:
    raise ValueError(f"an attempt repeats the lineage's arguments {recorded}, not {given}; give the same ones "
                     "or only --run (start a new lineage from it with --resume)")
  ours, theirs = split(recorded)
  return [*ours, *flags, *(["--", *theirs] if "--" in recorded else [])]


def wandb_mode(given: str | None, block: dict | None, check: bool) -> str:
  """This attempt's W&B mode: off for a verification run, else `--wandb`, else the
  lineage's recorded mode, else `DEFAULT_MODE`."""
  if check:
    return "disabled"
  return given or (block or {}).get("mode") or DEFAULT_MODE


def legacy_attempts(run: Path, record: dict) -> list[dict]:
  """A run recorded before `attempts`, as its one attempt (each own log directory)."""
  env = record.get("environment", {})
  dirs = sorted({p.parent for _, p in own_checkpoints(run)})
  start = checkpoint_iteration(record["resume_from"]) if record.get("resume_from") else 0
  return [{"n": n + 1, "slurm_job": env.get("slurm_job"), "host": env.get("host"),
           "log_dir": str(d.relative_to(run)), "start_iteration": int(start or 0), "outcome": record.get("status"),
           "recorded": "before attempts"} for n, d in enumerate(dirs)]


def main(argv: list[str] | None = None) -> int:
  held: list[WriterLock] = []
  try:
    return train(list(argv or []), held)
  finally:  # every exit, an error or "nothing to train" too, releases the lineage's lock
    for lock in held:
      lock.release()


def train(argv: list[str], held: list[WriterLock]) -> int:
  p = argparse.ArgumentParser(prog="crowd train traversal", description=__doc__.split("\n\n")[0])
  p.add_argument("name", nargs="?", default="train_traversal")
  p.add_argument("--graph", help="saved motion graph (default: $CROWD_GRAPH or artifact graph-default)")
  p.add_argument("--rotations", help="the graph's joint rotations (default: $CROWD_GRAPH_ROTATIONS)")
  p.add_argument("--behavior", help=f"encounter-design behavior config for the humans (default: {DEFAULT_BEHAVIOR.name})")
  p.add_argument("--humans", type=int, help="humans per env (default: 8)")
  p.add_argument("--robot-blind", action="store_true", default=None,
                 help="humans do not perceive the robot (every human is oblivious to it)")
  p.add_argument("--forward-first", action="store_true", default=None,
                 help="scale the goal command's speed by max(cos(heading error), 0): turn, then walk")
  p.add_argument("--teacher-lidar", action=argparse.BooleanOptionalAction, default=None,
                 help="the teacher also reads the LiDAR, a superset of the student's input (default: on)")
  p.add_argument("--no-avoidance", action="store_true", default=None,
                 help="b0: learn nothing about the humans (no LiDAR, crowd rewards or collision termination)")
  p.add_argument("--penalty-ramp", type=int, metavar="N",
                 help="ramp collision_end and fall_end in from 0 over N iterations (default: 0, off)")
  p.add_argument("--planner-rate", type=float, choices=(1.0, 2.0), metavar="HZ",
                 help="a planning arm: the policy steers the released kinematic planner over frozen SONIC, "
                      "deciding at 1 or 2 Hz (default: off)")
  p.add_argument("--crowd-curriculum", type=float, metavar="RATE",
                 help="raise the robot-blind share of the crowd while training collisions stay at or below RATE "
                      "per robot-minute (default: 0, off)")
  p.add_argument("--navigation", help="strict JSON config for the persistent-goal planning task; embedded in run.json")
  mode = p.add_mutually_exclusive_group()
  mode.add_argument("--teacher", action="store_true", help="train the privileged teacher instead of the LiDAR policy")
  mode.add_argument("--distill", help="an earlier --teacher run whose newest checkpoint the LiDAR student imitates")
  mode.add_argument("--finetune", help="an earlier --distill run whose student starts PPO on the LiDAR policy")
  p.add_argument("--resume", help="an earlier `crowd train traversal` run whose newest checkpoint a new lineage "
                                  "continues from (a branch; `--run` continues a lineage in place)")
  where = p.add_mutually_exclusive_group()
  where.add_argument("--run", type=Path, help="the lineage's directory: created when absent, resumed in place "
                                              "when it holds a lineage")
  where.add_argument("--out", type=Path, help="a fresh run directory (must not exist)")
  p.add_argument("--wandb", choices=MODES, help=f"mirror the lineage to W&B (default: the lineage's recorded mode; "
                                                f"new lineages {DEFAULT_MODE})")
  args = p.parse_args(split(argv)[0])

  record = lock = None
  if args.run is not None and (args.run / "run.json").exists():
    lock = WriterLock(args.run)
    if not lock.acquire(f"crowd train traversal (job {os.environ.get('SLURM_JOB_ID')})", timeout=LOCK_WAIT_S):
      p.error(f"{args.run}: its writer lock is held: {lock.holder()}")
    held.append(lock)
    record = json.loads((args.run / "run.json").read_text())
    if record.get("kind") != "train_traversal":
      p.error(f"{args.run} is not a `crowd train traversal` run")
    try:
      argv = attempt_argv(argv, record)
    except ValueError as exc:
      p.error(str(exc))
    args = p.parse_args(split(argv)[0])
  elif args.run is not None and args.run.exists():
    p.error(f"{args.run} exists but holds no run.json")
  theirs = split(argv)[1]
  if (args.distill or args.finetune) and args.resume:
    p.error("--distill and --finetune start fresh runs; they do not combine with --resume")
  if args.teacher_lidar is not None and not (args.teacher or args.resume or args.distill or args.finetune):
    p.error("--teacher-lidar applies to --teacher runs")
  if args.no_avoidance and (args.teacher or args.distill or args.finetune):
    p.error("--no-avoidance trains the plain policy; it has no teacher or student")
  if args.no_avoidance and args.planner_rate:
    p.error("--planner-rate trains a planning arm, which learns avoidance; --no-avoidance is b0")
  if args.crowd_curriculum is not None and not args.crowd_curriculum >= 0:
    p.error("--crowd-curriculum takes a collision rate (0 = off)")
  if args.crowd_curriculum and (args.robot_blind or args.no_avoidance):
    p.error("--crowd-curriculum makes a robot-aware crowd blinder; it needs collision ends (not --no-avoidance) "
            "and not --robot-blind")
  start = args.resume or args.distill or args.finetune
  earlier = Path(start) if start else None
  if args.resume:
    theirs = env_overrides(json.loads((earlier / "run.json").read_text()).get("argv", [])) + theirs
  try:  # a further attempt keeps the lineage's own recorded env (graph, robot options), not today's defaults
    env = env_options(args, args.run if record is not None else earlier)
  except ValueError as exc:
    p.error(str(exc))
  if record is not None:
    target, own = lineage_target(record), own_checkpoints(args.run)
    if record.get("status") == "done" or (own and own[-1][0] >= target - 1):
      print(f"{args.run.name}: reached its target ({target} iterations, {own[-1][1].name if own else 'done'}); "
            "nothing to train", flush=True)
      return 0

  import mjlab
  import mjlab.tasks  # noqa: F401  (populates the registry)
  import tyro
  from mjlab.scripts.train import TrainConfig, launch_training

  from crowdmotion.core.provenance import environment, git_diff
  from crowdmotion.sim.mjlab import wandb_log
  from crowdmotion.sim.mjlab.traversal import register

  graph = str(env["graph"] or default_graph())
  rotations = str(env["rotations"] or default_graph_rotations())
  behavior, humans = env["behavior"], env["humans"]
  policy = "student" if args.distill else "teacher" if args.teacher else "lidar"
  options = dict(perceive_robot=env["perceive_robot"], forward_first=env["forward_first"],
                 teacher_lidar=env["teacher_lidar"], no_avoidance=env["no_avoidance"],
                 penalty_ramp=env["penalty_ramp"], planner_rate=env["planner_rate"],
                 crowd_curriculum=env["crowd_curriculum"], navigation=env["navigation"])
  task = register(graph, behavior, rotations, humans, policy=policy, **options)
  option = next((k for k in STARTS if getattr(args, k)), None)
  resume = None  # the earlier run's checkpoint this lineage starts from: as first recorded
  if earlier is not None:
    resume = Path(record[STARTS[option]]) if record and record.get(STARTS[option]) else newest_checkpoint(earlier)
  if args.finetune:
    teacher_record = json.loads((earlier / "run.json").read_text())
    if teacher_record.get("policy") != "student" or not teacher_record.get("teacher"):
      p.error(f"--finetune needs a --distill run; {earlier.name} is not one")
  if record is None:
    run = RunDir.create(args.name, args.run or args.out, record={
      "kind": "train_traversal", "task": task, "argv": argv, "graph": graph, "rotations": rotations,
      "behavior": behavior, "humans": humans, "policy": policy, **options,
      "resume_from": str(resume) if args.resume else None, "teacher": str(resume) if args.distill else None,
      "finetune_from": str(resume) if args.finetune else None,
      "source_root": str(ROOT), "attempts": [],
    })
    lock = WriterLock(run.path)
    lock.acquire(f"crowd train traversal (job {os.environ.get('SLURM_JOB_ID')})")
    held.append(lock)
  else:
    run = RunDir(args.run, record)
    if not record.get("attempts"):
      record["attempts"] = legacy_attempts(run.path, record)
    for earlier_attempt in record["attempts"]:
      if earlier_attempt.get("outcome") == "running":  # killed without a word (SIGKILL, node failure)
        earlier_attempt["outcome"] = "lost"
    record.pop("error", None)
  attempts = run.record["attempts"]

  try:  # every failure after the run directory exists is recorded in it
    # The distillation runner config is parsed through the PPO task's TrainConfig (tyro
    # parses the declared agent type), then the shared agent options are carried over.
    parse_task = register(graph, behavior, rotations, humans, **options) if args.distill else task
    cfg = tyro.cli(TrainConfig, args=theirs, default=TrainConfig.from_task(parse_task),
                   prog="crowd train traversal --", config=mjlab.TYRO_FLAGS)
    if args.distill:
      from mjlab.tasks.registry import load_rl_cfg

      agent = load_rl_cfg(task)
      for name in ("max_iterations", "seed", "run_name", "logger", "save_interval", "num_steps_per_env"):
        setattr(agent, name, getattr(cfg.agent, name))
      cfg = replace(cfg, agent=agent)
    cfg = replace(cfg, log_root=str(run.path / "logs"))
    if env["navigation"] is not None:
      cfg.env.commands["crowd"].seed = cfg.agent.seed
      cfg.env.commands["twist"].paired_seed = cfg.agent.seed
      nav = cfg.env.commands["twist"].navigation
      expected = nav.gamma_per_s ** (1 / env["planner_rate"])
      if not args.distill and abs(cfg.agent.algorithm.gamma - expected) > 1e-9:
        raise ValueError("set navigation gamma in its JSON config so PPO and potential shaping share the discount")
    # Curricula set in iterations; the env counts steps of the parsed rollout length.
    substeps = round(1.0 / (env["planner_rate"] * cfg.env.decimation * cfg.env.sim.mujoco.timestep)) \
      if env["planner_rate"] else 1  # a planning arm's step is one decision
    ramp = cfg.env.curriculum.get("penalty_ramp")
    if ramp is not None:
      ramp.params["steps"] = env["penalty_ramp"] * cfg.agent.num_steps_per_env * substeps
    difficulty = cfg.env.curriculum.get("crowd_difficulty")
    if difficulty is not None:
      difficulty.params["interval"] = cfg.agent.num_steps_per_env * substeps
    own = own_checkpoints(run.path)
    if own:
      # In place: mjlab loads <log_root>/<experiment>/<load_run>/<load_checkpoint>, the
      # lineage's newest checkpoint, with its optimizer and iteration.
      start_iteration, loaded = own[-1]
      cfg.agent.resume = True
      cfg.agent.load_run = re.escape(loaded.parent.name)
      cfg.agent.load_checkpoint = re.escape(loaded.name)
    else:
      start_iteration, loaded = 0, resume
      if resume is not None:
        # mjlab loads <log_root>/<experiment>/<load_run>/<load_checkpoint>: the run to
        # continue, or (distillation) the teacher, whose PPO actor becomes the teacher model.
        prefix = "teacher" if args.distill else "init" if args.finetune else "resume"
        link = run.path / "logs" / cfg.agent.experiment_name / f"{prefix}_{resume.parent.name}"
        link.parent.mkdir(parents=True, exist_ok=True)
        if args.finetune:
          # Actor from the student, critic from its teacher; `TraversalPpoRunner` loads
          # only these (`init_only`).
          import torch

          if not (link / resume.name).exists():  # a repeated first attempt keeps the one written
            student = torch.load(resume, weights_only=False, map_location="cpu")
            teacher = torch.load(teacher_record["teacher"], weights_only=False, map_location="cpu")
            link.mkdir(exist_ok=True)
            torch.save({"actor_state_dict": student["student_state_dict"],
                        "critic_state_dict": teacher["critic_state_dict"], "init_only": True, "iter": 0,
                        "infos": None}, link / resume.name)
        elif not link.is_symlink():
          link.symlink_to(resume.parent.resolve(), target_is_directory=True)
        cfg.agent.resume = True
        cfg.agent.load_run = re.escape(link.name)
        cfg.agent.load_checkpoint = re.escape(resume.name)
        if args.resume:  # PPO continues at the checkpoint's iteration
          start_iteration = int(checkpoint_iteration(resume) or 0)
    first = not attempts
    if run.record.get("target_iterations") is None:  # a new lineage, or a run recorded before the key
      run.record["target_iterations"] = start_iteration + cfg.agent.max_iterations if first \
        else lineage_target(run.record)
    target = run.record["target_iterations"]
    cfg.agent.max_iterations = target - start_iteration  # RSL-RL repeats label start_iteration
    check = target - first_start(run.record) <= CHECK_ITERATIONS
    block = run.record.get("wandb")
    wandb = wandb_mode(args.wandb, block, check)
    if block is None:
      parent = lineage_id(earlier, json.loads((earlier / "run.json").read_text())) if earlier else None
      block = run.record["wandb"] = new_block(run.path, wandb, parent=parent, legacy=not first)
    block["mode"] = wandb
    if check:
      block["status"] = f"check (at most {CHECK_ITERATIONS} iterations): off W&B"
    attempt = {"n": len(attempts) + 1, "slurm_job": os.environ.get("SLURM_JOB_ID"),
               "slurm_restart": os.environ.get("SLURM_RESTART_COUNT"), "host": platform.node(),
               "start": utc_now(), "end": None, "checkpoint": str(loaded) if loaded else None,
               "start_iteration": start_iteration, "iterations": cfg.agent.max_iterations, "log_dir": None,
               "wandb": wandb, "outcome": "running"}
    if not first:
      attempt["environment"] = environment()
      diff = git_diff()
      before = run.path / "source.diff"
      if diff and (not before.exists() or before.read_text() != diff):
        attempt["source_diff"] = f"source_attempt{attempt['n']}.diff"
        (run.path / attempt["source_diff"]).write_text(diff)
    attempts.append(attempt)
    if first:
      run.record["train_config"] = {"num_envs": cfg.env.scene.num_envs, "max_iterations": cfg.agent.max_iterations,
                                    "seed": cfg.agent.seed, "logger": cfg.agent.logger}
    run.update(status="running")
    session = wandb_log.SESSION = wandb_log.Session(run, attempt, online=wandb == "online",
                                                    start_iteration=start_iteration)
    signal.signal(signal.SIGTERM, wandb_log.on_sigterm)
    print(f"{run.path}: attempt {attempt['n']}, iterations {start_iteration}-{target - 1}"
          + (f" from {loaded}" if loaded else "") + f", W&B {wandb}", flush=True)
  except BaseException as exc:
    run.update(status="failed", error=repr(exc))
    raise

  outcome, error = "failed", None
  try:
    try:
      launch_training(task, cfg)
      outcome = "done"
    finally:
      session.phase = "finishing"  # a further TERM only sets the flag
  except wandb_log.Checkpointed as exc:
    outcome = "checkpointed"
    attempt["checkpoint_saved"] = exc.iteration
  except BaseException as exc:
    outcome, error = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", repr(exc)
    raise
  finally:
    attempt.update(end=utc_now(), outcome=outcome, **({"error": error} if error else {}))
    run.update(status=outcome, **({"error": error} if error else {}))
    finished = session.close(preempting=outcome == "checkpointed", exit_code=0 if outcome == "done" else 1)
    run.save()
    lock.release()
    if not finished:  # W&B's finish hung: leave before the batch shell runs out of time
      print("W&B finish timed out; exiting", flush=True)
      sys.stdout.flush()
      sys.stderr.flush()
      os._exit(EXIT_CHECKPOINTED if outcome == "checkpointed" else 0 if outcome == "done" else 1)
  return EXIT_CHECKPOINTED if outcome == "checkpointed" else 0
