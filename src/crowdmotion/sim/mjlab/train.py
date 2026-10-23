"""`crowd train traversal`: train the crowd traversal policy with mjlab's PPO runner.

Each invocation writes a fresh `runs/<UTC>_<name>/` holding `run.json` and mjlab's log
directory `logs/g1_crowd_traversal/<time>/` (checkpoints `model_<iteration>.pt`,
`params/`, TensorBoard events). Arguments after `--` go to mjlab's `TrainConfig`
(tyro), for example `-- --env.scene.num-envs 4096 --agent.max-iterations 3000`.

`--teacher` trains the privileged teacher (task `Crowd-Traversal-Teacher-Flat-Unitree-G1`,
experiment `g1_crowd_traversal_teacher`) instead of the LiDAR policy. `--distill <teacher
run>` distills that run's newest teacher checkpoint into the LiDAR policy (experiment
`g1_crowd_traversal_student`); mjlab's agent options after `--` that both runners share
(iterations, seed, run name, logger, save interval) carry over.

`--resume <run>` continues from the newest checkpoint of an earlier traversal run in a
new run directory, so a chain of time-limited Slurm jobs never overwrites outputs.
The env is `crowdmotion.sim.mjlab.traversal`.
"""

from __future__ import annotations

import argparse
import re
import signal
from dataclasses import replace
from pathlib import Path

from crowdmotion.core.paths import ROOT, default_graph, default_graph_rotations
from crowdmotion.core.rundir import RunDir


def newest_checkpoint(run: Path) -> Path:
  """The highest-iteration `model_<i>.pt` under an earlier run's mjlab log directories."""
  found = [(int(m.group(1)), p) for p in Path(run).glob("logs/*/*/model_*.pt")
           if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))]
  if not found:
    raise FileNotFoundError(f"no logs/<experiment>/*/model_<i>.pt under {run}")
  return max(found)[1]


def main(argv: list[str] | None = None) -> int:
  argv = list(argv or [])
  ours, theirs = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
  p = argparse.ArgumentParser(prog="crowd train traversal", description=__doc__.split("\n\n")[0])
  p.add_argument("name", nargs="?", default="train_traversal")
  p.add_argument("--graph", help="saved motion graph (default: $CROWD_GRAPH or artifact graph-default)")
  p.add_argument("--rotations", help="the graph's joint rotations (default: $CROWD_GRAPH_ROTATIONS)")
  p.add_argument("--behavior", default=str(ROOT / "configs/behavior/encounter_turns.json"),
                 help="encounter-design behavior config for the humans")
  p.add_argument("--humans", type=int, default=8)
  mode = p.add_mutually_exclusive_group()
  mode.add_argument("--teacher", action="store_true", help="train the privileged teacher instead of the LiDAR policy")
  mode.add_argument("--distill", help="an earlier --teacher run whose newest checkpoint the LiDAR student imitates")
  p.add_argument("--resume", help="an earlier `crowd train traversal` run to continue from")
  p.add_argument("--out", type=Path)
  args = p.parse_args(ours)
  if args.distill and args.resume:
    p.error("--distill starts from a teacher; continuing a distillation run is not supported yet")

  import mjlab
  import mjlab.tasks  # noqa: F401  (populates the registry)
  import tyro
  from mjlab.scripts.train import TrainConfig, launch_training

  from crowdmotion.sim.mjlab.traversal import register

  graph = str(args.graph or default_graph())
  rotations = str(args.rotations or default_graph_rotations())
  policy = "student" if args.distill else "teacher" if args.teacher else "lidar"
  task = register(graph, args.behavior, rotations, args.humans, policy=policy)
  resume = newest_checkpoint(Path(args.resume or args.distill)) if (args.resume or args.distill) else None
  run = RunDir.create(args.name, args.out, record={
    "kind": "train_traversal", "task": task, "argv": argv, "graph": graph, "rotations": rotations,
    "behavior": args.behavior, "humans": args.humans, "policy": policy,
    "resume_from": str(resume) if args.resume else None, "teacher": str(resume) if args.distill else None,
    "source_root": str(ROOT),
  })
  # The distillation runner config is parsed through the PPO task's TrainConfig (tyro
  # parses the declared agent type), then the shared agent options are carried over.
  parse_task = register(graph, args.behavior, rotations, args.humans) if args.distill else task
  cfg = tyro.cli(TrainConfig, args=theirs, default=TrainConfig.from_task(parse_task),
                 prog="crowd train traversal --", config=mjlab.TYRO_FLAGS)
  if args.distill:
    from mjlab.tasks.registry import load_rl_cfg

    agent = load_rl_cfg(task)
    for name in ("max_iterations", "seed", "run_name", "logger", "save_interval", "num_steps_per_env"):
      setattr(agent, name, getattr(cfg.agent, name))
    cfg = replace(cfg, agent=agent)
  cfg = replace(cfg, log_root=str(run.path / "logs"))
  if resume is not None:
    # mjlab loads <log_root>/<experiment>/<load_run>/<load_checkpoint>: the run to
    # continue, or (distillation) the teacher, whose PPO actor becomes the teacher model.
    prefix = "teacher" if args.distill else "resume"
    link = run.path / "logs" / cfg.agent.experiment_name / f"{prefix}_{resume.parent.name}"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(resume.parent.resolve(), target_is_directory=True)
    cfg.agent.resume = True
    cfg.agent.load_run = re.escape(link.name)
    cfg.agent.load_checkpoint = re.escape(resume.name)
  run.update(train_config={"num_envs": cfg.env.scene.num_envs, "max_iterations": cfg.agent.max_iterations,
                           "seed": cfg.agent.seed, "logger": cfg.agent.logger})

  def interrupt(signum, frame):  # Slurm's SIGTERM before the time limit: record it
    raise KeyboardInterrupt(f"signal {signum}")

  signal.signal(signal.SIGTERM, interrupt)
  try:
    launch_training(task, cfg)
  except BaseException as exc:
    run.update(status="failed" if not isinstance(exc, KeyboardInterrupt) else "interrupted", error=repr(exc))
    raise
  run.update(status="done")
  return 0
