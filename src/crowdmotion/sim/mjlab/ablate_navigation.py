"""Plan/submit matched navigation experiments; array workers dispatch frozen arguments.

  crowd train navigation-ablations --submit
  crowd train navigation-arm <campaign> [--evaluate] --index N

The campaign snapshots package, mjlab and scripts before submission. Jobs keep that
source across requeues; configurations and all projected training/evaluation paths are
recorded before the arrays start. No environment construction happens in the launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from crowdmotion.core.config import from_dict
from crowdmotion.core.paths import ROOT, artifact_path
from crowdmotion.core.rundir import RunDir, utc_stamp
from crowdmotion.sim.mjlab.navigation import NavigationCfg


@dataclass(frozen=True)
class AblationCfg:
  iterations: int = 120
  envs: int = 4096
  seeds: tuple[int, ...] = (42, 43)
  train_parallel: int = 48
  eval_parallel: int = 8
  eval_envs: int = 1024
  eval_seconds: int = 60
  eval_last: int = 3
  # overcap admits GPU types outside the lab quota and up to 48 GPUs per user.
  partition: str = "overcap"
  train_gpu_type: str = "a40"
  eval_gpu_type: str = "a40"

  def __post_init__(self):
    if min(self.iterations, self.envs, self.train_parallel, self.eval_parallel,
           self.eval_envs, self.eval_seconds, self.eval_last) < 1 or not self.seeds:
      raise ValueError("ablation sizes and seeds must be nonempty/positive")
    if self.iterations < 2 or self.iterations % 2:
      raise ValueError("iterations must be even for exposure-matched rollout probes")
    allowed = {"a40", "l40s", "a5000", "rtx_6000", "2080_ti"}
    if self.train_gpu_type not in allowed or self.eval_gpu_type not in allowed:
      raise ValueError(f"GPU types must be one of {sorted(allowed)}; verify workload memory before switching")


def arms(cfg: AblationCfg) -> list[dict]:
  out = []

  def add(group, crowd, seed, nav, humans=8, flow=0.):
    tag = f"{group}_{crowd}_s{seed}"
    out.append(dict(name=tag, group=group, crowd=crowd, seed=seed, humans=humans, flow_fraction=flow,
                    navigation=asdict(nav), iterations=cfg.iterations * 16 // nav.rollout_s))

  for seed, gamma, reward, crowd in itertools.product(
      cfg.seeds, (.99, .75), ("sparse", "potential", "all"), ("reactive", "independent")):
    add(f"{reward}_g{int(gamma*100)}", crowd, seed, NavigationCfg(reward_mode=reward, gamma_per_s=gamma))
  first = cfg.seeds[0]
  for reward, crowd in itertools.product(("sparse",), ("reactive", "independent")):
    add(f"{reward}_g95", crowd, first, NavigationCfg(reward_mode=reward, gamma_per_s=.95))
  for reward, crowd in itertools.product(("tracking", "proximity", "regularized", "centered"),
                                        ("reactive", "independent")):
    add(f"{reward}_g99", crowd, first, NavigationCfg(reward_mode=reward))
  for probe, fields in (("lambda90", dict(gae_lambda_per_s=.9)), ("rollout32", dict(rollout_s=32)),
                        ("no_time_bonus", dict(time_bonus=0.)), ("failure10", dict(failure_reward=-10.))):
    for crowd in ("reactive", "independent"):
      add(probe, crowd, first, NavigationCfg(**fields))
  for seed in cfg.seeds:
    for humans in (4, 16):
      add(f"density{humans}", "independent", seed, NavigationCfg(), humans=humans)
    add("coherent90", "independent", seed, NavigationCfg(), flow=.9)
  assert len({x["name"] for x in out}) == len(out)
  return out


def snapshot(destination: Path):
  files = {}
  for folder in ("src", "mjlab/src", "scripts"):
    shutil.copytree(ROOT / folder, destination / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
  for path in sorted(destination.rglob("*")):
    if path.is_file():
      files[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
  return files


BASE_BEHAVIOR = ROOT / "configs/behavior/encounter_decisive_robot.json"


def navigation_conditions() -> list[dict]:
  """The fixed evaluation crowds: independent (non-reactive) walkers at 4, 8 and 16
  humans, 90% coherent-flow independent traffic at 8 and 16, and the reactive crowd at 8
  aware of the robot (the base behavior's robot traits) or blind to it. `crowd` is the
  evaluator's robot perception (`aware` or `blind`); `behavior` is added by `plan`."""
  conditions = []
  for humans in (4, 8, 16):
    conditions.append(dict(name=f"independent_n{humans}", reactive=False, humans=humans, flow=0., crowd="blind"))
  for humans in (8, 16):
    conditions.append(dict(name=f"coherent90_n{humans}", reactive=False, humans=humans, flow=.9, crowd="blind"))
  for crowd in ("aware", "blind"):
    conditions.append(dict(name=f"reactive_{crowd}_n8", reactive=True, humans=8, flow=0., crowd=crowd))
  return conditions


def condition_behavior(base: dict, condition: dict) -> dict:
  """The behavior config of `condition`: a copy of `base` (an encounter-design config)
  with its reactivity and flow. A fixed world heading samples goal-relative flow angles
  through paired random goals; the evaluator also reports outcomes by the initial
  goal/flow relative-angle bin."""
  behavior = json.loads(json.dumps(base))
  behavior["encounter"].update(reactive=condition["reactive"], flow_fraction=condition["flow"], flow_heading_deg=90.)
  return behavior


def plan(cfg: AblationCfg, out: Path | None = None) -> RunDir:
  run = RunDir.create("navigation_ablations", out=out, record={"kind": "navigation_ablations", "config": asdict(cfg)})
  rows = arms(cfg)
  stamp = utc_stamp()
  base = json.loads(BASE_BEHAVIOR.read_text())
  conditions = navigation_conditions()
  for c in conditions:
    c["behavior"] = str(run.write_json(f"conditions/{c['name']}.json", condition_behavior(base, c)).resolve())
  for row in rows:
    row["run"] = str(ROOT / "runs" / f"{stamp}_train_traversal_nav_{row['name']}")
    row["eval_run"] = str(ROOT / "runs" / f"{stamp}_eval_navigation_nav_{row['name']}")
    behavior = json.loads(json.dumps(base))
    behavior["encounter"].update(reactive=row["crowd"] == "reactive", flow_fraction=row["flow_fraction"])
    row["behavior"] = str(run.write_json(f"configs/{row['name']}_crowd.json", behavior).resolve())
    row["navigation_path"] = str(run.write_json(f"configs/{row['name']}.json", row["navigation"]).resolve())
    row["argv"] = [f"train_traversal_nav_{row['name']}", "--run", row["run"], "--graph",
                   str(artifact_path("graph-turns")), "--rotations", str(artifact_path("graph-turns-rotations")),
                   "--behavior", row["behavior"], "--humans", str(row["humans"]), "--robot-blind",
                   "--planner-rate", "1", "--navigation", row["navigation_path"], "--", "--env.scene.num-envs",
                   str(cfg.envs), "--agent.max-iterations", str(row["iterations"]), "--agent.seed", str(row["seed"]),
                   "--agent.run-name", row["name"]]
  files = snapshot(run.path / "source")
  run.write_json("source_sha256.json", files)
  run.update(status="planned", arms=rows, conditions=conditions,
             estimated_train_a40_hours=len(rows)*7.2, source_snapshot=str((run.path / "source").resolve()))
  return run


def submit(run: RunDir):
  cfg = AblationCfg(**run.record["config"])
  count = len(run.record["arms"])
  common = ["sbatch", "--parsable", f"--partition={cfg.partition}", "--qos=short", "--account=rl2-lab",
            "--cpus-per-task=8", "--exclude=bishop", "--requeue", "--open-mode=append",
            "--export=ALL", "--output=logs/slurm/%x-%A_%a.out"]
  script = str(run.path / "source/scripts/slurm/navigation_ablations.sbatch")
  train = subprocess.check_output([*common, f"--gres=gpu:{cfg.train_gpu_type}:1",
                                   f"--array=0-{count-1}%{cfg.train_parallel}", "--time=04:00:00",
                                   "--signal=B:TERM@1000", "--job-name=nav-train", script, str(run.path.resolve())],
                                  text=True).strip().split(";")[0]
  run.update(status="submitted", training_array=train)
  evaluation = subprocess.check_output([*common, f"--gres=gpu:{cfg.eval_gpu_type}:1",
                                        f"--array=0-{count-1}%{cfg.eval_parallel}", "--time=04:00:00",
                                        f"--dependency=aftercorr:{train}", "--kill-on-invalid-dep=yes",
                                        "--job-name=nav-eval", script, str(run.path.resolve()), "--evaluate"],
                                       text=True).strip().split(";")[0]
  run.update(evaluation_array=evaluation)


def main(argv=None):
  if argv and argv[0] == "dispatch":
    return dispatch(argv[1:])
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--config", type=Path)
  p.add_argument("--out", type=Path)
  p.add_argument("--submit", action="store_true")
  args = p.parse_args(argv)
  cfg = from_dict(AblationCfg, json.loads(args.config.read_text()) if args.config else {}, what="navigation ablations")
  run = plan(cfg, args.out)
  if args.submit:
    submit(run)
  print(json.dumps({"out": str(run.path), "arms": len(run.record["arms"]),
                    "training_array": run.record.get("training_array"), "evaluation_array": run.record.get("evaluation_array")}))
  return 0


def dispatch(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument("campaign", type=Path)
  p.add_argument("--index", type=int, required=True)
  p.add_argument("--evaluate", action="store_true")
  args = p.parse_args(argv)
  run = RunDir.open(args.campaign)
  row = run.record["arms"][args.index]
  source = Path(run.record["source_snapshot"])
  env = dict(os.environ, CROWD_ROOT=str(ROOT), CROWD_SIGNAL_LEAD_S="1000",
             CROWD_WANDB_TAGS="navigation-ablation,"+row["group"]+","+row["crowd"],
             PYTHONPATH=f"{source / 'src'}:{source / 'mjlab/src'}:"+os.environ.get("PYTHONPATH", ""))
  if args.evaluate:
    from crowdmotion.sim.mjlab.evaluate_navigation import evaluate
    return evaluate(run, row)
  script = source / "scripts/slurm/train_traversal.sbatch"
  os.execve("/bin/bash", ["bash", str(script), *row["argv"]], env)
