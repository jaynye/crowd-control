"""Resumable fixed-condition navigation evaluation for an ablation campaign."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from crowdmotion.core.rundir import RunDir
from crowdmotion.metrics.traversal import EPISODE_ENDS, TraversalEvalCfg
from crowdmotion.sim.mjlab.evaluate import _clean, iteration_of, make_env, rollout, run_checkpoints, write_tables
from crowdmotion.sim.mjlab.play import run_policy
from crowdmotion.sim.mjlab.train import actor_checkpoint
from crowdmotion.tracking import flush_outbox, queue_eval


def flow_summary(tables):
  """Outcomes by initial route vs +x flow; reflected traffic is still exogenous.

  World/library headings are converted by the normal crowd adapter. Library 90° is
  MuJoCo +x. These bins classify initial geometry, not a claim of measured coherence.
  """
  legs, ep = tables["legs"], tables["episodes"]
  lookup = {}
  for i in range(len(legs["env"])):
    key = (int(legs["env"][i]), int(legs["episode"][i]))
    if key not in lookup:
      dx, dy = legs["goal_x"][i]-legs["x0"][i], legs["goal_y"][i]-legs["y0"][i]
      angle = abs(np.degrees(np.arctan2(dy, dx)))
      lookup[key] = "coflow" if angle < 45 else "opposing" if angle > 135 else "crossing"
  labels = np.array([lookup.get((int(e), int(n)), "unknown") for e, n in zip(ep["env"], ep["episode"])])
  out = {}
  for label in ("coflow", "crossing", "opposing"):
    mask = (labels == label) & (ep["end"] != EPISODE_ENDS.index("eval_end"))
    n = int(mask.sum())
    if n:
      out[f"flow/{label}/episodes"] = n
      out[f"flow/{label}/success_fraction"] = float((ep["end"][mask] == EPISODE_ENDS.index("success")).mean())
      minutes = ep["duration"][mask].sum() / 60
      out[f"flow/{label}/collisions_per_robot_min"] = float(ep["collisions"][mask].sum() / minutes)
  return out


def evaluate(campaign: RunDir, row: dict):
  import mjlab.tasks  # noqa: F401
  from mjlab.rl import MjlabOnPolicyRunner

  from crowdmotion.sim.mjlab.traversal import traversal_rl_cfg

  source = Path(row["run"])
  _, plays_as, options = run_policy(source)
  cfg = campaign.record["config"]
  checkpoints = run_checkpoints(source)[-cfg["eval_last"]:]
  destination = Path(row["eval_run"])
  run = RunDir.open(destination) if (destination / "run.json").exists() else RunDir.create(
    "eval_navigation", out=destination, record={"kind": "eval_navigation", "training_run": str(source),
      "campaign": str(campaign.path), "conditions": campaign.record["conditions"], "config": cfg, "results": []})
  results = run.record["results"]
  if run.record["status"] == "completed":
    return 0
  run.update(status="running")
  done = {(r["condition"], r["iteration"]) for r in results}
  agent = traversal_rl_cfg(teacher=plays_as == "teacher", planner_rate=options["planner_rate"],
                           navigation=options["navigation"])
  for condition in campaign.record["conditions"]:
    pending = [p for p in checkpoints if (condition["name"], iteration_of(p)) not in done]
    if not pending:
      continue
    args = Namespace(behavior=condition["behavior"], humans=condition["humans"], seed=101,
                     envs=cfg["eval_envs"], seconds=cfg["eval_seconds"], episode_s=None,
                     paired=True, cfg=TraversalEvalCfg())
    raw, wrapped = make_env(options, condition["crowd"], "end", args, agent.clip_actions, "cuda:0")
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
    try:
      for checkpoint in pending:
        runner.load(str(actor_checkpoint(checkpoint)), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        metrics, tables = rollout(raw, wrapped, policy, round(args.seconds/raw.step_dt), args.cfg, args.seed)
        if condition["flow"]:
          metrics.update(flow_summary(tables))
        metrics = _clean(metrics)
        iteration = iteration_of(checkpoint)
        write_tables(run.path / "tables", f"{iteration}_{condition['name']}", tables)
        scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        attachment = queue_eval(source, run.path, iteration=iteration,
                                kind=f"navigation/{condition['name']}", scalars=scalars)
        results.append(dict(condition=condition["name"], iteration=iteration, checkpoint=str(checkpoint),
                            metrics=metrics, wandb=attachment))
        run.write_json("metrics.json", {"conditions": campaign.record["conditions"], "results": results})
        run.update(results=results)
        print(f"{source.name}: {condition['name']} model_{iteration}: "
              f"collisions/min={metrics['collision/per_robot_min']:.3f}, "
              f"goal success={metrics['goal/success_fraction']}, "
              f"goals/min={metrics['goal/success_per_robot_min']}", flush=True)
    finally:
      raw.close()
    del raw, wrapped, runner, policy
    torch.cuda.empty_cache()
  flush_outbox(source)
  run.update(status="completed", results=results)
  return 0


def main(argv=None):
  import argparse
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("campaign", type=Path)
  p.add_argument("--index", type=int, required=True)
  args = p.parse_args(argv)
  campaign = RunDir.open(args.campaign)
  return evaluate(campaign, campaign.record["arms"][args.index])
