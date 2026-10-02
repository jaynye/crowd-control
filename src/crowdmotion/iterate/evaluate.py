"""Evaluate plaza behavior on a saved motion graph: traces and body-relative metrics.

  crowd behavior evaluate --config configs/behavior/baseline.json \
      [--graph <graph.pt>] [--out runs/<UTC>_behavior_eval]

Each seed of the config's seed bank runs one plaza of `humans` humans for
`seconds` at the motion frame rate, through the same `CrowdController` (plaza
intent and steering, path planner, graph motion source and velocity update) as
the mjlab `GraphCrowd` term. It excludes mjlab, interpolation, sensors and the
reset lifecycle, so it is not a bit-exact replay of a simulator run.

Writes `run.json`, `skeleton.json` (for replay), and per seed `seed_<n>.npz`
(a `Trace`) and `seed_<n>.json` (metrics), plus `summary.json`. No pass
criterion is applied; compare paired seeds with `crowd behavior compare`.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from crowdmotion.behavior.controller import CrowdController
from crowdmotion.behavior.kinematics import BufferVelocity
from crowdmotion.behavior.plaza import PlazaBehavior, PlazaSteeringCfg
from crowdmotion.behavior.sources import GraphMotionSource
from crowdmotion.core.config import check_fields, load_json
from crowdmotion.core.paths import default_graph
from crowdmotion.core.provenance import file_sha256
from crowdmotion.core.rng import human_seeds
from crowdmotion.core.rundir import RunDir
from crowdmotion.core.timing import synchronize
from crowdmotion.core.trace import Trace
from crowdmotion.graph import GraphConfig, GraphRuntime, MotionGraph
from crowdmotion.metrics.behavior import summarize_trace

EVAL_FIELDS = {"description", "humans", "seconds", "seeds", "graph_overrides", "steering"}
DIAGNOSTICS = ("traj_err_m", "desc_dist_m", "vel_jump_mps", "natural")
SCOPE = "Isolated graph/steering behavior at motion FPS; no mjlab, sensors, renderer or prior."


def resolved_config(raw: dict) -> dict:
  """Defaults filled in and every field checked: a misspelled key is an error."""
  if extra := set(raw) - EVAL_FIELDS:
    raise ValueError(f"Unknown evaluation fields: {extra}")
  steering = raw.get("steering", {})
  check_fields(PlazaSteeringCfg, steering, "steering")
  overrides = raw.get("graph_overrides", {})
  check_fields(GraphConfig, overrides, "graph")
  cfg = {
    "humans": 16,
    "seconds": 20.0,
    "seeds": [0, 1, 2],
    **raw,
    "steering": PlazaSteeringCfg(**steering).__dict__,
    "graph_overrides": overrides,
  }
  if cfg["humans"] < 1 or not math.isfinite(cfg["seconds"]) or cfg["seconds"] <= 0:
    raise ValueError("Positive humans and seconds are required")
  if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
    raise ValueError("Supply a nonempty bank of distinct seeds")
  if any(not isinstance(s, int) or s < 0 or s >= 2**40 for s in cfg["seeds"]):
    raise ValueError("Seeds must be integers in [0, 2**40)")
  return cfg


@torch.no_grad()
def rollout(graph: MotionGraph, cfg: dict, seed: int) -> tuple[Trace, dict]:
  """One plaza of `cfg["humans"]` humans for `cfg["seconds"]`; returns its trace and metrics.

  The trace starts with each human's start pose (frame 0) and includes a final
  partial edge trimmed to the requested duration.
  """
  M, dev, fps, L = cfg["humans"], graph.device, graph.fps, graph.L
  behavior = PlazaBehavior(1, M, PlazaSteeringCfg(**cfg["steering"]), seed, dev)
  source = GraphMotionSource(GraphRuntime(graph))
  ctrl = CrowdController(behavior, source, BufferVelocity(fps), group=M)
  env_ids = torch.tensor([0], device=dev)
  xz, heading = behavior.reset(env_ids)
  state = ctrl.reset(human_seeds(seed, torch.arange(M, device=dev), "evaluator"), xz[0], heading[0])
  joints, roots, heads = [source.current_joints(state)[:, None]], [xz[0, :, None]], [heading[0, :, None]]
  contacts = [torch.zeros(M, 1, graph.edge_contacts.shape[-1], device=dev, dtype=torch.bool)]
  edges, commands, goals, waiting, diagnostics = [], [], [], [], []
  frames = round(cfg["seconds"] * fps)
  if frames < 2:
    raise ValueError("Evaluation requires at least two motion frames")
  start = time.perf_counter()
  for frame in range(0, frames, L):
    t = torch.tensor([frame / fps], device=dev)
    step = ctrl.advance(t, groups=env_ids, limit=frames - frame)
    out, take = step.motion, step.take
    goals.append(behavior.goal[0].clone())
    waiting.append((behavior.wait_until[0] >= t[0]).clone())
    commands.append(torch.stack((step.speed, step.heading), -1))
    joints.append(out.joints[:, :take])
    roots.append(out.root[:, :take, [0, 2]])
    heads.append(out.heading[:, :take])
    contacts.append(out.contacts[:, :take])
    edges.append(out.extras["edge"])
    diagnostics.append(torch.stack([out.extras[k] for k in DIAGNOSTICS], -1))
  synchronize(dev)
  steady_s = time.perf_counter() - start
  trace = Trace(
    joints=torch.cat(joints, 1).cpu().numpy(),
    root=torch.cat(roots, 1).cpu().numpy(),
    heading=torch.cat(heads, 1).cpu().numpy(),
    fps=fps,
    contacts=torch.cat(contacts, 1).cpu().numpy(),
    foot_idx=np.array(graph.foot_idx),
    extras={
      "edge": torch.stack(edges, 1).cpu().numpy(),
      "command": torch.stack(commands, 1).cpu().numpy(),
      "goal": torch.stack(goals, 1).cpu().numpy(),
      "waiting": torch.stack(waiting, 1).cpu().numpy(),
      "diagnostics": torch.stack(diagnostics, 1).cpu().numpy(),
    },
  )
  stats = summarize_trace(trace, fps)
  diag = trace.extras["diagnostics"]
  stats.update(
    seed=seed,
    seconds=frames / fps,
    rollout_wall_s=steady_s,
    isolated_motion_seconds_per_wall_s=M * frames / fps / steady_s,
    natural_successor_percent=100 * float(diag[..., 3].mean()),
    graph_path_error_m_mean=float(diag[..., 0].mean()),
  )
  return trace, stats


def evaluate(args) -> Path:
  torch.set_num_threads(args.threads)
  cfg = resolved_config(load_json(args.config))
  graph_path = Path(args.graph or default_graph())
  record = {
    "config": cfg,
    "graph_path": str(graph_path.resolve()),
    "graph_sha256": file_sha256(graph_path),
    "device": args.device,
    "threads": args.threads,
    "scope": SCOPE,
  }
  with RunDir.create("behavior_eval", out=args.out, record=record) as run:
    t0 = time.perf_counter()
    graph = MotionGraph.load(graph_path, device=args.device, cfg_overrides=cfg["graph_overrides"])
    synchronize(graph.device)
    run.update(graph_load_s=time.perf_counter() - t0, graph_config=graph.cfg.__dict__, fps=graph.fps)
    # Replay skeleton metadata; rendering does not need to reload the graph.
    run.write_json("skeleton.json", {"meta": graph.meta, "rest_joints": graph.rest_joints.tolist()})
    summaries = []
    for seed in cfg["seeds"]:
      trace, summary = rollout(graph, cfg, seed)
      trace.save(run.path / f"seed_{seed}.npz")
      run.write_json(f"seed_{seed}.json", summary)
      summaries.append(summary)
      print(f"seed {seed}: forward {summary['moving_forward_percent']:.1f}% of moving frames", flush=True)
    run.write_json("summary.json", {"seeds": summaries, "aggregation": "Compare paired seeds; no invented pass criterion."})
  print(run.path, flush=True)
  return run.path


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--graph", type=Path, help="saved graph (default: $CROWD_GRAPH or artifact graph-default)")
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_behavior_eval)")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--threads", type=int, default=1)
  evaluate(parser.parse_args(argv))


if __name__ == "__main__":
  main()
