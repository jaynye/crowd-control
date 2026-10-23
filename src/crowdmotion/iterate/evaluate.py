"""Evaluate plaza behavior on a saved motion graph: traces and body-relative metrics.

  crowd behavior evaluate --config configs/behavior/baseline.json \
      [--graph <graph.pt>] [--out runs/<UTC>_behavior_eval]

Each seed of the config's seed bank runs one plaza of `humans` humans for
`seconds` at the motion frame rate, through the same `CrowdController` (behavior
design, path planner, graph motion source and velocity update) as the mjlab
`GraphCrowd` term. It excludes mjlab, interpolation, sensors and the reset
lifecycle, so it is not a bit-exact replay of a simulator run.

`design` selects the behavior (`behavior.designs`): `plaza` (default; section
`steering`) or `encounter` (section `encounter`).

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
from crowdmotion.behavior.designs import design_of, make_behavior, section_cfg
from crowdmotion.behavior.encounter import EncounterCfg
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
from crowdmotion.metrics.encounters import encounter_metrics

EVAL_FIELDS = {"description", "humans", "seconds", "seeds", "graph_overrides", "steering", "design", "encounter"}
DIAGNOSTICS = ("traj_err_m", "desc_dist_m", "vel_jump_mps", "natural")
SCOPE = "Isolated graph/steering behavior at motion FPS; no mjlab, sensors, renderer or prior."


def _checked_run_fields(cfg: dict) -> dict:
  if cfg["humans"] < 1 or not math.isfinite(cfg["seconds"]) or cfg["seconds"] <= 0:
    raise ValueError("Positive humans and seconds are required")
  if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
    raise ValueError("Supply a nonempty bank of distinct seeds")
  if any(not isinstance(s, int) or s < 0 or s >= 2**40 for s in cfg["seeds"]):
    raise ValueError("Seeds must be integers in [0, 2**40)")
  return cfg


def resolved_config(raw: dict) -> dict:
  """Defaults filled in and every field checked: a misspelled key is an error."""
  if extra := set(raw) - EVAL_FIELDS:
    raise ValueError(f"Unknown evaluation fields: {extra}")
  if design_of(raw) == "encounter":
    if "steering" in raw:
      raise ValueError("The encounter design takes an `encounter` section, not `steering`")
    overrides = raw.get("graph_overrides", {})
    check_fields(GraphConfig, overrides, "graph")
    section = EncounterCfg.from_dict(raw.get("encounter", {})).as_json()
    cfg = _checked_run_fields({"humans": 16, "seconds": 20.0, "seeds": [0, 1, 2], **raw,
                               "encounter": section, "graph_overrides": overrides})
    for name in ("waypoints", "velocities", "starts"):
      if section[name] and len(section[name]) != cfg["humans"]:
        raise ValueError(f"encounter {name} must list one entry per human ({cfg['humans']})")
    return cfg
  if "encounter" in raw:
    raise ValueError("An `encounter` section needs \"design\": \"encounter\"")
  steering = raw.get("steering", {})
  check_fields(PlazaSteeringCfg, steering, "steering")
  overrides = raw.get("graph_overrides", {})
  check_fields(GraphConfig, overrides, "graph")
  resolved_steering = PlazaSteeringCfg(**steering).__dict__
  if not resolved_steering["committed_turns"]:
    # The opt-in intent must preserve the baseline's settings serialization as
    # well as its random draws and motion (comparison pages are also golden).
    resolved_steering = {k: v for k, v in resolved_steering.items()
                         if k != "committed_turns" and (not k.startswith("turn_") or k == "turn_before_walk")}
  if not resolved_steering["forward_steering"]:
    resolved_steering = {k: v for k, v in resolved_steering.items() if not k.startswith("forward_")}
  if not resolved_steering["relax_orientation_for_avoidance"]:
    resolved_steering.pop("relax_orientation_for_avoidance")
  if not resolved_steering["turn_before_walk"]:
    resolved_steering = {k: v for k, v in resolved_steering.items()
                         if k != "turn_before_walk" and not k.startswith("redirect_")}
  cfg = {
    "humans": 16,
    "seconds": 20.0,
    "seeds": [0, 1, 2],
    **raw,
    "steering": resolved_steering,
    "graph_overrides": overrides,
  }
  return _checked_run_fields(cfg)


@torch.no_grad()
def rollout(graph: MotionGraph, cfg: dict, seed: int) -> tuple[Trace, dict]:
  """One plaza of `cfg["humans"]` humans for `cfg["seconds"]`; returns its trace and metrics.

  The trace starts with each human's start pose (frame 0) and includes a final
  partial edge trimmed to the requested duration.
  """
  M, dev, fps, L = cfg["humans"], graph.device, graph.fps, graph.L
  design = design_of(cfg)
  if design == "plaza":
    behavior = PlazaBehavior(1, M, PlazaSteeringCfg(**cfg["steering"]), seed, dev)
    planner = None
  else:
    behavior, planner = make_behavior(design, section_cfg(cfg), 1, M, seed, dev)
  source = GraphMotionSource(GraphRuntime(graph))
  if planner is None:
    ctrl = CrowdController(behavior, source, BufferVelocity(fps), group=M)
  else:
    ctrl = CrowdController(behavior, source, BufferVelocity(fps), planner=planner, group=M)
  env_ids = torch.tensor([0], device=dev)
  xz, heading = behavior.reset(env_ids)
  state = ctrl.reset(human_seeds(seed, torch.arange(M, device=dev), "evaluator"), xz[0], heading[0])
  joints, roots, heads = [source.current_joints(state)[:, None]], [xz[0, :, None]], [heading[0, :, None]]
  contacts = [torch.zeros(M, 1, graph.edge_contacts.shape[-1], device=dev, dtype=torch.bool)]
  edges, commands, goals, waiting, diagnostics = [], [], [], [], []
  steering = cfg.get("steering", {})
  turn_records = {k: [] for k in ("mode", "target", "turn_count", "sampled_angle")} if steering.get("committed_turns") else {}
  evasive = []
  optional_records = ({k: [] for k in ("vertical_fallback", "boundary")}
                      if graph.cfg.vertical_mask or graph.cfg.twist_mask else {})
  if graph.cfg.posture_enabled:
    optional_records.update(posture_hunched=[], boundary=[])
  redirect_records = {k: [] for k in ("turn_active", "turn_target")} if steering.get("turn_before_walk") else {}
  encounter_records = {}
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
    for key in turn_records:
      turn_records[key].append(getattr(behavior.intent, key)[0].clone())
    if any(steering.get(k) for k in ("forward_steering", "relax_orientation_for_avoidance", "turn_before_walk")):
      evasive.append(behavior.steering.evasive[0].clone())
    for key in redirect_records:
      redirect_records[key].append(getattr(behavior.steering, key)[0].clone())
    if design == "encounter":
      for key, value in behavior.trace_records(0).items():
        encounter_records.setdefault(key, []).append(value)
    joints.append(out.joints[:, :take])
    roots.append(out.root[:, :take, [0, 2]])
    heads.append(out.heading[:, :take])
    contacts.append(out.contacts[:, :take])
    edges.append(out.extras["edge"])
    diagnostics.append(torch.stack([out.extras[k] for k in DIAGNOSTICS], -1))
    for key in optional_records:
      optional_records[key].append(out.extras[key].clone())
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
      **{f"intent_{k}": torch.stack(v, 1).cpu().numpy() for k, v in turn_records.items()},
      **({"steering_evasive": torch.stack(evasive, 1).cpu().numpy()} if evasive else {}),
      **{k: torch.stack(v, 1).cpu().numpy() for k, v in optional_records.items()},
      **{k: torch.stack(v, 1).cpu().numpy() for k, v in redirect_records.items()},
      **{k: torch.stack(v, 1).cpu().numpy() for k, v in encounter_records.items()},
    },
  )
  stats = summarize_trace(trace, fps)
  if design == "encounter":
    stats["encounter"] = encounter_metrics(trace.arrays(), fps, L, cfg["encounter"])
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
    run.update(graph_load_s=time.perf_counter() - t0, graph_config=graph.cfg.__dict__, fps=graph.fps, edge_frames=graph.L)
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
