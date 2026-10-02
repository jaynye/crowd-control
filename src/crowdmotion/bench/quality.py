"""Closed-loop continuation quality for online generators (ARDY, Kimodo) and motion graphs.

  source scripts/env.sh
  crowd bench quality --config configs/bench/quality_a40.json [--out runs/<UTC>_quality]

Each scenario gives every human a command schedule (speed, heading) in simulated
time. At every replan, a path is planned from the human's **executed** root state
toward the current command, the motion source steps, and the first `P` seconds
(or the whole chunk, for buffer refill; one edge for a graph) are executed.
Metrics are computed on the executed motion only. See docs/usage/benchmarks.md
for definitions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from crowdmotion.backends.base import MotionRequest
from crowdmotion.backends.registry import config_from_case, make_backend
from crowdmotion.behavior.commands import CHANGE_T, RESTART_T, ScenarioCommands
from crowdmotion.behavior.controller import CrowdController
from crowdmotion.behavior.kinematics import WindowVelocity
from crowdmotion.behavior.planner import plan_path
from crowdmotion.behavior.sources import GraphMotionSource, OnlineMotionSource
from crowdmotion.core.config import check_keys
from crowdmotion.core.rng import human_seeds
from crowdmotion.core.rundir import RunDir
from crowdmotion.core.timing import synchronize
from crowdmotion.metrics.feet import contact_skate, label_free_skate
from crowdmotion.metrics.graph_diag import DIAG_KEYS, graph_metrics, graph_window_split
from crowdmotion.metrics.posture import ground_stats, hunch_stats
from crowdmotion.metrics.response import replan_wait, response_time, smoothed_velocity
from crowdmotion.metrics.seams import seam_stats

RESPONSE_KEY = {
  "stop_start": "stop_response_s", "turn90": "turn_response_s",
  "turn180": "turn_response_s", "speed_step": "speed_response_s",
}

# Backends ---------------------------------------------------------------------
_LIBRARIES: dict = {}
# Defaults of quality cases, applied below the case's own keys, and the case keys
# quality configs set for each backend (others, e.g. to_soma77, are not quality settings).
ONLINE_DEFAULTS = {
  "ardy": {"precision": "tf32", "cfg_type": "separated"},
  "kimodo": {"precision": "tf32", "steps": 100},
}
# Every key a quality case may hold (unknown keys, e.g. typos, are errors).
QUALITY_CASE_KEYS = (
  "name", "description", "backend", "model", "steps", "precision", "cfg_type", "future", "history", "new_frames",
  "post_processing", "seam_blend_frames", "humans", "duration", "seed", "scenarios", "periods", "jitter_changes",
  "branch_check", "library", "graph",
)
CASE_KEYS = {
  "ardy": ("model", "steps", "precision", "cfg_type", "future"),
  "kimodo": ("model", "steps", "precision", "post_processing", "history", "seam_blend_frames", "new_frames"),
}


def make_case_backend(case: dict):
  """An online backend, or a `GraphRuntime` over a library for `backend: graph`."""
  if case["backend"] == "graph":
    from crowdmotion.graph import GraphConfig, GraphRuntime, MotionGraph

    path = case["library"]
    if path not in _LIBRARIES:
      _LIBRARIES[path] = torch.load(path, map_location="cpu", weights_only=False)
    t0 = time.perf_counter()
    graph = MotionGraph(_LIBRARIES[path], GraphConfig.from_dict(case.get("graph")))
    synchronize(graph.device)
    graph.stats["build_s"] = time.perf_counter() - t0
    return GraphRuntime(graph)
  kind = case["backend"]
  return make_backend(config_from_case(case, CASE_KEYS[kind], **ONLINE_DEFAULTS[kind]))


def history_frames(backend, cfg: dict) -> int:
  """Executed frames fed back each call. Must be positive: before 2026-10 a value of 0
  accidentally passed the whole executed history (a `[-0:]` slice)."""
  n = cfg.get("history", 4 if backend.name == "ardy" else 5)
  if n < 1:
    raise ValueError("history must be at least one executed frame")
  return n


# Rollout ----------------------------------------------------------------------
def run_rollout(backend, cfg: dict, scenario: str, period, humans: int, duration: float, seed: int) -> dict:
  """Executed motion of `humans` under `scenario`, replanning every `period`.

  `period` is seconds, "refill" (execute every generated frame) or, for a graph,
  "edge". The start states, command schedules and change offsets come from one
  seeded generator, so graph and online cases see identical schedules.
  """
  fps, dev = backend.fps, backend.device
  graph = backend.name == "graph"
  if graph:
    if period != "edge":
      raise ValueError("graph cases replan once per edge: use period 'edge'")
    source = GraphMotionSource(backend, timed=True)
    interval_s = backend.graph.L / fps
    cap = backend.graph.L
  else:
    source = OnlineMotionSource(backend, history_frames(backend, cfg), timed=True)
    interval_s = backend.frames_per_call / fps if period == "refill" else float(period)
    cap = None if period == "refill" else max(1, round(period * fps))
  commands, xz = ScenarioCommands.draw(scenario, humans, seed, interval_s, cfg.get("jitter_changes", True), dev)
  ctrl = CrowdController(commands, source, WindowVelocity(fps))
  ctrl.reset(human_seeds(seed, torch.arange(humans), "quality"), xz, commands.h0)
  total = int(round(duration * fps))
  joints, roots, contacts, planned, boundaries, call_walls = [], [], [], [], [], []
  diag: dict[str, list] = {k: [] for k in DIAG_KEYS}
  executed = 0
  while executed < total:
    step = ctrl.advance(executed / fps, take=cap, limit=total - executed)
    motion, take = step.motion, step.take
    call_walls.append(motion.extras["call_wall_s"])
    boundaries.append(executed)
    joints.append(motion.joints[:, :take])
    roots.append(motion.root[:, :take])
    contacts.append(motion.contacts[:, :take])
    planned.append(step.path[:, :take])
    if graph:
      for key in diag:
        diag[key].append(motion.extras[key])
    executed += take
  r = {
    "fps": fps,
    "h0": commands.h0,
    "joints": torch.cat(joints, 1),
    "root": torch.cat(roots, 1),
    "contacts": torch.cat(contacts, 1).bool(),
    "planned": torch.cat(planned, 1),
    "boundaries": boundaries,
    "call_walls": call_walls,
    "rng": commands.rng,
    "offset": commands.offset,
  }
  if graph:
    r["graph"] = {k: torch.stack(v, 1) for k, v in diag.items()}
  else:
    r["features"] = ctrl.state.features
  return r


# Metrics ----------------------------------------------------------------------
def metrics(r: dict, backend, scenario: str) -> dict:
  fps = r["fps"]
  root_xz = r["root"][..., [0, 2]]
  joints = r["joints"]
  n, frames = root_xz.shape[:2]
  info = backend.skeleton_info()
  out: dict = {"humans": n, "frames": frames, "seconds": frames / fps, "calls": len(r["boundaries"])}
  finite = torch.isfinite(joints).all(dim=(-1, -2)).all(dim=-1)
  out["failed_humans_nonfinite"] = int((~finite).sum())

  err = (root_xz - r["planned"]).norm(dim=-1)
  out["path_err_mean_m"] = float(err.mean())
  out["path_err_p95_m"] = float(torch.quantile(err.float().flatten(), 0.95)) if err.numel() else float("nan")
  out["path_err_max_m"] = float(err.max())
  out["failed_humans_path_gt_1m"] = int((err.max(dim=1).values > 1.0).sum())

  vel = smoothed_velocity(root_xz, fps)
  speed = vel.norm(dim=-1)
  planned_vel = smoothed_velocity(r["planned"], fps)
  out["speed_err_mean_mps"] = float((speed - planned_vel.norm(dim=-1)).abs().mean())

  fpos = joints[:, :, info["foot_idx"]]  # [N, T, 4, 3]
  out.update(contact_skate(fpos, r["contacts"], fps))
  out.update(label_free_skate(fpos, fps))
  if "head_idx" in info:
    out.update(hunch_stats(joints, info["head_idx"], info["neutral_head_height"]))
  out.update(ground_stats(fpos, r["contacts"]))
  out.update(seam_stats(joints, vel, root_xz, r["boundaries"], fps))

  # Command response for single-change scenarios, from each human's own change time.
  offset = r["offset"].cpu()
  start = torch.round((CHANGE_T + offset) * fps).long()
  if scenario != "cruise" and int(start.max()) < frames:
    out["replan_wait_s"] = replan_wait(start, r["boundaries"], fps)
  if scenario == "stop_start" and int(start.max()) < frames:
    out["stop_response_s"] = response_time(speed < 0.2, start, fps)
    restart = torch.round((RESTART_T + offset) * fps).long()
    if int(restart.max()) < frames:
      out["start_response_s"] = response_time(speed > 0.96, restart, fps)
  elif scenario in ("turn90", "turn180") and int(start.max()) < frames:
    target = r["h0"] + (math.pi / 2 if scenario == "turn90" else math.pi)
    travel = torch.atan2(vel[..., 0], vel[..., 1])
    diff = torch.remainder(travel - target[:, None] + math.pi, 2 * math.pi) - math.pi
    out["turn_response_s"] = response_time((diff.abs() < math.radians(20)) & (speed > 0.5), start, fps)
  elif scenario == "speed_step" and int(start.max()) < frames:
    out["speed_response_s"] = response_time((speed - 1.6).abs() < 0.2, start, fps)
  if "graph" in r and scenario in RESPONSE_KEY and RESPONSE_KEY[scenario] in out:
    out.update(graph_window_split(r["graph"], r["boundaries"], fps, start, out[RESPONSE_KEY[scenario]]))
  for key in [k for k, v in out.items() if torch.is_tensor(v)]:
    vals = out.pop(key)
    out[f"{key}_median"] = float(vals.nanmedian()) if (~vals.isnan()).any() else float("nan")
    out[f"{key}_p95"] = float(torch.nanquantile(vals, 0.95)) if (~vals.isnan()).any() else float("nan")
    out[f"{key}_never_frac"] = float(vals.isnan().float().mean())
  out["call_wall_mean_s"] = float(np.mean(r["call_walls"]))
  if "graph" in r:
    out.update(graph_metrics(r["graph"], backend.graph.stats))
  return out


def branch_check(backend, cfg: dict, humans: int, seed: int, branches: int = 4) -> dict:
  """Same executed history, different seeds: history untouched, same seed reproducible."""
  r = run_rollout(backend, cfg, "cruise", "refill", humans, 3.0, seed)
  history = r["features"][:, -history_frames(backend, cfg):].clone()
  frozen = history.clone()
  xz = r["root"][:, -1, [0, 2]]
  vel = smoothed_velocity(r["root"][..., [0, 2]], r["fps"])[:, -1]
  heading = torch.atan2(vel[:, 0], vel[:, 1])
  path = plan_path(xz, vel, vel.norm(dim=-1), heading, backend.path_frames, r["fps"])
  outs = []
  for b in list(range(branches)) + [0]:
    req = MotionRequest(
      text_feat=backend.empty_text(humans), seeds=torch.arange(humans) + 7_000_000 + 1000 * b,
      counters=torch.zeros(humans, dtype=torch.long), history=history, root_path=path,
    )
    outs.append(backend.generate(req).posed_joints)
  diverge = [
    float((outs[i][:, -1] - outs[j][:, -1]).norm(dim=-1).mean())
    for i, j in itertools.combinations(range(branches), 2)
  ]
  return {
    "history_unchanged": bool(torch.equal(history, frozen)),
    "same_seed_max_abs_diff": float((outs[0] - outs[-1]).abs().max()),
    "branch_final_pose_dist_mean_m": float(np.mean(diverge)),
    "branch_final_pose_dist_min_m": float(np.min(diverge)),
  }


def save_clips(path: Path, r: dict, backend, keep: int = 6) -> None:
  info = backend.skeleton_info()
  np.savez_compressed(
    path,
    joints=r["joints"][:keep].cpu().numpy(),
    planned=r["planned"][:keep].cpu().numpy(),
    contacts=r["contacts"][:keep].cpu().numpy(),
    boundaries=np.array(r["boundaries"]),
    fps=r["fps"],
    parents=np.array(info["parents"]),
    joint_names=np.array(info["joint_names"]),
    foot_idx=np.array(info["foot_idx"]),
    offset=r["offset"][:keep].cpu().numpy(),
  )


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_quality)")
  parser.add_argument("--only", nargs="*", help="case names to run")
  parser.add_argument("--group", help="run only this list of cases, e.g. 'ardy' or 'kimodo'")
  parser.add_argument("--library", help="library.pt for graph cases that do not name one")
  args = parser.parse_args(argv)
  config = json.loads(args.config.read_text())
  if args.library:
    config = {**config, "library": args.library}
  if args.group:
    config = {**config, "cases": config[args.group]}
  elif "cases" not in config:
    config = {**config, "cases": [c for key in ("ardy", "kimodo") for c in config.get(key, [])]}
  for case in config["cases"]:
    check_keys(case, QUALITY_CASE_KEYS, f"quality case {case.get('name')!r}")
  rows = []
  with RunDir.create("quality", out=args.out, record={"config": config}) as run:
    jsonl = (run.path / "rows.jsonl").open("a")
    for case in config["cases"]:
      if args.only and case["name"] not in args.only:
        continue
      if case["backend"] == "graph" and not case.get("library"):
        case = {**case, "library": args.library}
      backend = make_case_backend(case)
      for scenario, period in itertools.product(case["scenarios"], case["periods"]):
        t0 = time.perf_counter()
        r = run_rollout(backend, case, scenario, period, case.get("humans", 32), case.get("duration", 12.0),
                        case.get("seed", 0))
        row = {"name": case["name"], "scenario": scenario, "period": period, **metrics(r, backend, scenario)}
        row["elapsed_s"] = time.perf_counter() - t0
        tag = f"{case['name']}_{scenario}_{period}"
        save_clips(run.path / f"clips_{tag}.npz", r, backend)
        rows.append(row)
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()
        print(json.dumps({k: row[k] for k in row if not k.startswith("interior")}), flush=True)
      if case.get("branch_check", backend.name != "graph"):
        row = {"name": case["name"], "scenario": "branch_check", **branch_check(backend, case, 8, 123)}
        rows.append(row)
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()
        print(json.dumps(row), flush=True)
      del backend
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    jsonl.close()
    run.record["rows"] = len(rows)
  print("wrote", run.path, flush=True)


if __name__ == "__main__":
  main()
