"""Usable-motion throughput sweep for ARDY and Kimodo.

  source scripts/env.sh
  crowd bench throughput --config configs/bench/ardy_a40.json [--out runs/<UTC>_throughput]

A config lists cases; list-valued keys in a case expand to their Cartesian
product. Each configuration is warmed up, then timed for `reps` synchronized
calls (stopping early once `time_budget_s` is spent). Model loading is timed
separately and excluded. Writes `rows.csv`, `rows.jsonl` and `run.json`.

Definitions (see docs/usage/benchmarks.md):
  new_s          newly generated seconds per human per call (history excluded)
  R_new          B * new_s / wall_p50
  u(P)           consumed fraction when replanning every P simulated seconds:
                 1 for buffer refill or P >= new_s, else P / new_s
  R_usable(P)    u(P) * R_new
  S_star(P, m)   R_usable(P) / (4096 * m): the fastest simulator speed (simulated
                 seconds per wall second per env) one GPU could feed in isolation
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch

from crowdmotion.backends.base import MotionChunk, MotionRequest
from crowdmotion.backends.registry import config_from_case, make_backend
from crowdmotion.behavior.planner import random_arc_paths
from crowdmotion.behavior.sources import MotionStep, OnlineMotionSource
from crowdmotion.core.config import check_keys
from crowdmotion.core.rng import human_seeds
from crowdmotion.core.rundir import RunDir

REPLAN_PERIODS = ("refill", 0.2, 0.5, 1.0, 2.0)
HUMANS_PER_ENV = (1, 4, 8)
NUM_ENVS = 4096
# Defaults of throughput cases, applied below the case's own keys.
SWEEP_DEFAULTS = {
  "ardy": {"precision": "fp32", "cfg_type": "separated", "steps": None},
  "kimodo": {"precision": "fp32", "steps": 100},
}
# Every key a throughput case may hold (unknown keys, e.g. typos, are errors).
THROUGHPUT_CASE_KEYS = (
  "name", "description", "backend", "model", "batch", "microbatch", "precision", "cfg_type", "steps", "future",
  "conditions", "cfg_weight", "post_processing", "to_soma77", "history", "new_frames", "workload", "seed", "reps",
  "time_budget_s",
)
SWEEP_KEYS = {
  "ardy": ("model", "device", "precision", "cfg_type", "steps", "future", "conditions", "cfg_weight"),
  "kimodo": ("model", "device", "precision", "steps", "post_processing", "to_soma77", "history", "cfg_weight",
             "new_frames"),
}


def expand(case: dict) -> list[dict]:
  keys = list(case)
  values = [v if isinstance(v, list) else [v] for v in case.values()]
  return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


class Population:
  """A batch of humans streaming synthetic random arcs through an online backend.

  The executed-feature history lives in an `OnlineMotionSource`; paths are random
  arcs from each human's last root along its recent travel heading.
  """

  def __init__(self, backend, batch: int, seed: int, keep: int):
    self.backend = backend
    self.batch = batch
    dev = backend.device
    gen = torch.Generator().manual_seed(seed)
    seeds = human_seeds(seed, torch.arange(batch), "stream")
    self.start_xz = (torch.rand(batch, 2, generator=gen) * 20 - 10).to(dev)
    self.start_heading = (torch.rand(batch, generator=gen) * 2 * math.pi - math.pi).to(dev)
    self.source = OnlineMotionSource(backend, history_frames=0, keep=keep)
    self.state = self.source.reset(seeds, self.start_xz, self.start_heading)
    self.root_xz = self.start_xz.clone()
    self.heading = self.start_heading.clone()

  @property
  def seeds(self) -> torch.Tensor:
    return self.state.seeds

  @property
  def counters(self) -> torch.Tensor:
    return self.state.counters

  @property
  def features(self) -> torch.Tensor | None:
    return self.state.features

  def request(self, history_frames: int | None) -> MotionRequest:
    """Arcs for the next call; history of `history_frames` executed frames (None: fresh)."""
    path = random_arc_paths(
      self.root_xz, self.heading, self.backend.path_frames, self.backend.fps, self.seeds, self.counters
    )
    return self.source.request(self.state, path, history_frames or 0)

  def execute(self, chunk) -> None:
    """Every frame of `chunk` was executed (buffer refill)."""
    self.source.commit(self.state, MotionStep.from_chunk(chunk), chunk.frames)
    self.root_xz = chunk.root_pos[:, -1, [0, 2]]
    # Heading of travel over the last few frames; falls back to the old heading.
    delta = chunk.root_pos[:, -1, [0, 2]] - chunk.root_pos[:, max(0, chunk.frames - 5), [0, 2]]
    moving = delta.norm(dim=-1) > 1e-3
    self.heading = torch.where(moving, torch.atan2(delta[:, 0], delta[:, 1]), self.heading)


def build_backend(kind: str, model: str, cache: dict):
  """Load (or reuse) the backend for `model`; only one model stays in memory."""
  key = (kind, model)
  if key in cache:
    return cache[key], 0.0
  for other in list(cache):
    del cache[other]
  gc.collect()
  torch.cuda.empty_cache()
  t0 = time.perf_counter()
  backend = make_backend(config_from_case({"backend": kind, "model": model}))
  torch.cuda.synchronize()
  cache[key] = backend
  return backend, time.perf_counter() - t0


def configure(backend, cfg: dict) -> None:
  """Apply a sweep configuration to the loaded backend (no weight reload)."""
  case = {**cfg, "model": backend.model_name, "device": str(backend.device)}
  backend.configure(config_from_case(case, SWEEP_KEYS[backend.name], **SWEEP_DEFAULTS[backend.name]))


def history_frames(backend, cfg: dict) -> int:
  return cfg.get("history", 4) if backend.name == "ardy" else backend.transition_frames


def one_call(backend, pop: Population, cfg: dict):
  """One microbatched population refresh; returns (wall, stages, chunk list)."""
  continuing = cfg.get("workload", "continue") == "continue"
  req = pop.request(history_frames(backend, cfg) if continuing else None)
  micro = min(cfg.get("microbatch") or pop.batch, pop.batch)
  stages: dict[str, float] = {}
  chunks = []
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for lo in range(0, pop.batch, micro):
    sl = slice(lo, lo + micro)
    sub = MotionRequest(
      text_feat=req.text_feat[sl],
      seeds=req.seeds[sl],
      counters=req.counters[sl],
      history=None if req.history is None else req.history[sl],
      root_path=req.root_path[sl],
      start_xz=None if req.start_xz is None else req.start_xz[sl],
      start_heading=None if req.start_heading is None else req.start_heading[sl],
    )
    chunks.append(backend.generate(sub))
  torch.cuda.synchronize()
  wall = time.perf_counter() - t0
  for chunk in chunks:
    for name, sec in chunk.stage_seconds().items():
      stages[name] = stages.get(name, 0.0) + sec
  return wall, stages, chunks


def merge(chunks):
  if len(chunks) == 1:
    return chunks[0]
  cat = lambda name: torch.cat([getattr(c, name) for c in chunks])  # noqa: E731
  return MotionChunk(
    features=cat("features"), root_pos=cat("root_pos"), local_rot=cat("local_rot"),
    posed_joints=cat("posed_joints"), contacts=cat("contacts"), heading=cat("heading"),
    fps=chunks[0].fps,
  )


def run_config(backend, cfg: dict, defaults: dict) -> dict:
  configure(backend, cfg)
  reps = cfg.get("reps", defaults["reps"])
  budget = cfg.get("time_budget_s", defaults["time_budget_s"])
  batch = cfg["batch"]
  history = history_frames(backend, cfg)
  keep = max(history, 8) + 64
  pop = Population(backend, batch, cfg.get("seed", 0), keep)
  continuing = cfg.get("workload", "continue") == "continue"
  torch.cuda.reset_peak_memory_stats()
  # Warm-up; for continuation also fill the executed-history buffer.
  warm = 0
  while True:
    wall, _, chunks = one_call(backend, pop, cfg)
    warm += 1
    if continuing:
      pop.execute(merge(chunks))
    have = 0 if pop.features is None else pop.features.shape[1]
    if warm >= defaults["warmup"] and (not continuing or have >= history):
      break
    if warm > 64:
      raise RuntimeError("could not fill history buffer")
  walls, stage_rows = [], []
  new_frames = chunks[0].frames
  spent = 0.0
  for _ in range(reps):
    wall, stages, chunks = one_call(backend, pop, cfg)
    walls.append(wall)
    stage_rows.append(stages)
    spent += wall
    if continuing:
      pop.execute(merge(chunks))
    if spent > budget and len(walls) >= 2:
      break
  fps = backend.fps
  new_s = new_frames / fps
  p50 = statistics.median(walls)
  p95 = sorted(walls)[min(len(walls) - 1, math.ceil(0.95 * len(walls)) - 1)]
  row = {
    **{k: v for k, v in cfg.items() if k not in ("reps", "time_budget_s")},
    "fps": fps,
    "new_frames": new_frames,
    "new_s": new_s,
    "reps_timed": len(walls),
    "wall_p50": p50,
    "wall_p95": p95,
    "wall_mean": statistics.fmean(walls),
    "wall_min": min(walls),
    "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    "R_new": batch * new_s / p50,
  }
  names = sorted({n for s in stage_rows for n in s})
  for name in names:
    row[f"stage_{name}"] = statistics.fmean(s.get(name, 0.0) for s in stage_rows)
  row["stage_sum"] = sum(row[f"stage_{n}"] for n in names)
  for period in REPLAN_PERIODS:
    u = 1.0 if period == "refill" or period >= new_s else period / new_s
    tag = "refill" if period == "refill" else f"P{period:g}"
    row[f"u_{tag}"] = u
    row[f"R_usable_{tag}"] = u * row["R_new"]
    for m in HUMANS_PER_ENV:
      row[f"S_star_{tag}_m{m}"] = u * row["R_new"] / (NUM_ENVS * m)
  return row


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_throughput)")
  parser.add_argument("--only", nargs="*", help="run only cases with these names")
  args = parser.parse_args(argv)
  config = json.loads(args.config.read_text())
  defaults = {"reps": 5, "warmup": 1, "time_budget_s": 120.0, **config.get("defaults", {})}
  for case in config["cases"]:
    check_keys(case, THROUGHPUT_CASE_KEYS, f"throughput case {case.get('name')!r}")
  cache: dict = {}
  loads: dict = {}
  rows = []
  with RunDir.create("throughput", out=args.out, record={"config": config}) as run:
    env = run.record["environment"]
    print(json.dumps({k: env[k] for k in ("gpu", "driver", "torch", "torch_cuda") if k in env}), flush=True)
    jsonl = (run.path / "rows.jsonl").open("a")
    for case in config["cases"]:
      if args.only and case.get("name") not in args.only:
        continue
      for cfg in expand(case):
        backend, load_s = build_backend(cfg["backend"], cfg["model"], cache)
        if load_s:
          loads[cfg["model"]] = load_s
        while True:
          try:
            row = run_config(backend, cfg, defaults)
            row["status"] = "ok"
            break
          except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            micro = (cfg.get("microbatch") or cfg["batch"]) // 2
            if micro < 1:
              row = {**cfg, "status": "oom"}
              break
            print(f"OOM; retrying with microbatch {micro}", flush=True)
            cfg = {**cfg, "microbatch": micro}
        rows.append(row)
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()
        brief = {k: row.get(k) for k in ("name", "model", "batch", "precision", "cfg_type", "steps",
                                         "history", "future", "new_frames", "wall_p50", "R_new", "peak_gb", "status")}
        print(json.dumps(brief), flush=True)
    jsonl.close()
    fields = sorted({k for r in rows for k in r})
    with (run.path / "rows.csv").open("w", newline="") as fh:
      writer = csv.DictWriter(fh, fieldnames=fields)
      writer.writeheader()
      writer.writerows(rows)
    run.record.update(load_seconds=loads, rows=len(rows))
  print("wrote", run.path, flush=True)


if __name__ == "__main__":
  main()
