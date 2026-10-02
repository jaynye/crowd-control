"""Runtime cost of stepping a motion graph for a whole training population.

  source scripts/env.sh
  .venv/bin/python -m crowdmotion.bench.graph_speed --library <library.pt> \
      --out runs/<UTC>_graph_speed

Builds the graph (timed), then for each population size steps every human
through one edge per call toward a random straight path and reports the
synchronized wall time per call. One call advances every human by `edge_frames`
frames, so `motion_s_per_wall_s = humans * edge_s / wall`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import time
from pathlib import Path

import torch

from crowdmotion.behavior.planner import straight_paths
from crowdmotion.core.provenance import environment
from crowdmotion.core.rundir import write_json
from crowdmotion.graph import GraphConfig, GraphRuntime, MotionGraph


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--library", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--humans", type=int, nargs="+", default=[4096, 16384, 65536])
  parser.add_argument("--graph-config", type=Path, help="GraphConfig JSON (default: correction only)")
  parser.add_argument("--neighbors", type=int, nargs="+", default=[32, 128])
  parser.add_argument("--reps", type=int, default=20)
  args = parser.parse_args(argv)
  args.out.mkdir(parents=True, exist_ok=False)
  env = environment()
  lib = torch.load(args.library, map_location="cpu", weights_only=False)
  if args.graph_config:
    base = GraphConfig.from_dict(json.loads(args.graph_config.read_text()))
  else:
    base = GraphConfig(correct_yaw_deg=5.0, correct_speed=0.1)
  rows = []
  for k in args.neighbors:
    cfg = dataclasses.replace(base, neighbors=k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    graph = MotionGraph(lib, cfg)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    runtime = GraphRuntime(graph)
    edge_s = graph.L / graph.fps
    for humans in args.humans:
      torch.cuda.reset_peak_memory_stats()
      gen = torch.Generator().manual_seed(humans)
      state = runtime.reset(
        torch.arange(humans),
        torch.rand(humans, 2, generator=gen) * 20 - 10,
        torch.rand(humans, generator=gen) * 2 * math.pi - math.pi,
      )
      speed = (0.5 + torch.rand(humans, generator=gen)).to(graph.device)
      walls = []
      for rep in range(args.reps + 3):
        path = straight_paths(state.root_xz, state.heading, speed, runtime.path_frames, graph.fps)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runtime.step(state, path)
        torch.cuda.synchronize()
        if rep >= 3:
          walls.append(time.perf_counter() - t0)
      wall = statistics.median(walls)
      row = {
        "neighbors": k,
        "humans": humans,
        "edge_s": edge_s,
        "build_s": build_s,
        "wall_p50_s": wall,
        "wall_max_s": max(walls),
        "motion_s_per_wall_s": humans * edge_s / wall,
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        "graph_config": dataclasses.asdict(cfg),
        **{f"graph_{key}": v for key, v in graph.stats.items()},
      }
      rows.append(row)
      print(json.dumps(row), flush=True)
    del graph, runtime
    torch.cuda.empty_cache()
  (args.out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
  write_json(args.out / "run.json", {"args": {k: str(v) for k, v in vars(args).items()}, "environment": env, "status": "completed"})


if __name__ == "__main__":
  main()
