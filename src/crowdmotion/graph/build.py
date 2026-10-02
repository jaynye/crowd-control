"""kNN candidate build, and `crowd graph build`: build a graph from a library and save it.

  source scripts/env.sh
  crowd graph build --library <library.pt> \
      --config configs/graph/default.json --out runs/<UTC>_motion_graph/graph.pt

The saved graph loads on any device with `MotionGraph.load`, without the
library or the kNN build (which is slow on CPU).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from crowdmotion.core.provenance import environment
from crowdmotion.core.rundir import write_json
from crowdmotion.graph.config import GraphConfig


@torch.no_grad()
def build_candidates(graph, chunk: int = 2048) -> None:
  """Fill `graph.cand [B, k]` (edge ids) and `graph.cand_dist [B, k]` (descriptor RMS
  distance per joint, meters): the `neighbors` nearest valid boundaries of every
  boundary in descriptor space, computed in row chunks of `chunk`."""
  k = graph.cfg.neighbors
  targets = torch.nonzero(graph.valid).squeeze(1)
  tdesc = graph.desc[targets]
  cand, dist = [], []
  for start in range(0, graph.desc.shape[0], chunk):
    d = torch.cdist(graph.desc[start : start + chunk], tdesc)
    val, idx = torch.topk(d, k, dim=1, largest=False)
    cand.append(targets[idx])
    dist.append(val)
  graph.cand = torch.cat(cand)  # [B, k] edge ids
  # Descriptor distance as RMS per joint, in meters.
  graph.cand_dist = torch.cat(dist) / math.sqrt(graph.joints_n)
  own = graph.cand == torch.arange(graph.cand.shape[0], device=graph.device)[:, None]
  graph.stats["own_successor_in_candidates"] = float(own.any(1)[graph.valid].float().mean())
  graph.stats["neighbor_dist_rms_mean_m"] = float(graph.cand_dist[:, 1:].mean())


def main(argv=None) -> None:
  # Imported here: tables.py imports `build_candidates` from this module.
  from crowdmotion.graph.tables import MotionGraph

  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--library", type=Path, required=True)
  parser.add_argument("--config", type=Path, default=Path("configs/graph/default.json"))
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args(argv)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  if args.out.exists():
    raise FileExistsError(args.out)
  lib = torch.load(args.library, map_location="cpu", weights_only=False)
  cfg = GraphConfig.from_dict(json.loads(args.config.read_text()))
  t0 = time.perf_counter()
  graph = MotionGraph(lib, cfg, device=args.device)
  build_s = time.perf_counter() - t0
  graph.save(args.out, rest_joints=lib["posed_joints"][0, 0])
  info = {
    "library": str(args.library),
    "config": cfg.__dict__,
    "build_s": build_s,
    "device": args.device,
    "bytes": args.out.stat().st_size,
    "stats": graph.stats,
    "environment": environment(),
  }
  write_json(args.out.with_suffix(".json"), info)
  print(json.dumps({k: v for k, v in info.items() if k != "environment"}, indent=1))


if __name__ == "__main__":
  main()
