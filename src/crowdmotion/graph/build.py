"""kNN candidate build, and `crowd graph build`: build a graph from a library and save it.

  source scripts/env.sh
  crowd graph build --library <library.pt> [<library.pt> ...] \
      --config configs/graph/default.json --out runs/<UTC>_motion_graph/graph.pt

The saved graph loads on any device with `MotionGraph.load`, without the
library or the kNN build (which is slow on CPU).

Several libraries build one graph over all their streams (`load_libraries`): they
must share the frame rate, edge length and skeleton; streams are concatenated in
the order given, and every stream is truncated to the shortest library's length.
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


# What a graph build reads from a library; `crowd graph rotations` also reads local_rot.
GRAPH_KEYS = ("posed_joints", "root_pos", "heading", "contacts")
# Library meta that must agree for libraries to share one graph.
MATCH_META = ("fps", "edge_frames", "joint_names", "parents", "foot_idx", "head_idx", "neutral_head_height")


def load_libraries(paths, keys=GRAPH_KEYS, mmap: bool = False) -> dict:
  """One library dict from one or more `library.pt` files (CPU tensors).

  One path loads the file unchanged (`mmap` as given). Several are merged: their `MATCH_META` fields
  must be equal; the `keys` tensors are concatenated along streams in the given
  order, each truncated to the shortest library's whole edges (a warning is
  printed when a library loses frames). The merged `meta` is the first library's
  plus `sources`: `[{"library", "streams", "frames", "model", "command_source"}]`, in
  stream order.
  """
  paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
  if not paths:
    raise ValueError("no library given")
  if len(paths) == 1:
    return torch.load(paths[0], map_location="cpu", mmap=mmap, weights_only=False)
  # Memory-mapped: only the `keys` tensors are read, into the concatenation.
  libs = [torch.load(p, map_location="cpu", mmap=True, weights_only=False) for p in paths]
  meta = libs[0]["meta"]
  for p, lib in zip(paths[1:], libs[1:]):
    diff = [k for k in MATCH_META if lib["meta"].get(k) != meta.get(k)]
    if diff:
      raise ValueError(f"library {p} differs from {paths[0]} in {diff}; one graph needs one skeleton and timing")
  L = int(meta["edge_frames"])
  frames = min(lib["posed_joints"].shape[1] for lib in libs) // L * L
  for p, lib in zip(paths, libs):
    if lib["posed_joints"].shape[1] != frames:
      print(f"warning: {p}: streams truncated from {lib['posed_joints'].shape[1]} to {frames} frames", flush=True)
  merged = {k: torch.cat([lib[k][:, :frames] for lib in libs]) for k in keys}
  merged["meta"] = {**meta, "sources": [
    {"library": str(p), "streams": int(lib["posed_joints"].shape[0]), "frames": frames,
     "model": lib["meta"].get("model"), "command_source": lib["meta"].get("command_source", "random")}
    for p, lib in zip(paths, libs)]}
  return merged


def main(argv=None) -> None:
  # Imported here: tables.py imports `build_candidates` from this module.
  from crowdmotion.graph.tables import MotionGraph

  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--library", type=Path, nargs="+", action="extend", required=True,
                      help="library.pt; several build one graph over all their streams")
  parser.add_argument("--config", type=Path, default=Path("configs/graph/default.json"))
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args(argv)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  if args.out.exists():
    raise FileExistsError(args.out)
  lib = load_libraries(args.library)
  cfg = GraphConfig.from_dict(json.loads(args.config.read_text()))
  t0 = time.perf_counter()
  graph = MotionGraph(lib, cfg, device=args.device)
  build_s = time.perf_counter() - t0
  graph.save(args.out, rest_joints=lib["posed_joints"][0, 0])
  info = {
    "library": str(args.library[0]) if len(args.library) == 1 else [str(p) for p in args.library],
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
