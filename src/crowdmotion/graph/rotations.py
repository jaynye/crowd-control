"""Saved joint rotations for a motion graph (`crowd graph rotations`).

  crowd graph rotations [--graph <graph.pt>] [--library <library.pt> ...] [--out runs/<UTC>_graph_rotations]

The saved graph (`MotionGraph.save`) keeps joint positions only. Rigid mesh humans
(`crowdmotion.sim.body`) also need each bone's rotation, so this command derives them
from the library (or libraries, in `crowd graph build` order) that built the graph and
saves `rotations.pt`, aligned with the graph's tables:

- `edge_rot [B, L, J, 4]`: each edge's frames, like `edge_pose` (edge id = start
  boundary id; the last boundary of each stream is padding, filled with identity);
- `pred_rot [B, J, 4]`: the pose at each boundary, like `pred_pose`;
- `offsets [J, 3]`: each joint's rest offset in its parent's frame.

Rotations are global joint rotations (FK of the library's local rotations,
`sim.body.joint_transforms`) relative to each frame's heading, as `edge_pose` is
(`core.geometry.body_local`): the world rotation is `quat_yaw(heading) * rot`. They are
stored as float16 wxyz quaternions.

`GraphRotations.load` checks a file against its graph: for sampled valid edge frames,
each parent's rotation must carry the child's rest offset onto the graph's
parent-to-child vector. A file built from another library or graph fails the check.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from crowdmotion.sim.body import joint_transforms, quat_from_matrix, quat_to_matrix, quat_yaw

FORMAT = 1
# Bone direction error (degrees) a matching file stays under; float16 tables give
# about 0.5 degrees at the median.
MAX_MEDIAN_DEG = 2.0
MAX_P95_DEG = 10.0


def build(lib: dict, L: int, chunk: int = 64, device="cpu") -> dict:
  """Rotation tables for a graph with `L`-frame edges built from library `lib`."""
  parents = list(lib["meta"]["parents"])
  n, T, J = lib["posed_joints"].shape[:3]
  K = T // L
  edge = torch.empty(n, K, L, J, 4, dtype=torch.float16)
  pred = torch.empty(n, K, J, 4, dtype=torch.float16)
  edge[:, K - 1] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float16)  # padding row
  offsets = None
  for s in range(0, n, chunk):
    rot = joint_transforms(lib["local_rot"][s:s + chunk, :K * L].to(device),
                           lib["posed_joints"][s:s + chunk, :K * L].to(device), parents)[..., :3, :3]
    if offsets is None:
      # Rest offsets from the first chunk: offset_j = R_parent^T (p_j - p_parent).
      pos = lib["posed_joints"][s:s + chunk, :K * L].to(device).float()
      j = torch.tensor([i for i, p in enumerate(parents) if p >= 0], device=device)
      p = torch.tensor([parents[i] for i in j.tolist()], device=device)
      local = (rot[..., p, :, :].transpose(-1, -2) @ (pos[..., j, :] - pos[..., p, :])[..., None])[..., 0]
      offsets = torch.zeros(J, 3)
      offsets[j.cpu()] = local.flatten(0, 1).median(0).values.cpu()
    yaw = quat_to_matrix(quat_yaw(-lib["heading"][s:s + chunk, :K * L].to(device)))
    q = quat_from_matrix(yaw[..., None, :, :] @ rot).half().cpu()  # [c, K*L, J, 4]
    c = q.shape[0]
    pred[s:s + c] = q[:, L - 1::L]  # boundary k sits at frame k*L - 1
    edge[s:s + c, :K - 1] = q.view(c, K, L, J, 4)[:, 1:]  # edge j plays frames (j+1)*L ...
  return {
    "format": FORMAT,
    "edge_rot": edge.flatten(0, 1),
    "pred_rot": pred.flatten(0, 1),
    "offsets": offsets,
    "parents": parents,
    "joint_names": list(lib["meta"]["joint_names"]),
    "L": L, "streams": n, "K": K,
  }


def bone_direction_error(graph, edge_rot: torch.Tensor, offsets: torch.Tensor, parents,
                         samples: int = 512, seed: int = 0) -> torch.Tensor:
  """Angles (degrees) between each rotated rest offset and the graph's bone vector over
  `samples` random valid edges: `[samples * L, bones]`."""
  dev = graph.edge_pose.device
  valid = torch.nonzero(graph.valid).squeeze(-1)
  pick = torch.randint(len(valid), (samples,), generator=torch.Generator().manual_seed(seed))
  e = valid[pick.to(dev)]
  offsets = offsets.to(dev).float()
  j = torch.tensor([i for i, p in enumerate(parents) if p >= 0 and float(offsets[i].norm()) > 0.01], device=dev)
  p = torch.tensor([parents[i] for i in j.tolist()], device=dev)
  rot = quat_to_matrix(edge_rot[e].to(dev).float())  # [S, L, J, 3, 3]
  pose = graph.edge_pose[e].float()  # [S, L, J, 3]
  u = (rot[..., p, :, :] @ offsets[j][..., None])[..., 0]
  v = pose[..., j, :] - pose[..., p, :]
  cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-9)
  return torch.rad2deg(torch.acos(cos.clamp(-1, 1))).flatten(0, 1)


@dataclass
class GraphRotations:
  edge_rot: torch.Tensor  # [B, L, J, 4] float16 on the graph's device
  pred_rot: torch.Tensor  # [B, J, 4]
  offsets: torch.Tensor  # [J, 3] float32
  parents: list[int]

  @classmethod
  def load(cls, path, graph, check: bool = True) -> "GraphRotations":
    """Load `path` for `graph` (a `MotionGraph`) onto its device; raise if they differ."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    B, L, J = graph.edge_pose.shape[:3]
    if tuple(blob["edge_rot"].shape) != (B, L, J, 4) or tuple(blob["pred_rot"].shape) != (B, J, 4):
      raise ValueError(f"{path}: rotation tables {tuple(blob['edge_rot'].shape)} do not fit the graph "
                       f"({B} boundaries, {L} frames, {J} joints); rebuild with `crowd graph rotations`")
    dev = graph.edge_pose.device
    out = cls(blob["edge_rot"].to(dev), blob["pred_rot"].to(dev), blob["offsets"].float().to(dev),
              list(blob["parents"]))
    if check:
      err = bone_direction_error(graph, out.edge_rot, out.offsets, out.parents)
      med, p95 = float(err.median()), float(err.quantile(0.95))
      if med > MAX_MEDIAN_DEG or p95 > MAX_P95_DEG:
        raise ValueError(f"{path} does not match the graph: bone directions off by {med:.1f} deg "
                         f"(median), {p95:.1f} deg (95th percentile); rebuild with `crowd graph rotations`")
    return out


def main(argv=None) -> None:
  from crowdmotion.core.paths import artifact_path, default_graph
  from crowdmotion.core.rundir import RunDir
  from crowdmotion.graph.build import load_libraries
  from crowdmotion.graph.tables import MotionGraph

  parser = argparse.ArgumentParser(prog="crowd graph rotations", description=__doc__.split("\n", 1)[0])
  parser.add_argument("--graph", type=Path, help="saved graph (default: $CROWD_GRAPH or artifact graph-default)")
  parser.add_argument("--library", type=Path, nargs="+", action="extend",
                      help="library or libraries that built it, in build order (default: artifact library-ardy-h40)")
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_graph_rotations)")
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args(argv)
  graph_path = args.graph or default_graph()
  libraries = args.library or [artifact_path("library-ardy-h40")]
  library = str(libraries[0]) if len(libraries) == 1 else [str(p) for p in libraries]
  torch.set_num_threads(min(torch.get_num_threads(), 8))
  record = {"args": {"graph": str(graph_path), "library": library, "device": args.device}}
  with RunDir.create("graph_rotations", out=args.out, record=record) as run:
    graph = MotionGraph.load(graph_path, device="cpu")
    lib = load_libraries(libraries, keys=("posed_joints", "heading", "local_rot"), mmap=True)
    n, T, J = lib["posed_joints"].shape[:3]
    if (n, T // graph.L, J) != (graph.streams, graph.K, graph.joints_n):
      raise ValueError(f"library {library} ({n} streams x {T} frames) did not build graph {graph_path}")
    blob = build(lib, graph.L, device=args.device)
    blob.update(graph=str(graph_path), library=library)
    torch.save(blob, run.path / "rotations.pt")
    err = bone_direction_error(graph, blob["edge_rot"], blob["offsets"], blob["parents"])
    summary = {
      "bytes": (run.path / "rotations.pt").stat().st_size,
      "boundaries": int(blob["pred_rot"].shape[0]),
      "bone_direction_error_deg": {
        "median": float(err.median()), "p95": float(err.quantile(0.95)), "p99": float(err.quantile(0.99)),
        "max": float(err.max()),
      },
    }
    run.write_json("summary.json", summary)
  print(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
