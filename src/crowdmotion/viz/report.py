"""Audit a saved motion graph and write a self-contained local HTML report.

  crowd graph report [--graph <graph.pt>] [--library <library.pt>] [--video <crowd.mp4>] \
      [--out runs/<UTC>_behavior_report]

The report shows what motion the graph's tables contain (body-relative frame and
edge composition, turning, connectivity, transition availability) and example
edges. Composition is availability, not online visitation. Writes `audit.json`
and `index.html` (data embedded; no network access needed).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch

from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.core.paths import default_graph
from crowdmotion.core.provenance import file_sha256
from crowdmotion.core.rundir import RunDir
from crowdmotion.graph.audit import audit_graph
from crowdmotion.viz.html import render_template


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--graph", type=Path, help="saved graph (default: $CROWD_GRAPH or artifact graph-default)")
  parser.add_argument("--library", type=Path, help="library.pt, for command composition")
  parser.add_argument("--video", type=Path, help="a crowd.mp4 to embed (relative link)")
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_behavior_report)")
  args = parser.parse_args(argv)
  torch.set_num_threads(1)
  graph = args.graph or default_graph()
  with RunDir.create("behavior_report", out=args.out, record={"graph": str(Path(graph).resolve())}) as run:
    blob = torch.load(graph, map_location="cpu", weights_only=False, mmap=True)
    data = audit_graph(blob)
    data.update(
      created=datetime.now(UTC).isoformat(),
      graph=str(Path(graph).resolve()),
      environment=run.record["environment"],
      steering=PlazaSteeringCfg(obstacles=((-6.0, 0.0, 0.3),)).__dict__,
      graph_sha256=file_sha256(graph),
      # Source provenance is the repository state in `environment` (commit, diff when dirty).
      source_hashes={k: run.record["environment"].get(k) for k in ("repository", "source_fingerprint")},
      sources={},
    )
    if args.library:
      lib = torch.load(args.library, map_location="cpu", weights_only=False, mmap=True)
      data["library"] = {
        "path": str(args.library.resolve()),
        "streams": int(lib["cmd_mode"].shape[0]),
        "command_edges": int(lib["cmd_mode"].numel()),
        "stand_command_percent": 100 * float((lib["cmd_mode"] == 1).float().mean()),
        "walk_command_percent": 100 * float((lib["cmd_mode"] == 0).float().mean()),
      }
      provenance = args.library.parent / "run.json"
      if provenance.exists():
        data["library"]["provenance"] = json.loads(provenance.read_text())
    data["video"] = os.path.relpath(args.video.resolve(), run.path.resolve()) if args.video else None
    if args.video:
      for name in ("summary", "run"):
        path = args.video.parent / f"{name}.json"
        if path.exists():
          data[f"video_{name}"] = json.loads(path.read_text())
    run.write_json("audit.json", data)
    payload = json.dumps(data).replace("<", "\\u003c")
    (run.path / "index.html").write_text(render_template("graph_audit.html", report_data=payload))
  print(json.dumps({"report": str(run.path / "index.html"), "frames": data["frame_composition"],
                    "edges": data["edge_composition"]}, indent=2))


if __name__ == "__main__":
  main()
