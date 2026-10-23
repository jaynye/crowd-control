"""Paired comparison of two behavior evaluations on the same graph and seed bank.

  crowd behavior compare --baseline runs/<a> --candidate runs/<b> [--out runs/<UTC>_behavior_compare]

Writes `comparison.json` (per-seed metric deltas and both manifests) and a local
`index.html` with the deltas and any rendered videos of either run. Candidate
minus baseline; no empirical acceptance threshold is applied.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

from crowdmotion.core.rundir import RunDir
from crowdmotion.metrics.behavior import PAIRED_METRICS
from crowdmotion.viz.html import render_template

STANDOFF_METRICS = (
  "pair_standoffs.per_human_minute", "pair_standoffs.duration_s_p50_p95", "pair_standoffs.duration_s_max",
  "pair_stalls_including_maneuvers.per_human_minute", "pair_stalls_including_maneuvers.duration_s_p50_p95",
  "pair_stalls_including_maneuvers.duration_s_max", "pending_goal_stops.duration_s_p50_p95",
  "stopped_pending_goal_s_per_human_minute", "mean_speed_mps", "goal_arrivals_per_human_minute",
  "threat_encounters.resolution_s_p50_p95", "threat_encounters.censored_at_end",
  "threat_encounters.either_deviates_percent", "threat_encounters.neither_deviates_percent",
)


def _fmt(value) -> str:
  if value is None:
    return "—"
  if isinstance(value, list):
    return " / ".join(f"{v:.3f}" for v in value)
  return f"{value:.3f}"


def paired_rows(baseline: Path, candidate: Path, seeds) -> list[dict]:
  rows = []
  for seed in sorted(seeds):
    left = json.loads((baseline / f"seed_{seed}.json").read_text())
    right = json.loads((candidate / f"seed_{seed}.json").read_text())
    for metric in PAIRED_METRICS:
      x, y = left[metric], right[metric]
      delta = None if x is None or y is None else (np.asarray(y) - np.asarray(x)).tolist()
      rows.append({"seed": seed, "metric": metric, "baseline": x, "candidate": y, "delta": delta})
    if "standoffs" in left and "standoffs" in right:
      for metric in STANDOFF_METRICS:
        x, y = left["standoffs"], right["standoffs"]
        for key in metric.split("."):
          x, y = x[key], y[key]
        delta = None if x is None or y is None else (np.asarray(y) - np.asarray(x)).tolist()
        rows.append({"seed": seed, "metric": f"standoffs.{metric}", "baseline": x, "candidate": y, "delta": delta})
  return rows


def check_paired(a: dict, b: dict) -> None:
  if a["status"] != "completed" or b["status"] != "completed":
    raise ValueError("Both evaluations must be completed")
  for key in ("graph_sha256", "fps"):
    if a[key] != b[key]:
      raise ValueError(f"Paired comparison requires matching {key}")
  for key in ("humans", "seconds"):
    if a["config"][key] != b["config"][key]:
      raise ValueError(f"Paired comparison requires matching {key}")
  if set(a["config"]["seeds"]) != set(b["config"]["seeds"]):
    raise ValueError("Paired comparison requires the same seed bank")


def compare(args) -> Path:
  a = json.loads((args.baseline / "run.json").read_text())
  b = json.loads((args.candidate / "run.json").read_text())
  check_paired(a, b)
  rows = paired_rows(args.baseline, args.candidate, a["config"]["seeds"])
  record = {"baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve())}
  with RunDir.create("behavior_compare", out=args.out, record=record, env=False) as run:
    run.write_json("comparison.json", {
      **record,
      "baseline_manifest": a,
      "candidate_manifest": b,
      "paired_metrics": rows,
      "note": "Review motion and spacing/foot tradeoffs together; no empirical acceptance threshold.",
    })
    body = "".join(
      f"<tr><td>{r['seed']}</td><td>{html.escape(r['metric'])}</td><td>{_fmt(r['baseline'])}</td>"
      f"<td>{_fmt(r['candidate'])}</td><td>{_fmt(r['delta'])}</td></tr>"
      for r in rows
    )
    videos = ""
    for name, directory in (("Baseline", args.baseline), ("Candidate", args.candidate)):
      for video in sorted(directory.glob("render_seed_*_*/crowd.mp4")):
        url = html.escape(os.path.relpath(video.resolve(), run.path.resolve()), quote=True)
        videos += (f'<div><h3>{name} · {html.escape(video.parent.name)}</h3>'
                   f'<video controls preload="metadata" src="{url}"></video></div>')
    settings = html.escape(json.dumps({"baseline": a["config"], "candidate": b["config"]}, indent=2))
    (run.path / "index.html").write_text(
      render_template("compare.html", rows=body, videos=videos, settings=settings)
    )
  print(run.path / "index.html", flush=True)
  return run.path


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_behavior_compare)")
  compare(parser.parse_args(argv))


if __name__ == "__main__":
  main()
