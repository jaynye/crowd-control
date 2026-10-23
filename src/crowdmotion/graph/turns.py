"""Turning coverage of a saved graph under a flat-ground mask (`crowd graph turns`).

  crowd graph turns [--graph <graph.pt> ...] [--labels a b ...] [--overrides <json>] \\
      [--out runs/<UTC>_turn_audit]

Per edge (one `L`-frame edge): planar path length, net body yaw, radius = length /
|yaw|, and travel along the mid-edge facing. Classes (mutually exclusive): in place
(speed < 0.2 m/s and turning >= 25 deg/s), forward turns by radius bin (travel within
30 deg of facing), non-forward moving turns, straight or gentle. Per natural window of
three successive edges of one stream (1.2 s at 20 fps), all admissible: turns of at
least 45 degrees by speed and radius. Counts are split upright/hunched (signed trunk
pitch >= `posture_cutoff_deg` is hunched; a window is hunched when two of its edges
are) and, for a graph built from several libraries (`meta["sources"]`), per source
library; `per_motion_hour` normalizes by each part's motion duration.

Twisted edges: the heading feature that yields `edge_yaw` can turn while the feet stay
planted (the pelvis twists). Each edge's foot yaw is the mean over its frames of the
angle between the mean foot direction (toe minus foot, both feet) and body forward;
`twisted` counts admissible edges (windows: any of their edges) above `twist_deg`,
which defaults to the 99th percentile of the first graph's admissible edges;
`twisted_planted` also requires every foot-contact label on through the edge (no step).
Stepping turns also lag the feet behind the body for a while, so `twisted` is an upper
bound; `twisted_planted` isolates turning on planted feet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crowdmotion.graph.annotations import annotate_twist
from crowdmotion.graph.tables import MotionGraph


def _classes(dt, speed, net, radius, fwd_frac):
  turning = np.abs(net) >= 10.0 * (dt / 0.4)  # >= 25 deg/s
  out = {"in_place (speed<.2, turn>=25deg/s)": turning & (speed < 0.2)}
  moving_turn = turning & (speed >= 0.2)
  forwardish = fwd_frac >= np.cos(np.radians(30))
  for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 4), (4, np.inf)):
    out[f"forward turn r {lo}-{hi} m"] = moving_turn & forwardish & (radius >= lo) & (radius < hi)
  out["non-forward moving turn"] = moving_turn & ~forwardish
  out["straight/gentle (<25deg/s)"] = ~turning
  return out


def foot_yaw_deg(g: MotionGraph, chunk: int = 8192) -> np.ndarray:
  """Per edge: mean over its frames of |angle| between the mean foot direction (toe minus
  foot, both feet) and body forward (+Z of `edge_pose`), in degrees."""
  return annotate_twist(g, chunk).foot_yaw_deg.cpu().numpy()


def turn_coverage(g: MotionGraph, streams: slice | None = None, twist_deg: float = np.inf,
                  foot_yaw: np.ndarray | None = None) -> dict:
  """Turning-class counts over all streams, or the edges of the stream range `streams`."""
  L, fps, K = g.L, g.fps, g.K
  dt = L / fps
  rows = slice(None) if streams is None else slice(streams.start * K, streams.stop * K)
  S = (g.streams if streams is None else streams.stop - streams.start)
  allowed = g.admissible_edges().cpu().numpy()[rows]
  valid = g.valid.cpu().numpy()[rows]
  a = g.ensure_annotations()
  hunched = (a.trunk_pitch_deg >= g.cfg.posture_cutoff_deg).cpu().numpy()[rows]
  foot_yaw = (foot_yaw_deg(g) if foot_yaw is None else foot_yaw)[rows]
  twisted = foot_yaw > twist_deg
  planted_tw = twisted & g.edge_contacts.cpu().numpy()[rows].all(axis=(1, 2))
  root = g.edge_root.float().cpu().numpy()[rows]  # [B, L, 2], predecessor frame
  yaw = g.edge_yaw.float().cpu().numpy()[rows]  # [B, L]
  steps = np.diff(np.concatenate([np.zeros_like(root[:, :1]), root], 1), axis=1)
  length = np.linalg.norm(steps, axis=-1).sum(1)
  net = np.degrees(yaw[:, -1])
  mid = np.concatenate([np.zeros_like(yaw[:, :1]), yaw[:, :-1]], 1)
  mid = mid + 0.5 * (yaw - mid)
  fwd = steps[..., 0] * np.sin(mid) + steps[..., 1] * np.cos(mid)
  fwd_frac = np.where(length > 1e-6, fwd.sum(1) / np.maximum(length, 1e-6), 0)
  speed = length / dt
  radius = np.where(np.abs(net) > 1e-3, length / np.radians(np.maximum(np.abs(net), 1e-3)), np.inf)
  hours = S * K * L / fps / 3600

  def table(ok):
    res = {}
    for name, m in _classes(dt, speed, net, radius, fwd_frac).items():
      res[name] = {"upright": int((m & ok & ~hunched).sum()), "hunched": int((m & ok & hunched).sum()),
                   "twisted": int((m & ok & twisted).sum()), "twisted_planted": int((m & ok & planted_tw).sum())}
    return res

  edges = {"valid": table(valid), "admissible": table(allowed)}
  ip = allowed & (np.abs(net) >= 10) & (speed < 0.2)
  inplace_rates = np.sort(np.abs(net[ip]) / dt).tolist()

  W = 3
  ok = allowed.reshape(S, K)
  win_ok = ok[:, :K - W + 1].copy()
  for j in range(1, W):
    win_ok &= ok[:, j:K - W + 1 + j]
  netr = np.radians(net).reshape(S, K)
  tot_yaw = sum(netr[:, j:K - W + 1 + j] for j in range(W))
  tot_len = sum(length.reshape(S, K)[:, j:K - W + 1 + j] for j in range(W))
  tot_fwd = sum((fwd_frac * length).reshape(S, K)[:, j:K - W + 1 + j] for j in range(W))
  hun = hunched.reshape(S, K)
  win_h = sum(hun[:, j:K - W + 1 + j].astype(int) for j in range(W)) >= 2
  tw = twisted.reshape(S, K)
  win_t = sum(tw[:, j:K - W + 1 + j].astype(int) for j in range(W)) >= 1
  tp = planted_tw.reshape(S, K)
  win_tp = sum(tp[:, j:K - W + 1 + j].astype(int) for j in range(W)) >= 1
  wspeed, wnet = tot_len / (W * dt), np.degrees(tot_yaw)
  wrad = np.where(np.abs(tot_yaw) > 1e-3, tot_len / np.maximum(np.abs(tot_yaw), 1e-3), np.inf)
  wfwd = np.where(tot_len > 1e-6, tot_fwd / np.maximum(tot_len, 1e-6), 0)
  big = np.abs(wnet) >= 45
  windows = {"turn >= 45 deg in 1.2 s": {}}
  for name, m in {
      "in_place (speed<.2)": big & (wspeed < 0.2),
      "slow (.2-.6 m/s)": big & (wspeed >= .2) & (wspeed < .6),
      "walking forward r<1 m": big & (wspeed >= .6) & (wfwd >= .87) & (wrad < 1),
      "walking forward r 1-2 m": big & (wspeed >= .6) & (wfwd >= .87) & (wrad >= 1) & (wrad < 2),
      "walking forward r>=2 m": big & (wspeed >= .6) & (wfwd >= .87) & (wrad >= 2),
      "walking non-forward": big & (wspeed >= .6) & (wfwd < .87)}.items():
    windows["turn >= 45 deg in 1.2 s"][name] = {"upright": int((m & win_ok & ~win_h).sum()),
                                                "hunched": int((m & win_ok & win_h).sum()),
                                                "twisted": int((m & win_ok & win_t).sum()),
                                                "twisted_planted": int((m & win_ok & win_tp).sum())}
  per_hour = {
    "admissible_edges": {k: {c: v / hours for c, v in row.items()} for k, row in edges["admissible"].items()},
    "windows": {k: {c: v / hours for c, v in row.items()} for k, row in windows["turn >= 45 deg in 1.2 s"].items()},
  }
  ip_all = allowed & _classes(dt, speed, net, radius, fwd_frac)["in_place (speed<.2, turn>=25deg/s)"]
  pct = (50, 90, 99, 99.9)
  yaw_stats = {name: {f"p{q}": float(np.percentile(foot_yaw[m], q)) for q in pct} if m.any() else {}
               for name, m in (("admissible", allowed), ("admissible_in_place", ip_all))}
  return {"edge_seconds": dt, "streams": S, "motion_hours": hours, "edges": edges,
          "twist_deg": float(twist_deg), "foot_yaw_deg": yaw_stats,
          "inplace_admissible_turn_rates_deg_s": inplace_rates, "windows_3_edges": windows,
          "admissible_windows": int(win_ok.sum()), "admissible_edges": int(allowed.sum()), "per_motion_hour": per_hour,
          "notes": "radius = planar path length / |net yaw|; forward = travel within 30 deg of midpoint facing."}


def audit(g: MotionGraph, twist_deg: float = np.inf) -> dict:
  """Coverage of the whole graph and, for a merged graph, of each source library."""
  yaw = foot_yaw_deg(g)
  out = {"all": turn_coverage(g, twist_deg=twist_deg, foot_yaw=yaw)}
  start = 0
  for i, src in enumerate(g.meta.get("sources", [])):
    out[f"source{i}"] = {"library": src["library"], "command_source": src.get("command_source"),
                         **turn_coverage(g, slice(start, start + src["streams"]), twist_deg, yaw)}
    start += src["streams"]
  return out


def main(argv=None) -> None:
  from crowdmotion.core.paths import default_graph
  from crowdmotion.core.rundir import RunDir

  parser = argparse.ArgumentParser(prog="crowd graph turns", description=__doc__.split("\n", 1)[0])
  parser.add_argument("--graph", type=Path, nargs="+", help="saved graphs (default: $CROWD_GRAPH or graph-default)")
  parser.add_argument("--labels", nargs="+", help="one name per graph (default: graph0, graph1, ...)")
  parser.add_argument("--overrides", type=Path,
                      help='JSON graph overrides (a dict, or {"graph_overrides": {...}}), e.g. the flat-ground mask')
  parser.add_argument("--twist-deg", type=float,
                      help="foot-yaw threshold of a twisted edge (default: p99 over the first graph's admissible edges)")
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_turn_audit)")
  args = parser.parse_args(argv)
  graphs = args.graph or [default_graph()]
  labels = args.labels or [f"graph{i}" for i in range(len(graphs))]
  if len(labels) != len(graphs):
    raise ValueError("give one label per graph")
  overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
  overrides = overrides.get("graph_overrides", overrides)
  torch.set_num_threads(min(torch.get_num_threads(), 8))
  record = {"graphs": dict(zip(labels, map(str, graphs))), "graph_overrides": overrides,
            "scope": "turning radius coverage of admissible edges"}
  with RunDir.create("turn_audit", out=args.out, record=record) as run:
    result, twist = {}, args.twist_deg
    for label, path in zip(labels, graphs):
      g = MotionGraph.load(path, device="cpu", cfg_overrides=overrides)
      if twist is None:
        twist = float(np.percentile(foot_yaw_deg(g)[g.admissible_edges().cpu().numpy()], 99))
        run.record["twist_deg"] = {"value": twist, "from": label}
      result[label] = audit(g, twist)
      a = result[label]["all"]
      print(label, json.dumps({"admissible_edges": a["admissible_edges"], "windows_3_edges": a["windows_3_edges"],
                               "in_place": a["edges"]["admissible"]["in_place (speed<.2, turn>=25deg/s)"]}), flush=True)
    run.write_json("turn_audit.json", result)


if __name__ == "__main__":
  main()
