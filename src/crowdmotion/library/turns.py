"""Turn tracking, feet and ground checks of a turning library (`crowd library turns`).

  crowd library turns --library <library.pt> [<library.pt> ...] [--labels a b ...] \\
      [--overrides <json>] [--clips 6] [--out runs/<UTC>_turn_report]

For libraries generated with `command_source: "turns"` (per-edge `cmd_*` keys and
`heading_target`, see `library.generate.save_library`); a random-command library gets
the walk/stand rows only, as a reference. Per library:

- heading: the generator's returned heading (`heading`, decoded from its heading
  feature) against the dense heading reference, and both against the hip-line
  heading measured from the posed joints, which does not read that feature;
- segments: executed against commanded turn of in-place and arc segments (hip-line
  heading from the frame before the segment to `lag_s` after it), root drift (from
  `lag_s` into the segment, split by whether the human was at rest before), foot turn
  and lift-offs during in-place turns, executed arc radius and forward travel (from
  `lag_s` into the arc);
- feet: label-free and contact-labeled planted-foot speed per segment kind
  (`metrics.feet` definitions);
- ground: per-edge validity (hunch filter), flat-ground admissibility and posture
  class under the graph config and overrides, with the annotations and mask of
  `MotionGraph.admissible_edges`, per segment kind.

Writes `turn_report.json`; with `--clips N`, `clips_in_place_<label>.npz` of N
in-place turns for `crowd bench clips`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crowdmotion.behavior.commands import IN_PLACE, STAND, TURN_KINDS
from crowdmotion.core.geometry import body_local
from crowdmotion.core.rundir import RunDir
from crowdmotion.graph.annotations import annotate_edges
from crowdmotion.graph.config import GraphConfig
from crowdmotion.graph.tables import MotionGraph

ARC = TURN_KINDS.index("arc")
TWIST_TOE_RATIO = 0.4  # an in-place turn whose feet turn less than this share of the command is a twist


def wrap_deg(a):
  return (np.asarray(a) + 180.0) % 360.0 - 180.0


def line_heading(right: np.ndarray, left: np.ndarray) -> np.ndarray:
  """Facing from a right-minus-left body line (0 faces +Z, positive toward +X)."""
  d = right - left
  return np.arctan2(d[..., 2], -d[..., 0])


def toe_heading(joints: np.ndarray, names: list[str]) -> np.ndarray:
  """Mean foot-to-toe direction of both feet."""
  v = 0
  for side in ("Left", "Right"):
    d = joints[..., names.index(f"{side}ToeBase"), :] - joints[..., names.index(f"{side}Foot"), :]
    d = np.stack((d[..., 0], d[..., 2]), -1)
    v = v + d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
  return np.arctan2(v[..., 0], v[..., 1])


def stats(x, qs=(10, 50, 90)) -> dict:
  x = np.asarray(x, dtype=np.float64)
  x = x[np.isfinite(x)]
  if not len(x):
    return {"n": 0}
  return {"n": int(len(x)), "mean": float(x.mean()), **{f"p{q}": float(np.percentile(x, q)) for q in qs}}


def frac_above(x, limit) -> float:
  x = x[np.isfinite(x)]
  return float((x > limit).mean()) if len(x) else float("nan")


class _EdgeTables:
  """The attributes `annotate_edges` and `MotionGraph.admissible_edges` read."""

  def __init__(self, meta, edge_pose, pred_pose, valid, cfg):
    self.meta, self.edge_pose, self.pred_pose, self.valid, self.cfg = meta, edge_pose, pred_pose, valid, cfg
    self.foot_idx = list(meta["foot_idx"])
    self.annotations = None

  def ensure_annotations(self):
    if self.annotations is None:
      self.annotations = annotate_edges(self)
    return self.annotations


def ground(lib: dict, cfg: GraphConfig, chunk: int = 256) -> dict:
  """Per library edge e >= 1 (`[N, E]`, edge 0 False): validity, admissibility, hunched class."""
  meta = lib["meta"]
  L = int(meta["edge_frames"])
  n, t = lib["posed_joints"].shape[:2]
  E = t // L
  out = {k: np.zeros((n, E), bool) for k in ("valid", "admissible", "hunched")}
  for s in range(0, n, chunk):
    j = lib["posed_joints"][s:s + chunk, :E * L].float()
    root = lib["root_pos"][s:s + chunk, :E * L, [0, 2]].float()
    h = lib["heading"][s:s + chunk, :E * L].float()
    c = j.shape[0]
    local = body_local(j, root, h)  # [c, T, J, 3]
    edge = local.view(c, E, L, *local.shape[2:])[:, 1:]  # library edges 1.. (graph edges 0..)
    pred = local[:, L - 1:(E - 1) * L:L]  # the frame before each of them
    head = edge[..., meta["head_idx"], 1].mean(-1)
    valid = torch.isfinite(edge).all(dim=(-1, -2, -3))
    if cfg.hunch_ratio > 0:
      valid &= head >= cfg.hunch_ratio * meta["neutral_head_height"]
    tables = _EdgeTables(meta, edge.flatten(0, 1), pred.flatten(0, 1), valid.flatten(), cfg)
    allowed = MotionGraph.admissible_edges(tables, cfg).view(c, E - 1)
    pitch = tables.ensure_annotations().trunk_pitch_deg.view(c, E - 1)
    out["valid"][s:s + c, 1:] = valid.numpy()
    out["admissible"][s:s + c, 1:] = allowed.numpy()
    out["hunched"][s:s + c, 1:] = (pitch >= cfg.posture_cutoff_deg).numpy()
  return out


def analyze(lib: dict, cfg: GraphConfig, lag_s: float = 0.5) -> tuple[dict, list[tuple]]:
  """The report for one turning library, and its in-place segments for clips."""
  meta = lib["meta"]
  names = list(meta["joint_names"])
  L, fps = int(meta["edge_frames"]), float(meta["fps"])
  joints = lib["posed_joints"].float().numpy()
  n, T = joints.shape[:2]
  E = T // L
  mode = lib["cmd_mode"].numpy()
  heading = lib["heading"].float().numpy()
  turning = "cmd_segment" in lib  # else a random-command library: walk/stand feet, ground, heading only
  seg = lib["cmd_segment"].numpy() if turning else np.zeros_like(mode)
  angle = np.degrees(lib["cmd_angle"].numpy()) if turning else np.zeros(mode.shape)
  radius = lib["cmd_radius"].numpy() if turning else np.full(mode.shape, np.nan)
  ref = lib["heading_target"].float().numpy() if turning else np.full(heading.shape, np.nan)
  hip = line_heading(joints[..., names.index("RightUpLeg"), :], joints[..., names.index("LeftUpLeg"), :])
  hip_u, feat_u = np.unwrap(hip, axis=1), np.unwrap(heading, axis=1)
  toe_u = np.unwrap(toe_heading(joints, names), axis=1)
  xz = lib["root_pos"].float().numpy()[..., [0, 2]]
  kind_f = np.repeat(mode, L, axis=1)[:, :T]

  report = {"turning": turning, "streams": n, "frames": T, "motion_hours": n * T / fps / 3600,
            "edge_kind_fraction": {k: float((mode == i).mean()) for i, k in enumerate(TURN_KINDS)}}

  # Heading agreement per frame kind (degrees).
  pairs = {"returned_vs_reference": wrap_deg(np.degrees(heading - ref)),
           "hip_vs_reference": wrap_deg(np.degrees(hip - ref)),
           "returned_vs_hip": wrap_deg(np.degrees(heading - hip))}
  report["heading_abs_error_deg"] = {
    k: {name: {**stats(np.abs(v[kind_f == i]), (50, 95)), "frac_gt_20": frac_above(np.abs(v[kind_f == i]), 20)}
        for name, v in pairs.items()}
    for i, k in enumerate(TURN_KINDS) if (kind_f == i).any()}

  # Feet: label-free planted frames (within 2.5 cm of the joint's own 5th percentile height).
  fidx = list(meta["foot_idx"])
  feet = joints[:, :, fidx]
  h = feet[..., 1]
  planted = h < np.percentile(h, 5, axis=1, keepdims=True) + 0.025
  both = planted[:, 1:] & planted[:, :-1]
  fspeed = np.linalg.norm(np.diff(feet[..., [0, 2]], axis=1), axis=-1) * fps  # [N, T-1, F]
  contacts = lib["contacts"].numpy().astype(bool)
  cboth = contacts[:, 1:] & contacts[:, :-1]
  kf = kind_f[:, 1:, None].repeat(len(fidx), -1)
  report["planted_foot_speed_mps"] = {}
  for i, k in enumerate(TURN_KINDS):
    rows = {}
    for name, mask in (("label_free", both), ("contact_labeled", cboth)):
      v = fspeed[mask & (kf == i)]
      rows[name] = {**stats(v, (50, 95)), "frac_gt_0p1": float((v > 0.1).mean()) if len(v) else float("nan")}
    report["planted_foot_speed_mps"][k] = rows
  liftoff = planted[:, :-1] & ~planted[:, 1:]  # [N, T-1, F]

  # Segments: runs of one `cmd_segment` value; skip the first and last of each stream.
  lag = int(round(lag_s * fps))
  seg_rows = {"in_place": [], "arc": []}
  clips = []
  for i in range(n):
    cut = np.flatnonzero(np.diff(seg[i])) + 1
    for e0, e1 in zip(np.r_[0, cut], np.r_[cut, E] - 1):
      k = mode[i, e0]
      if e0 == 0 or e1 >= E - 1 or k not in (IN_PLACE, ARC):
        continue
      f0, f1 = e0 * L - 1, (e1 + 1) * L - 1
      fl = min(f1 + lag, T - 1)
      late = min(f1 + lag + int(fps), T - 1)
      row = {"cmd_turn_deg": float(angle[i, e0]), "seconds": (f1 - f0) / fps,
             "from_rest": bool(mode[i, e0 - 1] in (STAND, IN_PLACE)),
             "hip_turn_deg": float(np.degrees(hip_u[i, fl] - hip_u[i, f0])),
             "returned_turn_deg": float(np.degrees(feat_u[i, fl] - feat_u[i, f0])),
             "toe_turn_deg": float(np.degrees(toe_u[i, fl] - toe_u[i, f0])),
             "toe_turn_late_deg": float(np.degrees(toe_u[i, late] - toe_u[i, f0]))}
      w0 = min(f0 + lag, f1)
      step = np.diff(xz[i, w0:f1 + 1], axis=0)
      if k == IN_PLACE:
        row["drift_m"] = float(np.linalg.norm(xz[i, w0:f1 + 1] - xz[i, w0], axis=-1).max())
        row["mean_speed_mps"] = float(np.linalg.norm(np.diff(xz[i, f0:f1 + 1], axis=0), axis=-1).sum() / max(f1 - f0, 1)
                                      * fps)
        row["liftoffs"] = int(liftoff[i, f0:fl].sum())
        if abs(row["cmd_turn_deg"]) >= 90:
          clips.append((i, f0, row["from_rest"]))
      else:
        length = float(np.linalg.norm(step, axis=-1).sum())
        turn = float(hip_u[i, f1] - hip_u[i, w0])
        facing = np.stack((np.sin(hip_u[i, w0:f1]), np.cos(hip_u[i, w0:f1])), -1)
        row["cmd_radius_m"] = float(radius[i, e0])
        row["exec_radius_m"] = length / abs(turn) if abs(turn) > np.radians(5) else float("nan")
        row["forward_fraction"] = float((step * facing).sum() / length) if length > 1e-6 else float("nan")
      seg_rows["in_place" if k == IN_PLACE else "arc"].append(row)

  def twist_frac(rows):
    late = ratio(rows, "toe_turn_late_deg")
    return float(np.mean(np.array(late) < TWIST_TOE_RATIO)) if late else float("nan")

  def ratio(rows, key):
    return [r[key] / r["cmd_turn_deg"] for r in rows if abs(r["cmd_turn_deg"]) > 1e-6]

  ip, arc = seg_rows["in_place"], seg_rows["arc"]
  report["in_place"] = {
    "segments": len(ip),
    "hip_turn_ratio": stats(ratio(ip, "hip_turn_deg")),
    "returned_turn_ratio": stats(ratio(ip, "returned_turn_deg")),
    "toe_turn_ratio": stats(ratio(ip, "toe_turn_deg")),
    "toe_turn_ratio_late": stats(ratio(ip, "toe_turn_late_deg")),
    # Twist: the feet have turned less than 40% of the command 1 s after the hip-line check.
    "twist_frac": twist_frac(ip),
    "twist_frac_from_rest": twist_frac([r for r in ip if r["from_rest"]]),
    "twist_frac_from_motion": twist_frac([r for r in ip if not r["from_rest"]]),
    "hip_turn_error_abs_deg": stats([abs(r["hip_turn_deg"] - r["cmd_turn_deg"]) for r in ip]),
    "drift_m_from_rest": stats([r["drift_m"] for r in ip if r["from_rest"]]),
    "drift_m_from_motion": stats([r["drift_m"] for r in ip if not r["from_rest"]]),
    "mean_speed_mps": stats([r["mean_speed_mps"] for r in ip]),
    "liftoffs_per_segment": stats([r["liftoffs"] for r in ip]),
    "frac_with_liftoff": float(np.mean([r["liftoffs"] > 0 for r in ip])) if ip else float("nan"),
  }
  report["arc"] = {
    "segments": len(arc),
    "hip_turn_ratio": stats(ratio(arc, "hip_turn_deg")),
    "returned_turn_ratio": stats(ratio(arc, "returned_turn_deg")),
    "radius_ratio": stats([r["exec_radius_m"] / r["cmd_radius_m"] for r in arc]),
    "forward_fraction": stats([r["forward_fraction"] for r in arc]),
  }

  g = ground(lib, cfg)
  report["ground"] = {}
  for i, k in enumerate(TURN_KINDS):
    m = mode == i
    m[:, 0] = False
    if not m.any():
      continue
    adm = g["admissible"] & m
    report["ground"][k] = {
      "edges": int(m.sum()), "valid": float(g["valid"][m].mean()), "admissible": float(g["admissible"][m].mean()),
      "admissible_hunched": float(g["hunched"][adm].mean()) if adm.any() else float("nan"),
      "admissible_per_motion_hour": float(adm.sum() / report["motion_hours"]),
    }
  pelvis = joints[..., names.index("Hips"), 1]
  report["pelvis_height_m"] = {k: stats(pelvis[kind_f == i], (5, 50, 95)) for i, k in enumerate(TURN_KINDS)
                               if (kind_f == i).any()}
  return report, clips


def write_clips(path: Path, lib: dict, clips: list[tuple], count: int, frames: int = 80) -> None:
  """Stick-figure clip file (`viz.stick` layout) of `count` in-place turns of at least 90
  degrees: half entered from rest and half from motion, evenly spaced over each list."""
  meta = lib["meta"]
  L = int(meta["edge_frames"])
  T = lib["posed_joints"].shape[1]
  picked = []
  for rest, want in ((True, (count + 1) // 2), (False, count // 2)):
    pool = [(i, f0 - 10) for i, f0, r in clips if r == rest and f0 - 10 >= 0 and f0 - 10 + frames <= T]
    if pool and want:
      picked += [pool[k] for k in np.linspace(0, len(pool) - 1, min(want, len(pool))).round().astype(int)]
  if not picked:
    return
  joints = np.stack([lib["posed_joints"][i, s:s + frames].float().numpy() for i, s in picked])
  boundaries = [f for f in range(frames) if (picked[0][1] + f) % L == 0]
  np.savez_compressed(
    path, joints=joints, planned=joints[:, :, 0, [0, 2]],
    contacts=np.stack([lib["contacts"][i, s:s + frames].numpy() for i, s in picked]),
    joint_names=np.array(meta["joint_names"]),
    parents=np.array(meta["parents"]), fps=float(meta["fps"]), boundaries=np.array(boundaries),
    picked=np.array(picked))


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(prog="crowd library turns", description=__doc__.split("\n", 1)[0])
  parser.add_argument("--library", type=Path, nargs="+", required=True)
  parser.add_argument("--labels", nargs="+", help="one name per library (default: parent directory names)")
  parser.add_argument("--config", type=Path, default=Path("configs/graph/default.json"))
  parser.add_argument("--overrides", type=Path,
                      help='JSON graph overrides (a dict, or {"graph_overrides": {...}}) for the ground checks')
  parser.add_argument("--lag-s", type=float, default=0.5, help="seconds after a segment its turn may complete")
  parser.add_argument("--clips", type=int, default=0, help="write this many in-place turn clips per library")
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_turn_report)")
  args = parser.parse_args(argv)
  labels = args.labels or [p.parent.name for p in args.library]
  if len(labels) != len(args.library):
    raise ValueError("give one label per library")
  overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
  overrides = overrides.get("graph_overrides", overrides)
  cfg = GraphConfig.from_dict({**json.loads(args.config.read_text()), **overrides})
  torch.set_num_threads(min(torch.get_num_threads(), 8))
  record = {"libraries": dict(zip(labels, map(str, args.library))), "config": str(args.config),
            "overrides": overrides, "lag_s": args.lag_s}
  with RunDir.create("turn_report", out=args.out, record=record) as run:
    result = {}
    for label, path in zip(labels, args.library):
      lib = torch.load(path, map_location="cpu", weights_only=False)
      result[label], clips = analyze(lib, cfg, args.lag_s)
      if args.clips:
        write_clips(run.path / f"clips_in_place_{label}.npz", lib, clips, args.clips)
      print(label, json.dumps({k: result[label][k] for k in ("in_place", "arc")}), flush=True)
    run.write_json("turn_report.json", result)


if __name__ == "__main__":
  main()
