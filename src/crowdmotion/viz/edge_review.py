"""Audit posture/flat-ground measurements and draw sampled saved-edge contact sheets."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from crowdmotion.core.paths import default_graph
from crowdmotion.core.provenance import file_sha256
from crowdmotion.core.rundir import RunDir
from crowdmotion.graph.annotations import annotate_edges
from crowdmotion.graph.tables import MotionGraph
from crowdmotion.metrics.locomotion import EDGE_CLASSES, classify, motion_features


def distribution(x):
  return {str(q): float(np.percentile(x, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


def edge_classes(graph):
  root, yaw = graph.edge_root.cpu().numpy(), graph.edge_yaw.cpu().numpy()
  speed, angle, _ = motion_features(root, yaw, graph.fps, np.zeros((len(root), 2)), np.zeros(len(root)))
  label = classify(speed, angle)
  dominant = np.stack([(label == i).sum(1) for i in range(5)], 1).argmax(1)
  cls = np.choose(dominant, [0, 2, 4, 5, 6])
  turn = np.abs(np.degrees(yaw[:, -1])) >= 10
  cls[(dominant == 0) & turn] = 1
  cls[(dominant == 1) & turn] = 3
  return cls, speed.mean(1)


def draw_sheet(graph, ids, values, title, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  ncols, nrows = 4, max(1, (len(ids) + 3) // 4)
  fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.3 * nrows), squeeze=False)
  parents = graph.meta["parents"]
  records = []
  for ax, e in zip(axes.flat, ids):
    p = graph.edge_pose[e].cpu().numpy()
    # Local body poses show intrinsic posture; root displacement omitted explicitly.
    for k, color in zip((0, graph.L // 2, graph.L - 1), ("#adc8e8", "#548bc5", "#182c48")):
      for j, parent in enumerate(parents):
        if parent >= 0:
          ax.plot(p[k, [parent, j], 2], p[k, [parent, j], 1], color=color, lw=1.7)
      ax.scatter(p[k, graph.foot_idx, 2], p[k, graph.foot_idx, 1], color=color, s=8)
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlim=(-.8, 1.0), ylim=(-.12, max(2.05, float(p[..., 1].max()) + .08)),
           xlabel="body forward Z (m)", ylabel="ground height Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)
    pitch, ratio = values["trunk_pitch_deg"][e], values["head_ratio"][e]
    ax.set_title(f"edge {e}, stream {e // graph.K}\npitch {pitch:.1f}°, head {ratio:.3f}\n"
                 f"pelvis range {values['pelvis_range_m'][e]:.3f}m; "
                 f"low foot {values['lowest_foot_min_m'][e]:.3f}-{values['lowest_foot_max_m'][e]:.3f}m", fontsize=9)
    records.append({"edge": int(e), "stream": int(e // graph.K),
                    **{k: float(v[e]) for k, v in values.items()}})
  for ax in list(axes.flat)[len(ids):]:
    ax.axis("off")
  fig.suptitle(title + "\nSide projection, body-local XZ; Y absolute. Frames 1/mid/last: light to dark. No root translation.")
  fig.tight_layout(rect=(0, 0, 1, .94))
  fig.savefig(path, dpi=125)
  plt.close(fig)
  return records


def draw_twist_sheet(graph, ids, path):
  """Top views retain root displacement and yaw so planted feet are visible."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  from crowdmotion.core.geometry import body_world
  from crowdmotion.graph.turns import foot_yaw_deg

  names = graph.meta["joint_names"]
  feet = [names.index(n) for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
  hips = [names.index(n) for n in ("LeftUpLeg", "RightUpLeg")]
  yaw = foot_yaw_deg(graph)
  fig, axes = plt.subplots(max(1, (len(ids) + 3) // 4), 4,
                           figsize=(16, 4.4 * max(1, (len(ids) + 3) // 4)), squeeze=False)
  for ax, e in zip(axes.flat, ids):
    p = body_world(graph.edge_pose[e], graph.edge_root[e], graph.edge_yaw[e]).cpu().numpy()
    for k, color in zip((0, graph.L // 2, graph.L - 1), ("#abc9e9", "#548bc5", "#182c48")):
      for indices in (hips, feet[:2], feet[2:]):
        ax.plot(p[k, indices, 0], p[k, indices, 2], color=color, lw=3)
      center = p[k, hips].mean(0)
      heading = float(graph.edge_yaw[e, k])
      ax.arrow(center[0], center[2], .22 * np.sin(heading), .22 * np.cos(heading),
               color=color, width=.006, length_includes_head=True)
      ax.scatter(p[k, feet, 0], p[k, feet, 2], c=[color] * 4, s=20)
    contacts = graph.edge_contacts[e].float().mean().item()
    lifts = graph.edge_pose[e, :, feet, 1].amax(0).cpu().numpy()
    ax.set_title(f"edge {e}; foot/body yaw {yaw[e]:.1f}°\ncontact labels {contacts:.0%}; "
                 f"foot peaks {lifts[0]:.2f}/{lifts[2]:.2f}m", fontsize=9)
    ax.set(xlim=(-.65, .65), ylim=(-.65, .65), xlabel="predecessor X (m)", ylabel="predecessor Z (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
  for ax in list(axes.flat)[len(ids):]:
    ax.axis("off")
  fig.suptitle("Requested edges: feet and hip line, with body-forward arrows\n"
               "Frames first/mid/last: light to dark. Native root motion and yaw retained.")
  fig.tight_layout(rect=(0, 0, 1, .94), h_pad=3.5)
  fig.savefig(path, dpi=125)
  plt.close(fig)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--graph", type=Path)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--config", type=Path, help="behavior config supplying opt-in graph overrides")
  parser.add_argument("--edges", type=int, nargs="+", help="also draw these specific saved edge IDs")
  args = parser.parse_args(argv)
  torch.set_num_threads(1)
  path = args.graph or default_graph()
  with RunDir.create("edge_review", out=args.out,
                     record={"graph_path": str(Path(path).resolve()), "graph_sha256": file_sha256(path)}) as run:
    overrides = json.loads(args.config.read_text())["graph_overrides"] if args.config else {}
    graph = MotionGraph.load(path, device="cpu", cfg_overrides=overrides)
    run.update(graph_overrides=overrides)
    ann = annotate_edges(graph)
    values = {f.name: getattr(ann, f.name).cpu().numpy() for f in fields(ann)}
    twist = graph.ensure_twist_annotations()
    values["foot_yaw_deg"] = twist.foot_yaw_deg.cpu().numpy()
    valid = graph.valid.numpy()
    ids = np.flatnonzero(valid)
    cls, speed = edge_classes(graph)
    pitch = values["trunk_pitch_deg"]
    keys = ("head_ratio", "trunk_pitch_deg", "trunk_inclination_deg", "head_forward_m")
    means = np.zeros(graph.streams)
    for stream in range(graph.streams):
      pool = ids[ids // graph.K == stream]
      means[stream] = pitch[pool].mean() if len(pool) else 0
    between = np.var(means[ids // graph.K]) / np.var(pitch[ids])
    sweep = []
    for cutoff in (10, 15, 18, 20, 25):
      row = {"pitch_cutoff_deg": cutoff, "hunched_percent": 100 * float((pitch[valid] >= cutoff).mean()), "joint_motion_counts": {}}
      for label, name in enumerate(EDGE_CLASSES):
        row["joint_motion_counts"][name] = {style: int((valid & (cls == label) & mask).sum())
          for style, mask in (("upright", pitch < cutoff), ("hunched", pitch >= cutoff))}
      sweep.append(row)
    masks = []
    cand = graph.cand.numpy()
    retained = graph.admissible_edges().numpy()
    joint = {}
    for style, posture in (("upright", pitch < graph.cfg.posture_cutoff_deg),
                            ("hunched", pitch >= graph.cfg.posture_cutoff_deg)):
      eligible = retained & posture
      starts = eligible & ((values["start_pitch_deg"] >= graph.cfg.posture_cutoff_deg) == (style == "hunched"))
      after = np.flatnonzero(eligible) + 1
      candidates = cand[after]
      joint[style] = {
        "edges": int(eligible.sum()), "class_matched_starts": int(starts.sum()),
        "slow_class_matched_starts": int((starts & (graph.root_speed.numpy() < .15)).sum()),
        "motion_counts": {name: int((eligible & (cls == i)).sum()) for i, name in enumerate(EDGE_CLASSES)},
        "no_same_class_candidate_after_edge": int((~eligible[candidates].any(1)).sum()),
        "has_same_class_forward_turn_percent": 100 * float((eligible[candidates] & (cls[candidates] == 3)).any(1).mean()),
        "natural_successor_same_class_percent": 100 * float(eligible[after].mean()),
      }
    for pelvis in (.10, .15, .20):
      for airborne in (.12, .18):
        ok = valid & (values["pelvis_range_m"] <= pelvis) & (values["lowest_foot_min_m"] <= airborne) & (values["foot_max_m"] <= .60)
        dead = ~ok[cand].any(1)
        masks.append({"pelvis_range_max_m": pelvis, "lowest_foot_min_max_m": airborne, "foot_max_m": .60,
                      "edges_removed": int((valid & ~ok).sum()), "boundaries_with_removed_candidate": int((~ok[cand]).any(1).sum()),
                      "dead_end_boundaries_all": int(dead.sum()), "dead_end_after_valid_edge": int(dead[ids + 1].sum()),
                      "dead_end_after_retained_edge": int(dead[np.flatnonzero(ok) + 1].sum())})
    examples = {}
    groups = {
      "pitch_under10": valid & (pitch < 10),
      "pitch10to18": valid & (pitch >= 10) & (pitch < 18),
      "pitch18to25": valid & (pitch >= 18) & (pitch < 25),
      "pitch_over25": valid & (pitch >= 25),
      "low_head_low_pitch": valid & (values["head_ratio"] < .90) & (pitch < 15),
      "high_head_high_pitch": valid & (values["head_ratio"] >= .93) & (pitch >= 20),
      "pelvis_range_over015": valid & (values["pelvis_range_m"] > .15),
      "persistent_foot_over018": valid & (values["lowest_foot_min_m"] > .18),
      "foot_over060": valid & (values["foot_max_m"] > .60),
      "brief_foot_over012": valid & (values["lowest_foot_max_m"] > .12) & (values["lowest_foot_min_m"] <= .12),
    }
    for name, mask in groups.items():
      pool = np.flatnonzero(mask)
      chosen = pool[np.linspace(0, len(pool) - 1, min(8, len(pool))).astype(int)] if len(pool) else []
      examples[name] = draw_sheet(graph, chosen, values, name, run.path / f"{name}.png")
    if args.edges:
      if any(e < 0 or e >= len(valid) for e in args.edges):
        raise ValueError("Requested edge ID outside saved tables")
      examples["requested"] = draw_sheet(graph, args.edges, values, "Requested edges", run.path / "requested.png")
      draw_twist_sheet(graph, args.edges, run.path / "requested_top.png")
    dead = ~retained[cand].any(1)
    before_twist = graph.admissible_edges(replace(graph.cfg, twist_mask=False)).numpy()
    twist_removed = before_twist & ~retained
    before_dead = ~before_twist[cand].any(1)
    report = {
      "valid_edges": len(ids), "neutral_head_height_m": graph.meta["neutral_head_height"],
      "height_convention": "body_local copies joints Y unchanged; edge_pose and pred_pose Y are absolute above ground, not pelvis-relative.",
      "distributions": {k: distribution(v[valid]) for k, v in values.items()},
      "head_ratio_below090_percent": 100 * float((values["head_ratio"][valid] < .90).mean()),
      "correlation_names": [*keys, "speed"],
      "correlations": np.corrcoef(np.stack([*[values[k][valid] for k in keys], speed[valid]])).tolist(),
      "pitch_between_stream_variance_fraction": float(between), "posture_cutoff_sweep": sweep,
      "vertical_mask_sweep": masks, "review_examples": examples,
      "joint_coverage_with_configured_mask": joint,
      "configured_mask_counts": {
        "edges_removed": int((valid & ~retained).sum()),
        "boundaries_with_removed_candidate": int((~retained[cand]).any(1).sum()),
        "dead_end_boundaries_all": int(dead.sum()),
        "dead_end_after_valid_edge": int(dead[ids + 1].sum()),
        "dead_end_after_retained_edge": int(dead[np.flatnonzero(retained) + 1].sum()),
      },
      "twist_mask_counts": {
        "enabled": graph.cfg.twist_mask, "threshold_deg": graph.cfg.twist_max_foot_yaw_deg,
        "planted_definition": "every foot-contact label on at every edge frame",
        "edges_removed_after_other_masks": int(twist_removed.sum()),
        "boundaries_with_removed_candidate": int(twist_removed[cand].any(1).sum()),
        "new_dead_end_boundaries": int((dead & ~before_dead).sum()),
        "dead_end_after_retained_edge": int(dead[np.flatnonzero(retained) + 1].sum()),
        "by_motion_class": {
          name: {"removed_edges": int((twist_removed & (cls == i)).sum()),
                 "affected_boundaries": int((twist_removed & (cls == i))[cand].any(1).sum())}
          for i, name in enumerate(EDGE_CLASSES)
        },
      },
      "definitions": "Signed sagittal Hips-to-Neck pitch; mean over edge frames. Inclination includes lateral lean. "
                     "Height extrema include predecessor. Motion: majority frame class, >=10deg native net yaw splits turns. "
                     "Cutoffs are diagnostic candidates, not calibrated posture or acceptance thresholds. Samples evenly spaced by edge ID.",
    }
    run.write_json("summary.json", report)
    np.savez_compressed(run.path / "annotations.npz", **values, valid=valid, motion_class=cls, speed_mps=speed)
  print(run.path)
