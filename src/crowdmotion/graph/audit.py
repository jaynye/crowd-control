"""Graph audit: what motion the saved tables contain and how edges connect.

Measures saved graph tables on CPU, without building kNN or loading a prior.
Composition counts weight every valid edge and its frames equally; they describe
availability, not online visitation.
"""

from __future__ import annotations

import numpy as np

from crowdmotion.core.geometry import wrap_np
from crowdmotion.metrics.locomotion import CLASSES, EDGE_CLASSES, classify, composition, motion_features


def audit_graph(blob: dict) -> dict:
  """Measure saved graph tables on CPU without building kNN or loading a prior."""
  tab = blob["tables"]
  root = tab["edge_root"].float().numpy()
  heading = tab["edge_yaw"].float().numpy()
  valid = tab["valid"].numpy()
  ids = np.flatnonzero(valid)
  speed, angle, _turn = motion_features(
    root, heading, blob["fps"], np.zeros((len(root), 2)), np.zeros(len(root))
  )
  labels = classify(speed, angle)
  votes = np.stack([(labels == i).sum(1) for i in range(len(CLASSES))], 1)
  dominant = votes.argmax(1)  # documented low-index tie break
  net_turn = np.abs(np.degrees(wrap_np(heading[:, -1])))
  edge_class = np.choose(dominant, [0, 2, 4, 5, 6]).astype(np.int8)
  edge_class[(dominant == 0) & (net_turn >= 10)] = 1
  edge_class[(dominant == 1) & (net_turn >= 10)] = 3
  cand = tab["cand"].numpy()
  K = blob["K"]
  B, k = cand.shape
  indegree = np.zeros(B, dtype=np.int64)
  cross_stream = own = count = 0
  matrix = np.zeros((len(EDGE_CLASSES), len(EDGE_CLASSES)), dtype=np.int64)
  availability = np.zeros_like(matrix)
  for start in range(0, len(ids), 4096):
    selected = ids[start : start + 4096]
    # An edge ends at boundary edge_id + 1; transitions leave that boundary.
    targets = cand[selected + 1]
    source_class = edge_class[selected]
    target_class = edge_class[targets]
    np.add.at(
      matrix, (np.broadcast_to(source_class[:, None], targets.shape), target_class), 1
    )
    for j in range(len(EDGE_CLASSES)):
      np.add.at(availability[:, j], source_class, (target_class == j).any(1))
    indegree += np.bincount((targets + 1).ravel(), minlength=B)
    cross_stream += int((selected[:, None] // K != targets // K).sum())
    own += int((targets == selected[:, None] + 1).sum())
    count += targets.size
  samples = []
  for cls in range(len(EDGE_CLASSES)):
    pool = ids[edge_class[ids] == cls]
    # Examples are deterministic, evenly spaced in edge id, not curated for appearance.
    chosen = (
      pool[np.linspace(0, len(pool) - 1, min(3, len(pool))).astype(int)]
      if len(pool)
      else []
    )
    for e in chosen:
      targets = cand[e + 1]
      samples.append(
        {
          "edge": int(e),
          "stream": int(e // K),
          "boundary": int(e % K),
          "class": int(cls),
          "root": root[e].round(4).tolist(),
          "heading": heading[e].round(4).tolist(),
          "speed": speed[e].round(3).tolist(),
          "angle": angle[e].round(2).tolist(),
          "candidates": [
            {
              "edge": int(t),
              "class": int(edge_class[t]),
              "distance_m": round(float(tab["cand_dist"][e + 1, j]), 4),
            }
            for j, t in enumerate(targets[:16])
          ],
        }
      )
  moving = speed[valid] >= 0.15
  return {
    "streams": blob["streams"],
    "boundaries": B,
    "boundaries_per_stream": K,
    "valid_edges": len(ids),
    "edge_seconds": blob["L"] / blob["fps"],
    "valid_motion_hours": len(ids) * blob["L"] / blob["fps"] / 3600,
    "fps": blob["fps"],
    "edge_frames": blob["L"],
    "neighbors": k,
    "stats": blob["stats"],
    "config": blob["cfg"],
    "meta": blob["meta"],
    "frame_composition": composition(labels[valid]),
    "edge_composition": composition(edge_class[valid], EDGE_CLASSES),
    "moving_forward_percent": 100
    * float((labels[valid] == 1).sum())
    / max(1, int(moving.sum())),
    "moving_facing_error_deg_p50_p95": np.percentile(
      np.abs(angle[valid][moving]), [50, 95]
    ).tolist(),
    "edge_net_turn_deg_p50_p95": np.percentile(net_turn[valid], [50, 95]).tolist(),
    "candidate_slots_all_boundaries": int(cand.size),
    "candidate_slots_after_valid_edges": int(count),
    "cross_stream_candidate_percent": 100 * cross_stream / count,
    "natural_continuation_candidate_percent": 100 * own / count,
    "reachable_end_boundaries": int((indegree > 0).sum()),
    "end_boundary_indegree_p50_p95_max": [
      *np.percentile(indegree[indegree > 0], [50, 95]).tolist(),
      int(indegree.max()),
    ],
    "transition_counts": matrix.tolist(),
    "availability_counts": availability.tolist(),
    "samples": samples,
    "definitions": {
      "still_mps": 0.15,
      "forward_deg": 30,
      "sideways_deg": [60, 120],
      "turning_net_deg_per_edge": 10,
      "edge_class": "Majority frame class; ties use the lowest class index. Forward/still edges split by net yaw >=10°.",
      "velocity": "Per-frame root difference, including predecessor, relative to midpoint ARDY body heading.",
      "scope": "Equal weighting of valid edges and their frames, not online visitation probabilities.",
    },
  }
