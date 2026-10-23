"""Derived edge measurements; never persisted into or used to rebuild a graph.

Body-local tables remove planar translation/yaw only: Y is absolute ground height.
Posture measurements describe geometry, not a calibrated human posture classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TwistAnnotations:
  foot_yaw_deg: torch.Tensor
  all_feet_planted: torch.Tensor


@torch.no_grad()
def annotate_twist(graph, chunk: int = 8192) -> TwistAnnotations:
  """Saved-table definition shared with the turning audit, including held twists.

  Mean absolute angle of the normalized mean foot direction relative to body +Z.
  Every contact label must be on at every edge frame to call the edge planted.
  This conservative label test deliberately keeps stepping turns with foot/body lag.
  """
  names = list(graph.meta["joint_names"])
  required = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
  if any(n not in names for n in required):
    raise ValueError(f"Twist annotations require named joints {required}")
  idx = [names.index(n) for n in required]
  yaw = []
  for lo in range(0, graph.edge_pose.shape[0], chunk):
    p = graph.edge_pose[lo:lo + chunk][..., idx, :].float()
    v = 0
    for foot, toe in ((0, 1), (2, 3)):
      d = p[..., toe, :][..., [0, 2]] - p[..., foot, :][..., [0, 2]]
      v = v + d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    yaw.append(torch.rad2deg(torch.atan2(v[..., 0], v[..., 1])).abs().mean(-1))
  return TwistAnnotations(torch.cat(yaw), graph.edge_contacts.all(dim=(1, 2)))


@dataclass
class EdgeAnnotations:
  head_ratio: torch.Tensor
  trunk_pitch_deg: torch.Tensor
  trunk_inclination_deg: torch.Tensor
  head_forward_m: torch.Tensor
  pelvis_range_m: torch.Tensor
  pelvis_min_m: torch.Tensor
  pelvis_max_m: torch.Tensor
  lowest_foot_min_m: torch.Tensor
  lowest_foot_max_m: torch.Tensor
  foot_max_m: torch.Tensor
  start_pitch_deg: torch.Tensor


def posture_pitch(pose: torch.Tensor, hips: int, neck: int) -> torch.Tensor:
  """Signed sagittal Hips-to-Neck pitch from vertical; +Z is body forward."""
  trunk = pose[..., neck, :] - pose[..., hips, :]
  return torch.rad2deg(torch.atan2(trunk[..., 2], trunk[..., 1]))


@torch.no_grad()
def annotate_edges(graph) -> EdgeAnnotations:
  """Measure saved poses in chunks to bound temporary GPU/CPU memory.

  Height extrema include the predecessor, making edge entry jumps observable.
  Posture means use executed edge frames; starts use the predecessor separately.
  """
  names = graph.meta["joint_names"]
  if "Hips" not in names or "Neck" not in names:
    raise ValueError("Posture annotations require named Hips and Neck joints")
  hips, neck = names.index("Hips"), names.index("Neck")
  head, feet = graph.meta["head_idx"], graph.foot_idx
  rows = {k: [] for k in EdgeAnnotations.__dataclass_fields__}
  for lo in range(0, len(graph.valid), 4096):
    p = graph.edge_pose[lo:lo + 4096]
    prev = graph.pred_pose[lo:lo + 4096]
    trunk = p[..., neck, :] - p[..., hips, :]
    heights = torch.cat((prev[:, None, hips, 1], p[..., hips, 1]), 1)
    foot = torch.cat((prev[:, None, feet, 1], p[..., feet, 1]), 1)
    lowest = foot.amin(-1)
    vals = dict(
      head_ratio=p[..., head, 1].mean(1) / graph.meta["neutral_head_height"],
      trunk_pitch_deg=posture_pitch(p, hips, neck).mean(1),
      trunk_inclination_deg=torch.rad2deg(torch.atan2(trunk[..., [0, 2]].norm(dim=-1), trunk[..., 1])).mean(1),
      head_forward_m=(p[..., head, 2] - p[..., hips, 2]).mean(1),
      pelvis_range_m=heights.amax(1) - heights.amin(1),
      pelvis_min_m=heights.amin(1), pelvis_max_m=heights.amax(1),
      lowest_foot_min_m=lowest.amin(1), lowest_foot_max_m=lowest.amax(1),
      foot_max_m=foot.amax((1, 2)),
      start_pitch_deg=posture_pitch(prev, hips, neck),
    )
    for key, value in vals.items():
      rows[key].append(value)
  return EdgeAnnotations(**{k: torch.cat(v) for k, v in rows.items()})
