"""Skeleton facts shared by backends, libraries, graphs, metrics and capsule bodies."""

from __future__ import annotations

from dataclasses import dataclass

import torch

HEAD = "Head"
HUNCH_RATIO = 0.85  # head below this fraction of neutral standing head height counts as hunched


def neutral_head_height(skeleton, head: str = HEAD) -> float:
  """Head height above the lowest joint in an upstream skeleton's neutral pose."""
  joints = skeleton.neutral_joints
  return float(joints[skeleton.bone_index[head], 1] - joints[:, 1].min())


@dataclass
class SkeletonInfo:
  """Joint names, parents, foot joints (left chain then right) and the head joint.

  `head_idx` and `neutral_head_height` are None when the skeleton has no "Head".
  `rest_joints [J, 3]` is any posed frame (bone lengths are pose-invariant).
  """

  joint_names: list[str]
  parents: list[int]
  foot_idx: list[int]
  head_idx: int | None = None
  neutral_head_height: float | None = None
  rest_joints: torch.Tensor | None = None

  @classmethod
  def from_meta(cls, meta: dict, rest_joints: torch.Tensor | None = None) -> "SkeletonInfo":
    """From a library or graph `meta` dict."""
    return cls(
      joint_names=list(meta["joint_names"]),
      parents=list(meta["parents"]),
      foot_idx=list(meta["foot_idx"]),
      head_idx=meta.get("head_idx"),
      neutral_head_height=meta.get("neutral_head_height"),
      rest_joints=rest_joints,
    )

  @classmethod
  def from_upstream(cls, skeleton, joint_names: list[str]) -> "SkeletonInfo":
    """From an ARDY or Kimodo skeleton object."""
    info = cls(
      joint_names=list(joint_names),
      parents=skeleton.joint_parents.tolist(),
      foot_idx=list(skeleton.left_foot_joint_idx) + list(skeleton.right_foot_joint_idx),
    )
    try:
      info.head_idx = skeleton.bone_index[HEAD]
      info.neutral_head_height = neutral_head_height(skeleton)
    except (AttributeError, KeyError):
      pass
    return info

  def as_dict(self) -> dict:
    """The historical `skeleton_info()` dict (head keys only when known)."""
    out = {"parents": list(self.parents), "foot_idx": list(self.foot_idx), "joint_names": list(self.joint_names)}
    if self.head_idx is not None:
      out["head_idx"] = self.head_idx
      out["neutral_head_height"] = self.neutral_head_height
    return out
