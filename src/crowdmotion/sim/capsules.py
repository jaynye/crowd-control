"""Kinematic capsule humans for MuJoCo, posed from Core-skeleton joint positions.

Each bone (parent joint -> child joint) is one mocap body carrying one capsule,
plus a head capsule extending the neck-to-head direction. Capsule lengths come
from the skeleton's rest bone lengths, so posing only needs per-bone midpoints
and orientations. Poses take MuJoCo-frame joints (`core.geometry.motion_to_mujoco`).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch

# Capsule radius per bone, keyed by the bone's child joint (bone = parent -> child).
BONE_RADIUS = {
  "Spine": 0.12, "Spine1": 0.12, "Spine2": 0.13, "Spine3": 0.13,
  "Neck": 0.06, "Head": 0.055,
  "RightShoulder": 0.06, "RightArm": 0.055, "RightForeArm": 0.05,
  "RightHand": 0.04, "RightHandEnd": 0.035,
  "LeftShoulder": 0.06, "LeftArm": 0.055, "LeftForeArm": 0.05,
  "LeftHand": 0.04, "LeftHandEnd": 0.035,
  "RightUpLeg": 0.08, "RightLeg": 0.075, "RightFoot": 0.055, "RightToeBase": 0.045,
  "LeftUpLeg": 0.08, "LeftLeg": 0.075, "LeftFoot": 0.055, "LeftToeBase": 0.045,
}
HEAD_LENGTH = 0.17
HEAD_RADIUS = 0.095


@dataclass
class CapsuleSkeleton:
  """Bone table: names, endpoint joint indices, radii and rest half-lengths."""

  names: list[str]
  parent: torch.Tensor  # [B] joint index of each bone's start
  child: torch.Tensor  # [B] joint index of each bone's end; -1 for the head cap
  radius: torch.Tensor
  half_length: torch.Tensor
  neck: int
  head: int

  @classmethod
  def from_library(cls, lib: dict, device="cpu") -> "CapsuleSkeleton":
    return cls.from_rest(lib["meta"], lib["posed_joints"][0, 0], device)

  @classmethod
  def from_rest(cls, meta: dict, rest_joints: torch.Tensor, device="cpu") -> "CapsuleSkeleton":
    """`rest_joints [J, 3]`: any posed frame; bone lengths are pose-invariant."""
    names, parents = list(meta["joint_names"]), list(meta["parents"])
    rest = rest_joints.float().cpu()
    bone_names, pi, ci, radius, half = [], [], [], [], []
    for j, name in enumerate(names):
      if name not in BONE_RADIUS or parents[j] < 0:
        continue
      bone_names.append(name)
      pi.append(parents[j])
      ci.append(j)
      radius.append(BONE_RADIUS[name])
      half.append(float((rest[j] - rest[parents[j]]).norm()) / 2)
    head, neck = names.index("Head"), names.index("Neck")
    bone_names.append("HeadTop")
    pi.append(head)
    ci.append(-1)
    radius.append(HEAD_RADIUS)
    half.append(HEAD_LENGTH / 2)
    t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)  # noqa: E731
    return cls(bone_names, t(pi, torch.long), t(ci, torch.long), t(radius), t(half), neck, head)

  @property
  def bones(self) -> int:
    return len(self.names)

  def add_to_spec(self, spec: mujoco.MjSpec, humans: int, colors, group: int = 2) -> list[list[str]]:
    """Add `humans x bones` mocap bodies to the worldbody; returns body names."""
    out = []
    for i in range(humans):
      rgba = colors[i % len(colors)]
      row = []
      for b, name in enumerate(self.names):
        body_name = f"h{i:02d}_{name}"
        body = spec.worldbody.add_body(name=body_name, mocap=True, pos=(0, 0, -5.0 - i))
        body.add_geom(
          name=f"{body_name}_geom",
          type=mujoco.mjtGeom.mjGEOM_CAPSULE,
          size=(float(self.radius[b]), max(1e-3, float(self.half_length[b])), 0),
          rgba=rgba,
          group=group,
          contype=0,
          conaffinity=0,
        )
        row.append(body_name)
      out.append(row)
    return out

  def poses(self, joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mocap `pos [H, B, 3]` and `quat [H, B, 4]` (wxyz) from MuJoCo-frame joints [H, J, 3]."""
    start = joints[:, self.parent]
    up = joints[:, self.head] - joints[:, self.neck]
    up = up / up.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    head_end = joints[:, self.head] + HEAD_LENGTH * up
    end = torch.where(
      (self.child >= 0)[None, :, None], joints[:, self.child.clamp(min=0)], head_end[:, None]
    )
    pos = (start + end) / 2
    d = end - start
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # Shortest rotation taking +Z onto d.
    quat = torch.stack((1 + d[..., 2], -d[..., 1], d[..., 0], torch.zeros_like(d[..., 0])), -1)
    flip = quat[..., 0] < 1e-6  # d points straight down
    quat[flip] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=quat.device)
    return pos, quat / quat.norm(dim=-1, keepdim=True)
