"""Executed motion of a group of humans: the one format metrics and viewers read.

`Trace.save` writes the behavior evaluator's `seed_<n>.npz` layout. `Trace.load`
also reads the quality benchmark's `clips_*.npz`, which store joints but no root
or heading; those are derived from the root joint and marked in `extras`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CORE = ("joints", "root", "heading", "contacts", "foot_idx")


@dataclass
class Trace:
  joints: np.ndarray  # [H, T, J, 3] world, Y-up
  root: np.ndarray  # [H, T, 2] planar root XZ
  heading: np.ndarray  # [H, T] body heading (0 faces +Z)
  fps: float
  contacts: np.ndarray | None = None  # [H, T, C] bool
  foot_idx: np.ndarray | None = None
  extras: dict[str, np.ndarray] = field(default_factory=dict)

  @property
  def humans(self) -> int:
    return self.joints.shape[0]

  @property
  def frames(self) -> int:
    return self.joints.shape[1]

  def arrays(self) -> dict[str, np.ndarray]:
    """Arrays in the saved order: core fields, then extras."""
    out = {"joints": self.joints, "root": self.root, "heading": self.heading}
    if self.contacts is not None:
      out["contacts"] = self.contacts
    out.update(self.extras)
    if self.foot_idx is not None:
      out["foot_idx"] = self.foot_idx
    return out

  def save(self, path) -> None:
    np.savez_compressed(path, **self.arrays())

  @classmethod
  def load(cls, path, fps: float | None = None) -> "Trace":
    with np.load(Path(path)) as z:
      data = {k: z[k] for k in z.files}
    if "root" in data and "heading" in data:
      if fps is None:
        raise ValueError("evaluator traces store fps in run.json; pass fps")
      extras = {k: v for k, v in data.items() if k not in CORE}
      return cls(data["joints"], data["root"], data["heading"], float(fps), data.get("contacts"),
                 data.get("foot_idx"), extras)
    # Quality-benchmark clips: root from the root joint, heading from travel direction.
    joints = data["joints"]
    root = joints[:, :, 0, [0, 2]]
    delta = np.diff(root, axis=1, prepend=root[:, :1])
    heading = np.arctan2(delta[..., 0], delta[..., 1])
    extras = {k: v for k, v in data.items() if k not in ("joints", "contacts", "foot_idx", "fps")}
    extras["derived"] = np.array("root=joint 0; heading=travel direction")
    return cls(joints, root, heading, float(data["fps"]), data.get("contacts"), data.get("foot_idx"), extras)
