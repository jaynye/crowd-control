"""Body-relative locomotion: direction of travel relative to body facing, turning, speed.

Facing is the body (root) heading, independent of the root's direction of travel,
so a curved forward path is forward, not sideways.
"""

from __future__ import annotations

import numpy as np

from crowdmotion.core.geometry import wrap_np

CLASSES = ("Still / slow", "Forward", "Oblique", "Sideways", "Backward")
EDGE_CLASSES = (
  "Still / slow",
  "Turn in place / slow",
  "Forward · straight",
  "Forward · turning",
  "Oblique",
  "Sideways",
  "Backward",
)
STILL_MPS = 0.15


def motion_features(root, heading, fps, start_root=None, start_heading=None):
  """Input `[H,T,2]`, `[H,T]`; velocities include the predecessor when provided.

  Otherwise the first frame is excluded (no known predecessor), never labeled still.
  Direction uses midpoint body facing between adjacent frames.
  Returns `(speed m/s, angle of travel from facing in degrees, body turn deg/s)`.
  """
  if start_root is None:
    delta = np.diff(root, axis=1)
    yaw = wrap_np(np.diff(heading, axis=1))
    facing = heading[:, :-1] + yaw / 2
  else:
    delta = np.diff(np.concatenate((start_root[:, None], root), axis=1), axis=1)
    before = np.concatenate((start_heading[:, None], heading[:, :-1]), axis=1)
    yaw = wrap_np(heading - before)
    facing = before + yaw / 2
  speed = np.linalg.norm(delta, axis=-1) * fps
  angle = wrap_np(np.arctan2(delta[..., 0], delta[..., 1]) - facing)
  return speed, np.degrees(angle), np.degrees(yaw) * fps


def classify(speed, angle_deg, still_mps=STILL_MPS):
  """0 still/slow, 1 forward (<=30°), 2 oblique (<60°), 3 sideways (<120°), 4 backward."""
  angle = np.abs(angle_deg)
  out = np.full(speed.shape, 4, dtype=np.int8)
  out[angle < 120] = 3
  out[angle < 60] = 2
  out[angle <= 30] = 1
  out[speed < still_mps] = 0
  return out


def composition(labels, names=CLASSES):
  counts = np.bincount(labels.ravel(), minlength=len(names))
  total = int(counts.sum())
  return [
    {"name": name, "count": int(n), "percent": 100 * float(n) / max(1, total)}
    for name, n in zip(names, counts)
  ]
