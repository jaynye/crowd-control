"""Pairwise spacing between humans of one group."""

from __future__ import annotations

import numpy as np


def root_gaps(root: np.ndarray) -> np.ndarray:
  """`[T, H, H]` planar root distances per frame for `root [H, T, 2]`; self pairs are inf."""
  pair = root.transpose(1, 0, 2)
  gaps = np.linalg.norm(pair[:, :, None] - pair[:, None], axis=-1)
  gaps[:, np.arange(len(pair[0])), np.arange(len(pair[0]))] = np.inf
  return gaps
