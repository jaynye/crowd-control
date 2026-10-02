"""Command response: smoothed root velocity, time to respond, wait for the next replan."""

from __future__ import annotations

import torch


def smoothed_velocity(root_xz: torch.Tensor, fps: float, k: int = 3) -> torch.Tensor:
  """`[N, T, 2]` velocity over `k`-frame differences (the first k frames copy frame k)."""
  v = torch.zeros_like(root_xz)
  v[:, k:] = (root_xz[:, k:] - root_xz[:, :-k]) * fps / k
  v[:, :k] = v[:, k : k + 1]
  return v


def response_time(ok: torch.Tensor, start: torch.Tensor, fps: float) -> torch.Tensor:
  """Seconds after each human's `start` frame until `ok` holds for 0.25 s; NaN if never."""
  hold = max(1, round(0.25 * fps))
  n, frames = ok.shape
  out = torch.full((n,), float("nan"))
  okc = ok.cpu()
  for i in range(n):
    run = 0
    for f in range(int(start[i]), frames):
      run = run + 1 if okc[i, f] else 0
      if run >= hold:
        out[i] = (f - hold + 1 - int(start[i])) / fps
        break
  return out


def replan_wait(start: torch.Tensor, boundaries: list[int], fps: float) -> torch.Tensor:
  """Seconds from each change frame to the first replan at or after it."""
  b = torch.tensor(boundaries + [10**9], dtype=torch.float32)
  idx = torch.searchsorted(b, start.float())
  return (b[idx] - start.float()) / fps
