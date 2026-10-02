"""Synchronized wall-clock and per-stage CUDA timing."""

from __future__ import annotations

import contextlib
import time

import torch


class StageTimer:
  """CUDA-event timing of named stages inside one synchronized call.

  Events measure the GPU timeline, so a stage that starves the GPU with Python
  work still shows its full duration. Call `result()` after the caller syncs.
  Disabled (all zeros) without CUDA.
  """

  def __init__(self, enabled: bool = True):
    self.enabled = enabled and torch.cuda.is_available()
    self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

  @contextlib.contextmanager
  def stage(self, name: str):
    if not self.enabled:
      yield
      return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
      yield
    finally:
      end.record()
      self._events.append((name, start, end))

  def result(self) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, start, end in self._events:
      out[name] = out.get(name, 0.0) + start.elapsed_time(end) / 1e3
    return out


def synchronize(device=None) -> None:
  """`torch.cuda.synchronize` when timing a CUDA device; a no-op on CPU."""
  if torch.cuda.is_available() and (device is None or torch.device(device).type == "cuda"):
    torch.cuda.synchronize(device)


def timed_call(fn):
  """Run `fn` between device syncs; return (wall seconds, output)."""
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  out = fn()
  torch.cuda.synchronize()
  return time.perf_counter() - t0, out
