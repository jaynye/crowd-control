"""Command sources: per-human `(speed m/s, heading rad)` requests in simulated time.

- `RandomCommands`: independent semi-Markov segments per human (library generation).
- `ScenarioCommands`: the quality benchmark's fixed schedules (cruise, stop_start,
  turn90, turn180, speed_step).
- `PlazaBehavior` (in `behavior.plaza`): goal intent plus local steering.

A command source implements `command(t, kin, groups) -> (speed [H], heading [H])`.
Heading 0 faces +Z; positive turns toward +X.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from crowdmotion.core.config import from_dict
from crowdmotion.core.rng import stream_seed


# Random segments (library generation) -------------------------------------------
@dataclass
class CommandRanges:
  """Segment distribution. Speeds in m/s, angles in degrees, times in seconds."""

  p_stand: float = 0.2
  stand_s: tuple[float, float] = (1.0, 4.0)
  walk_s: tuple[float, float] = (1.5, 6.0)
  # (probability, low, high) speed bands for walk segments.
  speed_bands: tuple[tuple[float, float, float], ...] = (
    (0.15, 0.3, 0.8),
    (0.70, 0.8, 1.5),
    (0.15, 1.5, 2.0),
  )
  p_keep_heading: float = 0.35
  p_reverse_range: float = 0.2  # heading change drawn from +-180 instead of +-90
  p_curve: float = 0.25
  curve_rate_deg: float = 40.0

  @classmethod
  def from_dict(cls, d: dict | None) -> "CommandRanges":
    return from_dict(cls, d, what="command range")


class RandomCommands:
  """Per-human piecewise commands.

  Each human runs an independent semi-Markov process over segments: stand, or
  walk at a sampled speed toward a sampled heading change, optionally curving at
  a constant turn rate. Draws come from the human's own NumPy generator, seeded
  from its stream id, so commands never depend on batch composition.
  """

  def __init__(self, seeds: torch.Tensor, start_heading: torch.Tensor, ranges: CommandRanges):
    self.ranges = ranges
    self.rngs = [np.random.default_rng(stream_seed(int(s), 0)) for s in seeds.tolist()]
    n = len(self.rngs)
    self.mode = np.zeros(n, dtype=np.int64)  # 0 walk, 1 stand
    self.speed = np.zeros(n)
    self.heading = start_heading.detach().cpu().double().numpy().copy()
    self.rate = np.zeros(n)
    self.seg_start = np.zeros(n)
    self.seg_end = np.zeros(n)
    self.segment = np.zeros(n, dtype=np.int64)
    for i in range(n):
      self._draw(i, 0.0)

  def _draw(self, i: int, t: float) -> None:
    r, g = self.ranges, self.rngs[i]
    # Carry the heading reached so far into the next segment.
    self.heading[i] += self.rate[i] * (t - self.seg_start[i])
    if g.random() < r.p_stand:
      self.mode[i], self.speed[i], self.rate[i] = 1, 0.0, 0.0
      dwell = g.uniform(*r.stand_s)
    else:
      self.mode[i] = 0
      probs = np.array([b[0] for b in r.speed_bands])
      band = r.speed_bands[g.choice(len(probs), p=probs / probs.sum())]
      self.speed[i] = g.uniform(band[1], band[2])
      u = g.random()
      if u >= r.p_keep_heading:
        span = math.pi if g.random() < r.p_reverse_range else math.pi / 2
        self.heading[i] += g.uniform(-span, span)
      curve = g.random() < r.p_curve
      self.rate[i] = g.uniform(-1, 1) * math.radians(r.curve_rate_deg) if curve else 0.0
      dwell = g.uniform(*r.walk_s)
    self.seg_start[i], self.seg_end[i] = t, t + dwell
    self.segment[i] += 1

  def __call__(self, t: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    for i in np.nonzero(t >= self.seg_end)[0]:
      while t >= self.seg_end[i]:
        self._draw(int(i), float(self.seg_end[i]))
    heading = self.heading + self.rate * (t - self.seg_start)
    return (
      torch.as_tensor(self.speed, dtype=torch.float32, device=device),
      torch.as_tensor(heading, dtype=torch.float32, device=device),
    )

  def command(self, t: float, kin, groups=None):
    return self(t, kin.root_xz.device)

  def state(self) -> dict[str, np.ndarray]:
    return {"mode": self.mode.copy(), "segment": self.segment.copy(), "rate": self.rate.copy()}


# Benchmark scenarios -------------------------------------------------------------
CHANGE_T = 4.0  # seconds; single-command scenarios switch here
RESTART_T = 7.0
SCENARIOS = ("cruise", "stop_start", "turn90", "turn180", "speed_step")


class ScenarioCommands:
  """The quality benchmark's per-human command schedules.

  `draw` makes the start states, cruise schedule and change offsets from one
  seeded generator, in a fixed order, so every backend sees identical schedules.
  Single-change scenarios switch at CHANGE_T + offset[i] (and restart at
  RESTART_T + offset[i]); the offset spreads changes across a replan interval.
  """

  def __init__(self, scenario: str, h0: torch.Tensor, rng: dict, offset: torch.Tensor):
    if scenario not in SCENARIOS:
      raise ValueError(scenario)
    self.scenario, self.h0, self.rng, self.offset = scenario, h0, rng, offset

  @classmethod
  def draw(cls, scenario: str, humans: int, seed: int, interval_s: float, jitter: bool, device):
    """Commands and start states `(commands, start_xz)` for `humans` from `seed`."""
    gen = torch.Generator().manual_seed(seed)
    h0 = (torch.rand(humans, generator=gen) * 2 * math.pi - math.pi).to(device)
    xz = (torch.rand(humans, 2, generator=gen) * 10 - 5).to(device)
    rng = {
      "period": 4.0,
      "speed": (0.5 + 1.1 * torch.rand(humans, 16, generator=gen)).to(device),
      "heading": (h0[:, None].cpu() + torch.cumsum(
        (torch.rand(humans, 16, generator=gen) * 2 - 1) * math.pi / 2, 1)).to(device),
    }
    offset = (torch.rand(humans, generator=gen) * interval_s if jitter else torch.zeros(humans)).to(device)
    return cls(scenario, h0, rng, offset), xz

  def __call__(self, t: float) -> tuple[torch.Tensor, torch.Tensor]:
    h0, rng, offset = self.h0, self.rng, self.offset
    n = h0.shape[0]
    if self.scenario == "cruise":
      # Piecewise-constant random commands, redrawn every rng["period"] seconds.
      k = int(t // rng["period"])
      return rng["speed"][:, k % rng["speed"].shape[1]], rng["heading"][:, k % rng["heading"].shape[1]]
    after = t >= CHANGE_T + offset
    speed = torch.full((n,), 1.2, device=h0.device)
    heading = h0.clone()
    if self.scenario == "stop_start":
      speed = torch.where(after & (t < RESTART_T + offset), torch.zeros_like(speed), speed)
    elif self.scenario == "turn90":
      heading = torch.where(after, h0 + math.pi / 2, h0)
    elif self.scenario == "turn180":
      heading = torch.where(after, h0 + math.pi, h0)
    elif self.scenario == "speed_step":
      speed = torch.where(after, torch.full_like(speed, 1.6), torch.full_like(speed, 0.6))
    return speed, heading

  def command(self, t: float, kin, groups=None):
    return self(t)
