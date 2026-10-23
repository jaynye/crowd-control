"""Command sources: per-human `(speed m/s, heading rad)` requests in simulated time.

- `RandomCommands`: independent semi-Markov segments per human (library generation).
- `TurnCommands`: semi-Markov stand / walk / in-place turn / arc segments with per-frame
  body-heading targets (the supplementary turning library).
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

from crowdmotion.behavior.planner import plan_arc
from crowdmotion.core.config import from_dict
from crowdmotion.core.geometry import wrap
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


# Turning segments (supplementary turning library) --------------------------------
TURN_KINDS = ("walk", "stand", "in_place", "arc")  # `mode` codes 0..3 (0 walk, 1 stand as RandomCommands)
WALK, STAND, IN_PLACE, ARC = range(4)
TURN_STREAM = 1  # `stream_seed` counter of the turn segment draws (RandomCommands uses 0)
TANGENT_MIN_SPEED = 0.1  # m/s: below this planned speed, heading targets ignore the path tangent


@dataclass
class TurnRanges:
  """Turning segment distribution. Speeds in m/s, angles in degrees, times in seconds,
  radii in meters. `next_*` are next-segment probabilities in `TURN_KINDS` order
  (walk, stand, in_place, arc) after a segment of that kind; `start` is the first."""

  stand_s: tuple[float, float] = (0.6, 2.0)
  walk_s: tuple[float, float] = (1.0, 3.0)
  # (probability, low, high) speed bands for walk and arc segments.
  speed_bands: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.3, 0.8),
    (0.60, 0.8, 1.5),
    (0.15, 1.5, 2.0),
  )
  in_place_deg: tuple[float, float] = (45.0, 180.0)  # turn magnitude; the sign is drawn
  in_place_rate_deg: tuple[float, float] = (60.0, 180.0)  # deg/s
  settle_s: tuple[float, float] = (0.3, 0.8)  # standing after an in-place turn completes
  arc_radius_m: tuple[float, float] = (0.5, 4.0)
  arc_deg: tuple[float, float] = (45.0, 180.0)
  arc_max_rate_deg: float = 120.0  # deg/s; a tighter arc is walked slower (speed = radius * rate)
  arc_keep_speed: bool = True  # an arc after walking keeps that speed (before the rate cap)
  start: tuple[float, float, float, float] = (0.3, 0.4, 0.3, 0.0)
  next_walk: tuple[float, float, float, float] = (0.15, 0.25, 0.1, 0.5)
  next_stand: tuple[float, float, float, float] = (0.35, 0.05, 0.6, 0.0)
  next_in_place: tuple[float, float, float, float] = (0.6, 0.2, 0.1, 0.1)
  next_arc: tuple[float, float, float, float] = (0.55, 0.15, 0.05, 0.25)
  heading_targets: bool = True  # pass heading targets to the generator (False: path only)
  walk_heading: bool = True  # also constrain straight walking (forward facing); False: NaN
  stand_heading: bool = True  # also hold the heading while standing; False: NaN
  forward_speed: float = 0.5  # m/s of planned speed at which walk/arc targets fully follow the path tangent
  blend_s: float = 0.5  # path velocity and heading blend from the executed state
  heading_every: int = 1  # constrain every k-th planned frame's heading (k-1, 2k-1, ...); others NaN
  path_every: int = 1  # likewise for the root path (NaN frames are free; ARDY vectorized conditions only)
  in_place_path: str = "plan"  # in-place turns: "plan" (as above) or "end" (only the last planned frame)

  def __post_init__(self):
    if self.in_place_path not in ("plan", "end"):
      raise ValueError(f"in_place_path must be 'plan' or 'end', not {self.in_place_path!r}")
    if self.heading_every < 1 or self.path_every < 1:
      raise ValueError("heading_every and path_every must be at least 1")

  @classmethod
  def from_dict(cls, d: dict | None) -> "TurnRanges":
    return from_dict(cls, d, what="turn range")


class TurnCommands:
  """Per-human turning segments with body-heading targets, for library generation.

  Each human runs an independent semi-Markov chain over walk, stand, in-place turn
  (speed 0, a signed angle at a turn rate, then a short settle) and arc (constant speed
  along a circle of a drawn radius through a signed angle, facing along the arc).
  Typical sequences: stand -> in-place turn -> walk off; walk -> arc -> walk. Every
  draw comes from the human's own generator, so segment kinds, durations, angles and
  rates never depend on executed motion or batch composition; only the absolute
  headings do: a segment starts from the executed heading of the step that enters it.
  Segments therefore start at command reads (edge boundaries in library generation).

  Use it as both command source and planner (`CrowdController(cmds, src, vel,
  planner=cmds.plan)`): `plan` traces the segment with `planner.plan_arc` and stores
  per-frame heading targets, which `motion_options()` hands to the motion source
  (`root_heading`). Targets start at the executed heading and blend to the segment's
  schedule over `blend_s`; where the planned path moves, they turn toward its tangent
  (fully at `forward_speed`), so the body faces where it walks.
  """

  def __init__(self, seeds: torch.Tensor, start_heading: torch.Tensor, ranges: TurnRanges):
    self.ranges = ranges
    self.rngs = [np.random.default_rng(stream_seed(int(s), TURN_STREAM)) for s in seeds.tolist()]
    n = len(self.rngs)
    self.mode = np.full(n, -1, dtype=np.int64)  # no segment yet
    self.speed = np.zeros(n)
    self.anchor = start_heading.detach().cpu().double().numpy().copy()  # heading at segment start
    self.rate = np.zeros(n)  # signed rad/s
    self.angle = np.zeros(n)  # signed total turn of the segment, radians
    self.radius = np.full(n, np.nan)  # arcs only, meters
    self.seg_start = np.zeros(n)
    self.seg_end = np.zeros(n)  # 0: draw at the first read
    self.segment = np.zeros(n, dtype=np.int64)
    self._exec_heading: torch.Tensor | None = None
    self.reference: torch.Tensor | None = None  # [H, F] dense heading reference of the last plan
    self.targets: torch.Tensor | None = None  # [H, F] the constrained subset of it (NaN = free)

  # Segments -----------------------------------------------------------------------
  def _speed(self, g) -> float:
    bands = self.ranges.speed_bands
    probs = np.array([b[0] for b in bands])
    band = bands[g.choice(len(probs), p=probs / probs.sum())]
    return float(g.uniform(band[1], band[2]))

  def _draw(self, i: int, t: float, heading: float) -> None:
    r, g = self.ranges, self.rngs[i]
    prev = int(self.mode[i])
    probs = np.array(r.start if prev < 0 else (r.next_walk, r.next_stand, r.next_in_place, r.next_arc)[prev])
    kind = int(g.choice(4, p=probs / probs.sum()))
    prev_speed = self.speed[i]
    self.rate[i] = self.angle[i] = 0.0
    self.radius[i] = np.nan
    if kind == WALK:
      self.speed[i] = self._speed(g)
      dwell = g.uniform(*r.walk_s)
    elif kind == STAND:
      self.speed[i] = 0.0
      dwell = g.uniform(*r.stand_s)
    elif kind == IN_PLACE:
      sign = 1.0 if g.random() < 0.5 else -1.0
      angle = math.radians(g.uniform(*r.in_place_deg))
      rate = math.radians(g.uniform(*r.in_place_rate_deg))
      self.speed[i], self.angle[i], self.rate[i] = 0.0, sign * angle, sign * rate
      dwell = angle / rate + g.uniform(*r.settle_s)
    else:  # ARC
      sign = 1.0 if g.random() < 0.5 else -1.0
      radius = g.uniform(*r.arc_radius_m)
      angle = math.radians(g.uniform(*r.arc_deg))
      speed = self._speed(g)
      if r.arc_keep_speed and prev in (WALK, ARC) and prev_speed > 0:
        speed = prev_speed
      speed = min(speed, radius * math.radians(r.arc_max_rate_deg))
      self.speed[i], self.radius[i] = speed, radius
      self.angle[i], self.rate[i] = sign * angle, sign * speed / radius
      dwell = angle / abs(self.rate[i])
    self.mode[i] = kind
    # Keep the commanded heading continuous (unwrapped) while starting from the executed one.
    self.anchor[i] = heading
    self.seg_start[i], self.seg_end[i] = t, t + dwell
    self.segment[i] += 1

  def _turned(self, t: float) -> np.ndarray:
    """Signed turn completed so far in each human's segment."""
    done = np.abs(self.rate) * np.maximum(t - self.seg_start, 0.0)
    return np.sign(self.angle) * np.minimum(done, np.abs(self.angle))

  def command(self, t: float, kin, groups=None):
    executed = kin.heading.detach().cpu().double().numpy()
    current = self.anchor + self._turned(t)
    for i in np.nonzero(t >= self.seg_end)[0]:
      # Start from the executed heading, unwrapped next to the command so far.
      cur = current[i]
      self._draw(int(i), float(t), float(cur + math.remainder(executed[i] - cur, 2 * math.pi)))
    heading = self.anchor + self._turned(t)
    self._exec_heading = kin.heading
    self._remaining = np.abs(self.angle) - np.abs(self._turned(t))
    dev = kin.root_xz.device
    return (
      torch.as_tensor(self.speed, dtype=torch.float32, device=dev),
      torch.as_tensor(heading, dtype=torch.float32, device=dev),
    )

  # Planner and heading targets ---------------------------------------------------
  def plan(self, xz, vel, speed, heading, frames: int, fps: float):
    """`plan_arc` through the rest of each human's segment; stores heading targets."""
    r = self.ranges
    dev = xz.device
    rate = torch.as_tensor(self.rate, dtype=torch.float32, device=dev)
    remaining = torch.as_tensor(self._remaining, dtype=torch.float32, device=dev)
    path, sched = plan_arc(xz, vel, speed, heading, rate, frames, fps, blend_s=r.blend_s, max_turn=remaining)
    alpha = (torch.arange(1, frames + 1, device=dev, dtype=torch.float32)
             / max(1.0, r.blend_s * fps)).clamp(max=1.0)[None]
    executed = heading if self._exec_heading is None else self._exec_heading.to(dev).float()
    target = sched + (1 - alpha) * wrap(executed - heading)[:, None]
    # Walking and arcs: face along the planned path where it moves forward enough. A
    # dead zone and the forward half-plane keep velocity noise and backward drift out.
    mode = torch.as_tensor(self.mode, device=dev)
    step = torch.diff(torch.cat((xz[:, None], path), 1), dim=1) * fps  # planned velocity [H, F, 2]
    diff = wrap(torch.atan2(step[..., 0], step[..., 1]) - target)
    ramp = ((step.norm(dim=-1) - TANGENT_MIN_SPEED) / max(r.forward_speed - TANGENT_MIN_SPEED, 1e-6)).clamp(0, 1)
    moving = ((mode == WALK) | (mode == ARC))[:, None]
    target = target + moving * ramp * torch.cos(diff).clamp(min=0) * diff
    free = torch.zeros_like(mode, dtype=torch.bool)
    if not r.walk_heading:
      free |= mode == WALK
    if not r.stand_heading:
      free |= mode == STAND
    self.reference = target
    nan = torch.full_like(target, float("nan"))
    frame = torch.arange(1, frames + 1, device=dev)
    self.targets = torch.where(free[:, None] | (frame % r.heading_every != 0)[None], nan, target)
    free_path = (frame % r.path_every != 0)[None].expand(len(mode), -1)
    if r.in_place_path == "end":
      free_path = torch.where((mode == IN_PLACE)[:, None], (frame != frames)[None], free_path)
    if free_path.any():
      path = torch.where(free_path[..., None], float("nan"), path)
    return path

  def motion_options(self) -> dict:
    return {"root_heading": self.targets} if self.ranges.heading_targets else {}

  def state(self) -> dict[str, np.ndarray]:
    return {"mode": self.mode.copy(), "segment": self.segment.copy(), "rate": self.rate.copy(),
            "angle": self.angle.copy(), "radius": self.radius.copy()}


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
