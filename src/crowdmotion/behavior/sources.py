"""Motion sources: one interface over the motion graph and online generators.

A source turns a requested world root path into body motion for every human:

  state = source.reset(seeds, root_xz, heading)
  step = source.step(state, path)       # path [H, source.path_frames, 2]
  source.commit(state, step, take)      # the first `take` frames were executed

`GraphMotionSource` plays one graph edge per step. `OnlineMotionSource` calls an
online backend (ARDY, Kimodo) and keeps the executed-feature history it feeds back,
so only frames that were actually played condition the next call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from crowdmotion.backends.base import MotionRequest
from crowdmotion.core.geometry import body_world
from crowdmotion.core.timing import synchronize


@dataclass
class MotionStep:
  """Frames a source produced for one step (world coordinates, `[H, F, ...]`)."""

  joints: torch.Tensor  # [H, F, J, 3]
  root: torch.Tensor  # [H, F, 3]
  heading: torch.Tensor  # [H, F]
  contacts: torch.Tensor  # [H, F, C]
  features: torch.Tensor | None = None  # online backends: native normalized features
  local_rot: torch.Tensor | None = None  # online backends: [H, F, J, 3, 3]
  extras: dict = field(default_factory=dict)  # graph: edge ids and per-human diagnostics

  @property
  def frames(self) -> int:
    return self.root.shape[1]

  @classmethod
  def from_chunk(cls, chunk) -> "MotionStep":
    return cls(chunk.posed_joints, chunk.root_pos, chunk.heading, chunk.contacts,
               features=chunk.features, local_rot=chunk.local_rot, extras={"chunk": chunk})


class GraphMotionSource:
  """One motion-graph edge per step (`frames_per_step = L`).

  timed: synchronize around each step and record its wall time in
  `extras["call_wall_s"]` (benchmarks; off in simulation).
  """

  def __init__(self, runtime, timed: bool = False):
    self.runtime = runtime
    self.timed = timed
    self.graph = runtime.graph
    self.fps = runtime.fps
    self.frames_per_step = self.graph.L

  @property
  def path_frames(self) -> int:
    return self.runtime.path_frames

  def reset(self, seeds: torch.Tensor, root_xz: torch.Tensor, heading: torch.Tensor):
    return self.runtime.reset(seeds, root_xz, heading)

  def allocate(self, humans: int):
    """Placeholder state for `humans` rows, filled by later `reset`s of subsets."""
    from crowdmotion.graph import GraphState

    dev = self.runtime.device
    return GraphState(
      boundary=torch.zeros(humans, dtype=torch.long, device=dev),
      root_xz=torch.zeros(humans, 2, device=dev),
      heading=torch.zeros(humans, device=dev),
      pose=torch.zeros(humans, self.graph.joints_n, 3, device=dev),
      seeds=torch.zeros(humans, dtype=torch.long, device=dev),
      counters=torch.zeros(humans, dtype=torch.long, device=dev),
    )

  def step(self, state, path: torch.Tensor) -> MotionStep:
    if self.timed:
      synchronize(self.runtime.device)
      t0 = time.perf_counter()
    out = self.runtime.step(state, path)
    if self.timed:
      synchronize(self.runtime.device)
      out["call_wall_s"] = time.perf_counter() - t0
    joints, root, heading, contacts = (out.pop(k) for k in ("joints", "root", "heading", "contacts"))
    return MotionStep(joints, root, heading, contacts, extras=out)

  # The runtime advances its state through the whole edge in `step`, so a step whose
  # frames were only partly executed cannot be continued.
  resumes_partial = False

  def commit(self, state, step: MotionStep, take: int) -> None:
    pass

  def current_joints(self, state) -> torch.Tensor:
    """World joints `[H, J, 3]` of each human's last played pose."""
    return body_world(state.pose, state.root_xz, state.heading)

  @staticmethod
  def select(state, rows: torch.Tensor):
    return type(state)(*(getattr(state, n)[rows] for n in _GRAPH_FIELDS))

  @staticmethod
  def put(state, rows: torch.Tensor, sub) -> None:
    for name in _GRAPH_FIELDS:
      getattr(state, name)[rows] = getattr(sub, name)


_GRAPH_FIELDS = ("boundary", "root_xz", "heading", "pose", "seeds", "counters")


@dataclass
class OnlineState:
  seeds: torch.Tensor
  counters: torch.Tensor
  start_xz: torch.Tensor
  start_heading: torch.Tensor
  text: torch.Tensor
  features: torch.Tensor | None = None  # executed frames, oldest first


class OnlineMotionSource:
  """Closed-loop calls to an online backend, fed back only executed frames.

  history_frames: executed frames passed as history (0: every call starts fresh
    from the reset state). keep: frames of executed history retained (None: all).
  cast_history: pass history as float32 (library generation).
  timed: synchronize around each backend call and record its wall time in
    `extras["call_wall_s"]`.
  """

  def __init__(self, backend, history_frames: int, keep: int | None = None,
               path_frames: int | None = None, cast_history: bool = False, timed: bool = False):
    self.backend = backend
    self.timed = timed
    self.fps = backend.fps
    self.history_frames = history_frames
    self.keep = keep
    self.path_frames = path_frames if path_frames is not None else backend.path_frames
    self.cast_history = cast_history

  def reset(self, seeds: torch.Tensor, root_xz: torch.Tensor, heading: torch.Tensor) -> OnlineState:
    return OnlineState(
      seeds=seeds, counters=torch.zeros(len(seeds), dtype=torch.long), start_xz=root_xz,
      start_heading=heading, text=self.backend.empty_text(len(seeds)),
    )

  def request(self, state: OnlineState, path: torch.Tensor | None,
              history_frames: int | None = None) -> MotionRequest:
    """The next request; `history_frames` overrides the source's history length."""
    n = self.history_frames if history_frames is None else history_frames
    hist = None
    if n and state.features is not None:
      hist = state.features[:, -n:]
      if self.cast_history:
        hist = hist.float()
    return MotionRequest(
      text_feat=state.text, seeds=state.seeds, counters=state.counters.clone(), history=hist,
      root_path=path, start_xz=None if hist is not None else state.start_xz,
      start_heading=None if hist is not None else state.start_heading,
    )

  def generate(self, req: MotionRequest) -> MotionStep:
    if not self.timed:
      return MotionStep.from_chunk(self.backend.generate(req))
    synchronize(self.backend.device)
    t0 = time.perf_counter()
    chunk = self.backend.generate(req)
    synchronize(self.backend.device)
    step = MotionStep.from_chunk(chunk)
    step.extras["call_wall_s"] = time.perf_counter() - t0
    return step

  def step(self, state: OnlineState, path: torch.Tensor | None) -> MotionStep:
    return self.generate(self.request(state, path))

  resumes_partial = True

  def commit(self, state: OnlineState, step: MotionStep, take: int) -> None:
    feats = step.features[:, :take]
    state.features = feats if state.features is None else torch.cat((state.features, feats), 1)
    if self.keep is not None:
      state.features = state.features[:, -self.keep:]
    state.counters += 1
