"""Backend-neutral contract for batched human-motion generation.

    Generate(executed_history, prompt_embeddings, future_constraints, seeds)
        -> new_root_motion, joint_rotations, contacts, frame_rate

A backend (see `registry.make_backend`) exposes:
  name, device, fps, joint_names, skeleton_info() -> dict
  path_frames      frames of root path a request may constrain
  frames_per_call  new frames each call returns (history excluded)
  empty_text(batch) -> [B, 1, 4096]
  generate(req: MotionRequest) -> MotionChunk
  configure(cfg)   apply another config of the same model without reloading weights

Coordinates are each backend's native convention (both ARDY and Kimodo: Y-up,
meters, heading 0 facing +Z). Skeletons stay native; `joint_names` says which.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass
class MotionRequest:
  """One batched generation call. B = number of humans in the call.

  history: `[B, Th, D]` native normalized features of frames that were actually
    executed, oldest first, or None for a fresh start.
  text_feat: `[B, L, 4096]` cached prompt embeddings (zeros = empty prompt).
  root_path: `[B, F, 2]` world XZ targets for the F frames after the history
    (native frame rate), or None for no path constraint.
  root_heading: `[B, F]` world body-heading targets (radians, 0 faces +Z, positive
    toward +X) for the same frames as `root_path`; NaN leaves a frame unconstrained.
    None (the default) constrains no heading. Only ARDY supports it.
  seeds, counters: `[B]` int64 per-human stream id and draw counter.
  start_xz, start_heading: `[B, 2]`, `[B]` for fresh starts only.
  """

  text_feat: torch.Tensor
  seeds: torch.Tensor
  counters: torch.Tensor
  history: torch.Tensor | None = None
  root_path: torch.Tensor | None = None
  start_xz: torch.Tensor | None = None
  start_heading: torch.Tensor | None = None
  root_heading: torch.Tensor | None = None

  @property
  def batch(self) -> int:
    return self.text_feat.shape[0]


@dataclass
class MotionChunk:
  """Newly generated frames only; executed history is never included.

  features: `[B, H, D]` native normalized features (feed back as history).
  root_pos: `[B, H, 3]`; local_rot: `[B, H, J, 3, 3]`; posed_joints: `[B, H, J, 3]`;
  contacts: `[B, H, C]` bool; heading: `[B, H]` radians.
  """

  features: torch.Tensor
  root_pos: torch.Tensor
  local_rot: torch.Tensor
  posed_joints: torch.Tensor
  contacts: torch.Tensor
  heading: torch.Tensor
  fps: float
  timer: object | None = None

  @property
  def frames(self) -> int:
    return self.features.shape[1]

  def stage_seconds(self) -> dict[str, float]:
    """Per-stage GPU seconds; valid only after the caller synchronizes."""
    return self.timer.result() if self.timer is not None else {}


@contextlib.contextmanager
def injected_randn(module, noise: torch.Tensor) -> Iterator[list]:
  """Make `module.torch.randn` return `noise` once, without editing upstream code.

  The upstream samplers draw their initial DDIM state with one `torch.randn`
  call; DDIM here is deterministic (eta = 0), so that draw is the only
  randomness. Everything else resolves to the real torch module.
  """
  real = module.torch
  calls: list = []

  class _Proxy:
    def __getattr__(self, name):
      return getattr(real, name)

    @staticmethod
    def randn(*size, **kwargs):
      shape = tuple(size[0]) if len(size) == 1 and not isinstance(size[0], int) else size
      if tuple(shape) != tuple(noise.shape):
        raise RuntimeError(f"randn{tuple(shape)} does not match noise {tuple(noise.shape)}")
      if calls:
        raise RuntimeError("upstream drew noise more than once")
      calls.append(shape)
      return noise

  module.torch = _Proxy()
  try:
    yield calls
  finally:
    module.torch = real
  if len(calls) != 1:
    raise RuntimeError(f"expected one randn draw, saw {len(calls)}")


# Precision -----------------------------------------------------------------------
PRECISIONS = ("fp32", "tf32", "bf16")


def check_precision(precision: str) -> None:
  if precision not in PRECISIONS:
    raise ValueError(f"precision must be one of {PRECISIONS}")


@contextlib.contextmanager
def tf32_matmul(enabled: bool):
  """Set the CUDA TF32 matmul and cuDNN flags for one call, then restore them."""
  prev = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = enabled
  try:
    yield
  finally:
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def denoise_autocast(precision: str):
  """bf16 autocast around denoising for "bf16"; nothing otherwise."""
  if precision == "bf16":
    return torch.autocast("cuda", dtype=torch.bfloat16)
  return contextlib.nullcontext()


def identity(iterable):
  """Progress-bar stand-in for upstream samplers."""
  return iterable
