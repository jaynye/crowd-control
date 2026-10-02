"""Typed backend configs and one constructor for every online generator.

  backend = make_backend(ArdyConfig(steps=5, cfg_type="nocfg", precision="tf32"))
  backend = make_backend(config_from_case({"backend": "kimodo", "new_frames": 20}))

Benchmark and library configs are flat JSON cases; `config_from_case` maps their
keys (`future`, `history`, ...) onto the typed configs, with caller defaults.
Adding a generator means one config class, one adapter and one `BACKENDS` row.
"""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass


@dataclass
class ArdyConfig:
  model: str = "ARDY-Core-RP-20FPS-Horizon40"
  device: str = "cuda"
  steps: int | None = None  # None: the checkpoint's base step count
  cfg_type: str = "separated"  # separated (3 passes), regular (2) or nocfg (1)
  cfg_weight: tuple[float, float] = (2.0, 2.0)
  future_frames: int = 0  # root-path context beyond the horizon (multiple of 4)
  precision: str = "fp32"  # fp32, tf32 or bf16
  conditions: str = "vectorized"  # or "library" (upstream constraint objects)
  timing: bool = True  # per-stage CUDA timing

  kind = "ardy"


@dataclass
class KimodoConfig:
  model: str = "Kimodo-SOMA-RP-v1.1"
  device: str = "cuda"
  steps: int = 100
  cfg_weight: tuple[float, float] = (2.0, 2.0)
  transition_frames: int = 5  # executed frames constraining the new window
  new_frames: int = 30  # frames generated per call after the transition
  precision: str = "fp32"
  post_processing: bool = False
  to_soma77: bool = False
  seam_blend_frames: int = 0
  timing: bool = True

  kind = "kimodo"


BACKENDS = {
  "ardy": (ArdyConfig, "crowdmotion.backends.ardy:ArdyBackend"),
  "kimodo": (KimodoConfig, "crowdmotion.backends.kimodo:KimodoBackend"),
}
# Flat case keys that name a config field differently.
CASE_KEYS = {
  "ardy": {"future": "future_frames"},
  "kimodo": {"history": "transition_frames"},
}


def config_from_case(case: dict, keys=None, **defaults):
  """Typed config from a flat case dict: dataclass defaults < `defaults` < case keys.

  keys: the case keys this caller honors (None: every backend field). Callers pass
  the keys their configs have always meant, so a stray key cannot change a run.
  Keys that are not backend fields (humans, scenarios, ...) are ignored.
  """
  kind = case["backend"]
  cls = BACKENDS[kind][0]
  names = {f.name for f in dataclasses.fields(cls)}
  values = {k: v for k, v in defaults.items() if k in names}
  for key, value in case.items():
    if keys is not None and key not in keys:
      continue
    field = CASE_KEYS[kind].get(key, key)
    if field in names:
      values[field] = value
  if "cfg_weight" in values:
    values["cfg_weight"] = tuple(values["cfg_weight"])
  return cls(**values)


def backend_class(kind: str):
  module, name = BACKENDS[kind][1].split(":")
  return getattr(importlib.import_module(module), name)


def make_backend(cfg):
  """Load the backend for a typed config (imports ARDY or Kimodo only when needed)."""
  return backend_class(cfg.kind)(cfg)
