"""Typed configs from JSON dicts.

`from_dict(cls, d)` builds a dataclass, turning JSON lists into tuples where the
field's default is a tuple. Strict mode (the default) rejects unknown keys, so a
misspelled field cannot silently fall back to its default.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def unknown_fields(cls, d: dict | None) -> set[str]:
  return set(d or {}) - {f.name for f in dataclasses.fields(cls)}


def check_keys(d: dict | None, allowed, what: str) -> None:
  """Raise `ValueError("Unknown <what> keys: {...}")` for keys outside `allowed` (flat JSON configs)."""
  if extra := set(d or {}) - set(allowed):
    raise ValueError(f"Unknown {what} keys: {sorted(extra)}")


def check_fields(cls, d: dict | None, what: str) -> None:
  """Raise `ValueError("Unknown <what> fields: {...}")` for keys `cls` lacks."""
  if extra := unknown_fields(cls, d):
    raise ValueError(f"Unknown {what} fields: {extra}")


def _tupled(value, default):
  if isinstance(default, tuple) and isinstance(value, list):
    inner = default[0] if default else None
    return tuple(_tupled(v, inner) for v in value) if isinstance(inner, tuple) else tuple(
      tuple(v) if isinstance(v, list) else v for v in value
    )
  return value


def from_dict(cls: type[T], d: dict | None, *, strict: bool = True, what: str | None = None) -> T:
  d = dict(d or {})
  if strict:
    check_fields(cls, d, what or cls.__name__)
  else:
    d = {k: v for k, v in d.items() if k not in unknown_fields(cls, d)}
  defaults = {}
  for f in dataclasses.fields(cls):
    if f.default is not dataclasses.MISSING:
      defaults[f.name] = f.default
    elif f.default_factory is not dataclasses.MISSING:
      defaults[f.name] = f.default_factory()
  return cls(**{k: _tupled(v, defaults.get(k)) for k, v in d.items()})


def load_json(path) -> dict:
  return json.loads(Path(path).read_text())
