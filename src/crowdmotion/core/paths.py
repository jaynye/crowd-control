"""Repository locations and the artifact manifest (`artifacts.toml`)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def _is_root(path: Path) -> bool:
  return (path / "third_party.toml").exists() and (path / "pyproject.toml").exists()


def _find_root() -> Path:
  """`$CROWD_ROOT`, else the checkout containing this package (editable install), else
  the checkout containing the working directory, else the working directory."""
  if os.environ.get("CROWD_ROOT"):
    return Path(os.environ["CROWD_ROOT"]).resolve()
  for start in (Path(__file__).resolve(), Path.cwd().resolve()):
    for parent in (start, *start.parents):
      if _is_root(parent):
        return parent
  return Path.cwd()


ROOT = _find_root()
RUNS = ROOT / "runs"


def checkpoints_dir() -> Path:
  """Model weights: `$CROWD_CHECKPOINTS`, else `<root>/checkpoints` (as `scripts/env.sh`)."""
  return Path(os.environ.get("CROWD_CHECKPOINTS") or ROOT / "checkpoints")


def artifact_path(name: str) -> Path:
  """Local path of artifact `name` from `artifacts.toml` (may not exist yet)."""
  path = ROOT / "artifacts.toml"
  if not path.exists():
    raise FileNotFoundError(f"{path} not found: run inside the checkout or set CROWD_ROOT (or CROWD_GRAPH)")
  manifest = tomllib.loads(path.read_text())
  try:
    return ROOT / manifest["artifact"][name]["path"]
  except KeyError:
    raise KeyError(f"no artifact {name!r} in artifacts.toml") from None


def default_graph() -> Path:
  """The saved motion graph: `$CROWD_GRAPH`, else artifact `graph-default`."""
  env = os.environ.get("CROWD_GRAPH")
  return Path(env) if env else artifact_path("graph-default")


def default_graph_rotations() -> Path:
  """Joint rotations for the default graph: `$CROWD_GRAPH_ROTATIONS`, else artifact
  `graph-rotations` (`crowd graph rotations`)."""
  env = os.environ.get("CROWD_GRAPH_ROTATIONS")
  return Path(env) if env else artifact_path("graph-rotations")
