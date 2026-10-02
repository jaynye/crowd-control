"""Run directories: `runs/<UTC>_<name>/` with a `run.json` lifecycle record.

  with RunDir.create("behavior_eval", out=args.out, record={"config": cfg}) as run:
    run.write_json("summary.json", ...)

`run.json` holds status (running, then completed or failed with the error),
the caller's record fields and `environment()`. A dirty git tree also leaves
`source.diff`. Directories are never reused.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from crowdmotion.core.paths import RUNS
from crowdmotion.core.provenance import environment, git_diff


def utc_stamp() -> str:
  return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, default=str))


class RunDir:
  def __init__(self, path: Path, record: dict):
    self.path = Path(path)
    self.record = record

  @classmethod
  def create(cls, name: str, out: Path | None = None, record: dict | None = None,
             env: bool = True) -> "RunDir":
    """Make a fresh directory: `out` if given (must not exist), else `runs/<UTC>_<name>`,
    with `_2`, `_3`, ... appended when another run took that name in the same second."""
    if out is not None:
      path = Path(out)
      path.mkdir(parents=True, exist_ok=False)
    else:
      stem = RUNS / f"{utc_stamp()}_{name}"
      RUNS.mkdir(parents=True, exist_ok=True)
      for n in range(1, 1000):
        path = stem if n == 1 else stem.with_name(f"{stem.name}_{n}")
        try:
          path.mkdir(exist_ok=False)
          break
        except FileExistsError:
          continue
    rec = {"status": "running", **(record or {})}
    if env:
      rec["environment"] = environment()
      diff = git_diff()
      if diff:
        (path / "source.diff").write_text(diff)
    run = cls(path, rec)
    run.save()
    return run

  def save(self) -> None:
    write_json(self.path / "run.json", self.record)

  def update(self, **fields) -> None:
    self.record.update(fields)
    self.save()

  def write_json(self, relative: str, payload) -> Path:
    path = self.path / relative
    write_json(path, payload)
    return path

  def __enter__(self) -> "RunDir":
    return self

  def __exit__(self, exc_type, exc, tb) -> bool:
    if exc is None:
      self.record["status"] = "completed"
    else:
      self.record.update(status="failed", error=f"{exc_type.__name__}: {exc}")
    self.save()
    return False
