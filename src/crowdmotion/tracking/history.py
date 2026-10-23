"""A lineage's TensorBoard rows, each iteration taken from the attempt whose trajectory W&B keeps.

An attempt contributes only what its own checkpoints committed. Attempt k starts at
`s_k`: from a checkpoint an earlier attempt of the lineage saved, it repeats that label
(RSL-RL does), so its rows start above `s_k`; from the parent's checkpoint or from
scratch (a first attempt, or the retry of one that saved nothing) its rows start at
`s_k`. It contributes its rows up to its last own checkpoint `model_<c_k>.pt`, and none
that a later attempt's rows replace (a later attempt restarts from an earlier
checkpoint). So an attempt that died before its first save contributes nothing, and
rows logged past an attempt's last checkpoint belong to a discarded trajectory. Runs
recorded before `attempts` count each of their own log directories as one attempt.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from crowdmotion.tracking.lineage import LINKED, checkpoint_iteration


def read_scalars(log_dir: Path) -> dict[int, dict[str, float]]:
  """Every scalar of one log directory's TensorBoard files, by iteration. `.../time` tags,
  stepped by wall-clock seconds, are left out."""
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  steps: dict[int, dict[str, float]] = defaultdict(dict)
  for f in sorted(Path(log_dir).glob("events.*")):
    acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
    acc.Reload()
    for tag in acc.Tags()["scalars"]:
      if tag.endswith("/time"):
        continue
      for e in acc.Scalars(tag):
        steps[int(e.step)][tag] = float(e.value)
  return dict(steps)


def saved(log_dir: Path) -> list[int]:
  """The iterations of the checkpoints a log directory holds (written atomically)."""
  return sorted(i for p in Path(log_dir).glob("model_*.pt") if (i := checkpoint_iteration(p)) is not None)


def attempts_of(run: Path, record: dict) -> list[tuple[Path, int]]:
  """`(log dir, start iteration)` of each attempt, in order."""
  run = Path(run)
  out = [(run / a["log_dir"], int(a.get("start_iteration") or 0)) for a in record.get("attempts", [])
         if a.get("log_dir")]
  if not out:  # recorded before attempts: each own log directory, from its first row
    dirs = sorted(d for d in run.glob("logs/*/*") if d.is_dir() and not d.name.startswith(LINKED))
    out = [(d, 0 if k == 0 else min(read_scalars(d), default=0)) for k, d in enumerate(dirs)]
  return out


def bounds(attempts: list[tuple[Path, int]]) -> list[tuple[int, int | None]]:
  """`(after, upto)` per attempt: it starts above `after` and its last own checkpoint,
  cut where a later attempt starts, is `upto` (None: nothing committed)."""
  ckpts = [saved(d) for d, _ in attempts]
  after = [s if any(s in ckpts[j] for j in range(k)) else s - 1 for k, (_, s) in enumerate(attempts)]
  out = []
  for k in range(len(attempts)):
    upto = min([ckpts[k][-1], *after[k + 1:]]) if ckpts[k] else None
    out.append((after[k], upto))
  return out


def first_after(run: Path, record: dict) -> int:
  """The last attempt's rows start above this iteration."""
  attempts = attempts_of(run, record)
  return bounds(attempts)[-1][0] if attempts else -1


def segments(run: Path, record: dict, through: int | None) -> list[tuple[Path, int, int]]:
  """`(log dir, after, upto)`: the rows `after < i <= upto` each contributing attempt's
  log directory adds, `upto` at most `through` (None: no further bound)."""
  attempts = attempts_of(run, record)
  out = []
  for (log_dir, _), (after, upto) in zip(attempts, bounds(attempts), strict=True):
    if upto is not None and through is not None:
      upto = min(upto, through)
    if upto is not None and upto > after:
      out.append((log_dir, after, upto))
  return out


def expected(run: Path, record: dict, after: int, through: int | None) -> set[int]:
  """The committed iterations with `after < i <= through`: what `rows` returns when
  TensorBoard holds every row."""
  out: set[int] = set()
  for _, lo, hi in segments(run, record, through):
    out.update(range(max(lo, after) + 1, hi + 1))
  return out


def rows(run: Path, record: dict, after: int, through: int | None) -> dict[int, dict[str, float]]:
  """The lineage's committed rows with `after < i <= through`."""
  out: dict[int, dict[str, float]] = {}
  for log_dir, lo, hi in segments(run, record, through):
    lo = max(lo, after)
    if hi <= lo:
      continue
    for i, row in read_scalars(log_dir).items():
      if lo < i <= hi:
        out[i] = row
  return dict(sorted(out.items()))
