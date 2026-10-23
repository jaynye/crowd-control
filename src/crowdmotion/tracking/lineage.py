"""Lineage identity, settings, delivery and the one-writer lock, all kept in the run directory.

`run.json`'s `wandb` block (`new_block`) names the lineage's W&B run; its id is the
first run directory's name, as for the runs uploaded before this module existed. The
W&B display name is the label after `train_traversal_` (`teacher_t7`), the group the
policy (`lidar`, `teacher`, `student`), the tags the policy plus the block's `tags`
(`CROWD_WANDB_TAGS` when the lineage was created), the job type `train_traversal`, and
the config a whitelist of `run.json` (`config`: paths shortened to run-directory or
`configs/` names, no host details).

Delivery. W&B's `log()` only queues a row. `init_run` asks the SDK to bound `finish()`
(`finish_timeout`, raising on expiry), so a `finish()` that returns has uploaded the
session (`finish` returns True); one that timed out has stopped uploading, with an
unknown part delivered (False). A writer marks its session `unconfirmed` in the block
before its first row and records rows as published only once `finish` returned True; the
next holder reads what W&B holds (`remote_rows`, bounded by `REMOTE_TIMEOUT_S`) to
settle an unconfirmed session (`evals.reconcile`). A finish that does not return at all
keeps every writer lock of the process until it exits (`stuck`).

The floor. `published_through` is the iteration through which every committed row is on
W&B; rows known to be there above a gap are in `published_above`, and committed rows that
neither TensorBoard nor W&B holds (a kill before TensorBoard reached the disk) are
recorded in `lost`, which the floor may cross (`advance`).

`wandb` is imported only when a lineage is online. Tests set `SDK` to a stand-in.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import platform
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from crowdmotion.core.rundir import write_json

# New lineages' W&B mode: on (user, 2026-10-21), after a canary in a scratch project
# passed. `--wandb disabled` keeps a lineage off.
DEFAULT_MODE = "online"
MODES = ("online", "disabled")
ENTITY, PROJECT = "rl2-group", "crowd-control-traversal"
# A lineage training at most this many iterations is a verification run and stays off
# W&B whatever `--wandb` or `crowd wandb sync --enable` says (user, 2026-10-18).
CHECK_ITERATIONS = 20
FINISH_TIMEOUT_S = 120.0  # the SDK's bound on finish(): it must not eat the checkpoint window before a requeue
FINISH_GRACE_S = 60.0  # past it, a finish() that has not returned is stuck
REMOTE_TIMEOUT_S = 120.0  # a read of W&B's history must not hold an attempt past its signal lead
LOCK = "wandb.lock"
LINKED = ("resume_", "teacher_", "init_")  # log directories holding another run's checkpoint
CONFIG = ("kind", "task", "argv", "humans", "perceive_robot", "forward_first", "teacher_lidar", "no_avoidance",
          "penalty_ramp", "planner_rate", "crowd_curriculum", "status", "train_config", "target_iterations")
PATHS = ("graph", "rotations", "behavior", "teacher", "resume_from", "finetune_from")

SDK = None  # a stand-in for the `wandb` module (tests); None imports wandb
_STUCK: list[threading.Thread] = []  # finish() calls still running past their bound


class PublisherRunning(RuntimeError):
  """W&B's `finish()` did not return within its bound, so the session may still be
  uploading: the process keeps every writer lock it holds until it exits."""


def sdk(given=None):
  return given if given is not None else SDK if SDK is not None else importlib.import_module("wandb")


def settings() -> tuple[str, str]:
  """(entity, project) for new lineages: `CROWD_WANDB_ENTITY`, `CROWD_WANDB_PROJECT`."""
  return os.environ.get("CROWD_WANDB_ENTITY", ENTITY), os.environ.get("CROWD_WANDB_PROJECT", PROJECT)


def env_tags() -> list[str]:
  """Extra W&B tags for new lineages: `CROWD_WANDB_TAGS`, comma-separated (test runs)."""
  return [t.strip() for t in os.environ.get("CROWD_WANDB_TAGS", "").split(",") if t.strip()]


def utc_now() -> str:
  return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_record(run: Path) -> dict:
  return json.loads((Path(run) / "run.json").read_text())


def write_record(run: Path, record: dict) -> None:
  """Only the lock holder writes; the write is atomic for concurrent readers."""
  write_json(Path(run) / "run.json", record)


def env_overrides(argv: list[str]) -> list[str]:
  """The `--env.*` options (with their values) among the mjlab arguments after `--`."""
  theirs = argv[argv.index("--") + 1:] if "--" in argv else []
  out, keep = [], False
  for token in theirs:
    if token.startswith("--"):
      keep = token.startswith("--env.")
    if keep:
      out.append(token)
  return out


def policy_of(record: dict) -> str:
  """Runs recorded before the `policy` key: from the command line."""
  argv = record.get("argv", [])
  return record.get("policy") or ("teacher" if "--teacher" in argv else "student" if "--distill" in argv
                                  else "lidar")


def label(run: Path) -> str:
  return Path(run).name.split("_train_traversal_", 1)[-1]


def short(path: str) -> str:
  """From the run directory or `configs/` on: no machine-specific prefix."""
  parts = Path(path).parts
  for anchor in ("runs", "configs"):
    if anchor in parts:
      i = parts.index(anchor)
      return "/".join(parts[i + 1:] if anchor == "runs" else parts[i:])
  return Path(path).name


def config(run: Path, record: dict) -> dict:
  cfg = {k: record[k] for k in CONFIG if k in record}
  cfg.update({k: short(record[k]) for k in PATHS if record.get(k)})
  env = record.get("environment", {})
  repo = env.get("repository", {})
  cfg.update(policy=policy_of(record), run_dir=Path(run).name, env_overrides=env_overrides(record.get("argv", [])),
             gpu=env.get("gpu"), commit=repo.get("commit"), dirty_files=repo.get("dirty_files"),
             parent=(record.get("wandb") or {}).get("parent"))
  return cfg


def checkpoint_iteration(path: Path) -> int | None:
  m = re.fullmatch(r"model_(\d+)\.pt", Path(path).name)
  return int(m.group(1)) if m else None


def own_checkpoints(run: Path) -> list[tuple[int, Path]]:
  """`(iteration, model_<i>.pt)` the run wrote itself, skipping the log directories it
  links to (`LINKED`: the run it resumed, its teacher, its init)."""
  return sorted((i, p) for p in Path(run).glob("logs/*/*/model_*.pt")
                if (i := checkpoint_iteration(p)) is not None and not p.parent.name.startswith(LINKED))


def lineage_target(record: dict) -> int:
  """One past the lineage's last iteration: recorded, or for runs recorded before
  `target_iterations`, their start plus their `max_iterations`."""
  if record.get("target_iterations") is not None:
    return int(record["target_iterations"])
  start = checkpoint_iteration(record["resume_from"]) if record.get("resume_from") else 0
  return int(start or 0) + int(record["train_config"]["max_iterations"])


def first_start(record: dict) -> int:
  attempts = record.get("attempts") or [{}]
  if attempts[0].get("start_iteration") is not None:
    return int(attempts[0]["start_iteration"])
  return int(checkpoint_iteration(record["resume_from"]) or 0) if record.get("resume_from") else 0


def check_run(record: dict) -> bool:
  """A verification run (at most `CHECK_ITERATIONS` iterations), which never reaches W&B;
  so is a run whose length is not recorded."""
  try:
    return lineage_target(record) - first_start(record) <= CHECK_ITERATIONS
  except (KeyError, TypeError, ValueError):
    return True


def lineage_id(run: Path, record: dict | None = None) -> str:
  """The W&B id of a training run: recorded, else (runs before this module) its name."""
  return ((record or {}).get("wandb") or {}).get("id") or Path(run).name


def new_block(run: Path, mode: str, parent: str | None = None, status: str | None = None,
              legacy: bool = False) -> dict:
  entity, project = settings()
  run_id = Path(run).name
  out = {"entity": entity, "project": project, "id": run_id,
         "url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}", "mode": mode, "parent": parent,
         "tags": env_tags(), "published_through": None, "status": status or mode, "evals": {}}
  if legacy:  # uploaded by the TensorBoard script: its floor is W&B's last step
    out["legacy"] = True
  return out


def enabled(record: dict) -> bool:
  """Online and not a check run: the one test every publishing path applies."""
  return ((record.get("wandb") or {}).get("mode")) == "online" and not check_run(record)


def not_found(exc: Exception) -> bool:
  """`Api().run` failed because the run does not exist: wandb 0.30 wraps its
  `RunNotFoundError` in a `CommError` (`.exc`), as it does network errors."""
  return any(type(e).__name__ == "RunNotFoundError" for e in (exc, getattr(exc, "exc", None)))


def bounded(fn, timeout: float | None = None):
  """`fn()` in a daemon thread, awaited in short slices so that the main thread's signal
  handlers still run: its result, or TimeoutError after `timeout` seconds (default
  `REMOTE_TIMEOUT_S`; the abandoned thread only reads)."""
  out, errors = [], []

  def go():
    try:
      out.append(fn())
    except Exception as exc:  # re-raised in the caller's thread
      errors.append(exc)

  limit = REMOTE_TIMEOUT_S if timeout is None else timeout
  thread = threading.Thread(target=go, name="wandb-read", daemon=True)
  thread.start()
  deadline = time.monotonic() + limit
  while thread.is_alive() and time.monotonic() < deadline:
    thread.join(min(0.5, max(deadline - time.monotonic(), 0.01)))
  if thread.is_alive():
    raise TimeoutError(f"reading W&B took longer than {limit:g} s")
  if errors:
    raise errors[0]
  return out[0]


def api_run(block: dict, sdk_=None):
  """The lineage's run through W&B's public API; None if it does not exist. Raises when
  W&B cannot be asked. Unbounded: call it inside `bounded`."""
  wandb = sdk(sdk_)
  try:
    return wandb.Api(timeout=60).run(f"{block['entity']}/{block['project']}/{block['id']}")
  except Exception as exc:
    if not_found(exc):
      return None
    raise


def floor(block: dict, sdk_=None) -> int:
  """The highest iteration through which every row is on W&B: `published_through`, else
  for a lineage uploaded by the TensorBoard script its `lastHistoryStep` (its rows were
  logged at step = iteration), else -1. Rows above it may also be on W&B
  (`published_above`, after a reconciliation)."""
  if block.get("published_through") is not None:
    return int(block["published_through"])
  if not block.get("legacy"):
    return -1

  def last():
    run = api_run(block, sdk_)
    return None if run is None else run.lastHistoryStep

  step = bounded(last)
  return -1 if step is None else int(step)


def settle_floor(block: dict, sdk_=None) -> int:
  """`floor`, recorded as `published_through` (an adopted run's, so W&B is not asked again)."""
  low = floor(block, sdk_)
  if block.get("published_through") is None and low >= 0:
    block["published_through"] = low
  return low


def remote_rows(block: dict, sdk_=None) -> list[dict]:
  """Every history row W&B holds for the lineage (`scan_history`, uncached), [] if the
  run does not exist. Raises when W&B cannot be read within `REMOTE_TIMEOUT_S`."""
  def read():
    run = api_run(block, sdk_)
    return [] if run is None else list(run.scan_history(page_size=1000, use_cache=False))

  return bounded(read)


def remote_files(block: dict, names, sdk_=None) -> dict[str, int]:
  """`{name: size}` of those of the run's uploaded files W&B holds (media of logged rows)."""
  def read():
    run = api_run(block, sdk_)
    return {} if run is None else {f.name: int(f.size or 0) for f in run.files(names=sorted(names))}

  return bounded(read)


def advance(record: dict, rows) -> None:
  """Take the training `rows` as on W&B: the floor rises through every iteration covered
  without a gap (these rows, `published_above` and `lost`, from the lineage's first row
  on), and covered rows above a remaining gap stay in `published_above`."""
  block = record["wandb"]
  done = block.get("published_through")
  low = -1 if done is None else int(done)
  lost = set(block.get("lost") or ())
  covered = {int(i) for i in (*rows, *(block.get("published_above") or ()), *lost) if int(i) > low}
  top = base = max(low, first_start(record) - 1)  # rows below the lineage's first one never exist
  while top + 1 in covered:
    top += 1
  if top > base:
    block["published_through"] = low = top
  above = sorted(i for i in covered if i > low and i not in lost)
  if above:
    block["published_above"] = above
  else:
    block.pop("published_above", None)


def note_lost(record: dict, lost) -> None:
  """Record committed rows that neither TensorBoard nor W&B holds (they cannot be sent)."""
  lost = set(lost)
  if lost:
    block = record["wandb"]
    block["lost"] = sorted(lost | set(block.get("lost") or ()))
    print(f"W&B: {len(lost)} committed rows are in neither TensorBoard nor W&B and stay missing: "
          f"{sorted(lost)[:10]}{' ...' if len(lost) > 10 else ''}", flush=True)


def tags(block: dict, policy: str) -> list[str]:
  return list(dict.fromkeys([policy, *(block.get("tags") or ())]))


def init_run(run: Path, record: dict, sdk_=None):
  """Open (or resume) the lineage's W&B run, online, with `iteration` as every metric's
  step and the SDK's bounded finish. Refuses a check run and an inherited `WANDB_MODE`
  other than online, and checks that W&B returned this lineage's run, online."""
  if check_run(record):
    raise RuntimeError(f"{Path(run).name} is a check run (at most {CHECK_ITERATIONS} iterations): off W&B")
  inherited = os.environ.get("WANDB_MODE")
  if inherited not in (None, "", "online"):
    raise RuntimeError(f"WANDB_MODE={inherited} contradicts the online lineage {Path(run).name}; unset it")
  wandb = sdk(sdk_)
  block = record["wandb"]
  (Path(run) / "wandb").mkdir(parents=True, exist_ok=True)
  policy = policy_of(record)
  # reinit: 0.30's default outside notebooks returns a still-active run (another lineage's).
  handle = wandb.init(entity=block["entity"], project=block["project"], id=block["id"], resume="allow",
                      mode="online", reinit="finish_previous", name=label(run), group=policy,
                      job_type="train_traversal", tags=tags(block, policy), config=config(run, record),
                      dir=str(Path(run) / "wandb"),
                      settings=wandb.Settings(console="off", finish_timeout=FINISH_TIMEOUT_S,
                                              finish_timeout_raises=True))
  got = (handle.entity, handle.project, handle.id)
  if got != (block["entity"], block["project"], block["id"]) or handle.offline or handle.disabled:
    abandon(handle)
    raise RuntimeError(f"wandb.init returned {'/'.join(map(str, got))} (offline {handle.offline}, disabled "
                       f"{handle.disabled}), not the online run {block['entity']}/{block['project']}/{block['id']}")
  try:
    handle.define_metric("iteration")
    handle.define_metric("*", step_metric="iteration")
  except Exception:
    abandon(handle)
    raise
  return handle


def stuck() -> bool:
  """Whether a `finish()` of this process is still running past its bound."""
  return any(t.is_alive() for t in _STUCK)


def finish(handle, *, preempting: bool = False, exit_code: int = 0, timeout: float | None = None) -> bool:
  """`finish()` (after `mark_preempting()`): True once W&B has uploaded the session,
  False if the SDK gave up after `FINISH_TIMEOUT_S` (it stopped uploading; what it sent is
  unconfirmed). Raises PublisherRunning if `finish()` has not returned after `timeout`
  seconds (default `FINISH_TIMEOUT_S + FINISH_GRACE_S`), else what the SDK raised."""
  errors = []

  def go():
    try:
      if preempting:
        handle.mark_preempting()
      handle.finish(exit_code=exit_code)
    except Exception as exc:  # re-raised in the caller's thread
      errors.append(exc)

  thread = threading.Thread(target=go, name="wandb-finish", daemon=True)
  thread.start()
  thread.join(FINISH_TIMEOUT_S + FINISH_GRACE_S if timeout is None else timeout)
  if thread.is_alive():
    _STUCK.append(thread)
    raise PublisherRunning("W&B finish() has not returned; keeping the writer lock until this process exits")
  if errors:
    if isinstance(errors[0], TimeoutError):  # `finish_timeout_raises`
      return False
    raise errors[0]
  return True


def abandon(handle) -> None:
  """Finish a session that failed, confirming nothing; raises only PublisherRunning."""
  try:
    finish(handle, exit_code=1)
  except PublisherRunning:
    raise
  except Exception as exc:
    print(f"W&B finish after a failure: {exc!r}", flush=True)


class WriterLock:
  """`<run>/wandb.lock`, an exclusive flock: its holder is the only writer of the run's W&B
  run and of the `wandb` block of its `run.json`. The holder's description is written
  into the file (`holder`). On NFS, Linux emulates flock with POSIX locks, which a
  process drops when it closes any descriptor of the file: the holding process never
  opens the file again (`holder` is for other processes)."""

  def __init__(self, run: Path):
    self.path = Path(run) / LOCK
    self.fd: int | None = None

  def acquire(self, purpose: str, timeout: float = 0.0) -> bool:
    """Take the lock, waiting up to `timeout` seconds; False if another process holds it."""
    fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + timeout
    while True:
      try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
      except BlockingIOError:
        if time.monotonic() >= deadline:
          os.close(fd)
          return False
        time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))
    holder = {"purpose": purpose, "host": platform.node(), "pid": os.getpid(),
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "since": utc_now()}
    os.ftruncate(fd, 0)
    os.pwrite(fd, json.dumps(holder).encode(), 0)
    self.fd = fd
    return True

  def release(self) -> None:
    """Unlock, unless a W&B finish of this process is stuck (`stuck`): the session may
    still be uploading, so the lock is kept until the process exits."""
    if self.fd is None:
      return
    if stuck():
      print(f"{self.path}: kept until this process exits (a W&B finish is still running)", flush=True)
      return
    fcntl.flock(self.fd, fcntl.LOCK_UN)
    os.close(self.fd)
    self.fd = None

  def holder(self) -> dict | None:
    """Who took the lock last (it may since have been released)."""
    try:
      return json.loads(self.path.read_text() or "null")
    except (OSError, ValueError):
      return None

  def free(self) -> bool:
    """Whether no process holds the lock now (a probe; never call it while holding it)."""
    if not self.path.exists():
      return True
    fd = os.open(self.path, os.O_RDONLY)
    try:
      fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
      fcntl.flock(fd, fcntl.LOCK_UN)
      return True
    except BlockingIOError:
      return False
    finally:
      os.close(fd)
