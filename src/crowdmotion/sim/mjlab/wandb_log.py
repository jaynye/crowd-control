"""`crowd train traversal`'s log writer and runner hooks: TensorBoard locally, W&B rows only
through the last saved checkpoint, and a checkpoint at a Slurm signal.

`train.py` opens a `Session` for one attempt of a lineage (`SESSION`). Without one, the
traversal runners log TensorBoard as RSL-RL does (mjlab's own scripts, `crowd play`). With
one:

- `Tracked`, mixed into both traversal runners, selects `CrowdWandbWriter` through
  RSL-RL's dict logger (`train_cfg["logger"]`) and publishes in `save`. mjlab calls the
  writer's `save_model` only when `upload_model` is on, so `save` is the one place every
  checkpoint passes. After each iteration's log call it checks the SIGTERM flag: it then
  saves `model_<it>.pt`, publishes, and raises `Checkpointed`.
- `CrowdWandbWriter` writes TensorBoard and, for an online lineage, keeps one row per
  iteration in memory: `iteration`, every scalar except RSL-RL's `.../time` tags (their
  step is wall-clock seconds, which restarts on resume), and new training videos as
  `train/video`. At its start it settles an earlier session that ended unconfirmed
  (`tracking.evals.reconcile`), marks its own `unconfirmed` and fills the earlier
  attempts' committed rows W&B lacks from TensorBoard (`tracking.history`).
  `publish_through(c)` sends the rows W&B lacks up to the saved checkpoint `c` and drains
  the evaluation outbox; this attempt's rows start above its checkpoint's label.
  `published_through` and the drained ledger entries advance only when `stop()`'s
  `finish()` confirms the session. Checkpoints and files are not uploaded.
- A W&B failure is recorded in `run.json` (`wandb.status`, `wandb.error`); the writer
  stops sending, `stop()` finishes its session unconfirmed, training continues on
  TensorBoard, and the next holder (`crowd wandb sync`, the next attempt) catches up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsl_rl.utils.log_writer import LogWriter
from torch.utils.tensorboard import SummaryWriter

from crowdmotion.core.rundir import RunDir
from crowdmotion.tracking import evals, history
from crowdmotion.tracking.lineage import (
  PublisherRunning,
  advance,
  enabled,
  finish,
  init_run,
  note_lost,
  sdk,
  settle_floor,
  stuck,
)


class Checkpointed(BaseException):
  """The attempt stopped at a checkpoint: `model_<iteration>.pt` was just saved, or (None)
  the signal came before training, so the lineage's newest checkpoint still stands. A
  BaseException, like KeyboardInterrupt, so that no `except Exception` swallows it."""

  def __init__(self, iteration: int | None):
    self.iteration = iteration
    super().__init__("before training" if iteration is None else f"model_{iteration}.pt")


@dataclass
class Session:
  """One attempt of a lineage: its run directory (the only in-process `run.json` writer),
  its `attempts` entry, and whether it publishes to W&B."""

  run: RunDir
  attempt: dict
  online: bool
  start_iteration: int
  sdk: Any = None
  phase: str = "setup"  # setup, training (inside learn), finishing
  stop: bool = False  # a SIGTERM arrived during training
  writer: CrowdWandbWriter | None = None

  def close(self, preempting: bool = False, exit_code: int = 0) -> bool:
    """Close TensorBoard and finish W&B; False while a W&B finish of this process is still
    running past its bound, this writer's or any other (the process must then exit, which
    releases the writer lock)."""
    done = True if self.writer is None else self.writer.stop(preempting=preempting, exit_code=exit_code)
    return done and not stuck()


SESSION: Session | None = None


def on_sigterm(signum, frame) -> None:
  """Slurm's TERM before the time limit, or a preemption. During training it only sets a
  flag (a second TERM is harmless); before training it stops at once."""
  session = SESSION
  if session is not None and session.phase != "setup":
    if not session.stop:
      print(f"signal {signum}: checkpointing after this iteration", flush=True)
    session.stop = True
    return
  raise Checkpointed(None)


class CrowdWandbWriter(SummaryWriter, LogWriter):
  """TensorBoard, and W&B rows through the last saved checkpoint (module docstring)."""

  def __init__(self, log_dir: str, **_: Any) -> None:
    super().__init__(log_dir=log_dir, flush_secs=10)
    self.session = SESSION
    self.rows: dict[int, dict] = {}
    self.videos: set[str] = set()
    self.sent: list[dict] = []  # outbox entries logged; published once finish() confirms
    self.handle = None
    self.live = False  # sending rows (False after a W&B failure)
    self.floor, self.above = -1, set()  # rows W&B holds: through the floor, and these above it
    self.first = -1  # this attempt's rows start above it
    self.sent_rows: set[int] = set()  # the training rows this session logged
    self.result: bool | None = None  # stop()'s
    session = self.session
    if session is None:
      return
    session.writer = self
    session.attempt["log_dir"] = str(Path(log_dir).resolve().relative_to(session.run.path.resolve()))
    session.run.save()
    if session.online and enabled(session.run.record):
      self._go_live()

  @property
  def block(self) -> dict:
    return self.session.run.record["wandb"]

  def _go_live(self) -> None:
    """Settle, open and fill the session. Before training starts, so a TERM meanwhile stops
    the attempt at once (`Checkpointed(None)`) rather than after reads of W&B."""
    session = self.session
    record, path = session.run.record, session.run.path
    phase, session.phase = session.phase, "setup"
    try:
      evals.reconcile(path, record, session.sdk)  # an earlier session that ended unconfirmed
      self.floor, self.above = settle_floor(self.block, session.sdk), evals.published_above(self.block)
      self.first = history.first_after(path, record)
      self.handle = init_run(path, record, session.sdk)
      self.live = True
      evals.opened(record, f"crowd train traversal attempt {session.attempt.get('n')}")
      session.run.save()
      gap = {i: v for i, v in history.rows(path, record, after=self.floor, through=self.first).items()
             if i not in self.above}
      note_lost(record, history.expected(path, record, self.floor, self.first) - self.above - set(gap)
                - set(self.block.get("lost") or ()))
      for i, values in gap.items():
        self.handle.log({"iteration": i, **values})
        self._sent(i)
      if gap:
        print(f"W&B: filled {len(gap)} rows ({min(gap)}-{max(gap)}) from earlier attempts", flush=True)
      self._drain()
      self.block["status"] = "live"
      self.block.pop("error", None)
      session.run.save()
      print(f"W&B: {self.block['url']} (floor {self.floor}, this attempt from {self.first + 1})", flush=True)
    except Exception as exc:
      self._fail(exc)
    finally:
      session.phase = phase

  def _fail(self, exc: Exception) -> None:
    """Stop sending; `stop()` finishes the session unconfirmed (the next holder reconciles)."""
    print(f"W&B failed, training continues on TensorBoard: {exc!r}", flush=True)
    self.live = False
    self.rows.clear()
    self.block.update(status="failed", error=repr(exc))
    self.session.run.save()

  def _sent(self, i: int) -> None:
    self.sent_rows.add(i)
    self.block["unconfirmed"]["through"] = max(self.sent_rows)

  def _drain(self) -> None:
    logged = {e["key"] for e in self.sent}
    new = [e for e in evals.pending(self.session.run.path, self.session.run.record) if e["key"] not in logged]
    for e in new:
      self.handle.log(evals.row(e, self.session.sdk))
    evals.mark(self.session.run.record, new, "pending")
    self.sent.extend(new)

  def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False,
                 double_precision=False) -> None:
    super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style,
                       double_precision=double_precision)
    if not self.live or global_step is None or tag.endswith("/time"):
      return
    self.rows.setdefault(int(global_step), {})[tag] = float(scalar_value)

  def save_video(self, video: Path, it: int) -> None:
    """RSL-RL offers every MP4 under the log directory at every iteration: each once."""
    if video.name in self.videos:
      return
    self.videos.add(video.name)
    if self.live:
      self.rows.setdefault(int(it), {})["train/video"] = str(video)

  def publish_through(self, it: int) -> None:
    """After `model_<it>.pt` is saved: send the rows W&B lacks up to `it`."""
    if not self.live:
      return
    try:
      for i in sorted(i for i in self.rows if i <= it):
        values = self.rows.pop(i)
        if i <= max(self.floor, self.first) or i in self.above:
          continue
        if "train/video" in values:
          values["train/video"] = sdk(self.session.sdk).Video(values["train/video"], format="mp4")
        self.handle.log({"iteration": i, **values})
        self._sent(i)
      self._drain()
      self.session.run.save()
    except Exception as exc:
      self._fail(exc)

  def stop(self, preempting: bool = False, exit_code: int = 0) -> bool:
    """Close TensorBoard and finish the W&B session (RSL-RL calls it after the final save,
    `Session.close` again): False while the finish is still running past its bound. A
    repeated call returns the first call's result."""
    if self.result is None:
      self.close()
      self.result = self._finish(preempting, exit_code)
    return self.result

  def _finish(self, preempting: bool, exit_code: int) -> bool:
    handle, self.handle = self.handle, None
    if handle is None:
      return True
    session = self.session
    try:
      done = finish(handle, preempting=preempting, exit_code=exit_code if self.live else 1)
    except PublisherRunning as exc:
      self.block.update(status="failed", error=repr(exc))
      session.run.save()
      return False
    except Exception as exc:
      self.block.update(status="failed", error=repr(exc))
      session.run.save()
      return True
    if not done:
      self.block.update(status="failed", error="W&B finish timed out; the upload is unconfirmed")
      session.run.save()
      return True
    if not self.live:  # a failed session: unconfirmed, for the next holder to reconcile
      return True
    advance(session.run.record, self.sent_rows)
    self.block.pop("unconfirmed", None)
    evals.mark(session.run.record, self.sent, "published")
    self.block["status"] = "preempted" if preempting else "finished" if exit_code == 0 else "crashed"
    session.run.save()
    evals.retire(session.run.path, self.sent)
    return True

  def save_model(self, model_path: str, it: int) -> None:
    """No checkpoint uploads: `run.json` points at the local files."""

  def save_file(self, path: str) -> None:
    """No file uploads."""

  def store_config(self, env_cfg, train_cfg: dict) -> None:
    """The W&B config is the `run.json` whitelist (`tracking.lineage.config`)."""


class Tracked:
  """Mixin for the traversal runners (before the RSL-RL runner class): with a `SESSION`,
  the `CrowdWandbWriter`, publishing in `save` and the SIGTERM checkpoint."""

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    tracked = SESSION is not None and log_dir is not None
    if tracked:
      train_cfg["logger"] = {"class_name": f"{__name__}:CrowdWandbWriter"}
    super().__init__(env, train_cfg, log_dir, device)
    if tracked:
      log = self.logger.log

      def log_then_check(*args, **kwargs):
        log(*args, **kwargs)
        if SESSION is not None and SESSION.stop:
          it = kwargs["it"] if "it" in kwargs else args[0]
          self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
          raise Checkpointed(it)

      self.logger.log = log_then_check

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    session = SESSION
    if session is None:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    session.phase = "training"
    try:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      session.phase = "finishing"

  def save(self, path: str, infos: dict | None = None) -> None:
    """Flush TensorBoard (a recovery replays committed rows from it), write `<path>.tmp`,
    then rename it: a kill mid-write (a preemption's SIGKILL) leaves no truncated
    `model_<i>.pt` for a resume to load or count."""
    writer = getattr(self.logger, "writer", None)
    if writer is not None and hasattr(writer, "flush"):  # every row the checkpoint commits is on disk
      writer.flush()
    tmp = f"{path}.tmp"
    super().save(tmp, infos)
    os.replace(tmp, path)
    if isinstance(writer, CrowdWandbWriter):
      writer.publish_through(self.current_learning_iteration)
