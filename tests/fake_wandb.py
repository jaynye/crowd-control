"""A recording stand-in for the W&B SDK, and a runner that drives RSL-RL's own `Logger` the
way `OnPolicyRunner.learn` does (rollouts and updates left out). For the tracking tests:
nothing here reaches the W&B service.

The stand-in models the wandb 0.30.0 behavior the tracking code relies on:

- `init` with a run still active returns that run (0.30's default outside notebooks,
  `return_previous`), unless `reinit` is `finish_previous` (finish it first) or
  `create_new`; `mode=None` inherits `WANDB_MODE`.
- `log` only queues a row; while `link_up` the row reaches the server (`server`) at
  once, else it waits for `finish`. A logged `Video` reaches the server as a reference
  (`{"_type": "video-file", "path", "size"}`) and its file as an upload (`files`), which
  `media_fails` drops. `finish` delivers what waits if the link is up, else
  (with `finish_timeout_raises`) raises TimeoutError, as the SDK does when its
  `finish_timeout` expires; `hang` (an Event) makes `finish` block until it is set.
- `Api().run(path)` raises `CommError` wrapping a `RunNotFoundError` for an unknown run,
  and a plain `CommError` while `api_down`; `scan_history` returns the server's rows
  with their `_step` (after calling `scan_hook`, if set), `files(names)` the uploads,
  and `lastHistoryStep` the last row (or `history[path]`, a run uploaded by the former
  TensorBoard script).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


class FakeRun:
  def __init__(self, sdk, kwargs):
    self.sdk, self.kwargs, self.rows, self.metrics = sdk, kwargs, [], []
    self.finished, self.preempting, self.delivered = None, False, 0
    self.entity, self.project, self.id = kwargs["entity"], kwargs["project"], kwargs["id"]
    mode = kwargs.get("mode") or os.environ.get("WANDB_MODE") or "online"
    self.offline, self.disabled = mode in ("offline", "dryrun"), mode == "disabled"
    sdk.runs.append(self)
    sdk.active.append(self)

  @property
  def path(self) -> str:
    return f"{self.entity}/{self.project}/{self.id}"

  def log(self, row, **kw):
    assert not kw, "rows carry `iteration`; never `step=`"
    assert self.finished is None, "log() after finish()"
    self.rows.append(dict(row))
    if self.sdk.link_up:
      self._deliver()

  def _deliver(self):
    if self.offline or self.disabled:
      return
    server = self.sdk.server.setdefault(self.path, [])
    for r in self.rows[self.delivered:]:
      row = {**r, "_step": len(server)}
      for key, value in r.items():
        if isinstance(value, tuple) and value[0] == "video":
          name = f"media/videos/{key}_{row['_step']}_{value[1]}"
          row[key] = {"_type": "video-file", "path": name, "size": 3}
          if not self.sdk.media_fails:
            self.sdk.files.setdefault(self.path, {})[name] = 3
      server.append(row)
    self.delivered = len(self.rows)

  def define_metric(self, name, **kw):
    if self.sdk.metric_fails:
      raise RuntimeError("define_metric failed")
    self.metrics.append((name, kw))

  def mark_preempting(self):
    self.preempting = True

  def finish(self, exit_code=0):
    if self.finished is not None:
      return
    self.finished = exit_code
    if self in self.sdk.active:
      self.sdk.active.remove(self)
    if self.sdk.hang is not None:
      self.sdk.hang.wait()
    if self.sdk.link_up:
      self._deliver()
    elif (self.kwargs.get("settings") or {}).get("finish_timeout_raises"):
      raise TimeoutError("Timed out finishing run.")


class FakeSDK:
  class errors:
    class CommError(Exception):
      def __init__(self, msg, exc=None):
        super().__init__(msg)
        self.exc = exc

  class RunNotFoundError(ValueError):
    pass

  def __init__(self, history=None):
    self.runs: list[FakeRun] = []
    self.active: list[FakeRun] = []
    self.history = history or {}  # path -> lastHistoryStep of a run uploaded by the TensorBoard script
    self.server: dict[str, list[dict]] = {}  # path -> the rows W&B holds
    self.files: dict[str, dict[str, int]] = {}  # path -> {uploaded file: size}
    self.fail = False
    self.media_fails = False
    self.metric_fails = False
    self.scan_hook = None
    self.link_up = True
    self.api_down = False
    self.hang = None
    self.video_fails = False

  def init(self, **kw):
    if self.fail:
      raise RuntimeError("no network")
    reinit = kw.get("reinit") or "default"
    if self.active:
      if reinit == "finish_previous":
        for run in list(self.active):
          run.finish()
      elif reinit != "create_new":
        return self.active[-1]
    return FakeRun(self, kw)

  def Video(self, path, format=None):  # noqa: N802 (the SDK's name)
    if self.video_fails:
      raise ValueError(f"cannot encode {path}")
    return ("video", Path(path).name, format)

  def Settings(self, **kw):  # noqa: N802
    return kw

  def Api(self, timeout=None):  # noqa: N802
    sdk = self

    class Run:
      def __init__(self, path):
        self.path = path

      @property
      def lastHistoryStep(self):  # noqa: N802
        if self.path in sdk.history:
          return sdk.history[self.path]
        return len(sdk.server.get(self.path, [])) - 1

      def scan_history(self, keys=None, page_size=1000, min_step=0, max_step=None, use_cache=True):
        assert keys is None and not use_cache
        if sdk.scan_hook is not None:
          sdk.scan_hook()
        return [dict(r) for r in sdk.server.get(self.path, [])]

      def files(self, names=None):
        held = sdk.files.get(self.path, {})
        return [type("File", (), {"name": n, "size": held[n]})() for n in names or held if n in held]

    class Api:
      def run(self, path):
        if sdk.api_down:
          raise sdk.errors.CommError("HTTP 502", ConnectionError("bad gateway"))
        if path not in sdk.history and path not in sdk.server:
          raise sdk.errors.CommError(f"Could not find run {path}", sdk.RunNotFoundError(path))
        return Run(path)

    return Api()

  def rows(self, key="iteration"):
    """The rows logged (queued), in order, across every session."""
    return [r[key] for run in self.runs for r in run.rows if key in r]

  def held(self, path, kind=None):
    """The iterations of the training rows (kind None) or of `kind`'s evaluation rows W&B holds."""
    rows = self.server.get(path, [])
    if kind is None:
      return [r["iteration"] for r in rows if not any(k.startswith("eval/") for k in r)]
    return [r["iteration"] for r in rows if f"eval/{kind}/id" in r]


class MiniRunner:
  """`OnPolicyRunner.learn`'s logging and saving: `env` holds `start` (the loaded
  checkpoint's iteration), `offset` (added to the logged loss) and `hook(it, runner)`,
  called at the start of each iteration."""
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    from rsl_rl.utils.logger import Logger

    self.env, self.cfg, self.current_learning_iteration = env, train_cfg, env.get("start", 0)
    self.logger = Logger(log_dir, train_cfg, {}, 1, False, 1, 0, "cpu")
    self.logger.git_status_repos = []

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    self.logger.init_logging_writer()
    start = self.current_learning_iteration
    for it in range(start, start + num_learning_iterations):
      (self.env.get("hook") or (lambda it, runner: None))(it, self)
      self.current_learning_iteration = it
      self.logger.log(it=it, start_it=start, total_it=start + num_learning_iterations, collect_time=0.1,
                      learn_time=0.1, loss_dict={"value": it + self.env.get("offset", 0.0)}, learning_rate=1e-3,
                      action_std=torch.ones(1), rnd_weight=None)
      self.logger.writer.add_scalar("Train/mean_reward/time", float(it), 7 * it)
      if it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
    self.logger.stop_logging_writer()

  def save(self, path, infos=None):
    torch.save({"iter": self.current_learning_iteration}, path)
    self.logger.save_model(path, self.current_learning_iteration)
