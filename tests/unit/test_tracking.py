"""W&B run continuity (`crowdmotion.tracking`, `sim.mjlab.wandb_log`) against a fake W&B SDK.

No test reaches the W&B service: `lineage.SDK` is a recording stand-in.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path

import pytest
import torch

from crowdmotion.core.rundir import RunDir
from crowdmotion.tracking import evals, history, lineage, sync
from crowdmotion.tracking.evals import attach_eval
from fake_wandb import FakeRun, FakeSDK, MiniRunner

STAMP = "20261018T000000Z"


@pytest.fixture
def sdk(monkeypatch):
  fake = FakeSDK()
  monkeypatch.setattr(lineage, "SDK", fake)
  monkeypatch.delenv("WANDB_MODE", raising=False)
  monkeypatch.delenv("CROWD_WANDB_TAGS", raising=False)
  return fake


def make_run(tmp_path, name="train_traversal_a9", mode="online", **record) -> Path:
  run = tmp_path / f"{STAMP}_{name}"
  run.mkdir()
  rec = {"kind": "train_traversal", "policy": "lidar", "argv": [name], "status": "running", "attempts": [],
         "target_iterations": 4000, "train_config": {"max_iterations": 4000}, **record}
  if mode is not None:
    rec["wandb"] = lineage.new_block(run, mode)
  (run / "run.json").write_text(json.dumps(rec))
  return run


def write_tb(log_dir: Path, values: dict[int, float], tag="Loss/value"):
  from torch.utils.tensorboard import SummaryWriter

  w = SummaryWriter(str(log_dir))
  for i, v in values.items():
    w.add_scalar(tag, v, i)
    w.add_scalar("Train/mean_reward/time", v, 1000 + i)  # wall-clock stepped: never a row
  w.close()


def test_the_floor_is_published_through_else_the_old_uploads_last_step(sdk):
  block = {"entity": "e", "project": "p", "id": "r", "published_through": 3700}
  assert lineage.floor(block) == 3700
  assert lineage.floor({**block, "published_through": None}) == -1
  sdk.history["e/p/r"] = 3780
  assert lineage.floor({**block, "published_through": None, "legacy": True}) == 3780
  assert lineage.floor({**block, "id": "never", "published_through": None, "legacy": True}) == -1
  sdk.api_down = True  # a network error is not "never uploaded"
  with pytest.raises(sdk.errors.CommError):
    lineage.floor({**block, "published_through": None, "legacy": True})


def test_the_tensorboard_cut_keeps_each_iteration_from_the_attempt_that_continued(tmp_path):
  run = make_run(tmp_path)
  a1, a2 = run / "logs/x/a1", run / "logs/x/a2"
  write_tb(a1, {i: 1.0 for i in range(0, 3781)})  # 0-3780 logged, model_3700 the last checkpoint
  checkpoints(a1, range(0, 3701, 100))
  write_tb(a2, {i: 2.0 for i in range(3700, 3901)})  # resumed from model_3700, repeats label 3700
  checkpoints(a2, (3700, 3800, 3900))
  record = {"attempts": [{"log_dir": "logs/x/a1", "start_iteration": 0},
                         {"log_dir": "logs/x/a2", "start_iteration": 3700}]}
  assert history.segments(run, record, through=3900) == [(a1, -1, 3700), (a2, 3700, 3900)]
  rows = history.rows(run, record, after=-1, through=3900)
  assert list(rows) == list(range(0, 3901))
  assert rows[3700] == {"Loss/value": 1.0} and rows[3701]["Loss/value"] == 2.0 and rows[3780]["Loss/value"] == 2.0
  assert list(history.rows(run, record, after=3750, through=3800)) == list(range(3751, 3801))
  assert list(history.rows(run, record, after=-1, through=None))[-1] == 3900  # past the last checkpoint: none


def test_an_attempt_contributes_only_what_its_checkpoints_committed(tmp_path):
  """A branch from its parent's model_150 logs 150-160 and dies before its first save at
  200; the retry starts from the parent's checkpoint again (D20 finding 3)."""
  run = make_run(tmp_path)
  a1, a2, a3 = (run / f"logs/x/a{k}" for k in (1, 2, 3))
  write_tb(a1, {i: 1.0 for i in range(150, 161)})
  write_tb(a2, {i: 2.0 for i in range(150, 231)})
  checkpoints(a2, (200,))
  write_tb(a3, {i: 3.0 for i in range(200, 301)})  # resumed in place from a2's model_200
  checkpoints(a3, (300,))
  record = {"attempts": [{"log_dir": "logs/x/a1", "start_iteration": 150},
                         {"log_dir": "logs/x/a2", "start_iteration": 150},
                         {"log_dir": "logs/x/a3", "start_iteration": 200}]}
  assert history.segments(run, record, through=None) == [(a2, 149, 200), (a3, 200, 300)]
  rows = history.rows(run, record, after=-1, through=None)
  assert list(rows) == list(range(150, 301))
  assert {rows[i]["Loss/value"] for i in range(150, 201)} == {2.0} and rows[201]["Loss/value"] == 3.0
  assert history.first_after(run, record) == 200
  assert history.first_after(run, {"attempts": record["attempts"][:2]}) == 149  # the retry repeats no label
  # A fresh lineage whose first attempt died after logging row 0, before saving model_0.
  b1, b2 = run / "logs/y/b1", run / "logs/y/b2"
  write_tb(b1, {0: 1.0})
  write_tb(b2, {i: 2.0 for i in range(0, 11)})
  checkpoints(b2, (0, 10))
  fresh = {"attempts": [{"log_dir": "logs/y/b1", "start_iteration": 0}, {"log_dir": "logs/y/b2", "start_iteration": 0}]}
  assert {v["Loss/value"] for v in history.rows(run, fresh, after=-1, through=None).values()} == {2.0}


def test_disabled_lineages_write_nothing(tmp_path, sdk):
  for mode in ("disabled", None):  # W&B off, or a run from before the wandb block
    run = make_run(tmp_path, name=f"train_traversal_{mode}", mode=mode)
    before = (run / "run.json").read_bytes()
    assert attach_eval(run, tmp_path / "eval", iteration=100, kind="play", scalars={"x": 1.0}) == "disabled"
    assert (run / "run.json").read_bytes() == before and not (run / "wandb").exists()
    assert not (run / lineage.LOCK).exists()
  assert attach_eval(tmp_path / "missing", tmp_path / "eval", iteration=0, kind="play", scalars={}) == "disabled"
  assert sdk.runs == []


def test_an_evaluation_waits_in_the_outbox_while_a_writer_holds_the_lock(tmp_path, sdk):
  run = make_run(tmp_path)
  ev = tmp_path / f"{STAMP}_eval_a9"
  ev.mkdir()
  (ev / "play.mp4").write_bytes(b"mp4")
  live = lineage.WriterLock(run)
  assert live.acquire("training")
  before = (run / "run.json").read_bytes()
  kind = "traversal/aware/end"
  assert attach_eval(run, ev, iteration=2000, kind=kind, scalars={"rate": 0.5}, video=ev / "play.mp4") == "queued"
  key = f"{ev.name}/{kind}@2000"
  assert (run / "run.json").read_bytes() == before and sdk.runs == []
  (queued,) = evals.outbox(run).glob("*.json")
  assert "/" not in queued.name and json.loads(queued.read_text())["key"] == key
  assert attach_eval(run, ev, iteration=2000, kind=kind, scalars={"rate": 0.5}) == "queued"  # same key, same file
  assert len(list(evals.outbox(run).glob("*.json"))) == 1
  live.release()

  # The next holder drains the outbox: here another evaluation, out of iteration order.
  assert attach_eval(run, ev, iteration=1000, kind=kind, scalars={"rate": 0.7}) == "published"
  (session,) = sdk.runs
  assert session.kwargs["id"] == run.name and session.kwargs["resume"] == "allow" and session.finished == 0
  assert sorted((r["iteration"], r[f"eval/{kind}/rate"]) for r in session.rows) == [(1000, 0.7), (2000, 0.5)]
  assert all(r[f"eval/{kind}/id"] == ev.name for r in session.rows)
  ledger = json.loads((run / "run.json").read_text())["wandb"]["evals"]
  assert {k: v["status"] for k, v in ledger.items()} == {key: "published", f"{ev.name}/{kind}@1000": "published"}
  assert list(evals.outbox(run).glob("*.json")) == []
  # Idempotent: a published key is not logged again.
  assert attach_eval(run, ev, iteration=2000, kind=kind, scalars={"rate": 0.5}) == "published"
  assert len(sdk.runs) == 1


def test_many_rows_of_one_evaluation_go_out_in_one_session(tmp_path, sdk):
  from crowdmotion.tracking import flush_outbox, queue_eval

  run = make_run(tmp_path)
  ev = tmp_path / f"{STAMP}_eval_traversal_a9"
  rows = [(it, f"traversal/{crowd}/{mode}") for it in (2000, 3999) for crowd in ("aware", "blind")
          for mode in ("end", "continue")]
  assert {queue_eval(run, ev, iteration=it, kind=kind, scalars={"rate": 1.0}) for it, kind in rows} == {"queued"}
  assert sdk.runs == []
  result = flush_outbox(run)
  assert sorted(result) == sorted(f"{ev.name}/{kind}@{it}" for it, kind in rows)
  assert set(result.values()) == {"published"} and len(sdk.runs) == 1 and len(sdk.runs[0].rows) == 8
  assert flush_outbox(run) == {} and len(sdk.runs) == 1
  assert queue_eval(run, ev, iteration=2000, kind="traversal/aware/end", scalars={"rate": 1.0}) == "published"
  assert flush_outbox(make_run(tmp_path, name="train_traversal_off", mode="disabled")) == {}


def test_a_failed_attachment_is_recorded_and_kept_for_a_retry(tmp_path, sdk):
  run = make_run(tmp_path)
  ev = tmp_path / f"{STAMP}_play_a9"
  ev.mkdir()
  (ev / "play.mp4").write_bytes(b"mp4")
  sdk.fail = True
  assert attach_eval(run, ev, iteration=300, kind="play", scalars={"collisions_env0": 2}, video=ev / "play.mp4") \
    == "failed"
  entry = json.loads((run / "run.json").read_text())["wandb"]["evals"][f"{ev.name}/play@300"]
  assert entry["status"] == "failed" and "no network" in entry["error"]
  assert len(list(evals.outbox(run).glob("*.json"))) == 1
  sdk.fail = False
  assert sync.sync(run) == "synced"  # no checkpoints: only the outbox
  (session,) = sdk.runs
  assert session.rows == [{"iteration": 300, "eval/play/id": ev.name, "eval/play/collisions_env0": 2.0,
                           "eval/play/video": ("video", "play.mp4", "mp4")}]
  assert json.loads((run / "run.json").read_text())["wandb"]["evals"][f"{ev.name}/play@300"]["status"] == "published"


def checkpoints(log_dir: Path, iterations):
  log_dir.mkdir(parents=True, exist_ok=True)
  for i in iterations:
    (log_dir / f"model_{i}.pt").write_bytes(b"")


def test_sync_publishes_past_the_floor_up_to_the_newest_checkpoint(tmp_path, sdk):
  run = make_run(tmp_path, attempts=[{"log_dir": "logs/x/a1", "start_iteration": 0},
                                     {"log_dir": "logs/x/a2", "start_iteration": 300}])
  write_tb(run / "logs/x/a1", {i: 1.0 for i in range(0, 351)})
  write_tb(run / "logs/x/a2", {i: 2.0 for i in range(300, 481)})
  checkpoints(run / "logs/x/a1", (0, 100, 200, 300))
  checkpoints(run / "logs/x/a2", (300, 400))
  record = json.loads((run / "run.json").read_text())
  record["wandb"]["published_through"] = 200
  (run / "run.json").write_text(json.dumps(record))

  held = lineage.WriterLock(run)
  held.acquire("training")
  assert sync.sync(run) == "locked" and sdk.runs == []
  held.release()
  assert sync.sync(run, dry_run=True) == "dry-run" and sdk.runs == []
  assert sync.sync(run) == "synced"
  assert sdk.rows() == list(range(201, 401))  # 401-480 wait: a resume would replace them
  assert all("Train/mean_reward/time" not in r for r in sdk.runs[0].rows)
  assert json.loads((run / "run.json").read_text())["wandb"]["published_through"] == 400
  assert sync.sync(run) == "current" and len(sdk.runs) == 1


def test_sync_turns_on_only_explicitly_and_adopts_old_uploads(tmp_path, sdk):
  run = make_run(tmp_path, name="train_traversal_t9", mode=None)  # uploaded by the former script
  write_tb(run / "logs/x/2026-10-12_t9", {i: 1.0 for i in range(0, 251)})
  checkpoints(run / "logs/x/2026-10-12_t9", (0, 100, 200))
  assert sync.sync(run) == "disabled" and "wandb" not in json.loads((run / "run.json").read_text())
  sdk.history[f"rl2-group/crowd-control-traversal/{run.name}"] = 150
  assert sync.sync(run, enable=True) == "synced"
  assert sdk.rows() == list(range(151, 201))
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert block["mode"] == "online" and block["legacy"] and block["published_through"] == 200


# The live writer, driven by RSL-RL's own Logger through `MiniRunner`.

def attempt(run: Path, start: int, online=True, sdk_=None):
  from crowdmotion.sim.mjlab import wandb_log

  rd = RunDir.open(run)
  entry = {"n": len(rd.record["attempts"]) + 1, "start_iteration": start, "outcome": "running"}
  rd.record["attempts"].append(entry)
  rd.save()
  wandb_log.SESSION = wandb_log.Session(rd, entry, online=online, start_iteration=start, sdk=sdk_)
  return wandb_log.SESSION


@pytest.fixture
def tracked(monkeypatch):
  from crowdmotion.sim.mjlab import wandb_log

  monkeypatch.setattr(wandb_log, "SESSION", None)
  previous = signal.getsignal(signal.SIGTERM)
  yield type("Runner", (wandb_log.Tracked, MiniRunner), {}), wandb_log
  signal.signal(signal.SIGTERM, previous)


def train_cfg():
  return {"logger": "tensorboard", "algorithm": {}, "num_steps_per_env": 1, "save_interval": 100,
          "upload_model": True}


def test_without_a_session_the_runner_logs_tensorboard_only(tmp_path, tracked):
  Runner, _ = tracked
  runner = Runner({}, train_cfg(), str(tmp_path / "log"))
  runner.learn(3)
  assert type(runner.logger.writer).__name__ == "SummaryWriter"


def test_the_writer_publishes_one_row_per_iteration_through_each_checkpoint(tmp_path, sdk, tracked):
  Runner, wandb_log = tracked
  run = make_run(tmp_path)
  session = attempt(run, 0)
  log_dir = run / "logs/x/2026-10-18_a9"
  seen = {}

  def hook(it, runner):
    if it == 150:
      (log_dir / "videos").mkdir(exist_ok=True)
      (log_dir / "videos/rl-video-step-0.mp4").write_bytes(b"mp4")
    if it == 120:  # mid-interval: nothing past the last checkpoint is on W&B
      seen[120] = sdk.rows()

  runner = Runner({"hook": hook}, train_cfg(), str(log_dir))
  runner.learn(250)
  assert isinstance(runner.logger.writer, wandb_log.CrowdWandbWriter)
  assert seen[120] == list(range(0, 101))
  (w,) = sdk.runs
  assert ("*", {"step_metric": "iteration"}) in w.metrics and w.finished == 0 and not w.preempting
  assert w.kwargs["mode"] == "online" and w.kwargs["reinit"] == "finish_previous"
  assert w.kwargs["settings"]["finish_timeout"] == lineage.FINISH_TIMEOUT_S
  assert w.kwargs["settings"]["finish_timeout_raises"]
  assert sdk.rows() == list(range(250))  # one row per iteration, through the final model_249
  assert sdk.held(w.path) == list(range(250))
  assert all(not k.endswith("/time") for r in w.rows for k in r)
  assert [r["iteration"] for r in w.rows if "train/video" in r] == [150]  # offered every iteration, logged once
  assert {"Loss/value", "Perf/total_fps", "Policy/mean_std"} <= set(w.rows[5])
  record = json.loads((run / "run.json").read_text())
  assert record["wandb"]["published_through"] == 249 and record["wandb"]["status"] == "finished"
  assert "unconfirmed" not in record["wandb"]
  assert record["attempts"][0]["log_dir"] == "logs/x/2026-10-18_a9"
  assert session.writer.result is True and session.close() is True


def test_sigterm_checkpoints_then_a_resume_continues_past_the_floor(tmp_path, sdk, tracked):
  Runner, wandb_log = tracked
  run = make_run(tmp_path)
  signal.signal(signal.SIGTERM, wandb_log.on_sigterm)
  session = attempt(run, 0)
  with pytest.raises(wandb_log.Checkpointed) as stop:
    os.kill(os.getpid(), signal.SIGTERM)  # before training: stops at once
  assert stop.value.iteration is None

  def term(it, runner):
    if it == 137:
      os.kill(os.getpid(), signal.SIGTERM)
      os.kill(os.getpid(), signal.SIGTERM)  # a second TERM is harmless

  log1 = run / "logs/x/a1"
  with pytest.raises(wandb_log.Checkpointed) as stop:
    Runner({"hook": term}, train_cfg(), str(log1)).learn(300)
  assert stop.value.iteration == 137 and (log1 / "model_137.pt").exists()
  assert torch.load(log1 / "model_137.pt")["iter"] == 137
  assert session.close(preempting=True, exit_code=1)
  first = sdk.runs[0]
  assert first.preempting and [r["iteration"] for r in first.rows] == list(range(138))
  assert json.loads((run / "run.json").read_text())["wandb"]["published_through"] == 137

  # The requeued attempt resumes from model_137, repeats label 137 (dropped) and goes on.
  session = attempt(run, 137)
  log2 = run / "logs/x/a2"
  Runner({"start": 137}, train_cfg(), str(log2)).learn(300 - 137)
  assert sdk.rows() == list(range(300))  # 0-299 exactly once over both attempts
  assert sdk.runs[1].kwargs["id"] == sdk.runs[0].kwargs["id"] == run.name


def test_a_late_writer_fills_the_gap_from_earlier_attempts(tmp_path, sdk, tracked):
  Runner, _ = tracked
  run = make_run(tmp_path)
  attempt(run, 0, online=False)  # W&B off (or its init failed) in the first attempt
  Runner({}, train_cfg(), str(run / "logs/x/a1")).learn(230)  # TensorBoard 0-229, checkpoints to 229
  assert sdk.runs == []
  attempt(run, 200, online=True)  # resumed (here from model_200) with W&B on
  Runner({"start": 200, "offset": 0.5}, train_cfg(), str(run / "logs/x/a2")).learn(100)
  assert sdk.rows() == list(range(300))
  # From the first attempt through its checkpoint 200, from the second after it.
  assert [r["Loss/value"] for r in sdk.runs[0].rows if r["iteration"] in (200, 201)] == [200.0, 201.5]


def test_the_live_writer_drains_the_outbox_and_publishes_after_finish(tmp_path, sdk, tracked):
  Runner, _ = tracked
  run = make_run(tmp_path)
  ev = tmp_path / f"{STAMP}_eval"
  attempt(run, 0)
  states = {}

  def hook(it, runner):
    if it == 50:  # an evaluation during training: the trainer holds the lock
      states["attach"] = attach_eval(run, ev, iteration=0, kind="traversal/aware/end", scalars={"rate": 0.1})
    if it == 110:  # logged at checkpoint 100, published only once W&B's finish returns
      states["ledger"] = dict(runner.logger.writer.session.run.record["wandb"]["evals"])

  lock = lineage.WriterLock(run)
  assert lock.acquire("training")
  Runner({"hook": hook}, train_cfg(), str(run / "logs/x/a1")).learn(120)
  lock.release()
  assert states["attach"] == "queued"
  assert [v["status"] for v in states["ledger"].values()] == ["pending"]
  (w,) = sdk.runs
  assert [r["iteration"] for r in w.rows if "eval/traversal/aware/end/rate" in r] == [0]
  ledger = json.loads((run / "run.json").read_text())["wandb"]["evals"]
  assert ledger[f"{ev.name}/traversal/aware/end@0"]["status"] == "published"
  assert list(evals.outbox(run).glob("*.json")) == []


def test_a_wandb_failure_never_stops_training(tmp_path, sdk, tracked):
  Runner, _ = tracked
  run = make_run(tmp_path)
  sdk.fail = True
  attempt(run, 0)
  Runner({}, train_cfg(), str(run / "logs/x/a1")).learn(5)
  record = json.loads((run / "run.json").read_text())
  assert record["wandb"]["status"] == "failed" and "no network" in record["wandb"]["error"]
  assert history.read_scalars(run / "logs/x/a1")  # TensorBoard has every row


def test_a_save_killed_mid_write_leaves_no_checkpoint(tmp_path, tracked, monkeypatch):
  Runner, _ = tracked
  log_dir = tmp_path / "logs/x/a1"
  runner = Runner({}, train_cfg(), str(log_dir))
  runner.learn(3)
  assert sorted(p.name for p in log_dir.glob("model_*")) == ["model_0.pt", "model_2.pt"]  # no .tmp left

  def torn(obj, path):  # the process dies while writing
    Path(path).write_bytes(b"torn")
    raise KeyboardInterrupt

  monkeypatch.setattr(torch, "save", torn)
  with pytest.raises(KeyboardInterrupt):
    runner.save(str(log_dir / "model_3.pt"))
  assert not (log_dir / "model_3.pt").exists()
  assert [i for i, _ in lineage.own_checkpoints(tmp_path)] == [0, 2]


# The D20 findings of the 2026-10-18 review, one test (or more) each.

def kill(sdk):
  """The process died: its sessions deliver nothing more."""
  for run in sdk.active:
    run.finished = "killed"
  sdk.active.clear()


def test_a_crash_leaves_rows_unconfirmed_and_the_next_writer_replays_only_what_wandb_lacks(tmp_path, sdk, tracked):
  """Finding 1: `published_through` and the ledger advance only once W&B confirmed delivery."""
  Runner, wandb_log = tracked
  run = make_run(tmp_path)
  ev = tmp_path / f"{STAMP}_eval"
  kind = "traversal/aware/end"
  lock = lineage.WriterLock(run)
  assert lock.acquire("training")
  attempt(run, 0)

  def hook(it, runner):
    if it == 50:
      assert attach_eval(run, ev, iteration=0, kind=kind, scalars={"rate": 0.1}) == "queued"
    if it == 120:
      sdk.link_up = False  # rows 0-100 and the evaluation reached W&B; nothing after
    if it == 230:  # the node dies; TensorBoard reached the disk at each checkpoint (Tracked.save)
      raise RuntimeError("node failure")

  with pytest.raises(RuntimeError):
    Runner({"hook": hook}, train_cfg(), str(run / "logs/x/a1")).learn(300)
  kill(sdk)
  lock.release()
  block = json.loads((run / "run.json").read_text())["wandb"]
  path = sdk.runs[0].path
  assert sdk.rows("Loss/value") and sdk.held(path) == list(range(101))  # delivered 0-100
  assert [r["iteration"] for r in sdk.runs[0].rows if "Loss/value" in r] == list(range(201))  # logged 0-200
  assert block["published_through"] is None and block["unconfirmed"]["through"] == 200
  assert [e["status"] for e in block["evals"].values()] == ["pending"]

  sdk.link_up = True
  attempt(run, 200)  # resumed in place from model_200
  Runner({"start": 200, "offset": 0.5}, train_cfg(), str(run / "logs/x/a2")).learn(100)
  assert sdk.held(path) == list(range(300))  # each iteration once: 101-200 filled from TensorBoard
  assert sdk.held(path, kind) == [0]  # the evaluation W&B already held is not logged again
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert block["published_through"] == 299 and "unconfirmed" not in block
  assert [e["status"] for e in block["evals"].values()] == ["published"]
  assert list(evals.outbox(run).glob("*.json")) == []


def test_a_finish_timeout_is_unconfirmed_and_sync_replays_only_the_holes(tmp_path, sdk):
  """Findings 1 and 4: the SDK's finish timeout raises; nothing counts as published."""
  run = make_run(tmp_path, attempts=[{"log_dir": "logs/x/a1", "start_iteration": 0}])
  write_tb(run / "logs/x/a1", {i: 1.0 for i in range(0, 401)})
  checkpoints(run / "logs/x/a1", range(0, 401, 100))
  record = json.loads((run / "run.json").read_text())
  record["wandb"]["published_through"] = 200
  (run / "run.json").write_text(json.dumps(record))
  path = f"rl2-group/crowd-control-traversal/{run.name}"
  sdk.server[path] = [{"iteration": i, "Loss/value": 1.0, "_step": i} for i in range(201)]
  sdk.link_up = False
  assert sync.sync(run) == "failed"
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert block["published_through"] == 200 and block["unconfirmed"]["through"] == 400
  assert block["status"] == "failed" and "unconfirmed" in block["error"]
  assert lineage.WriterLock(run).free() and not lineage.stuck()

  # Part of the timed-out session landed, with a hole (251-270).
  sdk.server[path] += [{"iteration": i, "Loss/value": 1.0, "_step": 0} for i in [*range(201, 251), *range(271, 301)]]
  sdk.link_up = True
  assert sync.sync(run) == "synced"
  assert sdk.runs[-1].rows[0]["iteration"] == 251 and len(sdk.runs[-1].rows) == 20 + 100
  assert sorted(sdk.held(path)) == list(range(401))  # each iteration once
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert block["published_through"] == 400 and "published_above" not in block and "unconfirmed" not in block
  assert sync.sync(run) == "current"


def test_an_unreadable_wandb_is_never_guessed(tmp_path, sdk):
  run = make_run(tmp_path, attempts=[{"log_dir": "logs/x/a1", "start_iteration": 0}])
  write_tb(run / "logs/x/a1", {i: 1.0 for i in range(0, 101)})
  checkpoints(run / "logs/x/a1", (100,))
  record = json.loads((run / "run.json").read_text())
  evals.opened(record, "a killed trainer")
  (run / "run.json").write_text(json.dumps(record))
  sdk.api_down = True
  assert sync.sync(run) == "failed" and sdk.runs == []
  assert json.loads((run / "run.json").read_text())["wandb"]["unconfirmed"]["holder"] == "a killed trainer"
  ev = tmp_path / f"{STAMP}_eval"
  assert attach_eval(run, ev, iteration=100, kind="play", scalars={"x": 1.0}) == "failed" and sdk.runs == []
  sdk.api_down = False
  assert sync.sync(run) == "synced" and sorted(sdk.held(sdk.runs[0].path)) == list(range(101))
  assert sdk.held(sdk.runs[0].path, "play") == [100]


def test_a_failed_publication_is_finished_before_the_next_lineage(tmp_path, sdk):
  """Finding 2: with 0.30's default `reinit`, B's `wandb.init` would return A's open run."""
  a, b = make_run(tmp_path, name="train_traversal_a"), make_run(tmp_path, name="train_traversal_b")
  ev = tmp_path / f"{STAMP}_play"
  ev.mkdir()
  (ev / "play.mp4").write_bytes(b"mp4")
  assert evals.queue_eval(a, ev, iteration=10, kind="play", scalars={"x": 1.0}, video=ev / "play.mp4") == "queued"
  assert evals.queue_eval(b, ev, iteration=20, kind="play", scalars={"x": 2.0}) == "queued"
  sdk.video_fails = True  # raises after wandb.init succeeded
  assert sync.main(["sync", str(a), str(b)]) == 1  # A failed, B synced
  first, second = sdk.runs
  assert first.id == a.name and first.finished == 1 and second.id == b.name and second.finished == 0
  assert sdk.held(first.path, "play") == [] and sdk.held(second.path, "play") == [20]
  assert json.loads((a / "run.json").read_text())["wandb"]["evals"][f"{ev.name}/play@10"]["status"] == "failed"
  assert lineage.WriterLock(a).free() and lineage.WriterLock(b).free()


def test_wandb_must_return_the_lineages_own_online_run(tmp_path, sdk, monkeypatch):
  run = make_run(tmp_path)
  record = json.loads((run / "run.json").read_text())
  monkeypatch.setattr(sdk, "init", lambda **kw: FakeRun(sdk, {**kw, "id": "someone_else"}))
  with pytest.raises(RuntimeError, match="someone_else"):
    lineage.init_run(run, record)
  assert sdk.runs[0].finished == 1 and sdk.active == []
  monkeypatch.setattr(sdk, "init", lambda **kw: FakeRun(sdk, {**kw, "mode": "offline"}))
  with pytest.raises(RuntimeError, match="offline True"):
    lineage.init_run(run, record)


@pytest.fixture
def hung(sdk, monkeypatch):
  """W&B's finish() blocks until the test ends; the bounds are shortened."""
  monkeypatch.setattr(lineage, "FINISH_TIMEOUT_S", 0.05)
  monkeypatch.setattr(lineage, "FINISH_GRACE_S", 0.05)
  sdk.hang = threading.Event()
  yield sdk
  sdk.hang.set()
  for thread in lineage._STUCK:
    thread.join(5)
  lineage._STUCK.clear()


def test_a_stuck_finish_keeps_the_lock_and_stop_returns_its_first_result(tmp_path, hung, tracked):
  """Finding 4: the lock outlives a finish still running; a second stop() does not report success."""
  Runner, _ = tracked
  run = make_run(tmp_path)
  lock = lineage.WriterLock(run)
  assert lock.acquire("training")
  session = attempt(run, 0)
  Runner({}, train_cfg(), str(run / "logs/x/a1")).learn(5)  # RSL-RL's stop() at the end
  assert session.writer.result is False and session.writer.stop() is False and session.close() is False
  assert lineage.stuck()
  lock.release()
  assert lock.fd is not None  # kept until the process exits
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert block["published_through"] is None and "PublisherRunning" in block["error"] and block["unconfirmed"]
  assert sync.sync(make_run(tmp_path, name="train_traversal_other")) == "failed"  # no new session meanwhile
  hung.hang.set()
  for thread in lineage._STUCK:
    thread.join(5)
  lock.release()
  assert lock.fd is None


def test_check_runs_never_reach_wandb_by_any_path(tmp_path, sdk, tracked):
  """Finding 5: a lineage of at most CHECK_ITERATIONS iterations, whatever its recorded mode."""
  Runner, _ = tracked
  check = make_run(tmp_path, name="train_traversal_check", target_iterations=20,
                   train_config={"max_iterations": 20}, attempts=[{"log_dir": "logs/x/a1", "start_iteration": 0}])
  write_tb(check / "logs/x/a1", {i: 1.0 for i in range(20)})
  checkpoints(check / "logs/x/a1", (0, 19))
  ev = tmp_path / f"{STAMP}_eval"
  before = (check / "run.json").read_bytes()
  assert attach_eval(check, ev, iteration=19, kind="play", scalars={"x": 1.0}) == "disabled"
  assert evals.flush_outbox(check) == {}
  assert sync.sync(check) == sync.sync(check, enable=True) == "disabled"
  assert (check / "run.json").read_bytes() == before
  with pytest.raises(RuntimeError, match="check run"):
    lineage.init_run(check, json.loads(before))
  # An old check run without a wandb block is not adopted.
  old = make_run(tmp_path, name="train_traversal_old_check", mode=None, target_iterations=None,
                 train_config={"max_iterations": 5})
  assert sync.sync(old, enable=True) == "disabled" and "wandb" not in json.loads((old / "run.json").read_text())
  assert lineage.check_run({"kind": "train_traversal"})  # length unknown: not published
  # The live writer of a check lineage does not go live even when asked.
  live = make_run(tmp_path, name="train_traversal_live_check", target_iterations=5, train_config={"max_iterations": 5})
  attempt(live, 0)
  Runner({}, train_cfg(), str(live / "logs/x/a1")).learn(5)
  assert sdk.runs == []


def test_an_inherited_wandb_mode_is_refused_before_anything_is_recorded(tmp_path, sdk, tracked, monkeypatch):
  """Finding 6: `WANDB_MODE=offline` must not turn an online lineage's publication into a no-op."""
  Runner, _ = tracked
  monkeypatch.setenv("WANDB_MODE", "offline")
  run = make_run(tmp_path)
  attempt(run, 0)
  Runner({}, train_cfg(), str(run / "logs/x/a1")).learn(5)
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert sdk.runs == [] and block["status"] == "failed" and "WANDB_MODE=offline" in block["error"]
  assert block["published_through"] is None and "unconfirmed" not in block
  ev = tmp_path / f"{STAMP}_eval"
  assert attach_eval(run, ev, iteration=4, kind="play", scalars={"x": 1.0}) == "failed"
  assert json.loads((run / "run.json").read_text())["wandb"]["evals"][f"{ev.name}/play@4"]["status"] == "failed"
  monkeypatch.setenv("WANDB_MODE", "online")
  assert sync.sync(run) == "synced" and sdk.runs[0].kwargs["mode"] == "online"


def test_a_lineage_created_with_out_resumes_with_run():
  """Finding 7: the recorded `--out` gives way to the attempt's `--run`."""
  from crowdmotion.sim.mjlab.train import attempt_argv

  for out in (["--out", "runs/x"], ["--out=runs/x"]):
    record = {"argv": ["a3", *out, "--humans", "8", "--", "--agent.max-iterations", "4000"]}
    assert attempt_argv(["--run", "runs/x"], record) == ["a3", "--humans", "8", "--run", "runs/x", "--",
                                                         "--agent.max-iterations", "4000"]
    assert attempt_argv(record["argv"][:1] + ["--run", "runs/x", "--humans", "8", "--", "--agent.max-iterations",
                                               "4000"], record)[1:3] == ["--humans", "8"]


def test_test_runs_carry_their_tags_across_attempts(tmp_path, sdk, monkeypatch):
  monkeypatch.setenv("CROWD_WANDB_TAGS", "test, canary,,debug")
  monkeypatch.setenv("CROWD_WANDB_PROJECT", "crowd-control-traversal-debug")
  run = make_run(tmp_path, name="train_traversal_TEST_canary")
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert block["tags"] == ["test", "canary", "debug"] and block["project"] == "crowd-control-traversal-debug"
  monkeypatch.delenv("CROWD_WANDB_TAGS")  # a later attempt or an evaluation without the variable
  handle = lineage.init_run(run, json.loads((run / "run.json").read_text()))
  assert handle.kwargs["tags"] == ["lidar", "test", "canary", "debug"] and handle.kwargs["name"] == "TEST_canary"
  assert lineage.finish(handle)


# The Astra review of the fixes (logs of 2026-10-18): R1-R4.

def test_reading_wandb_is_bounded_and_a_term_meanwhile_stops_at_once(tmp_path, sdk, tracked, monkeypatch):
  """R1: a reconciliation that hangs must neither hold the attempt nor swallow its TERM."""
  Runner, wandb_log = tracked
  monkeypatch.setattr(lineage, "REMOTE_TIMEOUT_S", 0.3)
  release = threading.Event()
  sdk.scan_hook = lambda: release.wait(10)
  run = make_run(tmp_path)
  sdk.server[f"rl2-group/crowd-control-traversal/{run.name}"] = []  # the run exists on W&B
  record = json.loads((run / "run.json").read_text())
  evals.opened(record, "a killed trainer")
  (run / "run.json").write_text(json.dumps(record))
  attempt(run, 0)
  Runner({}, train_cfg(), str(run / "logs/x/a1")).learn(3)  # W&B gives up, training goes on
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert sdk.runs == [] and "TimeoutError" in block["error"] and block["unconfirmed"]["holder"] == "a killed trainer"
  assert sync.sync(run) == "failed"
  release.set()

  signal.signal(signal.SIGTERM, wandb_log.on_sigterm)
  sdk.scan_hook = lambda: (os.kill(os.getpid(), signal.SIGTERM), threading.Event().wait(10))
  session = attempt(run, 2)
  with pytest.raises(wandb_log.Checkpointed) as stop:
    Runner({"start": 2}, train_cfg(), str(run / "logs/x/a2")).learn(3)
  assert stop.value.iteration is None and session.phase == "finishing" and sdk.runs == []


def test_the_floor_crosses_only_rows_wandb_holds_or_that_are_lost(tmp_path, sdk):
  """R2: W&B holds 0 and 3, TensorBoard lacks 2 (a kill before it reached the disk)."""
  run = make_run(tmp_path, attempts=[{"log_dir": "logs/x/a1", "start_iteration": 0}])
  write_tb(run / "logs/x/a1", {i: 1.0 for i in (0, 1, *range(3, 11))})
  checkpoints(run / "logs/x/a1", (0, 10))
  record = json.loads((run / "run.json").read_text())
  record["wandb"]["published_through"] = 0
  evals.opened(record, "a killed trainer")
  (run / "run.json").write_text(json.dumps(record))
  path = f"rl2-group/crowd-control-traversal/{run.name}"
  sdk.server[path] = [{"iteration": i, "Loss/value": 1.0, "_step": k} for k, i in enumerate((0, 3))]
  record = json.loads((run / "run.json").read_text())
  evals.reconcile(run, record)
  assert record["wandb"]["published_through"] == 0 and record["wandb"]["published_above"] == [3]
  lineage.advance(record, {1})  # a confirmed row 1 does not carry the floor over the hole at 2
  assert record["wandb"]["published_through"] == 1 and record["wandb"]["published_above"] == [3]
  assert sync.sync(run) == "synced"
  assert sorted(sdk.held(path)) == [0, 1, *range(3, 11)]
  block = json.loads((run / "run.json").read_text())["wandb"]
  assert block["lost"] == [2] and block["published_through"] == 10 and "published_above" not in block


def test_an_evaluation_whose_video_did_not_upload_is_sent_again(tmp_path, sdk):
  """R3: W&B holds the row, which names the MP4, but not the file."""
  run = make_run(tmp_path)
  ev = tmp_path / f"{STAMP}_play"
  ev.mkdir()
  (ev / "play.mp4").write_bytes(b"mp4")
  assert evals.queue_eval(run, ev, iteration=5, kind="play", scalars={"x": 1.0}, video=ev / "play.mp4") == "queued"
  record = json.loads((run / "run.json").read_text())
  handle = lineage.init_run(run, record)
  evals.opened(record, "an attachment that was killed")
  (run / "run.json").write_text(json.dumps(record))
  sdk.media_fails = True
  handle.log(evals.row(evals.pending(run, record)[0]))
  kill(sdk)  # the row reached W&B, its video did not
  sdk.media_fails = False
  assert sync.sync(run) == "synced"
  path = handle.path
  assert sdk.held(path, "play") == [5, 5] and len(sdk.files[path]) == 1  # the scalars twice, the video once
  assert json.loads((run / "run.json").read_text())["wandb"]["evals"][f"{ev.name}/play@5"]["status"] == "published"
  # With its file on W&B, a delivered row is not sent again.
  assert evals.queue_eval(run, ev, iteration=6, kind="play", scalars={"x": 1.0}, video=ev / "play.mp4") == "queued"
  record = json.loads((run / "run.json").read_text())
  handle = lineage.init_run(run, record)
  evals.opened(record, "an attachment that was killed")
  (run / "run.json").write_text(json.dumps(record))
  handle.log(evals.row(evals.pending(run, record)[0]))
  kill(sdk)
  assert sync.sync(run) == "current" and sdk.held(path, "play") == [5, 5, 6]


def test_a_finish_stuck_while_opening_keeps_the_attempt_from_exiting_normally(tmp_path, hung, tracked):
  """R4: wandb.init returned, define_metric raised and the cleanup finish hung."""
  Runner, _ = tracked
  hung.metric_fails = True
  run = make_run(tmp_path)
  session = attempt(run, 0)
  Runner({}, train_cfg(), str(run / "logs/x/a1")).learn(3)
  assert session.writer.handle is None and session.writer.result is True and lineage.stuck()
  assert session.close() is False  # train.py then leaves through os._exit, holding the lock
