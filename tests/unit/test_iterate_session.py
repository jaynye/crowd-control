"""Iteration sessions: local execution limits, immutable names, journal, and the
Slurm watcher and worker with a mocked scheduler (no Slurm calls occur)."""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from crowdmotion.iterate import executors, session

PY = sys.executable


def _start(tmp_path, hours=1.0):
  return session.create("t", tmp_path / "s", hours, "local", 300, (), "plan text", None)


def _result(s, name):
  return json.loads((s / "queue" / f"{name}.result.json").read_text())


def test_local_request_runs_and_names_are_immutable(tmp_path):
  s = _start(tmp_path)
  out = session.queue_request(s, "000_hello", [PY, "-c", "print('hi {name}')", "{out}"], 60, "rule A")
  assert out["status"] == "completed" and out["returncode"] == 0
  assert (s / "queue" / "000_hello.done.json").exists()
  assert "hi 000_hello" in (s / "queue" / "000_hello.log").read_text()
  assert _result(s, "000_hello")["argv"][-1] == str(s / "iterations" / "000_hello")
  text = (s / "journal.md").read_text()
  assert text.index("Rule change tested by 000_hello") < text.index("Request 000_hello") < text.index("Result 000_hello")
  with pytest.raises(SystemExit, match="immutable"):
    session.queue_request(s, "000_hello", [PY, "-c", "pass"], 60, None)
  failed = session.queue_request(s, "001_fail", [PY, "-c", "raise SystemExit(3)"], 60, None)
  assert failed["status"] == "failed" and failed["returncode"] == 3
  assert [r["status"] for r in session.requests(s)] == ["completed", "failed"]


def test_local_timeout_terminates_the_request(tmp_path):
  s = _start(tmp_path)
  t0 = time.time()
  out = session.queue_request(s, "slow", [PY, "-c", "import time; time.sleep(60)"], 1, None)
  assert out["status"] == "timeout" and time.time() - t0 < 15


def test_stop_file_ends_a_running_request_and_closes_the_session(tmp_path):
  s = _start(tmp_path)
  box = {}
  worker = threading.Thread(
    target=lambda: box.update(r=session.queue_request(s, "long", [PY, "-c", "import time; time.sleep(60)"], 600, None))
  )
  worker.start()
  time.sleep(1.5)
  (s / "STOP").touch()
  worker.join(30)
  assert box["r"]["status"] == "stopped"
  with pytest.raises(SystemExit, match="stopped"):
    session.queue_request(s, "after", [PY, "-c", "pass"], 60, None)
  assert json.loads((s / "session.json").read_text())["stop_reason"] == "stop_file"


def test_deadline_caps_requests_and_then_rejects_them(tmp_path):
  s = _start(tmp_path, hours=2.0 / 3600)
  t0 = time.time()
  out = session.queue_request(s, "capped", [PY, "-c", "import time; time.sleep(60)"], 600, None)
  assert out["status"] == "timeout" and time.time() - t0 < 15
  time.sleep(max(0.0, 2.5 - (time.time() - t0)))
  with pytest.raises(SystemExit, match="deadline"):
    session.queue_request(s, "late", [PY, "-c", "pass"], 60, None)
  config = json.loads((s / "session.json").read_text())
  assert config["status"] == "stopped" and config["stop_reason"] == "deadline"


def test_status_and_stop(tmp_path, capsys):
  s = _start(tmp_path)
  assert session.main(["status", str(s)]) == 0
  assert json.loads(capsys.readouterr().out)["session"]["status"] == "open"
  session.main(["stop", str(s), "--reason", "done"])
  assert (s / "STOP").exists()
  assert "Reason: done" in (s / "journal.md").read_text()


def test_worker_drains_the_queue_until_the_deadline(tmp_path):
  s = tmp_path / "s"
  (s / "queue").mkdir(parents=True)
  now = datetime.now(UTC)
  config = {"deadline_utc": (now + timedelta(seconds=4)).isoformat(), "slurm_job": "1", "status": "running"}
  (s / "session.json").write_text(json.dumps(config))
  for name in ("a", "b"):
    executors.save(s / "queue" / f"{name}.request.json", {"argv": [PY, "-c", f"print('{name}')"], "timeout_s": 30})
  assert executors.worker(s) == "deadline"
  assert [_result(s, n)["status"] for n in ("a", "b")] == ["completed", "completed"]
  assert (s / "worker_ready.json").exists() and (s / "worker_stopped.json").exists()


def _slurm_session(tmp_path, **config):
  (tmp_path / "session.json").write_text(json.dumps(config))
  (tmp_path / "queue").mkdir()
  (tmp_path / "journal.md").write_text("# journal\n")


def _fake_scheduler(monkeypatch, state):
  calls = []

  def fake_run(argv, **kwargs):
    calls.append(argv)
    return SimpleNamespace(stdout=f"{state}\n" if argv[0] == "squeue" else "", returncode=0)

  monkeypatch.setattr(executors.subprocess, "run", fake_run)
  return calls


def test_gpu_wait_cutoff_cancels_only_the_session_job(tmp_path, monkeypatch):
  now = datetime.now(UTC)
  _slurm_session(
    tmp_path, executor="slurm", slurm_job="123", status="waiting_gpu", gpu_wait_limit_s=300,
    submitted_utc=(now - timedelta(seconds=301)).isoformat(),
    deadline_utc=(now + timedelta(hours=4)).isoformat(),
  )
  calls = _fake_scheduler(monkeypatch, "PENDING")
  executors.SlurmExecutor().watch(tmp_path)
  result = json.loads((tmp_path / "session.json").read_text())
  assert result["stop_reason"] == "gpu_wait_limit"
  assert (tmp_path / "STOP").exists()
  assert calls[-1] == ["scancel", "123"]
  assert "gpu_wait_limit" in (tmp_path / "journal.md").read_text()


def test_deadline_stops_running_job_and_journals_results(tmp_path, monkeypatch):
  now = datetime.now(UTC)
  _slurm_session(
    tmp_path, executor="slurm", slurm_job="456", status="running", gpu_wait_limit_s=300,
    submitted_utc=(now - timedelta(hours=4)).isoformat(),
    deadline_utc=(now - timedelta(seconds=1)).isoformat(),
    gpu_acquired_utc=(now - timedelta(hours=4)).isoformat(),
  )
  executors.write_json(tmp_path / "queue" / "000_x.result.json",
                       {"status": "completed", "wall_s": 3.0, "log": "queue/000_x.log"})
  calls = _fake_scheduler(monkeypatch, "RUNNING")
  executors.SlurmExecutor().watch(tmp_path)
  result = json.loads((tmp_path / "session.json").read_text())
  assert result["stop_reason"] == "deadline"
  assert result["recorded_requests"] == ["000_x.result.json"]
  assert calls[-1] == ["scancel", "456"]
  assert "Result 000_x" in (tmp_path / "journal.md").read_text()


def test_slurm_requests_are_queued_for_the_worker(tmp_path):
  now = datetime.now(UTC)
  _slurm_session(tmp_path, executor="slurm", slurm_job="7", status="waiting_gpu",
                 deadline_utc=(now + timedelta(hours=1)).isoformat())
  out = session.queue_request(tmp_path, "000_q", [PY, "-c", "pass"], 60, None)
  assert out["status"] == "queued"
  assert (tmp_path / "queue" / "000_q.request.json").exists()


def test_cli_run_splits_options_from_the_request_command(tmp_path, capsys):
  s = _start(tmp_path)
  code = session.main(["run", str(s), "000_cli", "--timeout", "60", "--rule", "r", "--",
                       PY, "-c", "import sys; print(sys.argv[1:])", "--timeout", "{name}"])
  assert code == 0
  assert json.loads(capsys.readouterr().out)["status"] == "completed"
  assert "['--timeout', '000_cli']" in (s / "queue" / "000_cli.log").read_text()
  assert session.main(["run", str(s), "001_bad", "--", PY, "-c", "raise SystemExit(2)"]) == 1
