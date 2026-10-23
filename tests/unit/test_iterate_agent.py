"""Exercise the agent/session lifecycle with a fake Codex, without inference or GPUs."""

from __future__ import annotations

import json
import os
import signal
import sys

import pytest

from crowdmotion.iterate import agent, session


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
  executable = tmp_path / "codex"
  executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, signal, sys, time
from pathlib import Path
if sys.argv[1:3] == ["debug", "models"]:
    if os.environ.get("FAKE_CODEX_CATALOG") == "broken":
        print("not json")
        raise SystemExit(1)
    levels = lambda *efforts: [{"effort": effort} for effort in efforts]
    print(json.dumps({"models": [
        {"slug": "gpt-6-astra", "supported_reasoning_levels": levels("low", "medium", "high", "xhigh", "ultra")},
        {"slug": "configured-test-model", "supported_reasoning_levels": levels("low", "high", "xhigh")},
    ]}))
    raise SystemExit(0)
if "--version" in sys.argv:
    print("fake-codex")
    raise SystemExit(0)
# Mirror the real CLI (0.160.0): argument conflicts fail before the prompt is read.
if "--sandbox" in sys.argv and "--approve-for-me" in sys.argv:
    print("error: the argument '--sandbox <SANDBOX_MODE>' cannot be used with '--approve-for-me'",
          file=sys.stderr)
    raise SystemExit(2)
prompt = sys.stdin.read()
if not prompt.strip():
    print("No prompt provided via stdin.", file=sys.stderr)
    raise SystemExit(1)
run = Path(os.environ["CROWD_ITERATION_RUN"])
session_path = Path(os.environ["CROWD_ITERATION_SESSION"])
(run / "received.json").write_text(json.dumps({
    "argv": sys.argv[1:], "prompt": prompt, "cwd": os.getcwd(),
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(),
}))
print('{"type":"thread.started","thread_id":"fake"}', flush=True)
print("agent diagnostic", file=sys.stderr, flush=True)
mode = os.environ.get("FAKE_CODEX_MODE", "complete")
if mode == "fail":
    raise SystemExit(3)
if mode == "stop":
    (session_path / "STOP").touch()
elif mode == "signal":
    os.kill(os.getppid(), signal.SIGTERM)
elif mode == "timeout":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode not in ("complete", "request_stop"):
    time.sleep(60)
from crowdmotion.iterate.session import queue_request
if mode == "request_stop":
    # The request's separate process group must see STOP before the agent is killed.
    code = "import os, time; from pathlib import Path; " + \
        "Path(" + repr(str(run / "request.pid")) + ").write_text(str(os.getpid())); " + \
        "time.sleep(60)"
    import threading
    def stop_after_child_starts():
        while not (run / "request.pid").exists():
            time.sleep(0.01)
        (session_path / "STOP").touch()
    threading.Thread(target=stop_after_child_starts, daemon=True).start()
    queue_request(session_path, "000_active", [sys.executable, "-c", code], 60, None)
    time.sleep(60)
queue_request(session_path, "000_smoke", [sys.executable, "-c", "print('request executed')"], 30, None)
Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text("finished")
''')
  executable.chmod(0o755)
  monkeypatch.setattr("crowdmotion.core.rundir.environment", lambda: {"host": "test"})
  monkeypatch.setattr("crowdmotion.core.rundir.git_diff", lambda: "initial diff")
  monkeypatch.setattr(session, "environment", lambda: {"host": "test"})
  monkeypatch.setattr(agent, "git_diff", lambda: "final diff")
  monkeypatch.setattr(agent, "TERMINATE_GRACE_S", 0.2)
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
  return executable


def _launch(tmp_path, fake_codex, hours=1.0, options=("--model", "configured-test-model")):
  prompt = tmp_path / "task.txt"
  prompt.write_text("Evaluate and render, then refine behavior.\n")
  return agent.main(["behavior_iteration", "--out", str(tmp_path / "run"), "--hours", str(hours),
                     "--prompt", str(prompt), "--codex-bin", str(fake_codex),
                     "--graph", "/test/graph.pt", *options])


def test_agent_runs_requests_locally_and_preserves_outputs(tmp_path, fake_codex):
  previous = signal.getsignal(signal.SIGTERM)
  assert _launch(tmp_path, fake_codex) == 0
  run = tmp_path / "run"
  record = json.loads((run / "run.json").read_text())
  config = json.loads((run / "session/session.json").read_text())
  received = json.loads((run / "received.json").read_text())
  assert record["status"] == "completed" and record["codex_version"] == "fake-codex"
  assert record["session"] == str(run / "session")
  assert config["executor"] == "local" and "slurm_job" not in config
  assert config["stop_reason"] == "completed" and (run / "session/STOP").exists()
  assert received["gpu"] == "7" and received["argv"][0] == "exec"
  assert received["argv"] == record["argv"][1:] and "--sandbox" not in received["argv"]
  assert received["argv"][received["argv"].index("--model") + 1] == "configured-test-model"
  assert received["argv"][received["argv"].index("-c") + 1] == 'model_reasoning_effort="xhigh"'
  assert (record["config"]["model"], record["config"]["effort"]) == ("configured-test-model", "xhigh")
  assert received["cwd"] == str(agent.ROOT)
  assert str(run / "session") in received["prompt"] and config["deadline_utc"] in received["prompt"]
  assert "Graph: /test/graph.pt" in received["prompt"]
  assert received["prompt"] == (run / "agent_prompt.txt").read_text()
  assert (run / "final.md").read_text() == "finished"
  assert json.loads((run / "agent.jsonl").read_text())["thread_id"] == "fake"
  assert "agent diagnostic" in (run / "agent.stderr.log").read_text()
  assert "request executed" in (run / "session/queue/000_smoke.log").read_text()
  assert (run / "source.diff").read_text() == "initial diff"
  assert (run / "changes.diff").read_text() == "final diff"
  assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("mode,outcome", [("fail", "failed"), ("stop", "stopped"), ("signal", "signal_sigterm")])
def test_agent_failure_stop_and_signal_close_the_session(tmp_path, fake_codex, monkeypatch, mode, outcome):
  monkeypatch.setenv("FAKE_CODEX_MODE", mode)
  previous = signal.getsignal(signal.SIGTERM)
  assert _launch(tmp_path, fake_codex) == 1
  run = tmp_path / "run"
  record = json.loads((run / "run.json").read_text())
  config = json.loads((run / "session/session.json").read_text())
  assert record["status"] == "failed" and record["outcome"] == outcome
  assert config["status"] == "stopped" and config["stop_reason"] == outcome
  assert signal.getsignal(signal.SIGTERM) == previous
  if mode == "fail":
    assert record["returncode"] == 3


def test_deadline_kills_an_agent_that_ignores_sigterm(tmp_path, fake_codex, monkeypatch):
  monkeypatch.setenv("FAKE_CODEX_MODE", "timeout")
  assert _launch(tmp_path, fake_codex, hours=1.0 / 3600) == 1
  run = tmp_path / "run"
  record = json.loads((run / "run.json").read_text())
  # wall_s starts after the Codex preflight, whose CLI startups are slow on shared storage.
  assert record["wall_s"] < 10
  assert record["outcome"] == "deadline" and record["returncode"] == -signal.SIGKILL
  pid = json.loads((run / "received.json").read_text())["pid"]
  with pytest.raises(ProcessLookupError):
    os.kill(pid, 0)


def test_stop_reaps_an_active_request_in_a_separate_process_group(tmp_path, fake_codex, monkeypatch):
  monkeypatch.setenv("FAKE_CODEX_MODE", "request_stop")
  assert _launch(tmp_path, fake_codex) == 1
  run = tmp_path / "run"
  result = json.loads((run / "session/queue/000_active.result.json").read_text())
  assert result["status"] == "stopped"
  assert (run / "session/queue/000_active.done.json").exists()
  pid = int((run / "request.pid").read_text())
  with pytest.raises(ProcessLookupError):
    os.kill(pid, 0)


@pytest.mark.parametrize("hours", ["0", "-1", "nan", "inf"])
def test_invalid_budget_creates_no_run(tmp_path, fake_codex, hours):
  with pytest.raises(SystemExit, match="finite and positive"):
    _launch(tmp_path, fake_codex, hours=hours)
  assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("options,effort", [((), "xhigh"), (("--effort", "ultra"), "ultra")])
def test_model_defaults_to_gpt6_astra_and_effort_is_selectable(tmp_path, fake_codex, options, effort):
  assert _launch(tmp_path, fake_codex, options=options) == 0
  run = tmp_path / "run"
  argv = json.loads((run / "received.json").read_text())["argv"]
  config = json.loads((run / "run.json").read_text())["config"]
  assert argv[argv.index("--model") + 1] == "gpt-6-astra" == config["model"]
  assert argv[argv.index("-c") + 1] == f'model_reasoning_effort="{effort}"' and config["effort"] == effort


@pytest.mark.parametrize("options,catalog,message", [
  (("--model", "unknown-model"), None, "not in the Codex catalog: gpt-6-astra, configured-test-model"),
  (("--model", "configured-test-model", "--effort", "ultra"), None, "does not support reasoning effort 'ultra'"),
  ((), "broken", "cannot read the Codex model catalog"),
])
def test_unavailable_model_or_effort_opens_no_run(tmp_path, fake_codex, monkeypatch, options, catalog, message):
  if catalog:
    monkeypatch.setenv("FAKE_CODEX_CATALOG", catalog)
  with pytest.raises(SystemExit, match=message):
    _launch(tmp_path, fake_codex, options=options)
  assert not (tmp_path / "run").exists()


def test_arguments_codex_rejects_open_no_run(tmp_path, fake_codex, monkeypatch):
  # The 2026-10-04 batch run failed this way: --sandbox conflicts with --approve-for-me.
  build = agent.codex_argv
  monkeypatch.setattr(agent, "codex_argv",
                      lambda *args: [*build(*args)[:2], "--sandbox", "workspace-write", *build(*args)[2:]])
  with pytest.raises(SystemExit, match="cannot be used with '--approve-for-me'"):
    _launch(tmp_path, fake_codex)
  assert not (tmp_path / "run").exists()


def test_check_validates_without_a_run_or_prompt(tmp_path, fake_codex, capsys):
  assert _launch(tmp_path, fake_codex, options=("--check",)) == 0
  assert not (tmp_path / "run").exists() and not (tmp_path / "received.json").exists()
  out = capsys.readouterr().out
  assert "Codex check passed: fake-codex, model gpt-6-astra, reasoning effort xhigh" in out
  assert "--approve-for-me" in out and "--sandbox" not in out
