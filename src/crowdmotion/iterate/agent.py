"""Run a noninteractive Codex agent in the caller's existing CPU/GPU allocation.

The outer RunDir holds provenance, the exact prompt, JSON events and the final
message. Its `session/` uses the local executor: no nested Slurm submission. The
runner bounds the agent itself as well as the requests, and closes the session on
completion, failure, deadline, STOP, SIGINT or SIGTERM. Before opening the run it
checks the model, reasoning effort and exact `codex exec` arguments against the
installed CLI, without any model request.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from crowdmotion.core.paths import ROOT
from crowdmotion.core.provenance import git_diff
from crowdmotion.core.rundir import RunDir
from crowdmotion.iterate import session
from crowdmotion.iterate.executors import deadline_of, load_config, now, stopped

DEFAULT_PROMPT = ROOT / "configs/behavior/iterate_codex.txt"
DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_EFFORT = "xhigh"
CHECK_TIMEOUT_S = 60
NO_PROMPT = "No prompt provided"  # codex exec's error for empty stdin, after argument parsing
POLL_S = 0.25
TERMINATE_GRACE_S = 5.0
REQUEST_STOP_GRACE_S = 6.0  # requests poll every 0.5 s and allow 5 s for termination


def codex_argv(executable: str, final: Path | str, model: str, effort: str) -> list[str]:
  """The launch argv; the preflight runs it with only the final-message path changed."""
  # --approve-for-me selects the workspace-write sandbox itself and rejects --sandbox.
  return [executable, "exec", "--approve-for-me", "--json", "--cd", str(ROOT),
          "--output-last-message", str(final), "--model", model,
          "-c", f'model_reasoning_effort="{effort}"', "-"]


def _codex(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
  try:
    return subprocess.run(argv, capture_output=True, text=True, timeout=CHECK_TIMEOUT_S, **kwargs)
  except (OSError, subprocess.TimeoutExpired) as exc:
    raise SystemExit(f"Codex check failed: {shlex.join(argv)}: {exc}") from None


def _errors(stderr: str) -> str:
  return "\n".join(line for line in stderr.splitlines() if not line.startswith("WARNING:")).strip()


def check_codex(executable: str, model: str, effort: str) -> None:
  """Validate model, effort and launch arguments against the installed Codex, without inference."""
  listed = _codex([executable, "debug", "models"])
  try:
    catalog = {entry["slug"]: [level["effort"] for level in entry["supported_reasoning_levels"]]
               for entry in json.loads(listed.stdout)["models"]}
  except (ValueError, KeyError, TypeError) as exc:
    raise SystemExit(f"cannot read the Codex model catalog (codex debug models, exit {listed.returncode}): "
                     f"{type(exc).__name__}: {exc}\n{_errors(listed.stderr)}") from None
  if model not in catalog:
    raise SystemExit(f"model {model!r} is not in the Codex catalog: {', '.join(catalog)}")
  if effort not in catalog[model]:
    raise SystemExit(f"{model} does not support reasoning effort {effort!r}: {', '.join(catalog[model])}")
  # Codex parses its arguments before it reads the prompt from stdin, so empty stdin
  # exposes rejected flags (exit 2) and stops any valid launch before a model request.
  argv = codex_argv(executable, os.devnull, model, effort)
  parsed = _codex(argv, cwd=ROOT, stdin=subprocess.DEVNULL)
  if parsed.returncode == 0 or NO_PROMPT not in parsed.stderr:
    raise SystemExit(f"codex exec rejected the launch arguments (exit {parsed.returncode}): {shlex.join(argv)}\n"
                     f"{_errors(parsed.stderr) or parsed.stdout.strip()}")


def _terminate_group(child: subprocess.Popen) -> None:
  """Reap the agent and kill any children remaining in its process group."""
  try:
    os.killpg(child.pid, signal.SIGTERM)
  except ProcessLookupError:
    return
  try:
    child.wait(timeout=TERMINATE_GRACE_S)
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(child.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  child.wait()


def _close_session(agent_session: Path, reason: str) -> None:
  session.stop(agent_session, reason)
  deadline = time.monotonic() + REQUEST_STOP_GRACE_S
  # Request processes start separate groups. Let their STOP polling terminate and
  # reap those children before killing the agent and its command runners.
  while any((agent_session / "queue").glob("*.working.json")) and time.monotonic() < deadline:
    time.sleep(POLL_S)


def run_agent(*, name: str, hours: float, prompt: Path, out: Path | None = None, codex_bin: str = "codex",
              model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, graph: str | None = None,
              check: bool = False) -> int:
  if not session.NAME.fullmatch(name):
    raise SystemExit("session name: letters, digits, '_' and '-' only")
  if not math.isfinite(hours) or hours <= 0:
    raise SystemExit("--hours must be finite and positive")
  executable = shutil.which(codex_bin)
  if not executable:
    raise SystemExit(f"Codex executable not found: {codex_bin}")
  task = Path(prompt).read_text()
  if not task.strip():
    raise SystemExit("prompt file is empty")
  # Check model, effort and arguments before opening a run, without any model request.
  check_codex(executable, model, effort)
  version = subprocess.run([executable, "--version"], capture_output=True, text=True,
                           timeout=10, check=True).stdout.strip()
  if check:
    print(f"Codex check passed: {version}, model {model}, reasoning effort {effort}\n"
          f"{shlex.join(codex_argv(executable, '<run>/final.md', model, effort))}")
    return 0
  run = RunDir.create(name, out=out, record={
    "kind": "codex_iteration", "started_utc": now(), "codex_version": version,
    "config": {"name": name, "hours": hours, "prompt": str(prompt), "model": model, "effort": effort,
               "graph": graph},
  })
  run.path = run.path.resolve()
  agent_session = None
  child = None
  reason, returncode, error = "failed", None, None
  started = time.monotonic()
  interrupted = []
  previous_handlers = {}

  def receive_signal(signum, _frame):
    interrupted.append(signal.Signals(signum).name)

  try:
    for sig in (signal.SIGINT, signal.SIGTERM):
      previous_handlers[sig] = signal.signal(sig, receive_signal)
    agent_session = session.create(name, run.path / "session", hours, "local", 300, (),
                                   "Codex agent runs inside the existing allocation.", graph)
    config = load_config(agent_session)
    remaining = deadline_of(config) - time.time()
    deadline = time.monotonic() + remaining
    run.update(session=str(agent_session), deadline_utc=config["deadline_utc"])
    context = (
      f"\n\nLauncher context:\nRun directory: {run.path}\n"
      f"Existing iteration session: {agent_session}\n"
      f"Deadline (UTC): {config['deadline_utc']}\n"
      "Use this existing local session for every evaluation, render and comparison.\n"
      "Finish the report and required project bookkeeping before the deadline.\n"
    )
    if graph:
      context += f"Graph: {graph}\nUse --graph with this path for each evaluation.\n"
    prompt_path = run.path / "agent_prompt.txt"
    prompt_path.write_text(task + context)
    argv = codex_argv(executable, run.path / "final.md", model, effort)
    run.update(argv=argv)
    env = dict(os.environ, CROWD_ITERATION_RUN=str(run.path), CROWD_ITERATION_SESSION=str(agent_session),
               CROWD_ITERATION_DEADLINE_UTC=config["deadline_utc"])
    print(f"Agent run: {run.path}\nIteration session: {agent_session}\nDeadline: {config['deadline_utc']}", flush=True)
    with prompt_path.open() as instructions, (run.path / "agent.jsonl").open("w") as events, \
         (run.path / "agent.stderr.log").open("w") as errors:
      child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=instructions, stdout=events,
                               stderr=errors, start_new_session=True)
      reason = "completed"
      while child.poll() is None:
        if interrupted:
          reason = f"signal_{interrupted[0].lower()}"
        elif stopped(agent_session):
          reason = "stopped"
        elif time.monotonic() >= deadline:
          reason = "deadline"
        if reason != "completed":
          break
        time.sleep(POLL_S)
      returncode = child.poll()
      if reason == "completed" and returncode != 0:
        reason = "failed"
  except Exception as exc:
    reason, error = "failed", f"{type(exc).__name__}: {exc}"
    print(error, file=sys.stderr)
  finally:
    # STOP lets request runners terminate their own children, including those that
    # have a separate process group. Then shut down the agent's process group.
    try:
      if agent_session is not None:
        _close_session(agent_session, reason)
    finally:
      if child is not None:
        _terminate_group(child)
        returncode = child.returncode
      for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
      (run.path / "changes.diff").write_text(git_diff())
      run.update(status="completed" if reason == "completed" else "failed", outcome=reason,
                 returncode=returncode, error=error, ended_utc=now(), wall_s=time.monotonic() - started)
  return 0 if reason == "completed" else 1


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("name", nargs="?", default="behavior_iteration")
  parser.add_argument("--hours", type=float, default=2.75, help="agent/session deadline after launch (default: 2.75)")
  parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help="task prompt file")
  parser.add_argument("--out", type=Path, help="fresh outer run directory")
  parser.add_argument("--codex-bin", default="codex", help="Codex executable (default: PATH)")
  parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Codex model slug (default: {DEFAULT_MODEL})")
  parser.add_argument("--effort", default=DEFAULT_EFFORT,
                      help=f"Codex reasoning effort for the model (default: {DEFAULT_EFFORT})")
  parser.add_argument("--graph", help="record and use this graph; otherwise use the configured/default artifact")
  parser.add_argument("--check", action="store_true",
                      help="only check the Codex model, effort and launch arguments; open no run")
  args = parser.parse_args(argv)
  return run_agent(**vars(args))


if __name__ == "__main__":
  sys.exit(main())
