"""How an iteration session executes its requests: locally, or inside a Slurm allocation.

A request is a command line saved as `queue/<name>.request.json`. Executing it
renames it to `.working.json`, runs it as a subprocess from the repository root
with its output in `queue/<name>.log`, then writes `queue/<name>.result.json`
and renames the request to `.done.json`. Both executors use `run_request`, so a
local and a cluster session leave identical records.

- `LocalExecutor` (default) runs each request immediately, in the caller's
  process, bounded by the request timeout, the session deadline and `STOP`.
- `SlurmExecutor` submits one allocation running `crowd iterate worker`, which
  drains the queue until the deadline or `STOP`. `watch` (run on the login node)
  cancels only the recorded job when the GPU-wait cutoff or deadline passes, and
  journals results as they appear.

  crowd iterate worker <session>     # inside the allocation (scripts/slurm/iterate_worker.sbatch)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from crowdmotion.core.paths import ROOT
from crowdmotion.core.provenance import environment
from crowdmotion.core.rundir import write_json

WORKER_SBATCH = "scripts/slurm/iterate_worker.sbatch"
DEFAULT_SBATCH_ARGS = ("--partition=rl2-lab,overcap", "--qos=short")
POLL_S = 0.5


# Session files -------------------------------------------------------------------
def now() -> str:
  return datetime.now(UTC).isoformat()


def save(path: Path, value) -> None:
  """Atomic JSON write (readers on other hosts never see a partial file)."""
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(value, indent=2))
  tmp.replace(path)


def load_config(session: Path) -> dict:
  return json.loads((session / "session.json").read_text())


def journal(session: Path, title: str, message: str) -> None:
  with (session / "journal.md").open("a") as stream:
    stream.write(f"\n## {now()} — {title}\n\n{message}\n")


def deadline_of(config: dict) -> float:
  return datetime.fromisoformat(config["deadline_utc"]).timestamp()


def stopped(session: Path) -> bool:
  return (session / "STOP").exists()


def resolve_argv(argv: list[str]) -> list[str]:
  """`crowd ...` runs through this interpreter, so no PATH setup is needed."""
  if argv and argv[0] == "crowd":
    return [sys.executable, "-m", "crowdmotion.cli", *argv[1:]]
  return list(argv)


# One request -----------------------------------------------------------------------
def run_request(session: Path, name: str, deadline: float) -> dict:
  """Execute queued request `name`; return its result record."""
  queue = session / "queue"
  request = queue / f"{name}.request.json"
  spec = json.loads(request.read_text())
  working = request.with_name(f"{name}.working.json")
  request.rename(working)
  log = queue / f"{name}.log"
  start = time.time()
  result = {"name": name, "started_utc": now(), "argv": spec["argv"], "log": str(log.relative_to(session))}
  try:
    timeout = min(float(spec.get("timeout_s", 900)), max(1.0, deadline - start))
    with log.open("w") as output:
      child = subprocess.Popen(
        resolve_argv(spec["argv"]), cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
        start_new_session=True,
      )
      reason = None
      while child.poll() is None:
        if time.time() - start >= timeout:
          reason = "timeout"
        elif stopped(session):
          reason = "stopped"
        if reason:
          _terminate(child)
          break
        time.sleep(POLL_S)
    if reason:
      result.update(status=reason, returncode=child.returncode,
                    error=f"request {'reached its timeout' if reason == 'timeout' else 'stopped by STOP'}")
    else:
      result.update(returncode=child.returncode, status="completed" if child.returncode == 0 else "failed")
  except (OSError, ValueError, subprocess.SubprocessError) as exc:
    result.update(status="failed", error=str(exc))
  result.update(ended_utc=now(), wall_s=time.time() - start)
  write_json(queue / f"{name}.result.json", result)
  working.rename(queue / f"{name}.done.json")
  return result


def _terminate(child: subprocess.Popen) -> None:
  os.killpg(child.pid, signal.SIGTERM)
  try:
    child.wait(timeout=5)
  except subprocess.TimeoutExpired:
    os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def result_message(result: dict) -> str:
  lines = [f"Status: {result['status']}; wall time: {result['wall_s']:.1f} s; log: {result['log']}."]
  if result.get("returncode") not in (None, 0):
    lines.append(f"Return code: {result['returncode']}.")
  if result.get("error"):
    lines.append(f"Error: {result['error']}.")
  return "\n".join(lines)


# Executors ---------------------------------------------------------------------------
class LocalExecutor:
  name = "local"

  def start(self, session: Path, config: dict) -> None:
    config["status"] = "open"

  def submit(self, session: Path, name: str) -> dict:
    """Run request `name` now; journal and return its result."""
    config = load_config(session)
    result = run_request(session, name, deadline_of(config))
    journal(session, f"Result {name}", result_message(result))
    return result

  def stop(self, session: Path, config: dict) -> None:
    pass  # a running local request sees STOP and terminates its child


class SlurmExecutor:
  name = "slurm"

  def __init__(self, sbatch_args: tuple[str, ...] = DEFAULT_SBATCH_ARGS):
    self.sbatch_args = tuple(sbatch_args)

  def start(self, session: Path, config: dict) -> None:
    """Submit the worker allocation and record the session-owned job."""
    argv = ["sbatch", "--parsable", *self.sbatch_args, WORKER_SBATCH, str(session)]
    (ROOT / "logs" / "slurm").mkdir(parents=True, exist_ok=True)  # the job's --output directory
    out = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
    config.update(
      slurm_job=out.stdout.strip().split(";")[0], submitted_utc=now(), status="waiting_gpu",
      sbatch_argv=argv,
    )

  def submit(self, session: Path, name: str) -> dict:
    """Leave the request queued for the worker; `watch` journals its result."""
    return {"name": name, "status": "queued"}

  def stop(self, session: Path, config: dict) -> None:
    if config.get("slurm_job"):
      subprocess.run(["scancel", str(int(config["slurm_job"]))], check=False, timeout=10)

  def watch(self, session: Path) -> dict:
    """Enforce the GPU-wait cutoff and deadline; cancel only the recorded job."""
    config = load_config(session)
    job = str(int(config["slurm_job"]))
    deadline = deadline_of(config)
    wait_deadline = datetime.fromisoformat(config["submitted_utc"]).timestamp() + config["gpu_wait_limit_s"]
    acquired = config.get("gpu_acquired_utc")
    seen = set(config.get("recorded_requests", []))
    while True:
      result = subprocess.run(
        ["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True, timeout=10, check=False
      )
      state = result.stdout.strip()
      if state == "RUNNING" and acquired is None:
        acquired = now()
        config.update(gpu_acquired_utc=acquired, status="running")
        journal(session, "GPU acquired", f"Slurm job {job} is running; queue wait ended.")
      for path in sorted((session / "queue").glob("*.result.json")):
        if path.name not in seen:
          record = json.loads(path.read_text())
          journal(session, "Result " + path.name.removesuffix(".result.json"), result_message(record))
          seen.add(path.name)
      config["recorded_requests"] = sorted(seen)
      config["last_scheduler_state"] = state
      save(session / "session.json", config)
      reason = None
      if stopped(session):
        reason = "explicit_stop"
      elif time.time() >= deadline:
        reason = "deadline"
      elif acquired is None and time.time() >= wait_deadline:
        reason = "gpu_wait_limit"
      elif not state and result.returncode == 0:
        reason = "allocation_ended"
      if reason:
        (session / "STOP").touch()
        subprocess.run(["scancel", job], check=False, timeout=10)
        config.update(status="stopped", stop_reason=reason, stopped_utc=now())
        save(session / "session.json", config)
        journal(session, "Session stopped",
                f"Reason: {reason}. Canceled session-owned Slurm job {job}; no more requests will run.")
        return config
      time.sleep(min(2, max(0.1, (deadline if acquired else min(deadline, wait_deadline)) - time.time())))


EXECUTORS = {"local": LocalExecutor, "slurm": SlurmExecutor}


def executor_for(config: dict):
  kind = config.get("executor") or ("slurm" if config.get("slurm_job") else "local")
  if kind == "slurm":
    return SlurmExecutor(tuple(config.get("sbatch_args", DEFAULT_SBATCH_ARGS)))
  return LocalExecutor()


# Worker -----------------------------------------------------------------------------
def worker(session: Path) -> str:
  """Drain the queue in name order until the deadline or `STOP` (one allocation)."""
  config = load_config(session)
  deadline = deadline_of(config)
  queue = session / "queue"
  queue.mkdir(exist_ok=True)
  write_json(session / "worker_ready.json", {"time": now(), "environment": environment()})
  while time.time() < deadline and not stopped(session):
    requests = sorted(queue.glob("*.request.json"))
    if not requests:
      time.sleep(POLL_S)
      continue
    run_request(session, requests[0].name.removesuffix(".request.json"), deadline)
  reason = "stop_file" if stopped(session) else "deadline"
  write_json(session / "worker_stopped.json", {"time": now(), "reason": reason})
  return reason


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(prog="crowd iterate", description="Run queued iteration requests inside one allocation.")
  sub = parser.add_subparsers(dest="command", required=True)
  w = sub.add_parser("worker", help="drain a session's queue until its deadline or STOP")
  w.add_argument("session", type=Path)
  args = parser.parse_args(argv)
  print(worker(args.session.resolve()), flush=True)


if __name__ == "__main__":
  main()
