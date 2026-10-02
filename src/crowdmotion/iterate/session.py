"""Iteration sessions: one run folder holding a bounded series of named requests.

  crowd iterate start turn-commit --hours 4 --note "Plan: committed turns biased to 90°"
  crowd iterate run runs/<UTC>_turn-commit 000_baseline --timeout 600 -- \
      crowd behavior evaluate --config configs/behavior/baseline.json --out {out}
  crowd iterate run runs/<UTC>_turn-commit 001_turns --rule "Committed turns, 90° mode" -- \
      crowd behavior evaluate --config <candidate.json> --out {out}
  crowd iterate status runs/<UTC>_turn-commit
  crowd iterate stop runs/<UTC>_turn-commit --reason "done"

The folder holds `session.json` (deadline, executor, status), `journal.md`,
`queue/` (request, result and log of every request) and `iterations/<name>/`
(the `{out}` of each request). Rules:

- request names are immutable: a name is used once, whatever the outcome;
- a stopped session (status, `STOP` file or past its deadline) accepts nothing;
- the journal records rule changes (`--rule`, `note --kind rule`) separately from
  results, so a proposed setting is never presented as a tested change.

`--executor local` (default) runs each request immediately as a subprocess.
`--executor slurm` submits one worker allocation (`crowd iterate worker`) that
drains the queue; run `crowd iterate watch <session>` on the login node to enforce
the GPU-wait cutoff and deadline. In request argv, `{session}`, `{name}` and `{out}`
(= `{session}/iterations/{name}`) are substituted; a leading `crowd` runs this
interpreter's CLI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crowdmotion.core.paths import RUNS
from crowdmotion.core.provenance import environment
from crowdmotion.core.rundir import utc_stamp
from crowdmotion.iterate.executors import (
  DEFAULT_SBATCH_ARGS,
  EXECUTORS,
  SlurmExecutor,
  executor_for,
  journal,
  load_config,
  now,
  save,
  stopped,
)

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
NOTE_KINDS = ("plan", "rule", "observation", "decision")


def create(name: str, out: Path | None, hours: float, executor: str, gpu_wait_limit_s: float,
           sbatch_args: tuple[str, ...], note: str | None, graph: str | None) -> Path:
  if not NAME.match(name):
    raise SystemExit("session name: letters, digits, '_' and '-' only")
  session = (Path(out) if out else RUNS / f"{utc_stamp()}_{name}").resolve()
  session.mkdir(parents=True, exist_ok=False)
  (session / "queue").mkdir()
  (session / "iterations").mkdir()
  start = datetime.now(UTC)
  config = {
    "name": name,
    "executor": executor,
    "start_utc": start.isoformat(),
    "deadline_utc": (start + timedelta(hours=hours)).isoformat(),
    "graph": graph,
    "environment": environment(),
  }
  if executor == "slurm":
    config.update(gpu_wait_limit_s=gpu_wait_limit_s, sbatch_args=list(sbatch_args))
  (session / "journal.md").write_text(
    f"# Iteration session {name}\n\nExecutor: {executor}. Deadline: {config['deadline_utc']}.\n"
    "Rule changes and results are separate entries; a rule is untested until a result cites it.\n"
  )
  save(session / "session.json", config)
  executor_for(config).start(session, config)
  save(session / "session.json", config)
  message = f"Executor {executor}; deadline {config['deadline_utc']}."
  if executor == "slurm":
    message += f" Slurm job {config['slurm_job']}; GPU-wait limit {gpu_wait_limit_s:g} s."
  journal(session, "Session started", message + (f"\n\n{note}" if note else ""))
  return session


def closed_reason(session: Path, config: dict) -> str | None:
  if config.get("status") == "stopped":
    return config.get("stop_reason", "stopped")
  if stopped(session):
    return "stop_file"
  if datetime.now(UTC) >= datetime.fromisoformat(config["deadline_utc"]):
    return "deadline"
  return None


def stop(session: Path, reason: str) -> dict:
  config = load_config(session)
  (session / "STOP").touch()
  if config.get("status") != "stopped":
    executor_for(config).stop(session, config)
    config.update(status="stopped", stop_reason=reason, stopped_utc=now())
    save(session / "session.json", config)
    journal(session, "Session stopped", f"Reason: {reason}. No more requests will run.")
  return config


def queue_request(session: Path, name: str, argv: list[str], timeout_s: float, rule: str | None) -> dict:
  config = load_config(session)
  reason = closed_reason(session, config)
  if reason:
    if config.get("status") != "stopped":
      stop(session, reason)
    raise SystemExit(f"session is stopped ({reason}); no requests accepted")
  if not NAME.match(name):
    raise SystemExit("request name: letters, digits, '_' and '-' only")
  if not argv:
    raise SystemExit("supply the request command after --")
  queue = session / "queue"
  if list(queue.glob(f"{name}.*")):
    raise SystemExit(f"request name {name!r} already exists; names are immutable")
  out = session / "iterations" / name
  fields = {"session": str(session), "name": name, "out": str(out)}
  argv = [a.format(**fields) if "{" in a else a for a in argv]
  spec = {"argv": argv, "timeout_s": timeout_s, "rule": rule, "queued_utc": now()}
  save(queue / f"{name}.request.json", spec)
  if rule:
    journal(session, f"Rule change tested by {name}", rule)
  journal(session, f"Request {name}", f"Timeout {timeout_s:g} s.\n\n    {' '.join(argv)}")
  return executor_for(config).submit(session, name)


def requests(session: Path) -> list[dict]:
  rows = {}
  for path in sorted((session / "queue").glob("*.json")):
    name, kind = path.name.removesuffix(".json").rsplit(".", 1)
    row = rows.setdefault(name, {"name": name, "status": "queued"})
    if kind == "working":
      row["status"] = "running"
    elif kind == "result":
      record = json.loads(path.read_text())
      row.update(status=record["status"], wall_s=round(record["wall_s"], 1), log=record["log"])
  return list(rows.values())


def main(argv=None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  command: list[str] = []
  if "--" in argv:  # `run ... -- <request command>`: everything after `--` is the request
    cut = argv.index("--")
    argv, command = argv[:cut], argv[cut + 1:]
  parser = argparse.ArgumentParser(prog="crowd iterate", description="Iteration sessions: one folder, named requests, a deadline.")
  sub = parser.add_subparsers(dest="command", required=True)
  p = sub.add_parser("start", help="create a session folder")
  p.add_argument("name")
  p.add_argument("--out", type=Path, help="session folder (default: runs/<UTC>_<name>)")
  p.add_argument("--hours", type=float, default=4.0, help="deadline after start")
  p.add_argument("--executor", choices=sorted(EXECUTORS), default="local")
  p.add_argument("--gpu-wait-limit", type=float, default=300.0, help="slurm: seconds to wait for the GPU")
  p.add_argument("--sbatch-arg", action="append", dest="sbatch_args",
                 help=f"slurm: sbatch option (repeatable; default {' '.join(DEFAULT_SBATCH_ARGS)})")
  p.add_argument("--graph", help="graph the session evaluates (recorded only)")
  p.add_argument("--note", help="plan or authorization text for the journal")
  p = sub.add_parser("run", help="add a named request; local sessions run it now")
  p.add_argument("session", type=Path)
  p.add_argument("name")
  p.add_argument("--timeout", type=float, default=900.0, help="seconds (capped by the deadline)")
  p.add_argument("--rule", help="the rule change this request tests (journaled separately)")
  p = sub.add_parser("status", help="print the session record and its requests")
  p.add_argument("session", type=Path)
  p = sub.add_parser("stop", help="stop the session (and cancel its Slurm job)")
  p.add_argument("session", type=Path)
  p.add_argument("--reason", default="explicit_stop")
  p = sub.add_parser("watch", help="slurm: enforce the GPU-wait cutoff and deadline")
  p.add_argument("session", type=Path)
  p = sub.add_parser("note", help="journal a plan, rule, observation or decision")
  p.add_argument("session", type=Path)
  p.add_argument("--kind", choices=NOTE_KINDS, required=True)
  p.add_argument("title")
  p.add_argument("text")
  args = parser.parse_args(argv)

  if args.command == "start":
    session = create(args.name, args.out, args.hours, args.executor, args.gpu_wait_limit,
                     tuple(args.sbatch_args or DEFAULT_SBATCH_ARGS), args.note, args.graph)
    print(session)
    return 0
  session = args.session.resolve()
  if args.command == "run":
    result = queue_request(session, args.name, command, args.timeout, args.rule)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("completed", "queued") else 1
  if args.command == "status":
    config = load_config(session)
    brief = {k: v for k, v in config.items() if k != "environment"}
    print(json.dumps({"session": brief, "requests": requests(session)}, indent=2))
    return 0
  if args.command == "stop":
    print(json.dumps(stop(session, args.reason), indent=2, default=str))
    return 0
  if args.command == "watch":
    config = load_config(session)
    if not config.get("slurm_job"):
      raise SystemExit("watch is for slurm sessions; local requests enforce limits themselves")
    print(json.dumps(SlurmExecutor().watch(session), indent=2, default=str))
    return 0
  if args.command == "note":
    journal(session, f"{args.kind.capitalize()}: {args.title}", args.text)
    return 0
  return 2


if __name__ == "__main__":
  sys.exit(main())
