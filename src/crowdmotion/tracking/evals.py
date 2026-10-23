"""Evaluation rows on a training lineage's W&B run: outbox, ledger, `attach_eval`, and the
one-session publication and reconciliation that sync and attachment share.

An evaluation writes its own run directory, then attaches one row per (evaluation,
kind, checkpoint) to the training lineage: `iteration` = the checkpoint's iteration,
`eval/<kind>/<name>` per scalar, `eval/<kind>/video` (an MP4) and `eval/<kind>/id` = the
evaluation directory's name. Each row first goes to `<train run>/wandb/outbox/` (one
file per ledger key). The writer-lock holder publishes it: the evaluation itself when
the lock is free, else the live trainer at its next checkpoint, or `crowd wandb sync`.
The ledger `wandb.evals[<eval dir>/<kind>@<iteration>]` turns `published` only after
W&B's `finish()` confirmed the session (`lineage.finish`), and a published key is never
logged again. A session that ended unconfirmed is settled by the next holder against
what W&B holds (`reconcile`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

from crowdmotion.core.rundir import write_json
from crowdmotion.tracking.lineage import (
  WriterLock,
  abandon,
  advance,
  enabled,
  finish,
  init_run,
  read_record,
  remote_files,
  remote_rows,
  sdk,
  settle_floor,
  stuck,
  utc_now,
  write_record,
)


def ledger_key(eval_run: Path, kind: str, iteration: int) -> str:
  return f"{Path(eval_run).name}/{kind}@{int(iteration)}"


def outbox(train_run: Path) -> Path:
  return Path(train_run) / "wandb" / "outbox"


def outbox_file(train_run: Path, key: str) -> Path:
  return outbox(train_run) / f"{quote(key, safe='')}.json"  # keys hold `/` (kinds such as traversal/aware/end)


def pending(train_run: Path, record: dict) -> list[dict]:
  """Outbox entries not yet published, oldest first; files of published keys are removed."""
  ledger = record["wandb"].setdefault("evals", {})
  out = []
  files = sorted(outbox(train_run).glob("*.json"), key=lambda f: (f.stat().st_mtime, f.name))
  for f in files:
    try:
      entry = json.loads(f.read_text())
    except (OSError, ValueError):
      continue
    if ledger.get(entry["key"], {}).get("status") == "published":
      f.unlink(missing_ok=True)
      continue
    out.append(entry)
  return out


def row(entry: dict, sdk_=None) -> dict:
  prefix = f"eval/{entry['kind']}"
  out = {"iteration": entry["iteration"], f"{prefix}/id": entry["eval_id"]}
  out.update({f"{prefix}/{name}": value for name, value in entry["scalars"].items()})
  if entry.get("video") and Path(entry["video"]).is_file():
    out[f"{prefix}/video"] = sdk(sdk_).Video(entry["video"], format="mp4")
  return out


def mark(record: dict, entries: list[dict], status: str, error: str | None = None) -> None:
  ledger = record["wandb"].setdefault("evals", {})
  for e in entries:
    ledger[e["key"]] = {"status": status, "eval_id": e["eval_id"], "kind": e["kind"], "iteration": e["iteration"],
                        "at": utc_now(), **({"error": error} if error else {})}


def retire(train_run: Path, entries: list[dict]) -> None:
  for e in entries:
    outbox_file(train_run, e["key"]).unlink(missing_ok=True)


def opened(record: dict, holder: str, through: int | None = None) -> None:
  """Mark a session unconfirmed before its first row (`reconcile` settles it if it never
  confirms); `through` is the highest training row logged so far."""
  record["wandb"]["unconfirmed"] = {"since": utc_now(), "holder": holder, "through": through}


def published_above(block: dict) -> set[int]:
  return set(block.get("published_above") or ())


def reconcile(train_run: Path, record: dict, sdk_=None, write: bool = True) -> bool:
  """Holding the lock, after a session that ended unconfirmed (`wandb.unconfirmed`: a
  crash, a kill, a finish timeout or a W&B error): take the rows W&B holds as published.
  The floor rises through the training iterations W&B holds without a gap; those above a
  gap go to `published_above`, so that only the missing ones are replayed. Evaluation
  entries whose row W&B holds turn published, a row with a video only once W&B also holds
  the video's file at its size (else the row is logged again). Returns whether it read
  W&B; raises when W&B cannot be read (nothing is then assumed)."""
  block = record["wandb"]
  if not block.get("unconfirmed"):
    return False
  legacy = block.get("legacy")
  train, seen, media = set(), set(), {}
  for r in remote_rows(block, sdk_):
    it = r.get("iteration", r.get("_step") if legacy else None)
    if it is None:
      continue
    ids = {k[len("eval/"):-len("/id")]: v for k, v in r.items() if k.startswith("eval/") and k.endswith("/id")}
    for kind, eval_id in ids.items():
      seen.add((kind, eval_id, int(it)))
      video = r.get(f"eval/{kind}/video")
      if isinstance(video, dict) and video.get("path"):
        media[(kind, eval_id, int(it))] = (video["path"], video.get("size"))
    if not ids and not any(k.startswith("eval/") for k in r):
      train.add(int(it))
  settle_floor(block, sdk_)
  advance(record, train)
  landed = [e for e in pending(train_run, record) if (e["kind"], e["eval_id"], int(e["iteration"])) in seen]
  videos = [media[k] for e in landed if (k := (e["kind"], e["eval_id"], int(e["iteration"]))) in media]
  held = remote_files(block, {path for path, _ in videos}, sdk_) if videos else {}
  incomplete = {path for path, size in videos if path not in held or (size is not None and held[path] != int(size))}
  landed = [e for e in landed if media.get((e["kind"], e["eval_id"], int(e["iteration"])), ("",))[0] not in incomplete]
  mark(record, landed, "published")
  print(f"W&B: reconciled an unconfirmed session ({block['unconfirmed'].get('holder')}): published through "
        f"{block.get('published_through')}, {len(block.get('published_above') or ())} rows above, {len(landed)} "
        f"evaluations already there" + (f", {len(incomplete)} videos to send again" if incomplete else ""),
        flush=True)
  block.pop("unconfirmed")
  if write:
    write_record(train_run, record)
    retire(train_run, landed)
  return True


def publish(train_run: Path, record: dict, rows: Mapping[int, dict] | None = None, sdk_=None,
            holder: str = "crowd wandb sync") -> dict[str, str]:
  """Holding the lock, after `reconcile`: one W&B session that logs the training `rows`
  (ascending iterations) and every pending outbox entry, then updates and writes
  `record`. Once `finish()` confirms the session, the floor covers the rows and the
  entries turn published. Returns each entry's ledger status. Any failure finishes the
  session (unconfirmed: the next holder reconciles), is recorded (failed ledger entries,
  outbox files kept for a retry; `wandb.status` when training rows failed) and re-raised."""
  rows = dict(sorted((rows or {}).items()))
  entries = pending(train_run, record)
  if not rows and not entries:
    return {}
  block = record["wandb"]
  handle = None
  try:
    handle = init_run(train_run, record, sdk_)
    opened(record, holder, max(rows) if rows else None)
    write_record(train_run, record)
    for i, values in rows.items():
      handle.log({"iteration": i, **values})
    for e in entries:
      handle.log(row(e, sdk_))
    session, handle = handle, None
    if not finish(session):
      raise TimeoutError("W&B finish timed out; the upload is unconfirmed")
  except Exception as exc:
    try:
      if handle is not None:
        abandon(handle)
    finally:
      mark(record, entries, "failed", repr(exc))
      if rows:
        block.update(status="failed", error=repr(exc))
      write_record(train_run, record)
    raise
  advance(record, rows)
  block.pop("unconfirmed", None)
  mark(record, entries, "published")
  write_record(train_run, record)
  retire(train_run, entries)
  return {e["key"]: "published" for e in entries}


def queue_eval(train_run: Path, eval_run: Path, *, iteration: int, kind: str, scalars: Mapping[str, float],
               video: Path | None = None) -> str:
  """Queue one evaluation row in train_run's outbox without publishing it: returns
  "queued", "published" (already in the ledger), "disabled" (no W&B identity, W&B
  disabled or a check run; nothing written) or "failed" (the outbox is not writable).
  `flush_outbox`, the live writer or `crowd wandb sync` publishes it."""
  train_run, eval_run = Path(train_run), Path(eval_run)
  try:
    record = read_record(train_run)
  except (OSError, ValueError):
    return "disabled"
  if not enabled(record):
    return "disabled"
  key = ledger_key(eval_run, kind, iteration)
  if record["wandb"].get("evals", {}).get(key, {}).get("status") == "published":
    return "published"
  entry = {"key": key, "eval_run": str(eval_run.resolve()), "eval_id": eval_run.name, "kind": kind,
           "iteration": int(iteration), "scalars": {str(k): float(v) for k, v in scalars.items()},
           "video": str(Path(video).resolve()) if video else None, "queued": utc_now()}
  try:
    write_json(outbox_file(train_run, key), entry)
  except OSError as exc:
    print(f"attach_eval {key}: cannot queue in {outbox(train_run)}: {exc!r}", flush=True)
    return "failed"
  return "queued"


def _flush(train_run: Path, sdk=None) -> dict[str, str] | None:
  """Publish the outbox in one W&B session; None when another process holds the lock."""
  if stuck():  # an earlier finish of this process may still be uploading: no new session
    return {e["key"]: "failed" for e in pending(train_run, read_record(train_run))}
  lock = WriterLock(train_run)
  if not lock.acquire("publish evaluations"):
    return None
  try:
    record = read_record(train_run)  # as the last holder left it
    entries = pending(train_run, record)
    try:
      reconcile(train_run, record, sdk)
      entries = pending(train_run, record)
      return publish(train_run, record, sdk_=sdk, holder="an evaluation's attachment")
    except Exception as exc:
      print(f"evaluations of {Path(train_run).name}: W&B failed ({exc!r}); kept in the outbox for "
            "`crowd wandb sync`", flush=True)
      ledger = record["wandb"].setdefault("evals", {})
      if any(ledger.get(e["key"], {}).get("status") != "failed" for e in entries):  # publish records its own
        mark(record, entries, "failed", repr(exc))
        write_record(train_run, record)
      return {e["key"]: "failed" for e in entries}
  finally:
    lock.release()


def flush_outbox(train_run: Path, sdk=None) -> dict[str, str]:
  """Publish train_run's queued evaluation rows in one W&B session: `{key: "published" |
  "failed"}`, empty when nothing was queued or a writer holds the lock (it publishes
  them). For many rows of one evaluation: `queue_eval` each, then one `flush_outbox`."""
  try:
    record = read_record(train_run)
  except (OSError, ValueError):
    return {}
  return (_flush(train_run, sdk) or {}) if enabled(record) else {}


def attach_eval(train_run: Path, eval_run: Path, *, iteration: int, kind: str, scalars: Mapping[str, float],
                video: Path | None = None, sdk=None) -> str:
  """Attach one evaluation row to train_run's W&B lineage: returns "published", "queued"
  (outbox, a live writer holds the lock), "disabled" (no W&B identity, W&B disabled or a
  check run) or "failed" (recorded in the ledger, never raised). "disabled" writes nothing."""
  status = queue_eval(train_run, eval_run, iteration=iteration, kind=kind, scalars=scalars, video=video)
  if status != "queued":
    return status
  key = ledger_key(eval_run, kind, iteration)
  result = _flush(Path(train_run), sdk)
  if result is None:
    return "queued"
  if key in result:
    return result[key]
  published = read_record(train_run)["wandb"].get("evals", {}).get(key, {}).get("status") == "published"
  return "published" if published else "queued"
