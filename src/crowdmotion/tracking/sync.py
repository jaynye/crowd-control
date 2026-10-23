"""`crowd wandb sync|status`: catch traversal lineages' W&B runs up from their local files.

  crowd wandb status [runs ...]                  # every training lineage under runs/ by default
  crowd wandb sync runs/<UTC>_train_traversal_<name> ... [--dry-run]
  crowd wandb sync --all                         # every online lineage under runs/
  crowd wandb sync <runs ...> --enable           # also turn W&B on for disabled or older runs

For each lineage whose writer lock is free, `sync` first settles a session that ended
unconfirmed against what W&B holds (`evals.reconcile`), then publishes the committed
TensorBoard rows W&B lacks (past `published_through`, except `published_above`) up to
the newest checkpoint, each iteration from the attempt W&B keeps (`tracking.history`),
and drains the evaluation outbox. A lineage that is training holds the lock and is
skipped: its live writer publishes and drains by itself. Rows past the newest
checkpoint wait, since a resume would replace them.

`--enable` sets a lineage's mode to `online`. A run from before `run.json` had a
`wandb` block (uploaded by the former `scripts/wandb_sync.py`) is adopted with its
directory name as the id, and W&B's last step as the floor. A check run (at most
`CHECK_ITERATIONS` iterations) is never enabled or published. `status` only reads.
Neither command writes a run directory: what a sync did is recorded in the lineage's
own `run.json` (`wandb.synced`, `published_through`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crowdmotion.tracking import history
from crowdmotion.tracking.evals import pending, publish, published_above, reconcile
from crowdmotion.tracking.lineage import (
  CHECK_ITERATIONS,
  WriterLock,
  check_run,
  enabled,
  new_block,
  note_lost,
  own_checkpoints,
  read_record,
  settle_floor,
  short,
  stuck,
  utc_now,
  write_record,
)


def training_runs(root: Path) -> list[Path]:
  """Run directories under `root` whose `run.json` is a `crowd train traversal` record."""
  out = []
  for run in sorted(Path(root).glob("*/run.json")):
    try:
      if read_record(run.parent).get("kind") == "train_traversal":
        out.append(run.parent)
    except (OSError, ValueError):
      continue
  return out


def parent_of(record: dict) -> str | None:
  """The run directory a pre-`wandb` run started from (its W&B id), from its recorded paths."""
  for key in ("resume_from", "teacher", "finetune_from"):
    if record.get(key) and "runs" in Path(record[key]).parts:
      return short(record[key]).split("/")[0]
  return None


def sync(run: Path, enable: bool = False, dry_run: bool = False, sdk=None) -> str:
  """Catch one lineage up; returns "synced", "current", "dry-run", "disabled", "locked",
  "skipped" or "failed"."""
  run = Path(run)
  try:
    record = read_record(run)
  except (OSError, ValueError) as exc:
    print(f"{run.name}: no readable run.json ({exc!r}); skipped")
    return "skipped"
  if record.get("kind") != "train_traversal":
    print(f"{run.name}: not a `crowd train traversal` run; skipped")
    return "skipped"
  if check_run(record):
    print(f"{run.name}: a check run (at most {CHECK_ITERATIONS} iterations) stays off W&B; skipped")
    return "disabled"
  if not enabled(record) and not enable:
    mode = (record.get("wandb") or {}).get("mode", "not set up")
    print(f"{run.name}: W&B {mode}; skipped (`--enable` turns it on)")
    return "disabled"
  if stuck():
    print(f"{run.name}: an earlier W&B finish of this process is still running; skipped")
    return "failed"
  lock = WriterLock(run)
  if not lock.acquire("crowd wandb sync"):
    print(f"{run.name}: the writer lock is held ({lock.holder()}); skipped, the holder publishes")
    return "locked"
  try:
    record = read_record(run)  # as the last holder left it
    if check_run(record):
      print(f"{run.name}: a check run (at most {CHECK_ITERATIONS} iterations) stays off W&B; skipped")
      return "disabled"
    if record.get("wandb") is not None:
      reconcile(run, record, sdk, write=not dry_run)  # a session that ended unconfirmed
    else:
      record["wandb"] = new_block(run, "online", parent=parent_of(record), legacy=True)
    block = record["wandb"]
    block["mode"] = "online"
    after, above = settle_floor(block, sdk), published_above(block)
    ckpts = own_checkpoints(run)
    cap = ckpts[-1][0] if ckpts else None
    rows = history.rows(run, record, after=after, through=cap) if cap is not None and cap > after else {}
    rows = {i: values for i, values in rows.items() if i not in above}
    if cap is not None and cap > after:  # committed rows TensorBoard lacks and W&B does not hold
      note_lost(record, history.expected(run, record, after, cap) - above - set(rows) - set(block.get("lost") or ()))
    queued = pending(run, record)
    print(f"{run.name}: W&B {block['entity']}/{block['project']}/{block['id']}, published through {after}"
          + (f" and {len(above)} rows above" if above else "") + f", newest checkpoint {cap}: {len(rows)} rows"
          + (f" ({min(rows)}-{max(rows)})" if rows else "") + f" and {len(queued)} evaluations to publish",
          flush=True)
    if dry_run:
      return "dry-run"
    if not rows and not queued:
      write_record(run, record)  # an --enable is kept
      return "current"
    publish(run, record, rows, sdk)
    block.update(status="synced", synced=utc_now())
    block.pop("error", None)
    write_record(run, record)
    return "synced"
  except Exception as exc:
    print(f"{run.name}: sync failed: {exc!r}", flush=True)
    return "failed"
  finally:
    lock.release()


def status_line(run: Path) -> str:
  record = read_record(run)
  block = record.get("wandb")
  ckpts = own_checkpoints(run)
  cap = ckpts[-1][0] if ckpts else None
  lock = WriterLock(run)
  holder = "free" if lock.free() else f"held: {(lock.holder() or {}).get('purpose', '?')}"
  attempts = len(record.get("attempts", [])) or 1
  if block is None:
    return f"{run.name:<58} {'(none)':<9} {'pre-wandb':<10} {'-':>9} {cap!s:>7} {'-':>6} {attempts:>3} {holder}"
  done = block.get("published_through")
  behind = "" if not enabled(record) or cap is None else f" behind {cap - (done if done is not None else -1)}"
  out = sum(1 for _ in (run / "wandb" / "outbox").glob("*.json")) if (run / "wandb" / "outbox").is_dir() else 0
  open_ = block.get("unconfirmed")
  flags = (" check run" if check_run(record) else "") + (f" unconfirmed since {open_.get('since')}" if open_ else "")
  return (f"{run.name:<58} {block.get('mode', '?'):<9} {block.get('status', '?'):<10} {done!s:>9} {cap!s:>7} "
          f"{out:>6} {attempts:>3} {holder}{behind}{flags}")


def main(argv: list[str] | None = None) -> int:
  from crowdmotion.core.paths import RUNS

  p = argparse.ArgumentParser(prog="crowd wandb", description=__doc__.split("\n\n")[0])
  sub = p.add_subparsers(dest="command", required=True)
  s = sub.add_parser("sync", help="publish TensorBoard rows and queued evaluations of training lineages")
  s.add_argument("runs", nargs="*", type=Path)
  s.add_argument("--all", action="store_true", help="every online training lineage under runs/")
  s.add_argument("--enable", action="store_true", help="turn W&B on for the given disabled or older runs")
  s.add_argument("--dry-run", action="store_true", help="print what would be published")
  t = sub.add_parser("status", help="W&B state of training lineages (reads only)")
  t.add_argument("runs", nargs="*", type=Path)
  args = p.parse_args(argv)
  if args.command == "status":
    print(f"{'run':<58} {'mode':<9} {'status':<10} {'published':>9} {'ckpt':>7} {'outbox':>6} {'att':>3} lock")
    for run in args.runs or training_runs(RUNS):
      print(status_line(run))
    return 0
  if args.all == bool(args.runs):
    p.error("give run directories or --all")
  if args.all and args.enable:
    p.error("--enable takes explicit run directories")
  results = [sync(run, enable=args.enable, dry_run=args.dry_run) for run in (args.runs or training_runs(RUNS))]
  return 1 if "failed" in results else 0
