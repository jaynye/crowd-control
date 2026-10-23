"""W&B run continuity for traversal training lineages; no mjlab import.

A **lineage** is one `crowd train traversal` run directory with every attempt resumed in
it (`--run`, `run.json`'s `attempts`). It maps to one W&B run in `CROWD_WANDB_ENTITY` /
`CROWD_WANDB_PROJECT` (default `rl2-group/crowd-control-traversal`) whose id is the
directory's name, recorded in `run.json`'s `wandb` block:

  {"entity", "project", "id", "url", "mode", "parent", "tags", "published_through",
   "published_above", "lost", "unconfirmed", "status", "evals"}

Local files are authoritative and W&B is a mirror of them. Rules:

- **One writer per W&B run.** `<run>/wandb.lock` (flock) is held by the trainer for its
  whole attempt, or briefly by an evaluation attachment or `crowd wandb sync`. Only the
  holder writes the `wandb` block of `run.json`, and it finishes every W&B session it
  opened before it releases the lock.
- **Rows only through the last checkpoint.** The live writer (`sim.mjlab.wandb_log`)
  sends a row only once a checkpoint of its attempt covers it; each attempt contributes
  only rows its own checkpoints committed (`tracking.history`), so W&B never shows a
  trajectory that a resume discarded. Rows carry `iteration` (the step metric), never
  `step=`.
- **Published means confirmed.** `published_through` (every row up to it is on W&B) and
  the ledger advance only after `finish()` returned within the SDK's `finish_timeout`. A
  session that ends otherwise (a crash, a kill, a finish timeout, a W&B error) stays
  `unconfirmed`; the next holder reads the run's history from W&B and replays only the
  rows it lacks (`evals.reconcile`, bounded; W&B's rows above a gap in `published_above`,
  committed rows neither TensorBoard nor W&B holds in `lost`). An evaluation row with a
  video counts only once W&B also holds its file. A finish that never returns keeps the
  lock until the process exits.
- **Evaluations** attach one row each (`attach_eval`) under `eval/<kind>/*` at the
  checkpoint's iteration. Without the lock they wait in `<run>/wandb/outbox/` for the
  next holder. The ledger, `wandb.evals`, makes attachment idempotent. Many rows of one
  evaluation go out in one W&B session with `queue_eval` for each, then `flush_outbox`.
- **Catch-up** from TensorBoard: `crowd wandb sync|status` (`tracking.sync`).
- **Off W&B:** check runs (at most `CHECK_ITERATIONS` iterations) on every path, and any
  process whose `WANDB_MODE` is set to something other than `online` (refused, not
  overridden). Sessions open with `mode="online"`.

`mode` is `online` or `disabled` per lineage. `DEFAULT_MODE` (in `lineage`, `online`) sets
it for new lineages; a lineage keeps its recorded mode.
`CROWD_WANDB_TAGS` (comma-separated) adds tags to new lineages, recorded in `tags`.
"""

from crowdmotion.tracking.evals import attach_eval, flush_outbox, queue_eval

__all__ = ["attach_eval", "flush_outbox", "queue_eval"]
