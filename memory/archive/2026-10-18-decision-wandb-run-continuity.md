# W&B run continuity

**Resolved 2026-10-18 (user), ledger D20; archived.** Answers: (1) in-place resume, yes:
`--run`, `attempts[]`, `target_iterations`; a branch (`--resume`) keeps a new lineage with
the parent's id. (2) Verification runs stay off W&B. (3) No checkpoint uploads. (4) No
real W&B writes yet: fake SDK only, writer off by default until a canary in a scratch
project. (5) Signal lead time 300 s, with requeue on `checkpointed`. Implemented in
`crowdmotion.tracking`, `sim.mjlab.wandb_log`, `crowd train traversal` and
`train_traversal.sbatch`.

Opened 2026-10-18 at the user's request: sync every future run to
`rl2-group/crowd-control-traversal` (proposed scope: traversal training lineages and
their evaluations, not other `crowd` run kinds), resume interrupted jobs into the same W&B run, and
attach evaluations (videos, metrics) to it. **Whose call:** user. **Blocking:** the
implementation; nothing else.

Proposal (Claude doc "W&B run continuity for crowd-control",
https://claude.ai/code/artifact/cc67b673-6b01-44f1-a980-6e1b611f478c):

- One W&B run per training lineage. Its id is the first run directory's name, as for the 15
  runs `scripts/wandb_sync.py` uploaded. It is recorded in `run.json`.
- An in-process rsl_rl log writer, selected through rsl_rl 5.5.1's dict logger, writes
  TensorBoard and W&B. Its axis is the custom `iteration` (no `step=`). It publishes rows
  only through the last saved checkpoint: the flush is in the traversal runners' `save`
  overrides, because mjlab calls `save_model` only when `upload_model` is on. A resumed
  job drops rows at or below `published_through`, so W&B never shows a discarded
  trajectory and no rewind is needed.
- On SIGTERM the run checkpoints and exits as `checkpointed`, and the sbatch file
  requeues it. A resume trains to a recorded `target_iterations`.
- Evaluations attach under `eval/*` at the checkpoint's iteration. One writer per W&B run
  (flock): an evaluation during training leaves an outbox entry that the live writer
  drains. The eval ledger lives in `run.json`.
- A TensorBoard-based `crowd wandb sync|status` replaces `scripts/wandb_sync.py` as the
  catch-up. No W&B offline mode: offline ignores `resume`.

Compared with `~/flash/hrc` (commit `01949f3`, `docs/TRACKING.md`, ledger D29):

- **Kept:** hrc's rules (local files authoritative, one writer per run, custom step axis,
  checkpoint-then-requeue, ledgered evaluations).
- **Dropped:** its separate mirror process. It is unsignalled, was killed at requeue, and
  left 4 sidecars stuck at `publishing`.
- **Not copied:** its live publishing with replayed-step skips. That relies on
  bit-exact resume, and mjlab does not checkpoint env state.
- **Not yet proven in hrc:** eval attachment has made no real W&B writes.

Open questions for the user:

1. In-place resume (all attempts in one run directory, `attempts[]` in `run.json`, as hrc
   D29) versus a fresh directory per job inheriting `wandb.id`. In place reverses "never
   reused" (`AGENTS.md`, `RunDir`) and the one-row-per-directory run log.
2. Whether verification runs (2–5 iterations) go to the project (`job_type="check"`).
3. Checkpoint uploads: none (recommended), final only, or every save.
4. Permission for real writes to a scratch project as a canary before the default switches.
5. Signal lead time: 120 s (today) or hrc's 300 s.

The user's answers decide it. Then record the outcome in the ledger and `AGENTS.md` (one
writer per W&B run; lineage id), and implement in the proposal's order.
