# 2026-10-18: W&B run continuity

Status: done (retired 2026-10-18). Implements the proposal of the
[W&B run continuity](../archive/2026-10-18-decision-wandb-run-continuity.md) decision with the user's
answers of 2026-10-18: in-place resume (`--run`, `attempts[]`, `target_iterations`);
verification runs stay off W&B; no checkpoint uploads; no real W&B writes yet (fake SDK
only, writer off by default until a canary); SIGTERM 300 s before the limit with requeue.
Runs in parallel with the traversal evaluation harness (parent agent, its own journal);
file ownership: `tracking/`, `sim/mjlab/wandb_log.py`, `train.py`, `play.py`,
`scripts/wandb_sync.py`, `train_traversal.sbatch`, `docs/slurm.md` are mine; in
`traversal.py` only the runners and RL configs; `cli.py`, `docs/cli.md`, `AGENTS.md`
shared (edit only).

## Plan

1. `crowdmotion.tracking` (L3, no mjlab): lineage identity and settings in `run.json`,
   writer lock (flock), outbox, eval ledger and `attach_eval`, TensorBoard reader cut per
   attempt, `crowd wandb sync|status`.
2. `sim/mjlab/wandb_log.py`: `CrowdWandbWriter` (TensorBoard + buffered W&B rows,
   `*/time` dropped, floor rule), runner mixin (flush in `save`, SIGTERM check after
   each iteration's log call); wire into `_ppo_runner`, `_distillation_runner`.
3. `train.py`: `--run` in place, `--resume` as the branch path, `--wandb`, `attempts[]`,
   `target_iterations`, SIGTERM flag, `checkpointed` exit code.
4. `play.py --video` attaches with kind `play`.
5. `cli.py` `wandb` group; `scripts/wandb_sync.py` becomes a shim.
6. `train_traversal.sbatch`: requeue, TERM@300, trap/forward/wait, conditional requeue.
7. Unit tests with a fake SDK; docs; optional GPU check (SIGTERM, requeue, resume).
8. Ledger entry, `AGENTS.md`, archive the decision, retire this entry.

## Log

- Step 1 code: `core/rundir.py` (atomic `write_json`, `RunDir.open`, docstring), `tracking/{__init__,lineage,history,evals,sync}.py` written (tests pending).
- Step 2 code: `sim/mjlab/wandb_log.py` (`Session`, `on_sigterm`, `Checkpointed`, `CrowdWandbWriter`, `Tracked`); `traversal.py` runners mix in `Tracked` (only `_ppo_runner`, `_distillation_runner` edited).
- Tests: tests/unit/test_tracking.py (fake SDK; floor, TB cut 0-3780/3700, disabled writes nothing, outbox+lock, failed attach retry, sync cap/locked/dry-run/enable, writer one row per iteration, */time, video once, SIGTERM checkpoint + resume 0-299 once, gap fill, outbox drain by live writer, W&B failure keeps training): 13 passed.
- Step 3 code: `train.py` rewritten (`--run` in place, `attempt_argv`, `lineage_target`, `wandb_mode`, `CHECK_ITERATIONS`, attempts, lock held for the attempt and released on every exit, `EXIT_CHECKPOINTED`=75); `play.py` attaches kind `play`; `cli.py` `wandb` group; `scripts/wandb_sync.py` shim; layering + architecture rows; `train_traversal.sbatch` (requeue, TERM@300, trap/forward/wait, requeue only when checkpointed at the time limit). Tests: `tests/fake_wandb.py` shared; full unit suite 304 passed, 3 skipped. Fixed a lock leak on early exit found by the tests.
- GPU check submitted: job 4133159, run `runs/20261018T153936Z_train_traversal_wandb_check` (64 envs, 20 iterations, save interval 5, W&B off as a check run; watcher in the scratchpad sends `scancel --signal=TERM --batch` at iteration 3 with CROWD_SIGNAL_LEAD_S=86400 so the requeue branch runs; then probes the lock from sky1).
- Docs: `docs/cli.md` (train row, Lineages and W&B paragraphs, `## wandb`, shim row), `docs/slurm.md` (job row, Training lineages section), `docs/architecture.md` row, `AGENTS.md` (runs row, layers, commands exception, Training lineages and W&B bullet, run-log rule), `scripts/index_runs.py` (attempt jobs, kinds). Ledger D20 added; decision moved (untracked file, plain mv) to `memory/archive/2026-10-18-decision-wandb-run-continuity.md`; decisions/archive READMEs updated. GPU check attempt 1: TERM at it 16, checkpoint model_17, exit 75, requeued.
- Lock probe job 4133216 (CPU-only, holds `logs/slurm/lockprobe/wandb.lock` 60 s on a compute node while sky1 probes).
- Lock probe 4133216 inconclusive (holder still importing torch at its 5 min limit); retry job 4133247 with a stdlib holder.
- Added `queue_eval` + `flush_outbox` (one W&B session for many rows of one evaluation; test added; 14 tracking tests). Design rationale section in `memory/core/design/runtime-architecture.md`; run-log row for the GPU check. GPU check result: passed (attempt 2 resumed from model_17 to model_19, status done). Unit suite 305 passed, 3 skipped; check_memory clean.
- Cross-node lock: job 4133247 (stdlib flock holder on hal) held `logs/slurm/lockprobe/wandb.lock`; the sky1 probe was refused: flock is shared across nodes on /coc/flash7. Recorded in runtime-architecture.md.
- Review fix: an in-place attempt now takes its env options from the lineage's own `run.json` (`env_options(args, args.run)`), so a manual `--run` or a pre-option run keeps its graph and robot options; test added (17 in test_train_options). AGENTS.md names `crowd wandb sync|status` as writing no run directory.
- Done. Promoted: ledger D20, `AGENTS.md` (training lineages and W&B), `memory/core/design/runtime-architecture.md` (training lineages and W&B), docs (`cli.md`, `slurm.md`, `architecture.md`), run log. Open: the canary in a scratch project (the user's go-ahead), then `DEFAULT_MODE`; the online writer, the distillation runner and the finetune in-place path have not run on a GPU. Retired to the archive.
