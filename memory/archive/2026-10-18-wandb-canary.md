# W&B continuity fixes and canary (2026-10-18)

Request (main session, for the user): fix the seven open W&B run-continuity findings of
ledger D20, add `CROWD_WANDB_TAGS`, have one GPT-6 Astra reviewer check the fixes, then run
the user-approved real W&B canary in the scratch project `rl2-group/crowd-control-traversal-debug`.

Constraints: no commits; no heavy work on the login node (CPU unit suite only; everything
else through `sbatch --partition=rl2-lab --qos=short`); `DEFAULT_MODE` stays `disabled`;
never write to `rl2-group/crowd-control-traversal`, never delete a W&B run; do not touch
jobs 4137004–4137006, 4137182–4137190, 4137777; keep `crowdmotion.tracking` importable and
`queue_eval`, `flush_outbox`, `attach_eval`, `ledger_key` compatible. Owned files:
`src/crowdmotion/tracking/*`, `sim/mjlab/wandb_log.py`, `scripts/wandb_sync.py`,
`tests/fake_wandb.py`, `tests/unit/test_tracking.py`, W&B sections of `docs/`, and in
`train.py` the `lineage_flags`/`attempt_argv` region plus minimal call sites. A baseline copy
of these files before the fixes: `wandb_fix_base/` in the session scratchpad (not durable).

## Plan

1. Fixes (numbering of D20):
   1. Confirmed floor: `published_through` advances only after `finish()` returned without
      timeout; a session marks `wandb.unconfirmed` before its first row; the next holder
      reconciles against the remote history (`scan_history`) before choosing rows.
   2. Every opened session is finished in `finally`; `reinit="finish_previous"`; the
      handle's id, entity, project and mode are checked.
   3. Attempt contribution from committed checkpoints (`history.segments`).
   4. SDK `finish_timeout` + `finish_timeout_raises`; a finish past its bound keeps every
      writer lock until the process exits; `stop()` returns its first result.
   5. `lineage.check_run` gates `enabled`, `init_run` and `sync --enable`.
   6. `mode="online"` passed to `wandb.init`; a conflicting `WANDB_MODE` is refused.
   7. `attempt_argv` drops the recorded `--out`.
   Plus `CROWD_WANDB_TAGS` (block `tags`, appended at every init).
2. Unit tests per fix; `pytest -q tests/unit`; `ruff check src tests scripts`.
3. One Astra review (`logs/wandb_canary/review_fixes.md`); fix or reject findings.
4. Canary: `runs/<UTC>_train_traversal_TEST_canary_debug`, 64 envs, 40 iterations, save
   interval 10, TERM at about iteration 15, requeue, eval attach, `crowd wandb status|sync`,
   read-back through `wandb.Api()` in a CPU job (outputs in `logs/wandb_canary/`).
5. Records: run log rows, D20, docs, `check_memory.py --check`; retire this entry.

## Log

All times 2026-10-19 UTC (corrected at retirement from file times and sacct; the first
draft of this log had guessed times for the steps before the canary).


- 00:30 UTC: read the code, the review and wandb 0.30.0 (`reinit` default is
  `return_previous` outside notebooks; `finish_timeout`/`finish_timeout_raises` bound
  `finish()` in the service; `Api.run` turns `RunNotFoundError` into `CommError`;
  `scan_history(use_cache=False)` reads the full history). Plan above.
- 00:32 UTC: baseline of the owned files copied to `logs/wandb_canary/base/` (durable).
- 00:50 UTC: Task 1 code done: `tracking/{lineage,history,evals,sync}.py`, `wandb_log.py`
  writer, `train.py` (`--out` in `LINEAGE_FLAGS`; `lineage_target`/`first_start` moved to
  `lineage`), `tests/fake_wandb.py` (0.30 reuse, mode inheritance, link/delivery, finish
  timeout and hang, API errors), 10 new tests in `test_tracking.py`. Unit suite 327 passed,
  ruff clean. The new tests fail against the baseline (overlay in the scratchpad).
- 00:52 UTC: docs updated (`docs/cli.md` train W&B paragraph and wandb table, `docs/slurm.md`
  W&B bullets with `CROWD_WANDB_*`). Review diff `logs/wandb_canary/fixes.diff`; Astra review
  launched (PID 3323970; prompt `logs/wandb_canary/review_fixes_prompt.txt`, output
  `review_fixes.md`, exit code in `review_fixes.exit`). Canary helpers ready:
  `logs/wandb_canary/readback.py`, `post_canary.sh`.
- 01:00 UTC: Astra review done (`logs/wandb_canary/review_fixes.md`, exit 0): fixes 2, 3, 5,
  6, 7 and tags correct; 1 and 4 incomplete through four new findings, all accepted:
  (R1, P1) the reconciliation read is unbounded and a TERM during it is only flagged ->
  bound remote reads (thread, sliced join) and let a TERM during go-live stop at once;
  (R2, P2) `confirm` could cross a hole that neither TensorBoard nor W&B holds -> flush
  TensorBoard before each checkpoint, confirm only across covered rows, record such
  holes as `lost`; (R3, P2) reconciliation trusted an eval row whose MP4 may not have
  uploaded -> check the referenced media file before retiring; (R4, P2) a finish stuck in
  init cleanup was invisible to `Session.close` -> it returns False while `stuck()`.
- 01:12 UTC: R1-R4 fixed (`lineage.bounded`/`REMOTE_TIMEOUT_S`, go-live in phase `setup`,
  `Tracked.save` flushes TensorBoard, `lineage.advance`/`note_lost`, media check in
  `evals.reconcile`, `Session.close` checks `stuck()`), 4 more tests: 30 tracking tests
  pass; full unit suite 331 passed + 1 flaky timing test outside my files
  (`test_iterate_session.py::test_deadline_caps...`, passes alone); ruff clean.
  `logs/wandb_canary/fixes.diff` refreshed. Next: the canary.
- 01:20 UTC: canary submitted: job 4138449, `runs/20261019T012044Z_train_traversal_TEST_canary_debug`
  (CROWD_WANDB_* in the sbatch environment, `--export=ALL,CROWD_SIGNAL_LEAD_S=86400,CROWD_GRAPH,...`).
  Plan: TERM (`scancel --signal=TERM --batch`) once `model_10.pt` exists, requeue, then eval, then
  `logs/wandb_canary/post_canary.sh` in a CPU job.
- 01:22 UTC: TERM sent after model_10 (NFS showed it late; model_20 already there); attempt 1 saved model_24, published_through 24 (confirmed, status preempted), exit 75, requeued.
- 01:26 UTC: attempt 2 (restart 1, crushinator) resumed from model_24 (floor 24, rows from 25), done at 39; published_through 39, status finished; end-of-job sync: 0 rows, 0 evaluations.
- 01:28 UTC: eval job 4138491 -> `runs/20261019T012706Z_eval_traversal_TEST_canary_debug`, attached `traversal/aware/end@39`: published.
- 01:50 UTC: post-canary job 4138523 (hal, overloaded): status ok (finished, published 39, lock free, behind 0), sync 'current' (exit 0); cancelled during the slow read-back; read-back resubmitted.
- 01:54 UTC: read-back job 4138774 (t1000): PASS. 41 rows: training 0-39 once each, eval row at 39; no /time tags; id/name/project/tags/group/job type as labeled; define_metric iteration; 3 local sessions same id; scan_history = GraphQL history = remote_rows; not-found probe CommError(RunNotFoundError). JSON: logs/wandb_canary/readback_<run>.json.
- 01:58 UTC: records: run-log rows for both canary directories, ledger D20, runtime
  architecture (training lineages and W&B), status. Left open: `DEFAULT_MODE` (user);
  canary did not exercise an SDK finish timeout, a kill mid-session or a partial media
  upload; AGENTS.md's W&B paragraph still says "until the user approves a canary".
  Retired to the archive.
