# 2026-10-22: live traversal viewer

## Request (user via lead agent, 2026-10-22)

An mjlab Viser viewer to watch trained traversal policies walk through crowds live,
with a live performance panel, and to switch crowd condition and policy from the GUI
without restarting the process. Build on `crowd play traversal`
(`src/crowdmotion/sim/mjlab/play.py`). Constraints: login node `sky1` (no env builds or
rollouts here; GPU checks via Slurm), do not touch Slurm job 4178666 or any existing run
directory, never re-capture goldens, legacy defaults unchanged, viewer stays off W&B.
Avoid editing `memory/core/status.md`, experiment notes and the ledger (another agent
is writing a report).

## Plan

1. `ablate_navigation`: factor the campaign's condition list and condition behavior
   into functions (`plan()` output unchanged), so the viewer and evaluator share them.
2. `evaluate.make_env`: split into `eval_env_cfg` + `build_env` (pure refactor) so the
   viewer builds exactly the evaluator's env, then adjusts viewer-only fields.
3. `planning.PlanningVecEnv`: step as a generator of substeps (same ops/order), plus a
   per-env-step adapter so Viser can play planning arms (play refused them before).
4. `metrics.traversal.TraversalMetrics.snapshot()`: non-destructive summary for the
   live panel (same definitions as the evaluator).
5. New `sim/mjlab/live.py`: run discovery, crowd condition registry (`trained` + the
   seven campaign conditions), compatibility checks, switch planning (load / live
   queued crowd edit / env rebuild), live metrics, `LiveViewer` (Viser) with an outer
   rebuild loop that keeps the server up.
6. `tuning.LiveCrowdTuning`: queued multi-field encounter edit for live crowd switches.
7. `play.py`: Viser path -> live viewer (`--crowd`, `--runs`, `--conditions`); native
   and `--video` paths unchanged.
8. CPU unit tests; docs (`docs/cli.md`, `docs/usage/demos.md`, `docs/slurm.md` if needed).
9. GPU Slurm check driving the viewer headless; log; runs.md row if a run dir appears.

## Progress

- (step log below)
- Steps 1-7 coded (sources backed up before editing under the session scratchpad):
  `ablate_navigation.navigation_conditions/condition_behavior` (+ `BASE_BEHAVIOR`);
  `evaluate.eval_env_cfg/build_env` (make_env = build_env(eval_env_cfg(...)));
  `planning.PlanningVecEnv.decide` (generator; `step` drains it);
  `metrics.traversal.TraversalMetrics.snapshot` (+ `_Table.numpy` keeps joined columns);
  `tuning.LiveCrowdTuning.submit_encounter`; new `sim/mjlab/live.py` (pure) and
  `sim/mjlab/live_viewer.py` (Session, LiveViewer, Live loop); `play.py` Viser path ->
  `live_viewer.serve` (`--crowd`, `--runs`, `--conditions`, `--frame-rate`).
- Login-node check (file reads only): the code-built conditions equal the frozen
  campaign manifest's 7 conditions (metadata and behavior JSON), all True.
- `discover_runs(runs/)`: 97 of 101 traversal runs playable.
- GPU check script `scripts/smoke/live_viewer_check.py`; job 4191549 submitted
  (A40, rl2-lab/overcap, 1 h): nav arm sparse_g99_independent_s42 + planning arm
  plan1hz_curr + teacher_plan1hz refusal. Log `logs/slurm/live-viewer-check-4191549.out`.
- Job 4191549 (A40) cancelled by me while pending on the lab A40 quota (QOSGrpGRES);
  4191643 (L40S) failed at once (`--wrap` runs /bin/sh: `source` not found); checks
  now go through a bash script in the scratchpad (`live_check.sh`).
- Job 4191648 (L40S, zhurong): first build 48 s, nav arm 0.98x real time at 1x with
  1 env; in-place crowd edits trained -> independent_n8 -> coherent90_n8 ->
  reactive_blind_n8 worked (0.1 s, counters restarted, no problems). The first rebuild
  failed: viser 1.0.27 `gui.reset()` cannot remove a tab group (it marks the group
  removed before its tabs). Fix: `live_viewer._remove_tabs` removes tabs innermost
  first, then reset (verified against real viser on the login node; unit test added).
- CPU unit suite (before that fix): 359 passed, 3 skipped.
- Job 4191715 resubmitted with the fix.
- Job 4191715 (L40S, zhurong, 7 min) PASSED: `runs/20261022T161322Z_live_viewer_check`
  (check.json): 21 stages, 12 env builds, no failures or step errors. Nav arm
  sparse_g99_independent_s42 through all 7 crowds + trained (4 in-place edits,
  4 rebuilds), mid-run switch to plan1hz_curr, plan arm through all crowds (3 in place,
  6 rebuilds), checkpoint model_119 -> model_117 in place, teacher_plan1hz in
  independent_n4 refused, nav arm back via the path field. Every panel: seconds ==
  viewer steps x 0.02 (counters restart with the switch). Totals: 16 collisions,
  23 arrivals, nav outcomes 17 success / 16 collision. Builds 1-3.5 s after the first
  (5.4 s with warm Warp cache; 48 s cold); CUDA memory flat across rebuilds (2.35 GB nav;
  3.93 GB after the plan arm: its run.json names the graph via /nethome/jni66/flash, a
  second graph-cache entry -> fixed: `live.load_run` resolves graph/rotations paths).
  Real-time factor ~0.9-1.2x with 1 env even at an 8x target (1x is the ceiling).
- Job 4191726 (L40S, grom): `crowd play traversal <plan1hz> --video --seconds 4` still
  works -> `runs/20261022T162026Z_play_traversal_plan1hz` (W&B: disabled lineage, nothing
  written).
- Failed run dir from 4191648: `runs/20261022T160845Z_live_viewer_check` (tab-group bug).
- TODO: runs.md rows (3 dirs + final rerun), final GPU rerun on final code, check_memory.
- Job 4191861 (L40S, grom): `runs/20261022T162508Z_live_viewer_check`, all 30 viewer
  stages passed (adds: nav_potential in place (rebuilt false), a2 50 Hz in aware n8 /
  trained (in place) / n16, teacher_plan1hz plays in n16, teacher_t7 (50 Hz teacher),
  student_plan1hz, plan2hz, b0). CUDA 2.29-2.35 GB for every policy (path fix works).
  Exit 1 only from the check's websocket client (missing `viser-v1.0.27` subprotocol);
  fixed; rerun as job 4192060 with `--seconds 10`.
- CPU unit suite after all code changes: 361 passed, 3 skipped.
- Docs updated: docs/cli.md (play row), docs/usage/demos.md (live viewer section:
  launch on Slurm/local, tunnel, panel, switch kinds, refusals, measured speed),
  docs/slurm.md, docs/architecture.md, scripts/slurm/play_traversal.sbatch header.
- Job 4192060 (L40S, grom) PASSED, exit 0: `runs/20261022T163632Z_live_viewer_check`,
  30 stages, 19 builds, websocket clients through every rebuild (20 connections, 27k
  messages, no errors, no server exceptions); real-time ~0.75x with clients streaming.
- AGENTS.md "Live crowd settings": one sentence on the viewer sharing eval_env_cfg and
  the campaign condition functions, and crowd switches being queued all-or-none edits
  or rebuilds.
- Remaining: final check_memory; report. Suggested core-memory updates go to the lead.
- Regression checks of the shared refactors (advisor): GPU tests
  `tests/gpu/test_traversal_eval.py tests/gpu/test_navigation.py` job 4192174 (L40S,
  grom): 2 passed. Goldens `pytest -m golden` job 4192173 (CPU, t1000; timed out at
  45 min after all 14 results): evaluate, runtime_step, graph_build, quality_graph and
  compare MISMATCH (~1e-10 relative float diffs), the other 7 fast cases pass (incl.
  mjlab_plaza, which runs GraphCrowd/apply_pending), 2 slow skipped. A/B job 4192590
  (CPU, t1000): the four failing fast cases mismatch identically with an src overlay
  that reverts all of my edits -> pre-existing, not from this work (compare not A/B'd;
  same behavior-evaluator path as evaluate). Note: the session scratchpad is
  login-node /tmp, invisible to compute nodes (first A/B 4192491 silently used repo src).
- Rescan now refreshes the selected run's checkpoints (GUI-only, not driven on GPU).
