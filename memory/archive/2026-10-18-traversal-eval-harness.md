# 2026-10-18: traversal evaluation harness

Status: done (retired 2026-10-18). Promoted to the [traversal evaluation](../decisions/traversal-evaluation.md)
decision ("Built"), ledger D20 and `memory/core/status.md`.

Original note: User (2026-10-18): build the metric suite and evaluation harness of
the [traversal evaluation](../decisions/traversal-evaluation.md) decision, with envs,
episode length and the rest configurable. W&B run continuity is built in parallel by a
subagent (its own journal, `2026-10-18-wandb-continuity.md`); the harness attaches rows
through `crowdmotion.tracking.attach_eval`.

## Plan

1. Env options for the next training arms (done before the parallel split): `fall_end`
   terminal penalty, `penalty_ramp_term` curriculum (`--penalty-ramp N`), b0
   (`--no-avoidance`), `collision_termination()`.
2. Paired goals: `CrowdGoalCommandCfg.paired_seed` draws spawns, yaws, goal chains,
   speeds, standing flags and resample times from per-env `core.rng` streams, each goal
   at least `min_goal_m` from the previous goal. Off by default.
3. `crowdmotion.metrics.traversal` (L1, no mjlab): command following per leg, collisions,
   clearance-rate time to contact, encounters, reaction latency, falls and their
   attribution, Kaplan–Meier time to first collision, exposure.
4. `crowdmotion.sim.mjlab.evaluate` (`crowd eval traversal`): rollouts per checkpoint ×
   crowd (aware, blind) × mode (end, continue); a `"step"` event captures each terminal
   state before the auto-reset; writes `metrics.json` and CSV tables; attaches to W&B.
5. Unit tests (CPU), a GPU smoke on Slurm, docs, an Astra review.

## Log

- Step 1 done: `traversal.py`, `train.py`, `play.py`; `tests/unit/test_train_options.py`
  11 passed.
- Step 2 done: `CrowdGoalCommandCfg.paired_seed` (`traversal_goals` seed scheme in
  `core.rng`); the default path makes the same global-generator calls in the same order.
- Step 3 done: `src/crowdmotion/metrics/traversal.py`; `tests/unit/test_traversal_metrics.py`
  8 passed.
- Step 4 done: `src/crowdmotion/sim/mjlab/evaluate.py` (`crowd eval traversal`), CLI entry,
  `scripts/slurm/eval_traversal.sbatch`; `tests/unit/test_traversal_eval.py` 3 passed.
- GPU jobs submitted: 4132687 (`tests/gpu/test_traversal_eval.py`), 4132688 (a2
  `model_3999`, 64 envs × 10 s, both crowds and modes, `--no-attach`).
- Jobs 4132687/4132688 and 4132694/4132695 failed at env build: `replace(sensor,
  debug_vis=False)` breaks on mjlab's contact-sensor configs; the override was removed
  (debug drawing only matters with a viewer). Failed run directories:
  `runs/20261018T145915Z_eval_traversal_a2`, `runs/20261018T150023Z_eval_traversal_a2`.
- A 2-env × 0.4 s CPU smoke ran on the login node (output in the session scratchpad, no
  run directory) and completed. The user then ruled out heavy computation on login nodes;
  the rule is in `AGENTS.md` (Slurm). Resubmitted: 4132715 (GPU test), 4132716 (a2 smoke).
- Astra reviewer 1 (gpt-6-astra, xhigh, read-only) is reviewing the harness.
- 4132715 (GPU test) passed; 4132716 (a2 smoke, `runs/20261018T150431Z_eval_traversal_a2`)
  completed all four rollouts in 1.5 min. Its numbers exposed: clearance-rate TTC flickers
  with the gait (267 encounters per robot-minute, 17% of moving steps under 1 s TTC,
  0.14 s reaction latency); zero-length episodes and legs at the rollout's end (128
  episodes, success diluted to 0.2). Fixed: TTC from the line-of-sight closing speed of
  root and hips, encounters end after a 0.5 s gap, reaction velocity smoothed (0.3 s,
  seeded per episode), zero-step records dropped, default time-out beyond the rollout.
- Astra review 1 (eval harness): 10 findings. Fixed: timers and standing flags keyed by
  the scheduled-resample count (were keyed by arrivals, so policy-dependent); collisions
  latched until separation; episodes carry an independent `fell` flag for fall survival;
  fall controls weighted to the falls' strata (standing × episode age); reactions measured
  against the velocity at the threat and restarted by a new goal; exact deviation and
  drift sums (histograms for quantiles only, range 20 m); infinite TTC kept in the median;
  the penalty ramp's steps set from the parsed `num_steps_per_env` (`train.py`). Already
  fixed: zero-length episodes and legs. Not changed: per-step host syncs (measured below).
  Unit tests: 13 metric tests, 3 eval tests, 17 train-option tests pass.
- W&B subagent finished (ledger D20; see its archived journal). The harness now queues
  rows with `queue_eval` and publishes them with one `flush_outbox`.
- Submitted 4133387 (GPU test) and 4133388 (a2 `model_3999`, 4,096 envs × 60 s, aware,
  both modes, attach on: expected `disabled`).
- 4133387 (GPU test) passed; 4133388 completed (`runs/20261018T160808Z_eval_traversal_a2`,
  0.390 collisions per robot-minute in end mode). Astra review 2 (W&B): 8 findings;
  checkpoint saves made atomic and the legacy sync shim gated; 7 recorded in D20 as canary
  blockers. Unit suite and memory check pass. The training arms are in the experiment
  2026-10-18-fall-fix-and-b0.
