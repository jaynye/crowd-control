# Saved-graph behavior iteration on trublu

Completed 2026-10-05. **Direction improved; retain the selected setting as opt-in
because long-rollout spacing and response delay remain unresolved.**

Outer run: `runs/20261005T201450Z_behavior_iteration/`, existing local session;
launcher allocation 4018337, NVIDIA A40 on trublu, driver 580.178.04,
torch 2.9.0+cu128 / CUDA 12.8. Evaluations used CPU float32, one thread; GPU was
used for replay and the small mjlab check. No jobs submitted, priors loaded,
assets rebuilt, dependencies changed, commits or pushes. Graph sha256 starts
`acbc5df3f295`; full provenance/config hashes are in the run.

The [earlier unfinished session](2026-10-05-graph-behavior-iteration.md) already
contained opt-in committed intent, facing/heading costs, forward steering,
avoidance priority and diagnostics. Initial modified files were snapshotted and
preserved. This session added `GraphConfig.heading_deadband_deg` (default zero),
its `HeadingCost` calculation and focused tests, and documented the existing
controls. Legacy plaza RNG and `GraphConfig` leniency remain unchanged.

## Iterations and selection

All 80 named requests completed in the existing session, with unique names,
explicit rule notes and timeouts. The original full configuration and graph hash
were frozen before edits; baseline `000_baseline` used seeds 0/1/2, 16 humans,
20 seconds. The session journal records each observation/decision.

1. `006_turns_only`: forward 55–62%, but only 8/49 turn attempts completed; wall
   accumulation in inspected overview frames. Rejected as a final behavior.
   Forward native turns of at least 10° were available at only 25–39% of visited
   large-turn states; this does not prove that longer feasible graph routes are absent.
2. `011_facing010`: facing weight 0.1 alone produced only 24–33% forward travel.
   `016_heading050` added heading weight 0.5 and reached 46–51% without the
   turns-only wall accumulation.
3. `021_forward`: forward steering reached 75–84%, but minimum gaps deteriorated
   to 0.371/0.631/0.421 m. `026_evasive` released orientation costs during urgent
   avoidance; spacing was still worse than baseline.
4. `031_contact120`: contact radius 0.9 → 1.2 m removed sub-0.6 m pair frames in
   development but lowered forward share to 62–72% and median speed to 0.47–0.49 m/s.
5. `037_heading100`: heading weight 1.0 completed all controlled 180° body turns
   in development; small spacing and foot tradeoffs remained.
6. `043_deadband10`: new 10° heading deadband lowered foot-speed p95 on all three
   development seeds, but one controlled 180° response was censored.
   `049_deadband05` reduced the deadband to 5° and retained all 48/48 controlled
   responses with minimum development gaps 0.817/0.788/0.732 m. Foot-speed p95 was
   0.486/0.437/0.425 m/s versus baseline 0.483/0.507/0.453: mixed, not uniformly better.

The selected config was frozen in `configs/selected_candidate.json` and
`heldout_freeze.json` before any held-out evaluation. No retuning followed.
It retains original destination intent; committed sampled turns are disabled.

## Held-out evidence

Seeds 11/23/47; same graph, population, duration and initial seed bank per pair.
All values are baseline → selected candidate. Direction is relative to midpoint
body facing, not the global path tangent. Cutoffs are diagnostics, not pass limits.

| Duration / seed | Moving forward % | Moving backward % | Minimum root gap m | Median speed m/s | Planted-foot speed p95 m/s |
|---|---:|---:|---:|---:|---:|
| 20 s / 11 | 23.88 → 69.98 | 38.62 → 6.55 | .719 → .736 | .695 → .604 | .457 → .465 |
| 20 s / 23 | 19.97 → 70.36 | 36.15 → 3.80 | .590 → .874 | .705 → .495 | .468 → .446 |
| 20 s / 47 | 22.90 → 63.67 | 27.98 → 9.13 | .510 → .608 | .647 → .506 | .474 → .457 |
| 60 s / 11 | 24.99 → 66.15 | 37.34 → 8.10 | .670 → .501 | .677 → .577 | .462 → .458 |
| 60 s / 23 | 22.46 → 68.13 | 33.93 → 5.98 | .590 → .580 | .661 → .508 | .458 → .449 |
| 60 s / 47 | 24.31 → 63.15 | 28.70 → 9.93 | .510 → .573 | .634 → .506 | .456 → .442 |

Paired reports: `session/iterations/057_heldout_compare/` and
`063_long_compare/`; diagnostics `058`, `059`, `064`, `065` in the outer run.
At 60 s, candidate ordinary-command travel was 80.72–83.15% forward and
1.60–2.28% backward; evasive commands retained 25–29% sideways and 16–24% backward.
Candidate acceleration p95 fell on all held-out seeds, but root seam acceleration
was mixed (seed 11: 11.38 → 11.94 m/s²). Pose-step p95 fell on all three long seeds.
Seed 11 had 0.104% outside-wall human frames. Root gaps are not capsule penetration
measurements. The worst gap was 0.501 m at 34.4 s, humans 5/8, both already evasive.

`060_heldout_probe` used fixed 90°/180° requests in separate empty plazas, 48 humans
per scenario. Baseline root medians 0.90/1.06 s became 2.33/5.83 s (candidate p95
4.08/9.60 s). Baseline body response reached 28/48 and 0/48; candidate reached
48/48 in each case, median 2.77/6.10 s and p95 4.79/10.28 s. Delays include replan
waiting; no acceptable delay was invented.

## Visual review and integration

Inspected 17 contact sheets (102 stills), not just MP4 existence:

- Development seed 0: baseline overview and follow human 4; each candidate's saved
  overview or follow view, at 0/4/8/12/16/19.5 s (requests `004`–`053`).
- Frozen baseline/candidate 60 s: seed 23 overview (`067`, `069`), seed 11 follow
  human 4 (`071`, `073`), at 0/12/24/36/48/59.5 s.
- Worst long candidate encounter: seed 11 follow human 5 (`076`), 33.6–34.6 s at
  0.2 s intervals. Close/overlapping silhouettes persist near the measured minimum.
- GPU mjlab seed 11 overview (`077`), 0/1.5/3/4/6/7.5 s.

The selected goal-based candidate continues using the interior, unlike turns-only
wall accumulation. Leaning/hunched poses and close passes persist. The available
image tool inspects stills, not continuous video; temporal smoothness is not
visually established. Foot/seam diagnostics provide separate numeric evidence.

`074_mjlab_candidate_check`: 2 environments × 4 humans, 8 s / 400 steps on A40,
80 finite-pose checks, 1,440 LiDAR rays and human hits in both environments;
resetting env 1 preserved env 0 state. Startup 2.08 s, mean step 2.05 ms, rendering
8.54 ms. This is a small integration check, not a 4,096-environment throughput claim.

## Validation and disposition

- 136 final CPU unit tests passed; focused cost/layer tests and relevant lint passed.
- Frozen baseline recheck `078`/`079`: every trace array on seeds 0/1/2 is bit-exact;
  configuration and saved-graph hashes unchanged.
- Original committed code and working defaults match 158 arrays and six JSON outputs
  on this host. Both still differ from historical golden hashes in small numeric
  diagnostics/table distances; the reference checks did **not** pass. No expected
  files were changed or recaptured. Evidence: `validation/golden_compatibility.json`.
- The first standalone compatibility helper failed on an incorrect outer-directory
  path before running checks. Its script/log were preserved; a separately named
  corrected helper completed the checks. Experiments were already finished.
  Initial sandbox loopback failure had been recovered through reviewed execution.

Memory validation also found a pre-existing, stopped, empty session missing from the
run log: `runs/20261005T162004Z_v0-1/`. Its recorded provenance was logged without
starting, submitting or changing any job.

Keep the selected config opt-in; no global default, baseline or golden changes.
Next useful experiment: candidate-edge-aware separation/braking over the executed
0.4 s horizon, using saved close encounters as regression cases and a fresh seed
bank. Committed-turn boundary handling and additional turn coverage remain separate
experiments; they were not implemented as hidden fixes here.

The outer run's `report.md`, `session/journal.md`, comparison pages, saved traces,
frames and validation logs are the detailed record. Project memory and the run log
were finalized before the launcher deadline (23:59 UTC).
