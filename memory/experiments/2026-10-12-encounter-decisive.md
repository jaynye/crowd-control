# 2026-10-12: decisive encounter behavior

The opt-in development candidate `configs/behavior/encounter_decisive.json` reduces
close-pair stalls and joint-sphere contacts, but does not solve all long stops.
Ordinary turns beside the west wall/scanner can remain nearly stationary for 48 s.
Keep the candidate in development; [progress recovery remains open](../decisions/encounter-progress.md).

Run: `runs/20261012T152601Z_encounter_decisive`. Its `report.md` contains the full
comparison, trace hypotheses, visual-review index and limitations. The existing local
session/allocation 4058374 was used throughout; no new session or job was submitted.
The outer run was logged before experiments. Pre-existing work was preserved in its
`snapshot/`; every request has a unique name, timeout and explicit rule.

## Setup and protocol

Host consu, AMD EPYC 7513, NVIDIA A40, driver 580.178.04, Python 3.13.15,
torch 2.9.0+cu128/CUDA 12.8, one CPU thread, TF32 off. Base commit
`5c1ef9cb87f5d7c15174ef0af8a2dd002182df45`, with the preserved dirty tree and per-request
source fingerprints. Graph-turns and graph-turns-rotations paths and SHA256 were
verified against artifacts.toml before evaluation. Every evaluation explicitly passed
the launcher graph. No priors, dependencies, third-party clones or saved graphs changed.

Frozen references: incumbent `encounter_turns.json` and original
`plaza_forward_posture.json`, development seeds 0/1/2 at 20 and 60 s; original head-on
and crossing probes at eight seeds/10 s. A slow close-start blocked head-on probe was
added and promoted as `configs/behavior/encounter_blocked_head_on.json`. Requests 00–06
contain these evaluations. Every candidate was diagnosed and compared on matching
seeds; `plaza_*` comparisons cover the original design. Request 114 indexes all
16-human development variants. No output was overwritten.

Shared standoff diagnostics were added before runtime changes. Pair standoffs require
both mean edge speeds <0.15 m/s, root gap <=1.1 m, duration >=1.2 s, at least one pending
goal, and neither human mid-locomotor maneuver/turn. Yields count as idle. A second
measure includes stalled maneuvers. Individual pending-goal stops are separate.
Threat resolution is passing or an increasing gap 0.25 m above its running minimum;
end-censored events are excluded from successful-delay quantiles. Definitions are
stored with outputs. All are diagnostic cutoffs, not pass thresholds.

## Selected result and costs

Request 113 consolidates saved traces with full-frame joint-sphere checks. At 60 s,
16 humans, development seeds 0/1/2 (event quantiles pooled; quality p95 values averaged
across seeds):

| Measure | Incumbent | Selected |
|---|---:|---:|
| Pair stalls including maneuvers, count / human-minute | 43 / 0.896 | 19 / 0.396 |
| Pair-stall p50 / p95 / max, s | 2.0 / 4.36 / 5.2 | 1.2 / 2.4 / 2.4 |
| Idle standoffs | 5 | 1 |
| Threat-resolution p50 / p95, s | 2.8 / 7.2 | 2.8 / 6.0 |
| Censored / total threat encounters | 39 / 616 | 27 / 529 |
| Either deviates / neither, % | 94.64 / 5.36 | 96.60 / 3.40 |
| Individual stop p50 / p95 / max, s | 0.8 / 4.4 / 9.6 | 0.8 / 2.6 / 48.0 |
| Stopped seconds per human-minute, pending goal | 12.058 | 12.083 |
| Mean speed, m/s | 0.616 | 0.643 |
| Goal arrivals per human-minute | 3.354 | 3.250 |
| Minimum joint-sphere clearance, m | -0.0272 | +0.0168 |
| Joint-sphere contact frames, mean % | 1.361 | 0 |
| Minimum root gap, m | 0.666 | 0.634 |
| Planted-foot speed p95, m/s | 0.381 | 0.412 |
| Acceleration p95, m/s² | 3.337 | 3.584 |
| Seam joint-step p95, m | 0.0694 | 0.0703 |
| Seam root-acceleration p95, m/s² | 6.932 | 7.414 |
| Moving forward / side / backward, % | 95.77 / 0.98 / 0.26 | 95.83 / 0.59 / 0.19 |
| Redirect turn-forward / side-back / unresolved | 145 / 15 / 198 | 152 / 5 / 179 |

At 20 s, stalls fall 16→5 and maximum 5.2→2.4 s; contact 3.833→0%, speed
0.622→0.633 m/s, goals 2.750→2.625/human-minute. The original plaza has fewer stalls
(4 at 60 s) but more side/back travel, lower speed and worse minimum joint clearance
(-0.136 m). This is a whole-design comparison, not a single mechanism ablation.

The selected stall-count reduction holds across root-gap cutoffs 1.0–1.5 m, but at
1.5 m its maximum remains 6.4 s. Goal-directed progress falls 0.378→0.368 m/s. Root
cylinders still overlap by up to 0.166 m with the larger selected envelope: the
0.01 m stop target is not guaranteed by graph execution. Positive joint-sphere
clearance does not prove full-bone or rigid-skin clearance.

## Causes, mechanisms and rejected candidates

Requests 11/43 retain stepwise CSV traces of real reference standoffs. Seed0/human7
commands sidesteps near 0.49–0.65 m/s but executes roughly 0–0.01; the stop flag can
clear while execution remains stationary. Seed1/human13's veer loses yaw under
approach limiting and has no retained body target. Seed2/humans8/15 both yield around
24–25.2 s. Return-to-goal cycles re-form conflicts; intended-velocity threats can vanish
while people wait/stop. Low assertiveness alone does not prevent these failures.
Pivots often execute in 0.4–0.8 s, so missing pivot coverage is not the sole cause.

Retained veer headings, mutual execution-stall recovery and all-neighbor geometric
headings were selected together with bounded goal-path-aware bypass completion.
Nearest-threat-only geometry caused a 28.4 s three-person jam. Opposite local passing
sides and identical deadlines caused co-motion/reversal in the slow probe; the
selected own-right convention and independently shortened deadlines for unproductive
parallel motion resolve its eight seeds in 2.4–3.6 s. Both may act simultaneously.

Selected settings: 0.8 s recovery, geometric margin 0.02–0.08 m, maximum bypass 3 s
with conditional jitter 0.75, rare sidesteps/yields, body radius 0.4 m, passing margin
0.2 m, bubble 2.5 m, existing graph weights end velocity 2 / forward 3 / clearance 60.
Stop clearance remains 0.01 m. The geometric search minimizes predicted heading
change toward the goal, not total executed path length. Every new mechanism defaults
off. Higher hazard, immediate blocked-veer pivots, exact velocity projection, smaller
ordinary radii, shorter arcs and extra braking forecasts had mixed/regressing results;
they were not selected. The selected hazard remains 2 Hz, exponent 1, patience 3 s.

The 5% stubborn reset trait uses an independent counter in the existing per-human
encounter stream. Active stubborn humans never initiate interventions; finite 8–14 s
stall patience retires the trait and forces an attempt. The development draw contains
2/48 stubborn humans, held-out 1/48. Active stubborn switches are 0/127 eligible
steps at development 60 s. Non-stubborn switch percentage given threat rises
22.48→28.09%; 27 forced recoveries are counted separately. Strategy counts change
175/118/106/130→24/182/201/56 for sidestep/veer/pivot/yield. Request 113 records the
observed/expected hazard by urgency and effective assertiveness; no human calibration
is claimed. All eight 30 s all-stubborn stress pairs resolve (request83/88), without
joint-sphere contacts; maximum pair stall is 14.8 s.

## Backward audit, held-out and visual review

Request 14 audits the saved tables and inspects twelve edge examples. There are 402
admissible short backward/back-diagonal edges with entry <=0.15 m/s, 3,218 <=0.5 m/s;
20,706 of 26,970 standing boundaries have a candidate. Yet 32 fixed-facing backward
commands from standing produce median 0.004 m backward progress over 2.4 s (p95 0.188).
No backstep was added. Audit reachability/selection before proposing an offline
standing-to-backward/back-diagonal start-stop supplement.

T1 was frozen at 16:52:46 UTC, then seeds101/307/911 used once at 20 and 60 s
(requests80/81). Source/config hashes remain unchanged (request113). Held-out 60 s:
25 close-pair stalls, p95/max2.8 s, speed0.607 m/s, goals2.875/human-minute,
stopped13.350 s/human-minute, zero joint-sphere contact, minimum+0.0228 m. At20 s:
7 stalls, max2.0 s, speed0.651 m/s, zero contacts. No paired held-out reference exists.

Consolidation exposed the remaining individual stop tail. Request109 traces
seed1/human2 at12–60 s and seed101/human3 at9.2–56.4 s. Both are non-stubborn in
ordinary turns beside the west wall/scanner, with commands but negligible execution;
often no nearby human threat exists and the pair-recovery clock stays zero. Inspected
follow sheets110/111 confirm the stationary poses. This remains unfinished; no
post-held-out tuning was performed.

Actual image inspection: reference follow12/13 (seed0/h7 and seed2/h15), matching
selected90/91; overview29/89 (seed0); selected small-turn93 (seed1/h2), stubborn-pass92
(seed1/h3), slow blocked94 (seed7/h0), intermediate L3 follow55, backward sheet14,
residual failures110/111 and rigid demo sheet101. Stills show separation/orientation
and remaining hunched poses, but do not establish animation smoothness; walls sometimes
occlude feet. Rigid mesh GPU demo85 runs16 humans for12 s with paired rotations and
mean139 human LiDAR hits / 5760 rays. Browser interaction and scale timing unverified.

## Validation and next work

254 CPU tests pass (request98), including shared metrics, RNG independence,
simultaneous stubborn release and queued controls. Final all-off encounter/plaza
60 s traces match all165 frozen arrays (request99). All283 fast-golden captured arrays
and the mismatch report match the untouched same-host snapshot (request100).
Seven cases match expected files; five have inherited tiny diagnostic rounding
mismatches. Expected files are unchanged; no clean-golden claim is made. CPU tests and
goldens unset both graph environment variables. Real-prior goldens were out of scope.
Two invocation bugs were repaired under fresh names: pytest output parent48→51 and
CUDA device index82→85. No unfixable tool/rendering failure occurred.

Next: a one-human wall/obstacle failure probe with a free-space control, then trace
ordinary-turn target retention, confinement, wall-mirror clearance and reachable graph
costs. Test one executable-progress recovery mechanism and compare full stop tails,
contacts, detour, feet/seams and redirects. The previously unseen seed bank has now
been used; a later promotion needs a separately reserved bank. Training readiness,
calibrated realism, browser interaction and reliable backward execution remain open.

Prior evidence: [encounter](2026-10-06-encounter-design.md),
[turning library](2026-10-06-turning-library.md),
[turning-graph iteration](2026-10-07-encounter-turns-iteration.md).
