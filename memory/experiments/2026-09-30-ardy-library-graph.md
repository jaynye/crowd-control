# 2026-09-30: offline ARDY library and the ARDY-tabulated motion graph

> Code paths and commands below are as run. Since the 2026-10-01 reorganization the
> code lives in `src/crowdmotion/` and runs through the `crowd` command line
> ([architecture](../../docs/architecture.md), [CLI](../../docs/cli.md)).


## Outcome

Numbers are from the held-out check (job 3993140: seed 1, 128 humans, not used for
tuning) unless marked.

- **Runtime.** Over an 18 h ARDY library (2 min of A40 time; 1.9 GB on disk), the graph
  at the recommended setting steps 65,536 humans in 47 ms per 0.4 s edge at k = 128
  (20 ms at k = 32). Peak memory is 3.1 GB.
  - At m = 16 and `S = 0.4` that is about 5% of one A40, isolated stepping only.
  - Online ARDY at the same responsiveness (replanned every 0.4 s, so u = 0.2) would
    need about 46 A40s. The floor is 9.3 A40s, for refill only (1.3–1.6 s responses).
- **Recommended setting** (`configs/library/graph_default.json`, now
  `configs/graph/default.json`; chosen in iteration 2):
  - k = 128;
  - ±5° yaw / ±10% speed root correction;
  - end-velocity cost weight 0.25 s;
  - hunched-edge filter at 85% of neutral head height.
- **Held-out, against online ARDY replanned every 0.4 s:**
  - path error 0.7–1.4 cm (online 0.6–0.75); p95 1.9–4.6 cm (1.4–1.7); rare
    excursions up to 18–29 cm (3–4);
  - speed error 0.04–0.09 m/s (0.04–0.05);
  - median responses within 0.0–0.4 s of online: stop 0.95 / start 0.95 / turn-90 0.9
    / turn-180 1.1 / speed step 1.2 s, against 0.9 / 0.95 / 0.75 / 0.7 / 1.0. The
    turn-180 p95 is 2.1 s (online 0.95). 2 of 128 humans (1.6%) never complete the
    turn-180;
  - hunched frames 0.2–0.6% (60 s: 0.4%), against 3–17% (60 s: 24%) online without
    text. The filter removes stooped gait by construction, since the filter and the
    metric both use head height;
  - clean pose seams: joint-step ratio 0.99–1.0.
- **Costs of stitching (held-out).**
  - Small planted-foot sliding is more frequent: the fraction above 0.1 m/s is
    0.39–0.52, against 0.22–0.35 online. Mean speeds are similar.
  - Root-speed seams are 1.4–2.0× the interior root-step change (p95), against 0.9–1.4×
    online.
- **Root-correction ablation (held-out; reported, not used to re-pick).** Without
  correction:
  - foot sliding drops to 0.30–0.40;
  - path error rises to 1.3–2.1 cm and speed error to 0.06–0.11 m/s;
  - the speed-step response slows to 1.6 s, with 2.3% never responding.
- **The iteration-1 response gap came from selection, not coverage.** Humans who never
  stopped had a candidate within 0.06 m/s of the requested end velocity, but chose one
  0.45 m/s off. Position-only path cost over one edge barely separates "start
  decelerating" from "keep walking". An end-velocity term fixes this without new motion.
  At k = 32, coverage does limit turn-180 (31% never; best candidate 0.83 m/s off), so
  k stays at 128.
- **Scope.** The evaluation covered walking at 0.5–1.6 m/s with stops, starts, 90°/180°
  turns and speed steps. Never tested:
  - starting from standing toward a new direction (wait, yield, reroute);
  - turning in place, sidestepping and walking backward as deliberate behaviors.
    Correction 2026-10-01: no deliberate labels were generated, but the library does
    contain body-relative lateral/backward travel under root-only constraints; see
    the [behavior audit](2026-10-01-graph-behavior-audit.md).

## Why

The user set 4,096 environments with 4–16 humans each and moved ARDY and Kimodo offline
(ledger D09–D10). This experiment builds the shared piece (an offline ARDY library under
random commands) and the first runtime over it (option 3 in
[historical runtime proposal](../archive/2026-10-01-decision-offline-runtime.md)). It scores the graph against online
ARDY in the same harness.

## Plan

1. **Library** (`crowdmotion/library/generate.py`, `configs/library/ardy_a40.json`):
   - ARDY H40 at the recommended setting: TF32, no guidance, 5 steps, 4-frame history.
   - 2,048 streams × 32 s = 18.2 h of motion.
   - Every 8-frame (0.4 s) edge is one ARDY call from the executed history, toward a
     path planned from a random semi-Markov command (stand; walk at 0.3–2.0 m/s; heading
     changes up to ±180°; some curving).
2. **Graph** (`crowdmotion/library/graph.py`):
   - Nodes are boundaries (the frame before each edge).
   - Candidates are the successors of the k nearest boundaries in a joint position +
     velocity descriptor; the own successor is always a candidate at distance 0.
   - Each human samples by root-path error plus descriptor distance with per-human
     Gumbel noise. Optional bounded yaw/speed root correction; smoothstep decay of the
     pose offset over the edge.
   - The prototype uses k-nearest neighbors in place of the proposal's k-means
     clusters: no cluster-boundary effects, same runtime cost.
3. **Quality** (`configs/bench/graph_a40.json`): same seeds, commands and planner as
   online ARDY replanned every 0.4 s.
   - k ∈ {8, 32, 128}; with and without inertialization (pre-inertialization seams);
     with and without root correction; greedy vs sampled; hunched-edge filter; 60 s
     cruise.
   - Graph diagnostics: chosen and best path error per edge, natural-successor
     fraction, pose offset and velocity jump at jumps, edge reuse, correction size.
4. **Runtime cost** (`crowdmotion/bench/graph_speed.py`): graph build time, and per-step
   wall time at 4,096, 16,384 and 65,536 humans.

## Jobs

- Pipeline smoke: job 3992993 failed at metrics (the runtime lacked `skeleton_info`; run
  dirs renamed `runs/20261001T004506Z_smoke_*`). Rerun as job 3992999
  (`runs/20261001T004557Z_smoke_*`).
- Iteration 1: job 3993000 on an A40 (`chitti`): `runs/20261001T004918Z_ardy_library`,
  `runs/20261001T004918Z_graph_quality`, `runs/20261001T004918Z_graph_speed`.
- Iteration 2 (tuning sweep): job 3993020, `runs/20261001T005830Z_graph_quality_v2`.
- Held-out check and runtime at the recommended setting: job 3993140
  (`scripts/graph_eval.sbatch`, `configs/bench/graph_heldout_a40.json`),
  `runs/20261001T010546Z_graph_quality_graph_heldout_a40`,
  `runs/20261001T010546Z_graph_speed`.

## Iteration 1 results (job 3993000, A40)

**Library.** 2,048 streams × 32 s = 18.2 h, 161,792 edges, in 121 s of generation:
542 kept motion-s per wall-s, since each 2 s call keeps 0.4 s. Peak 14.2 GB; no
non-finite streams. 26,972 edges (17%) have mean head height below 85% of neutral
standing height (hunched).

**Runtime cost** (`graph_speed`; build 2.4–2.7 s including kNN):

| k | Humans | Wall per step (one 0.4 s edge for everyone) | Motion-s per wall-s | Peak GB |
|---|---|---|---|---|
| 32 | 4,096 | 2.5 ms | 0.65M | 0.74 |
| 32 | 16,384 | 4.8 ms | 1.36M | 0.90 |
| 32 | 65,536 | 19 ms | 1.38M | 1.56 |
| 128 | 65,536 | 45 ms | 0.59M | 2.62 |

At m = 16 and `S = 0.4`, 65,536 humans need one step per wall-second: 2–4.5% of one
A40, isolated. Online ARDY would need about 46 A40s replanned every 0.4 s, or 9.3 A40s
with refill only.

**Quality** (32 humans, 12 s, same seeds and jittered commands; online ARDY replanned
every 0.4 s as reference). Response medians are computed over humans that ever
respond; "never" is the fraction that do not within the window.

| Case | Path err (cm) | Speed err (m/s) | h-gated skate mean (m/s) / frac > 0.1 | Seam joint-step ratio | Seam / interior root-step change p95 (mm) | Stop / start / turn90 / turn180 / speed response (s) | Never: stop / start / turn180 |
|---|---|---|---|---|---|---|---|
| Online ARDY, P = 0.4 | 0.6–0.75 | 0.04–0.05 | 0.14–0.37 / 0.20–0.33 | 0.97–1.0 | 4.7–8.6 / 5.1–7.4 | 0.95 / 0.9 / 0.75 / 0.7 / 1.05 | 0 / 0 / 0 |
| Graph k = 8 | 5.6–7.7 | 0.38–0.54 | 0.08–0.09 / 0.12–0.14 | 1.03 | 6.9–9.3 / 4.3–4.9 | 0 / 1.3 / 0.75 / 3.15 / 4.35 | 0.31 / 0.53 / 0.75 |
| Graph k = 32 | 3.7–5.8 | 0.22–0.39 | 0.12–0.17 / 0.21–0.30 | 1.0 | 8.9–12 / 5.2–5.6 | 0.95 / 1.35 / 1.85 / 1.95 / 3.55 | 0.44 / 0.16 / 0.56 |
| Graph k = 128 | 2.5–4.0 | 0.14–0.24 | 0.16–0.25 / 0.27–0.36 | 1.0 | 9.6–15 / 5.5–6.7 | 0.95 / 1.05 / 1.65 / 2.5 / 3.7 | 0.25 / 0.13 / 0.13 |
| Graph k = 128 + correction (±5° yaw, ±10% speed) | 1.5–3.0 | 0.10–0.19 | 0.16–0.27 / 0.41–0.57 | 0.99–1.02 | 9.2–14 / 5.6–6.6 | 1.15 / 1.1 / 1.2 / 1.75 / 1.75 | 0.22 / 0 / 0 |
| Graph k = 32, no inertialization | as k = 32 | as k = 32 | 0.13–0.17 / 0.16–0.22 | 1.09–1.22 | as k = 32 | as k = 32 | as k = 32 |

- **Seams.** Pose seams are clean once the offset is decayed over the edge: the joint-step
  ratio is 1.0, against 1.09–1.22 without decay. The decay and the root correction both
  add small foot sliding: the fraction of planted-foot frames above 0.1 m/s rises from
  0.16–0.22 (no decay) to 0.21–0.30 (decay), and from 0.27–0.36 to 0.41–0.57 with
  correction at k = 128. The pre-decay pose offset at
  non-natural transitions is 2.6–4.5 cm RMS per joint. Root-velocity seams are 1.5–2.5×
  the interior root-step change: the decay covers pose, not root velocity.
- **Following and stochasticity.** Humans follow their own stream 56–79% of the time.
  The best candidate tracks the request to 1.1–2.1 cm at k = 128, but the sampled one to
  3–5 cm. Temperature 0.01 m and pose weight 1 trade tracking for diversity. Greedy
  selection (k = 32) is not better than sampled.
- **The main gap is coverage of sharp transitions.** 22% of humans never stop within
  3 s even at k = 128 with correction. Turn-180 and speed-step responses are 0.7–1.0 s
  slower than online. In the library, commands change only every 1.5–6 s, so few
  edges decelerate to a stop or turn sharply from a given gait state.
- **The hunch filter** drops 17% of edges. In 60 s cruise with k = 32 and correction it
  cut path error from 3.15 to 2.54 cm; its effect on posture is not yet measured on the
  graph output.
- **60 s cruise is stable:** path error 3.2 cm (graph, k = 32 + correction) against
  0.8 cm online.

## Iteration 2: selection vs coverage (job 3993020, A40; tuning set: seed 0, 32 humans)

Same library (`runs/20261001T004918Z_ardy_library`), config `configs/bench/graph_a40_v2.json`,
run `runs/20261001T005830Z_graph_quality_v2`. Base: k = 128 with correction. New:
- an end-velocity cost (`end_vel_weight`);
- an optional one-edge lookahead;
- posture metrics (hunched frames, median head height);
- diagnostics in the first 1.2 s after each change, split by responders and
  never-responders.

| Case | Path err (cm) | Speed err (m/s) | Stop / start / turn-90 / turn-180 / speed response (s) | Never (worst scenario) | Hunched frames | h-gated skate frac > 0.1 |
|---|---|---|---|---|---|---|
| Online ARDY, P = 0.4 | 0.6–0.75 | 0.04–0.05 | 0.95 / 0.9 / 0.75 / 0.7 / 1.05 | 0 | 6.7–17% (60 s: 29%) | 0.20–0.33 |
| Base (iteration-1 best) | 1.5–3.0 | 0.10–0.19 | 1.15 / 1.1 / 1.2 / 1.75 / 1.75 | stop 22% | 11–16% | 0.41–0.57 |
| + end-vel 0.1 | 0.9–1.6 | 0.05–0.10 | 1.0 / 0.9 / 1.05 / 1.3 / 1.4 | 0 | 7–18% | 0.41–0.56 |
| + end-vel 0.25 | 0.7–1.3 | 0.04–0.07 | 1.05 / 0.95 / 1.0 / 1.0 / 1.2 | 0 | 5–22% (60 s: 24%) | 0.39–0.54 |
| + end-vel 0.5 | 0.7–1.3 | 0.04–0.08 | 0.95 / 0.9 / 0.9 / 0.95 / 1.1 | turn-180 3% | 7–20% | 0.40–0.53 |
| + pose weight 0.5 | 1.1–1.7 | 0.06–0.11 | 1.15 / 1.0 / 1.15 / 1.35 / 1.4 | stop 6% | 13–18% | 0.51–0.57 |
| + end-vel 0.25, pose weight 0.5 | 0.7–1.3 | 0.04–0.08 | 0.95 / 0.9 / 0.9 / 1.05 / 1.15 | 0 | 6–14% | 0.47–0.55 |
| + lookahead (no end-vel) | 1.3–3.7 | 0.09–0.27 | 0.95 / 0.95 / 0.9 / 1.7 / 1.2 | 28% | 10–15% | 0.28–0.43 |
| + lookahead + end-vel 0.25 | 0.7–1.5 | 0.04–0.09 | 1.0 / 0.85 / 0.95 / 1.05 / 1.15 | 0 | 11–16% | 0.36–0.51 |
| k = 32 + end-vel 0.25 | 1.1–3.1 | 0.06–0.22 | 1.05 / 0.95 / 1.2 / 1.3 / 1.3 | turn-180 31% | 7–20% | 0.42–0.55 |
| **+ end-vel 0.25 + hunch filter** | 0.7–1.3 | 0.04–0.08 | 1.0 / 0.95 / 1.0 / 1.05 / 1.2 | turn-180 3% | **0.2–0.7% (60 s: 0.3%)** | 0.38–0.51 |

- **Selection, not coverage, at k = 128.** In the base case, the 7 humans who never
  stopped had a best candidate 0.063 m/s from the requested end velocity, but chose
  0.45 m/s. With the end-velocity term, chosen and best agree within 0.01–0.10 m/s
  after a change.
- **At k = 32, turn-180 is coverage-limited.** The never-responders' best candidate is
  0.83 m/s off.
- **Lookahead** on its own hurts: it favours long steady continuations. With the
  end-velocity term it matches end-velocity alone, so it is not needed.
- **Hunch filter.** It removes 17% of edges and cuts hunched output frames by 20–70×,
  with no loss in response or tracking. Online ARDY without text stoops in 7–17% of
  frames over 12 s and 29% over 60 s (head below 85% of neutral; median head height
  90–93% of neutral).
- **Seams** for the recommended setting: pose joint-step ratio 0.98–1.01. Seam root-step
  change p95 is 8–13 mm, against 5–7 mm in the interior. Root acceleration p95 is the
  same at seams and in the interior (1.5–2.5 m/s²).
- **Open items:**
  - The runtime outputs no local rotations, so mesh skinning and mesh LiDAR are not
    supported yet.
  - Root-velocity seams are not inertialized.
  - Correction and pose-offset decay increase small foot sliding.
  - Branching from exact library states was planned in case coverage limited k = 128.
    It did not, so it is not built.
- **Videos** (6 humans each, stick figures, red title on new edges): iteration 1
  `runs/20261001T004918Z_graph_quality/clips_*.mp4`; recommended setting
  `runs/20261001T005830Z_graph_quality_v2/clips_g128c_ev0.25_hunch_{turn180,stop_start}_edge.mp4`.
  A 60 s cruise render (`..._long_cruise_edge.mp4`) was still running at the time of writing.
