# 2026-10-06: heading-conditioned turning library and merged graph

**Question.** The saved graph (`graph-default`, built over `library-ardy-h40`) had almost no
in-place or slow turns under the round-2 flat-ground mask
([transition/posture iteration](2026-10-05-transition-posture-iteration.md)). The library
was generated with root-XZ conditioning only, so a stationary path never asked ARDY to
rotate the body. Does ARDY follow body-heading targets well enough to generate in-place
turns and arcs? And does a supplementary turning library fill the graph's turning coverage?

## Outcome

- **ARDY follows body-heading targets.** It receives them through its
  `global_root_heading` feature, set to `[cos h, sin h]`. In in-place turns the hip line,
  measured from the posed joints, turns 0.98 of the commanded angle at the median
  (p10 0.89); without targets it turns 0.00.
  - The sign and zero follow our convention: 0 faces +Z, positive turns toward +X.
  - Arcs come out at 1.04 of the commanded radius at the median (p10 0.97, p90 1.22),
    with 0.99 of the travel forward.
- **Coverage, counting twists.** A 9.1 h turning supplement (`library-ardy-turns`) was
  merged with the base library into the opt-in graph `graph-turns`, whose rotations are in
  `graph-turns-rotations`. Under the round-2 flat-ground mask, with twisted edges removed
  from the counts:
  - In-place edges without a planted twist: 4,763 (gross 5,873). graph-default had 8
    in-place edges in total.
  - In-place windows turning at least 45° in 1.2 s without a planted twist: 4,259
    (gross 6,266). graph-default had none.
  - Forward-turn edges with radius below 1 m rise from 328 + 85 to 2,348 + 416
    (upright + hunched).
- **Twists in a graph consumer's view.** In a twist the pelvis turns on planted feet.
  - Per library turn, 14.9% of in-place turns twist (17.2% from rest, 7.7% from
    motion). Measure: feet below 0.4 of the commanded turn 1.5 s after the segment.
  - Per graph edge, the share is larger. A planted twist means foot yaw above
    graph-default's p99 of 21.4°, with every contact label on. 19% of admissible
    in-place edges (1,110 of 5,873) and 32% of in-place windows (2,007 of 6,266)
    contain one.
- **Held twists.** The supplement adds 2,354 straight or standing edges in which the body
  stays twisted on planted feet, against 67 in the base (about 35×). With `CROWD_GRAPH` =
  `graph-turns`, a human can therefore stand twisted even when no turn is requested.
  Nothing masks these edges yet: an opt-in twist mask is the proposed follow-up and is
  not implemented. Weigh this before adopting the graph.
- **Nothing existing changed.** Defaults, `graph-default`, `graph-rotations`,
  `core/paths.py` and the goldens are untouched. Every fast golden report is identical to
  the pre-edit snapshot's, and the slow real-ARDY and real-Kimodo CPU outputs are
  bit-identical to it.

## Code

- **Heading targets.** `MotionRequest.root_heading` (`[B, F]`, NaN = free frame, default
  None) passes through `OnlineMotionSource.request/step`.
  - ARDY `conditions(..., root_heading)` handles it in both modes. The library mode
    uses disjoint `Root2DConstraintSet`s around NaN frames.
  - The vectorized mode also accepts NaN path frames.
  - Kimodo raises `NotImplementedError`.
- **Turning commands.** `TurnRanges` and `TurnCommands` live in `behavior/commands.py`.
  - Segments form a semi-Markov chain over stand, walk, in-place turn and arc, drawn with
    `stream_seed(s, 1)`.
  - `plan()` traces each segment with `plan_arc` and stores the targets.
    `motion_options()` passes them on.
  - Targets start at the executed heading, blend to the segment's schedule over 0.5 s, and
    follow the path tangent while walking.
  - Switches: `heading_targets`, `walk_heading`, `stand_heading`, `heading_every`,
    `path_every`, `in_place_path`.
- **Library generation.** `library/generate.py` takes `command_source: "random" | "turns"`.
  Turning libraries add `heading_target` and `cmd_rate/angle/radius/segment`. The random
  path is unchanged.
- **`crowd library turns`** (`library/turns.py`) reports:
  - turn tracking on the hip line;
  - twist rates;
  - feet and pelvis;
  - ground admissibility, using the graph's own annotations and mask.
- **`crowd graph turns`** (`graph/turns.py`) is the main session's audit script as a
  command. It adds a per-source split, per-hour counts, and `twisted` and
  `twisted_planted` counts.
- **Multi-library graphs.** `graph.build.load_libraries` lets `crowd graph build` and
  `crowd graph rotations` take several libraries. They are concatenated in order and
  truncated to the shortest.
- **Smoke script.** `scripts/smoke/ardy_heading.py`, run by
  `scripts/slurm/ardy_heading.sbatch`.
- **Configs.** `configs/library/ardy_turns_smoke.json` and `ardy_turns_a40.json`.
- **Tests.** `tests/unit/test_turn_commands.py` (12) and `test_turn_library.py` (4).

## Runs

| Run | Job | Hardware | What |
|---|---|---|---|
| `runs/20261007T010942Z_golden_baseline` | 4024314 | CPU (cell) | Fast goldens of a pre-edit snapshot |
| `runs/20261007T012104Z_ardy_heading` | 4024358 | A40 (brainiac), driver 595.84 | Convention and parity check |
| `runs/20261007T013731Z_turn_smoke` | 4024394 | A40 (baymax), driver 595.91.07 | Smoke round 1: off / dense / hsparse / sparse |
| `runs/20261007T014003Z_golden_check` | 4024404 | CPU (cell) | Goldens after the edits |
| `runs/20261007T020945Z_turn_smoke2` | 4024606 | A40 (brainiac) | Smoke round 2: twist mitigations |
| `runs/20261007T022940Z_golden_slow_check` | 4024683 | CPU (cell) | Slow goldens (real ARDY/Kimodo, CPU) on both trees |
| `runs/20261007T021516Z_ardy_turns_library`, `_turn_report`, `_motion_graph_turns`, `_graph_turns_rotations`, `_turn_audit` | 4024645 | A40 (omgwth), driver 595.84 | Full library, report, merged graph, rotations, audit |

- **Generation setting.** All generation used ARDY-Core-RP-20FPS-Horizon40 (revision as in
  `artifacts.toml`): TF32, no guidance, 5 steps, 4 history frames, 8 frames kept per call,
  torch 2.9.0 cu128. The run directories record the environment, commit and
  `source.diff`.
- **Graph overrides.** All audits use the round-2 behavior config's overrides, copied to
  `runs/20261007T013731Z_turn_smoke/overrides.json`.
- **Reference audit.** The main session's default-graph audit is job 4024288
  (`runs/20261007T010435Z_turn_audit`). `crowd graph turns` on graph-default reproduces
  every one of its fields exactly.

## Heading check: convention and parity

- **Parity.** The library and vectorized condition paths give identical masks and
  observations (max difference 0), with and without NaN frames.
- **Convention.** Held from standing for 6 s, a +90° target turns the returned heading to
  +89.7° and the hip line to +83°.
- **Step targets twist.**
  - The feature snaps to the target on frame 0.
  - The hip line follows over about 6 s.
  - The toes turn only about 20°, and contacts stay at 100%.
  - At 180°, the feature reads −175° while the hip line reaches only −83°.

  The returned heading can therefore decouple from the body, so every turn metric uses
  the hip line and toes, never the feature alone.

## Smoke (256 streams × 32 s per variant, 2.28 h each)

**Round 1** compared four variants:

- **off**: no targets.
- **dense**: heading and path on every frame.
- **hsparse**: heading on every 10th frame.
- **sparse**: heading and path on every 10th frame.

| | off | dense | hsparse | sparse |
|---|---|---|---|---|
| In-place hip turn / command, p50 | 0.00 | 0.96 | 0.97 | 0.98 |
| In-place returned vs hip heading, p50/p95 (°) | 0.4/1.5 | 5.0/11.8 | 1.8/5.7 | 1.6/5.0 |
| Drift from rest, p90 (cm) | 1.5 | 2.1 | 2.4 | 5.2 |
| Arc radius ratio, p50 (p90) | 1.05 (1.96) | 0.96 (1.03) | 1.03 (1.21) | 1.05 (1.23) |
| In-place label-free planted-foot p95 (m/s) | 0.06 | 0.17 | 0.18 | 0.20 |
| In-place windows ≥ 45°, upright | 0 | 1,341 | 1,353 | 1,385 |
| Twist rate from rest | — | 0.219 | 0.207 | 0.191 |

- For reference, base-library stand edges give a planted-foot p95 of 0.10 m/s label-free
  and 0.08 m/s contact-labeled.
- In-place edges are 93–97% admissible.
- Twists concentrate in turns entered from rest, in turns under 90° (25–33%), and in the
  first 4 s of a stream.

**Round 2** tested twist mitigations on top of sparse, under a rule set before running:
a variant had to halve the from-rest twist rate, keep at least 95% of sparse's in-place
admissible edges per hour, and keep its in-place planted-foot p95 within 10% of sparse's.

| Variant | Change | From-rest twist rate | Note |
|---|---|---|---|
| `end` | In-place turns anchor only the last path frame | 0.626 | ARDY largely ignores the heading: hip ratio p50 0.46 |
| `nostart` | No in-place turn as the first segment | 0.167 | |
| `end_nostart` | Both | 0.591 | |

No variant qualified, so sparse ships as `configs/library/ardy_turns_a40.json`.

## Full library and merged graph

**Library** (`library-ardy-turns`): 1,024 streams × 32 s = 9.1 h.

- Generation took 63 s, plus 60 s of model load, with a 12.9 GB peak.
- Edge share: walk 31%, stand 12%, in place 14%, arc 43%.

| Measure | In place (2,188) | Arc (2,661) |
|---|---|---|
| Hip turn / command, p50 | 0.98 (p10 0.89) | 0.98 |
| Toe turn / command 1.5 s after, p50 | 0.89 (p10 0.23) | |
| Lift a foot | 90% | |
| Drift from rest, p50 / p90 | 2.5 / 5.6 cm | |
| Returned vs hip heading, p50 / p95 | 1.7° / 5.3° | |
| Radius ratio, p50 | | 1.04 |

- **Planted-foot p95 in place:** 0.22 m/s label-free, 0.10 contact-labeled. Stand edges
  give 0.09 and 0.07.
- **Admissible** under the mask: in place 96%, arc 94%, walk 92%.
- **Hunched among admissible edges:** 13% in place, 18% walk. Base-library walk edges
  give 39%.

**Merged graph** (`graph-turns`, `configs/graph/default.json`): 3,072 streams and
213,608 valid edges, built in 5.5 s. The hunch filter drops 2.6% of the turning library's
edges and 16.7% of the base's.

**Rotations** (`graph-turns-rotations`): bone direction error is 0.04° at the median and
0.51° at p99. `GraphRotations.load` accepts them for the merged graph.

## Coverage audit: graph-default vs graph-turns

Counts use the round-2 overrides. Each count is admissible edges (0.4 s) or 1.2 s
windows, given as upright + hunched. "Twisted" means foot yaw above 21.4°, the p99 of
graph-default. "Planted" also requires every contact label on.

| Class | graph-default (18.2 h) | graph-turns (27.3 h) | twisted / planted |
|---|---|---|---|
| In-place edges (speed < 0.2, ≥ 25°/s) | 5 + 3 | 5,178 + 695 | 2,429 / 1,110 |
| Forward-turn edges, r < 0.5 m | 13 + 2 | 442 + 79 | 30 / 0 |
| Forward-turn edges, r 0.5–1 m | 315 + 83 | 1,906 + 337 | 128 / 0 |
| Forward-turn edges, r 1–2 m | 1,588 + 542 | 6,233 + 1,164 | 298 / 0 |
| Forward-turn edges, r 2–4 m | 1,748 + 527 | 5,663 + 1,059 | 393 / 0 |
| Non-forward moving turns | 5,566 + 2,347 | 5,990 + 2,476 | 103 / 2 |
| Windows ≥ 45°: in place | 0 + 0 | 5,459 + 807 | 3,701 / 2,007 |
| Windows ≥ 45°: slow (0.2–0.6 m/s) | 17 + 8 | 774 + 193 | 119 / 49 |
| Windows ≥ 45°: walking forward, r < 1 m | 18 + 3 | 964 + 112 | 122 / 3 |
| Windows ≥ 45°: walking forward, r 1–2 m | 122 + 33 | 2,483 + 348 | 455 / 1 |

- **Base streams are unchanged.** The base-source half of the merged graph has exactly
  graph-default's counts.
- **Twists persist.** In the turning source, 2,354 straight or gentle edges are twisted
  and planted, against 67 in the base: twisted bodies stay twisted while standing.
- **`twisted` over-counts.** Stepping in-place edges also lag the feet behind the body
  (smoke p50 14–23°). The planted count isolates turns made on planted feet.

## Validation

- **Goldens.** `runs/20261007T014003Z_golden_check`: every fast case's report is identical
  to the pre-edit snapshot's.
  - Match: library_fake, quality_online, throughput_fake, audit, mjlab_plaza, corridor,
    replay.
  - The same pre-existing mismatches: evaluate (2), runtime_step (8), graph_build (1),
    quality_graph (8), compare (6).
- **Slow goldens** (real ARDY and Kimodo on CPU, root path only), in
  `runs/20261007T022940Z_golden_slow_check` (job 4024683). The edited tree and the
  pre-edit snapshot produce bit-identical arrays: ardy_cpu 21/21, kimodo_cpu 11/11.
  Both trees mismatch the expected files in the same pre-existing way (12 and 6
  differences). With `root_heading=None` the real adapters' outputs are therefore
  unchanged.
- **Unit tests.** `tests/unit`: 226 passed, 3 skipped (job 4024667, CPU node cell, after
  the last code edit).
- **Ruff.** `ruff check src tests scripts/smoke` passes in the same job.
- **Memory check.** `scripts/check_memory.py --check` flags only five unlogged
  `behavior_eval` run directories that belong to the concurrent main session.

## Caveats and follow-ups

- **Twist mask (proposed, not built).** An opt-in mask would drop edges whose foot yaw
  exceeds graph-default's p99 while every contact label is on: 1,110 in-place and 2,354
  straight edges. It is not built because it touches `graph/config.py`,
  `annotations.py` and `tables.py`.
- **Not evaluated in use.** The merged graph has not been run in crowd behavior or in
  mjlab. Coverage is availability, not visitation.
- **Heading cost may favor twists.** The runtime's heading cost may prefer exact-heading
  twists over stepping turns that drift 2–5 cm. Not measured.
- **Composition shift.** The supplement is 43% arcs and 31% walking, so the merged graph's
  mix changes. 2,048 streams instead of 1,024 would double the added graph and rotation
  size for about twice the turning windows.
- **Visual review is thin.** It is limited to the top-down stills
  (`report/clips_in_place_*.png`) and the round-1 stick MP4s. No MuJoCo render was made.
