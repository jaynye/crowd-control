# Current status

Snapshot updated 2026-10-06. Replace sections as work changes; do not
append history. Completed results move to dated
[experiment notes](../experiments/README.md), with their conclusions in core; open
questions move to [decisions](../decisions/README.md).

## Direction (user, 2026-09-30)

4,096 environments with 4–16 humans each. ARDY and Kimodo run **offline only** (ledger
D09–D10). Training may be RL in mjlab, or TANGO-style offline expert data plus imitation
([learning paradigm](../decisions/learning-paradigm.md)). Either way, the motion comes
from an offline library.

## Active work

### Recentering toward robot training (user, 2026-10-12)

- **Direction:** the crowd is close to ready for training the crowd-avoiding robot
  policy. One behavior problem remains: humans are not decisive. Two humans often bump
  into each other and stop, and neither acts. Avoidance should become more likely, with
  minimal deviation from each human's path. Candidate fixes are a step back on a
  detected collision and smallest-angle turns. A small population of stubborn,
  non-avoidant humans stays, and it must be easily tunable.
- **Round 4 (Astra):** prompt `configs/behavior/iterate_codex_decisive.txt`, on
  graph-turns, starting from `encounter_turns.json`. It measures standoffs first, then
  traces their causes, then adds opt-in mechanisms and `stubborn_fraction`. The agent
  records its run directory in the run log.
- **Robot policy:** the user asked which architecture to prioritize:
  - (a) a LiDAR policy from scratch;
  - (b) a residual on SONIC's planner and tracker;
  - (c) a LiDAR planner that replaces SONIC's kinematic planner;
  - (d) SONIC fine-tuned to also take velocity commands.

  The training environment is built in a separate snapshot checkout while Astra edits
  this one, and is merged afterwards.

### Repo migration to git and local iteration (Phases 0–9 done 2026-10-01)

- The repository now has commit `5c1ef9c`; the working tree includes preserved,
  uncommitted launcher and behavior work. No commit or push was made during iteration.
  Installation and migration steps remain in `docs/install.md`. Real-generator GPU
  migration checks are still outstanding; the saved graph passed a small GPU mjlab
  integration check on 2026-10-05.
- The refactor is behavior-preserving: CPU goldens from the pre-migration code
  (`tests/golden/`) match bit for bit after every phase, including real ARDY and Kimodo
  on CPU. One deliberate difference: a `history` of 0 is rejected (it used to pass the whole
  history by accident; no config used it). Design rationale:
  [runtime architecture](design/runtime-architecture.md). Plan, phase log and reviews:
  the repo-migration entry in the [journal](../journal/README.md).
- mjlab is first-party (ledger D13); its upstream history is outside the repo (plain
  tracked files, the plan's default; a fork or subtree can still be adopted).
- Next (Phase 11, user to agree): switch plaza steering to per-human streams and make
  `GraphConfig` strict and record a new baseline. The completed behavior iteration
  below kept both deferred changes out of scope.

### Encounter behavior design and turning library (2026-10-06)

- **User direction (ledger D16):** keep the random-goal plaza design and add an alternate
  `encounter` design that the iteration agent can also work on. It has:
  - waypoint or velocity conditioning, with random waypoints for comparison;
  - a configurable bubble;
  - an assertiveness- and distance-dependent switch hazard;
  - committed interventions, including gradual and in-place turns;
  - a stop rule with a 1 cm tolerance.

  Two humans may decide at once.
- **Implemented, opt-in.** The plaza path is bit-exact against the previous day's run.
  Configs: `configs/behavior/encounter*.json` and `plaza_forward_posture.json`.
  Usage: `docs/usage/behavior-iteration.md#behavior-designs`.
  [Results](../experiments/2026-10-06-encounter-design.md).
- **Results on development seeds 0/1/2:**
  - 94% forward travel against 70–83%;
  - lower foot sliding and acceleration;
  - joint-sphere contact at most about 1 cm, against plaza's −5 to −9 cm;
  - slower walking: 0.43–0.50 against 0.58–0.64 m/s at 60 s.

  The default graph realizes arcs but not in-place turns, which stand at a median
  2.8°/s, and sidesteps are mostly oblique.
- **Turning library (done, opt-in).** ARDY follows sparse body-heading targets: the hip
  line turns 0.98 of the command at the median, and arcs come out at 1.04 of the radius.
  The opt-in graph `graph-turns` (base plus a 9.1 h turning supplement; rotations in
  `graph-turns-rotations`) raises admissible in-place edges from 8 to 5,873, of which
  4,763 hold no planted twist. The supplement also adds 2,354 standing or straight edges
  that hold a twist (67 in the base), so on this graph a human can stand twisted with
  no turn requested.
  [Note](../experiments/2026-10-06-turning-library.md).
  - About 15% of generated in-place turns twist the pelvis while the feet stay planted.
  - On this graph the encounter design's in-place turns execute (median 83°/s against
    2.8°/s), and turn-then-walk resolutions roughly triple.
  - Joint-sphere contact regresses to 2–7 cm and sliding rises.
- **Round 3, Astra on graph-turns (2026-10-07; stopped after 13 of 225 min).**
  [Note](../experiments/2026-10-07-encounter-turns-iteration.md).
  - Opt-in planted-twist mask (`twist_mask`, `twist_max_foot_yaw_deg` 21.4°) removes
    3,533 graph-turns edges: 1,110 in-place and 2,421 straight or standing. Without it,
    2–4% of executed edges were planted twists.
  - `configs/behavior/encounter_turns.json` is `encounter.json` plus the mask: a
    development candidate, not frozen, held-out seeds unused. Minimum clearance
    improved in all 6 unpaired crowd runs (60 s: −0.027 / −0.010 / +0.043 m); sliding,
    acceleration and unresolved redirects did not.
  - The mask is a live Viser control. Rigid mjlab and the Viser demo run on
    graph-turns with `CROWD_GRAPH_ROTATIONS` set (launch: `docs/usage/demos.md`).
  - Comparison video (plaza and encounter on graph-default, encounter on graph-turns
    without and with the mask): `runs/20261007T165926Z_encounter_turns_video`.
  - The agent stopped on a KeyError in its own script and killed itself with
    `crowd iterate stop`. Both Codex prompts now treat own-script failures as bugs and
    forbid `crowd iterate stop`.
- **Next:**
  - Relaunch the agent with `iterate_codex_encounter.txt` (launch command in
    `docs/slurm.md`) from `encounter_turns.json`: contact tracing and tuning (stop
    look-ahead, approach time, body radius), unresolved redirects and jams, decision
    realism, then held-out seeds 101/307/911 once for a frozen candidate.
  - Tune the bubble and passing margin together with spacing and speed.

### Transition/posture iteration: development evidence, stopped before promotion (2026-10-05)

- The local A40 session completed 190 requests, then stopped after one frame-extraction
  invocation failed, following the user's explicit stop-on-failure instruction.
  Outputs: `runs/20261005T225742Z_behavior_iteration/report.md` and its session journal.
  [Results, inspected views and validation](../experiments/2026-10-05-transition-posture-iteration.md).
- **Incumbent unchanged:** the previously selected configuration remains
  `runs/20261005T201450Z_behavior_iteration/configs/selected_candidate.json`.
  Both it and the original baseline were frozen and evaluated before runtime edits.
  The new development candidate is
  `runs/20261005T225742Z_behavior_iteration/configs/forward_floor080_peak012.json`;
  it has not been promoted or frozen for held-out evaluation.
- Implemented opt-in load-time edge annotations, flat-ground masks with an explicit
  stationary-hold dead-end fallback, independent sticky per-human posture and matched
  starts, soft posture/turn/entry-velocity costs, turn-before-walk steering, executed
  height/posture/redirect metrics, and queued Viser live controls. All consumers still
  use `CrowdController`; graph/defaults/legacy plaza RNG/golden expected files unchanged.
- Three 60 s development seeds: ordinary forward travel 93.7–94.4% versus incumbent
  78.2–83.9%; posture-class crossings 4.3–5.8 versus 18.0–19.4 per human-minute;
  peak lowest-foot height .112 m versus 1.336 m. However seed-0 minimum root gap
  worsens .635→.433 m, feet/seams are mixed, and many redirect attempts remain unresolved.
  Controlled body responses are 47/48 versus incumbent 48/48 for both 90° and 180°.
- Coverage limits slow turns: after the chosen mask, only two upright and three hunched
  slow-turn edges remain. The .2 m/s explicit turn phase stalled; walking-turn phases
  improved controlled response but did not dominate crowd metrics. The main
  mostly-turn-then-walk objective is **unfinished**, not established by forward shares.
- Nine initial edge sheets plus brief-airborne examples and several seed-0 follow/
  overview variants were actually inspected. Seed-2 dense reversed follow exposed
  a raised-foot transition. Final candidate image review is unfinished: its MP4
  rendered, but extraction invoked base Python instead of the virtualenv and failed.
- **Validation:** final 165 CPU tests and Ruff pass. Earlier same-host compatibility
  matched 283 arrays/27 JSON outputs across all 12 fast golden cases and both frozen
  reference trace banks. Both trees retained historical expected-file mismatches;
  final peak-mask/diagnostic additions were not followed by another golden run.
  Viser bindings are headless-tested; browser and new GPU integration are unverified.
- **Comparison video (2026-10-06):** `runs/20261006T175357Z_behavior_video/comparison.mp4`
  shows baseline, round-1 selected and this candidate on the seen seeds 11/23/47.
  Posture and forward share improve. Seed 47 briefly exceeds the 0.12 m foot mask
  (0.138 m); the seed-11 wall excursion matches round 1. [Follow-up](../experiments/2026-10-05-transition-posture-iteration.md#follow-up-comparison-video-on-seen-seeds-2026-10-06).
- **Next:** repair the invocation by using the literal `.venv/bin/python` path, finish
  candidate image review, freeze it, evaluate still-unseen seeds 101/307/911 at 20/60 s,
  and run the prepared small mjlab/live-control check. Then test avoidance across the
  executed .4 s horizon and audit reachable upright walking-turn sequences. No new
  session or jobs were created in this run. [Calibration remains open](../decisions/posture-height-calibration.md).
- Historical preceding result: the [evening iteration](../experiments/2026-10-05-evening-graph-behavior-iteration.md)
  improved held-out forward shares but retained spacing/response limits; its GPU
  integration check applies to that earlier configuration, not the new controls.

### Rigid mesh humans in the plaza (adopted and implemented 2026-10-06, D15)

- **Decision (user):** full-resolution rigid Core-skin pieces, one per bone, with no
  simplification, are the human geometry for LiDAR and display. They are the plaza
  env-config default (`human_geometry`), which a training env will pick up; no
  training env exists yet.
  [Benchmark and timing](../experiments/2026-10-05-body-geometry-benchmark.md).
- **How it works:**
  - `crowd graph rotations` saves joint rotations per graph frame: artifact
    `graph-rotations`, 0.32 GB, checked against the graph at load.
  - `GraphCrowd` poses the pieces in one fused Warp pass (`sim.rays`).
  - A Warp ray target, registered through our mjlab's new
    `RayCastSensor.add_ray_target`, serves the LiDAR. The pieces are visual geoms in
    group 1.
  - The Viser demo builds both geometries; **Commands → Crowd body → Geometry**
    switches what is shown and what the LiDAR hits.
- **Cost (A40, 4,096 envs):** plaza step 57.3 ms (rigid) against 53.3 ms (capsules) at
  16 humans, and 29.3 ms for both at 4 humans.
- **Checks:**
  - hits equal CPU MuJoCo `mj_ray` on the visual pieces (CPU and A40);
  - the `mjlab_plaza` golden is unchanged (it pins capsules);
  - the unit suite passes.

  Browser interaction with the switch has not been exercised; a fake GUI drives it in
  the tests.
- **Still open:**
  - one body shape: the morphology set (SOMA-X refit or retargeting) is undecided;
  - robot contact geometry (the pieces have `contype` 0);
  - ray budget and scan rate;
  - an mjlab fix for mujoco_warp's off-centre mesh bounds (it affects traced robot
    meshes, not the rigid humans).

### Graph audit and review tools (completed)

- Report: `runs/20261001T055543Z_behavior_report/index.html`; graph mechanics,
  available motion composition, connectivity, stochastic rules and historical video.
  Regenerate with `crowd graph report`.
- Of valid library frames, 43.0% forward, 14.4% sideways, 17.4% backward; only 48.7%
  forward among moving frames. Five of 134,820 edges qualify as slow turning under
  explicit analysis cutoffs. Availability differs from runtime visitation.
  [Conclusions](design/motion-library.md#behavior-audit-and-review-loop-2026-10-01)
  and [audit evidence](../experiments/2026-10-01-graph-behavior-audit.md).
- The saved graph's default score still has zero facing/heading weights. Optional
  orientation terms now exist; legacy plaza goal/spawn RNG remains shared.

### Offline library and graph performance (historical prototype, 2026-09-30)

- **Result:** [motion library](design/motion-library.md#conclusions); numbers in the
  [experiment](../experiments/2026-09-30-ardy-library-graph.md).
  - An 18 h ARDY library takes 2 min of A40 time.
  - The kNN motion graph steps 65,536 humans in 47 ms per 0.4 s: about 5% of an A40
    at m = 16, `S = 0.4` (isolated). Online ARDY of the same responsiveness would need
    about 46 A40s.
  - On a held-out seed at the recommended setting
    (`configs/graph/default.json`), it tracks paths to 0.7–1.4 cm, and its
    median responses are within 0.0–0.4 s of online ARDY. The hunch filter removes
    stooped gait (0.2–0.6% of frames).
  - Costs: more frequent small foot sliding, root-speed seams 1.4–2.0× the interior,
    no local rotations yet, 20 fps output.
- **Waiting on the user:**
  - [learning paradigm](../decisions/learning-paradigm.md): RL vs TANGO-style;
  - [library generator](../decisions/library-generator.md): later ARDY/Kimodo expansion;
  - [graph response delay](../decisions/graph-response-delay.md): eventual acceptable
    simulated-time delay; does not block baseline review;
  - [text encoder access](../decisions/text-encoder-access.md): styles;
  - [human body model](../decisions/human-body-model.md): LiDAR geometry settled
    (D15, rigid pieces); still open: morphology source, robot contact geometry, ray
    budget and scan rate.
- **Remaining implementation candidates:**
  - mjlab smoke test: graph-driven humans in the crowd fixture with LiDAR, at m = 4–16;
    also measures simulator `S` at m = 16;
  - local-rotation output with offset decay (needed for mesh skinning);
  - root-velocity inertialization;
  - simulator frame-rate conversion (ARDY 20 fps);
  - a Kimodo/SOMA library through the same graph;
  - styles once prompts work.

## What exists

- **Code:** `src/crowdmotion` (installable; `crowd` CLI), layered as in
  [runtime architecture](design/runtime-architecture.md); module map and commands in
  `docs/architecture.md` and `docs/cli.md`.
  - `behavior/`: `CrowdController` with command sources (random segments, benchmark
    scenarios, plaza intent and steering), planner, graph and online motion sources.
    Library generation, the quality benchmark, the evaluator and mjlab's `GraphCrowd`
    share it.
  - `graph/`: kNN motion graph tables, build, cost terms, runtime, audit.
  - `backends/`: ARDY and Kimodo adapters with typed configs and a registry.
  - `iterate/`: evaluate, diagnose, controlled response probes, compare, sessions (local or Slurm). `bench/`, `library/`,
    `metrics/`, `viz/`, `sim/` (scene, capsules, `mjlab/` plaza, demos, corridor fixture).
  - Configs: `configs/{behavior,graph,library,bench}`; recommended graph setting
    `configs/graph/default.json`.
  - Tests: `tests/unit` (CPU), `tests/golden` (bit-exact, 14 cases), `tests/gpu`.
- **mjlab demos** (2026-09-30; one environment): a 14 × 10 m plaza in which 16
  graph-driven capsule humans walk, seen by an mjlab LiDAR (`crowd demo video|viser`;
  `docs/usage/demos.md`), from the saved graph
  `runs/20261001T013254Z_motion_graph/graph.pt` (441 MB, artifact `graph-default`).
  - **Video:** `runs/20261001T012334Z_mjlab_graph_demo/crowd.mp4`. Closest
    root-to-root gap 0.70 m; about 206 of 5,760 rays hit humans each step.
  - **Viser:** about 1× real time at 50 FPS on a GPU node with every 8th LiDAR hit
    drawn; 1.0× at 30 FPS on a 4-core CPU job; 0.4–0.7× on the busy login node.
  - Some stooped poses survive the mean-head-height filter.
  - This is a visual integration check, not the 4,096-environment smoke test.
- **Library on disk:** `runs/20261001T004918Z_ardy_library/library.pt` (18.2 h,
  2,048 × 32 s streams, Core skeleton; artifact `library-ardy-h40`).
- **Environment:** one uv workspace (crowd-control + mjlab + kimodo + py-soma-x) with
  ARDY from source; third-party pins in `third_party.toml`; see `AGENTS.md`.
- **Weights:** ARDY Core H8 and H40, Kimodo-SOMA-RP-v1.1 and SOMA-X v0.3.2 assets, all
  local and offline (`artifacts.toml` names the HF revisions). ARDY-SOMA is unreleased.
- **mjlab changes (first-party):** LiDAR pattern, minimum range, hit IDs and ray
  targets, with their tests and docs; the corridor fixture moved to `crowdmotion.sim.mjlab.corridor`. See
  [crowd architecture](design/crowd-architecture.md#corridor-fixture-and-our-mjlab-changes).

## Blockers and known issues

- **Gated Llama-3** (403 for the HF account): no prompts. Opt-in geometric posture
  selection now exists, but semantic styles and its geometric calibration remain open.
- **Behavior coverage and selection:** optional orientation costs improve forward
  travel, but slow-turn coverage, long-rollout spacing, wall excursions and slower
  path responses remain limitations. Shared legacy plaza RNG still does not meet
  per-human batch independence; it was preserved as explicitly requested.
- **No end-to-end measurement.** All speeds are isolated, on an A40. The simulator's
  own `S` at m = 16 with LiDAR is unmeasured.
- **`fpo-control/`'s git index stages every file for deletion** (the files are on
  disk). It is unrelated and was left untouched.
- **Old broken env:** `mjlab/.venv.stale-pre-migration` (2026-09-28, gitignored) awaits
  the user's deletion.
- **GPU paths unverified after the refactor** until Phase 10 on the local GPU: bf16
  autocast, ARDY `conditions="library"`, Kimodo post-processing and the throughput OOM
  retry are not covered by the CPU goldens.
