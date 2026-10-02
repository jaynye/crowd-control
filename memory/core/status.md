# Current status

Snapshot updated 2026-10-01. Replace sections as work changes; do not
append history. Completed results move to dated
[experiment notes](../experiments/README.md), with their conclusions in core; open
questions move to [decisions](../decisions/README.md).

## Direction (user, 2026-09-30)

4,096 environments with 4–16 humans each. ARDY and Kimodo run **offline only** (ledger
D09–D10). Training may be RL in mjlab, or TANGO-style offline expert data plus imitation
([learning paradigm](../decisions/learning-paradigm.md)). Either way, the motion comes
from an offline library.

## Active work

### Repo migration to git and local iteration (Phases 0–9 done 2026-10-01)

- The tree is commit-ready; **the user creates the git repository and initial commit**
  (Phase 10), then clones on the local Linux GPU machine: `scripts/bootstrap.sh`,
  `scripts/fetch_artifacts.sh` (graph from the cluster; checkpoints from HF), unit and
  golden tests, then the GPU checks: `tests/gpu`, and a small ARDY library plus a
  throughput row with the pre-migration snapshot and the migrated tree on the same card.
  Install steps: `docs/install.md`.
- The refactor is behavior-preserving: CPU goldens from the pre-migration code
  (`tests/golden/`) match bit for bit after every phase, including real ARDY and Kimodo
  on CPU. One deliberate difference: a `history` of 0 is rejected (it used to pass the whole
  history by accident; no config used it). Design rationale:
  [runtime architecture](design/runtime-architecture.md). Plan, phase log and reviews:
  the repo-migration entry in the [journal](../journal/README.md).
- mjlab is first-party (ledger D13); its upstream history is outside the repo (plain
  tracked files, the plan's default; a fork or subtree can still be adopted).
- Next (Phase 11, user to agree): switch plaza steering to per-human streams and make
  `GraphConfig` strict, record a new baseline, then resume the behavior iteration below
  locally with `crowd iterate`.

### Behavior iteration: paused (2026-10-01)

- User accepted the report and authorized continual iteration with one run folder and
  ephemeral journal, committed sampled turns biased toward 90°, a four-hour task limit.
  The first session (`runs/20261001T061749Z_behavior_iteration/`) stopped at its
  five-minute GPU-wait limit on Slurm (job 3994253 PENDING on `QOSGrpGRES`); its journal
  is archived in [memory/archive](../archive/README.md).
- **No real baseline/candidate evaluation or render has run; runtime behavior rules are
  unchanged.** The turn distribution is a proposal, not an improvement result.
  [Session evidence](../experiments/2026-10-01-graph-behavior-iteration.md).
- Sessions now run locally by default (`crowd iterate start|run|status|note|stop`;
  `docs/usage/behavior-iteration.md`); the Slurm executor keeps the GPU-wait and
  deadline safeguards.
- Next session: run the frozen baseline, implement committed turns and a facing cost
  (a `@cost_term` class plus a `GraphConfig` weight), review paired seeds, saved-motion
  renders and spacing/foot tradeoffs before mjlab integration.

### Graph audit and review tools (completed)

- Report: `runs/20261001T055543Z_behavior_report/index.html`; graph mechanics,
  available motion composition, connectivity, stochastic rules and historical video.
  Regenerate with `crowd graph report`.
- Of valid library frames, 43.0% forward, 14.4% sideways, 17.4% backward; only 48.7%
  forward among moving frames. Five of 134,820 edges qualify as slow turning under
  explicit analysis cutoffs. Availability differs from runtime visitation.
  [Conclusions](design/motion-library.md#behavior-audit-and-review-loop-2026-10-01)
  and [audit evidence](../experiments/2026-10-01-graph-behavior-audit.md).
- The graph has no facing cost; plaza steering can redirect velocity abruptly and
  still uses a shared RNG.

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
  - [human body model](../decisions/human-body-model.md): capsules vs mesh at 16k–65k
    humans.
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
  - `iterate/`: evaluate, compare, sessions (local or Slurm). `bench/`, `library/`,
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
- **mjlab changes (first-party):** LiDAR pattern, minimum range and hit IDs, with their
  tests and docs; the corridor fixture moved to `crowdmotion.sim.mjlab.corridor`. See
  [crowd architecture](design/crowd-architecture.md#corridor-fixture-and-our-mjlab-changes).

## Blockers and known issues

- **Gated Llama-3** (403 for the HF account): no prompts, so walking style is
  uncontrolled. Curation removes hunched gait but adds no styles.
- **Behavior coverage and selection:** body-facing travel is uncontrolled by the
  current path score; slow-turn coverage is scarce. Demo steering's shared RNG does
  not meet per-human batch independence; the review tools preserve it for baseline.
- **No end-to-end measurement.** All speeds are isolated, on an A40. The simulator's
  own `S` at m = 16 with LiDAR is unmeasured.
- **`fpo-control/`'s git index stages every file for deletion** (the files are on
  disk). It is unrelated and was left untouched.
- **Old broken env:** `mjlab/.venv.stale-pre-migration` (2026-09-28, gitignored) awaits
  the user's deletion.
- **GPU paths unverified after the refactor** until Phase 10 on the local GPU: bf16
  autocast, ARDY `conditions="library"`, Kimodo post-processing and the throughput OOM
  retry are not covered by the CPU goldens.
