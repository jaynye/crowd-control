# Learning paradigm: RL in mjlab, or TANGO-style offline expert data plus imitation?

Opened 2026-09-30. **Whose call:** the user, who raised it on 2026-09-30. Body motion
is offline either way (ledger D10); this decision sets what the crowd pipeline must
deliver beyond an offline library.

## Question

Is the traversal policy trained:

1. with **RL in mjlab**, 4,096 parallel environments with 4–16 humans each (ledger D09);
   or
2. **TANGO-style**: generate expert traversal episodes offline, render observations and
   train by imitation ([TANGO](../knowledge/tango.md): plan, edit and track a G1 expert
   in static clutter, then flow-matching behavior cloning, no RL)?

## What each branch needs from the crowd pipeline

| Need | RL in mjlab | TANGO-style |
|---|---|---|
| Who the humans react to | The **learner**, live. Needs an online runtime over the offline library (chosen graph, [motion library](../core/design/motion-library.md)) at 16k–65k humans | The **expert's** planned path, baked into each episode. The learner never perturbs them during data generation |
| Body-motion runtime | Required: pure-torch stitching at simulator rate | Not needed for data. Direct per-episode ARDY rollouts are cheap (below) |
| Closed-loop evaluation | Same runtime as training | Still needs reactive humans: an evaluation policy deviates from the expert, and baked humans would ignore it. Small-scale online ARDY or the graph runtime; DAgger-style data collection needs the same |
| Robot-side expert | None; the reward defines behavior | A planner that accounts for time and the humans' future motion. TANGO's expert is A* in static scenes plus IK editing plus SONIC tracking. **Outside the handoff; not started** |
| LiDAR geometry | At 4,096 × 16 = 65,536 humans, the 9,084-vertex Core skin would be about 600M vertices per step, which pushes toward capsule proxies ([human body model](../decisions/human-body-model.md)) | Offline rendering can afford full meshes |
| Body-model choice | ARDY Core (cheap library) unless SOMA is required | Kimodo/SOMA becomes affordable (below) |

**Baked-crowd cost at TANGO's scale** (agent estimate, 2026-09-30): 64,633 episodes
(TANGO's dataset size) × m humans × an assumed 20 s per episode. Generation only; no
rendering, robot planning or tracking. Rates are from the
[benchmark](../experiments/2026-09-30-motion-backend-benchmark.md), one A40 in
isolation:

| m | Human motion | ARDY H40 (recommended setting) | Kimodo (bf16, 10 steps) | Kimodo (25 steps) |
|---|---|---|---|---|
| 4 | 1,436 h | 0.5 A40-h | 24 A40-h | 55 A40-h |
| 16 | 5,745 h | 2.0 A40-h | 96 A40-h | 221 A40-h |

For comparison, TANGO reports 211 GPU-hours for its plan-edit-track and rendering
pipeline.

## Shared by both branches (no-regret work, built 2026-09-30)

- A random command generator covering speeds, turns, stops, starts and idles. Styles
  wait on prompts.
- An offline ARDY library generator with per-human seeded streams. Each edge is stored
  with the command it realized.
- The quality harness (`crowd bench quality`, `src/crowdmotion/bench/quality.py`).

The graph runtime ([motion library](../core/design/motion-library.md)) is specific to
the RL branch, and to closed-loop evaluation on the TANGO branch. It now steps 65,536
humans in 47 ms per 0.4 s with near online-ARDY quality, so stepping cost is small. The
open question on the RL branch is simulator cost at m = 16: posing humans in mjlab and
ray-casting LiDAR against them.

## What would decide it

The user's research direction. Relevant evidence the project can produce:

- Simulator speed with 16 humans and LiDAR (`crowd bench sim-speed`, which today
  measures the corridor with 4 and 8 humans). If `S` collapses at
  m = 16, the simulator, not the motion source, limits RL.
- Whether the graph runtime matches online ARDY's response and seam quality (the
  [motion graph](../core/design/motion-library.md) prototype).
