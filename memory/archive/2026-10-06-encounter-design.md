# Encounter behavior design and turning library

Opened 2026-10-06.

## Request (user, 2026-10-06)

Add an **alternate** behavior design beside the original random-goal plaza design.
The iteration agent must be able to iterate on either. The new design has four parts:

- **Conditioning.** Waypoints or velocities condition each human. Random waypoints
  give the comparison with the old design: directed, but still random.
- **Collision bubble.** The bubble radius is configurable. A human may intervene only
  when a neighbour inside the bubble is likely to collide.
- **Switch probability.** At each step, the chance that a human switches to an
  intervention depends on its own assertiveness and on the distance of the closest
  point in the bubble. Assertive humans may keep going.
- **Interventions.** Sidestep and continue, turn away and back, slow or yield, and
  turns of various radii. In-place turns and gradual turns are both strategies, and
  the choice depends on the collision.

Two further rules:

- Two humans deciding at the same moment is natural. **No tie-break is needed.**
  Humans must stop before they collide; the clearance tolerance starts at 1 cm and is
  configurable.
- Turning motions with various radii must be supported, and generated where the library
  lacks them.

## Constraints

- Everything is opt-in. The plaza design, its outputs, defaults, the default graph and
  the golden expected files stay unchanged.
- Every consumer stays on `CrowdController`, and all randomness uses per-human streams
  under a new named scheme.
- Do not commit. Keep the held-out seeds 101/307/911 unused.

## Plan

1. Turning-radius audit of the saved graph under the round-2 mask (job 4024271).
2. Behavior: `behavior/encounter.py` with conditioning, bubble, hazard, strategies and
   the stop rule, plus an arc planner. Thread it through a design factory into the
   evaluator, `GraphCrowd`, the demo, diagnose, live tuning, metrics, tests and docs.
3. Motion: heading targets in `MotionRequest` and ARDY conditions, and a turning command
   source for library generation. Then a GPU smoke test, a supplementary library, a
   merged graph with its rotations as opt-in artifacts, and an audit of the new graph.
4. Evaluate the encounter design against the plaza design on dev seeds 0/1/2, on both
   graphs, with encounter probes. Then update docs and the iteration prompt for both
   designs.

## Progress

- [x] 1 Audit: job 4024288 (4024271 cancelled on an overloaded node), `runs/20261007T010435Z_turn_audit/`.
  Under the round-2 mask, 8 admissible edges turn in place (5 upright, 3 hunched), none faster than
  31°/s, and no 1.2 s window turns 45° or more in place. Forward-turn edges by radius:
  - under 0.5 m: 15;
  - 0.5–1 m: 398;
  - 1–2 m: 2,130;
  - 2–4 m: 2,275.

  Forward 1.2 s windows turning 45° or more: 21 (r < 1 m), 155 (1–2 m), 13 (2 m or more).
- [x] 2 Behavior: done. Results, iterations and limits are in
  `memory/experiments/2026-10-06-encounter-design.md`; direction is recorded in AGENTS.md and ledger
  D16; status and the architecture note are updated.
- [ ] 3 Motion: background agent in progress; its note is `memory/experiments/2026-10-06-turning-library.md`.
  Its heading check found that the pelvis twists while the feet stay planted.
- [ ] 4 Evaluation on the turning graph, once it is validated.
