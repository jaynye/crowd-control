# Crowd architecture

How simulated humans are structured, independent of which motion backend realizes their
bodies. Sources: the user's [handoff](../handoff-2026-09-30.md) §5–6 (2026-09-30), and the
earlier proposal [Kimodo stochastic humans](../../archive/2026-09-28-mjlab-kimodo-stochastic-humans.md)
(2026-09-28, never reviewed; historical since ledger D11), whose intent-layer and
embodiment sections agree with the handoff.

## Layers

```text
intent (stochastic, per human)  ->  local steering  ->  body motion (backend)  ->  embodiment
goal, mode, traits, perception      TTC avoidance,       ARDY / Kimodo online,      kinematic proxies
semi-Markov switching               passing side,        or offline library +       (capsules or mesh)
                                    target path/heading  graph / matching           for LiDAR + contact
```

- **Intent** holds all behavioral stochasticity. It chooses behavior and targets from
  simulator state. Modes to start with: travel, wait, yield, reroute. Non-yielding
  pedestrians differ from deliberately intercepting ones; bounded interception is for
  labeled stress tests only.
- **Local steering** turns targets into a short future root path and heading. Neither
  generator understands scene obstacles; they must receive explicit constraints, and
  the result must be validated.
- **Body motion** realizes the path through the backend adapter
  ([motion backends](motion-backends.md)). It must not own behavior decisions.

## Per-human state

Goal, behavior mode, elapsed mode time `τ`, persistent traits `θ` (sampled once per
person per episode), perceived neighbor states (with configurable perception delay),
executed motion history, and a future-motion buffer.

Mode switching is duration-aware, for example
`P(switch) = 1 − exp[−λ_b(τ, c, θ) Δt]` with `Δt` in **simulated** time. Add minimum
dwell times and limits on acceleration and turning. These are controllable heuristics,
not calibrated human statistics.

## Behavior review

User direction (2026-10-01, ledger D12): ordinary travel is predominantly forward.
Use turning motion to change direction; reserve lateral/backward travel for deliberate
actions such as abrupt close avoidance. Do not equate a curved forward path with
sidestepping; measure executed velocity relative to body facing.

The baseline plaza retains random destinations/waits, closest-approach repulsion and
its shared RNG. Optional shared controls now add committed turn intent, body-facing
and terminal-heading graph costs, forward steering and evasive orientation-cost
relaxation. The graph heading cost also has an optional angular deadband. Every
consumer still uses `CrowdController`; no consumer-specific behavior loop was added.

The [2026-10-05 iteration](../../experiments/2026-10-05-evening-graph-behavior-iteration.md)
found held-out forward-travel gains, but longer runs exposed spacing regressions and
body-aligned turning slowed root-path response. Committed turns alone completed only
8/49 attempts and accumulated at walls. Keep these controls opt-in, preserve the
original baseline and measure direction, spacing, speed, feet and seams together.
The 2-env × 4-human GPU mjlab check verified execution, reset isolation and LiDAR;
training-scale performance and independent legacy plaza RNG remain unresolved.

The [graph audit](../../experiments/2026-10-01-graph-behavior-audit.md) remains the
coverage reference. The first [bounded session](../../experiments/2026-10-01-graph-behavior-iteration.md)
stopped waiting for a GPU; later local sessions executed the actual experiments.
Usage: [behavior iteration](../../../docs/usage/behavior-iteration.md).

## Posture, transitions and live controls (2026-10-05)

User direction in ledger D14 adds persistent upright/hunched posture, flat-ground
height feasibility and live stochasticity controls to the earlier forward-travel
objective. [Development evidence](../../experiments/2026-10-05-transition-posture-iteration.md)
supports useful opt-in mechanisms, but held-out promotion and GPU integration were
unfinished when the run stopped after a frame-extraction invocation failure.

- Posture is a per-human exact two-state CTMC observed at graph boundaries, with its
  own named `graph_posture` stream. Reset samples the initial distribution and matches
  the start class; live rate edits do not redraw existing humans. Plaza RNG is preserved.
- Turn-before-walk belongs in shared steering, with retained-target/native-yaw graph
  preferences. Scarce slow-turn coverage caused stalling; ordinary forward shares
  improved without that phase, but many measured redirects remain unresolved.
- Viser callbacks queue changes for the simulation thread. Each `GraphCrowd` owns
  mutable runtime settings while immutable tables may be cached. CPU bindings pass;
  browser interaction and the new GPU integration have not been verified.
- Measured posture persistence and vertical gains coexist with worse seed-0 spacing,
  mixed feet/seams and a controlled nonresponse. Next test avoidance over executed
  .4 s edges and retain long crowded traces alongside isolated response probes.

## Two behavior designs (2026-10-06)

The user's direction (ledger D16) keeps the random-goal plaza design and adds an
alternate **encounter** design. The iteration agent may work on either; both run
through `CrowdController` (`behavior/designs.py`).

Encounter humans follow a condition: random waypoints, fixed routes, or a commanded
velocity. They react to neighbors through decisions they keep once made:

- a collision bubble with a likely-collision test;
- a per-step switch hazard from assertiveness and the closest distance;
- a sidestep, veer (arc), pivot (in place) or yield;
- a stop rule with a 1 cm target that removes closing motion (sliding past is allowed).

Two design facts came out of the [experiment](../../experiments/2026-10-06-encounter-design.md):

- **Commands are requests.** The graph realizes them by cost, and it may pick a clip
  that walks into a neighbor. Safety therefore also lives in the opt-in graph
  `ClearanceCost`, to which the design passes neighbors and wall mirrors.
- **Body extent.** The no-contact rule needs a body radius near the swinging-arm reach
  (0.35 m), not the torso.

On development seeds the encounter design has 94% forward travel against 70–83% for
plaza. Lateral and backward travel nearly vanish, sliding and acceleration are lower,
and joint-sphere contact is less than plaza's. It is slower: 0.43–0.50 against
0.58–0.64 m/s at 60 s.

The default graph realizes arcs but **not** in-place turns: they stand, at a median
2.8°/s. Sidesteps are mostly oblique. Generating that motion is the
[turning library](../../experiments/2026-10-06-turning-library.md) work.

## Scheduling

Gather humans needing a refill or a changed plan, bucket compatible requests, generate in
batches, and scatter results back. Refill even without a behavior switch. Use per-human
random streams independent of batch order. Invalidate stale predictions on switches and
resets, and condition only on executed motion. Implement synchronously first.

Two different delays must be reported separately: wall-clock stalls (synchronous
inference slows training) and simulated-time reaction delay (committed buffered frames
delay responses inside the simulation).

## Geometry and contact

- Reuse an existing body model and skinning pipeline. For Kimodo/SOMA, evaluate the
  SOMA-X low and extra-low meshes first. The 2026-09-28 smoke posed the SOMA `xlo` mesh
  (612 vertices, 1,220 faces) at 2,400 poses in 24 ms on an L40S
  ([Kimodo smoke](../../experiments/2026-09-28-kimodo-smoke.md)).
- ARDY's released models use the 27-joint Core skeleton, which ships its own skin
  (`third_party/ardy/ardy/assets/skeletons/cskel27/skin_standard.npz`); it is not the
  SOMA mesh.
- Keep LiDAR geometry separate from the physics collision representation.
- Humans start kinematic (mocap). MuJoCo creates no contacts between mocap bodies and
  static geometry, or between two mocap bodies, so the intent layer must keep people
  out of walls and each other. A free-joint robot does collide with them (checked
  2026-10-05). The human acts as an infinitely heavy pusher that never reacts.
- LiDAR cost at scale (2026-10-06, A40, 4,096 envs × 4–16 humans, 5,760 rays per
  env; an A5000 set gives the same ordering):
  - Rays, not geometry updates, dominate: capsules take 32.6 ms against a 37.1 ms G1
    step.
  - With 32 shapes, rigid Core-skin pieces per bone cost 41.5–46.5 ms through a
    per-human kernel, with memory flat in the number of humans. A skinned surface
    with an SAH tree costs 45.3–108.7 ms at 600–2,500 faces; refit, memory and tree
    build all grow with humans × faces. Its shapes are per human and free.
  - Rigid pieces are the cheapest geometry for their accuracy at 4, 8 and 16 humans.
    Matching their 2.9% floor (joint blend regions and caps) takes skinning at 2,500
    faces, 2.3× the cost. At 4 humans all but skinned 4,500 cost 16–23 ms.
  - The same pieces as mujoco_warp mesh geoms cost 88–101 ms.
  - Adopted 2026-10-06 (D15): full-resolution rigid pieces are the plaza default.
    The plaza step at 4,096 envs costs 57.3 ms (rigid) against 53.3 ms (capsules) at
    16 humans, and 29.3 ms for both at 4 (A40). Rotations come from saved graph
    rotations; a Warp ray target serves the LiDAR, with visual geoms for viewers.
  - mjlab's compile-pose scene BVH groups each human's bones, so humans moving apart
    do not degrade it.
  - [Benchmark](../../experiments/2026-10-05-body-geometry-benchmark.md); open choices
    are in [human body model](../../decisions/human-body-model.md).

## Corridor fixture and our mjlab changes

- The corridor fixture (inherited from the 2026-09-26 to 09-28 sessions) now lives in
  `src/crowdmotion/sim/mjlab/corridor/`: `scene.py` (`make_crowd_env_cfg`, up to 8
  humans), `walking.py` (`CorridorWalk`, a seeded waypoint intent with private
  per-human, per-env RNG streams), `humans.py` (`PrimitiveHumanCfg`: primitive planar
  humans with finite-force actuators), and the demo and scan plot behind
  `crowd demo corridor`. Tests: `tests/unit/test_corridor.py`; usage:
  [corridor](../../../docs/usage/corridor.md).
- Simulator features stay in our mjlab copy (ledger D13): the LiDAR pattern, minimum
  range, hit geom IDs, visualization stride and ray targets (`add_ray_target`) in
  `mjlab/src/mjlab/sensor/raycast_sensor.py`, tested in
  `mjlab/tests/test_lidar_pattern.py` ([knowledge](../../knowledge/mjlab.md)).
