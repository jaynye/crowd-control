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

The current plaza demo has random destinations/waits and closest-approach repulsion,
but no turn-before-walk state, acceleration/yaw limits or stable per-person traits.
Its shared RNG depends on call order, so it is not training-ready. The graph has
per-human sampling but no explicit facing cost. The
[audit](../../experiments/2026-10-01-graph-behavior-audit.md) establishes coverage
problems before tuning; [behavior iteration](../../../docs/usage/behavior-iteration.md)
documents the separate evaluation, saved-motion replay and paired comparison
(`crowd behavior evaluate|render|compare`), which run the same `CrowdController` as
mjlab ([runtime architecture](runtime-architecture.md)). The user accepted the setup
and authorized a bounded iteration session, which stopped at five minutes pending
on the short-QoS GPU group limit; no stochastic rule change or real tuning/render
executed ([session](../../experiments/2026-10-01-graph-behavior-iteration.md)). The
committed-turn proposal samples once on entry, gives 90° strongest weight and
retains a target body heading. It is not implemented/evaluated yet.

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
  out of walls and each other.

## Corridor fixture and our mjlab changes

- The corridor fixture (inherited from the 2026-09-26 to 09-28 sessions) now lives in
  `src/crowdmotion/sim/mjlab/corridor/`: `scene.py` (`make_crowd_env_cfg`, up to 8
  humans), `walking.py` (`CorridorWalk`, a seeded waypoint intent with private
  per-human, per-env RNG streams), `humans.py` (`PrimitiveHumanCfg`: primitive planar
  humans with finite-force actuators), and the demo and scan plot behind
  `crowd demo corridor`. Tests: `tests/unit/test_corridor.py`; usage:
  [corridor](../../../docs/usage/corridor.md).
- Simulator features stay in our mjlab copy (ledger D13): the LiDAR pattern, minimum
  range, hit geom IDs and visualization stride in
  `mjlab/src/mjlab/sensor/raycast_sensor.py`, tested in
  `mjlab/tests/test_lidar_pattern.py` ([knowledge](../../knowledge/mjlab.md)).
