# Runtime architecture: why the code is shaped this way

Rationale and invariants behind the 2026-10-01 reorganization of `src/crowdmotion/`.
The module map, layers and APIs are in [architecture](../../../docs/architecture.md); the
graph's behavior and measured quality are in [motion library](motion-library.md); the
behavior layers are in [crowd architecture](crowd-architecture.md).

## One crowd loop

Before the reorganization the loop "command → plan path → step motion → record frames →
update root velocity" existed in six hand-maintained copies: the quality benchmark (twice,
online and graph), library generation, the behavior evaluator, the mjlab `GraphCrowd`
term, and the throughput benchmark. The evaluator's claim to measure what mjlab runs
rested only on copy discipline: a turn-commitment rule added to one copy would silently
make the evaluated crowd differ from the trained one.

Now every consumer drives humans through `behavior.controller.CrowdController`, built
from a command source, the planner, a motion source and a velocity estimator. The
evaluator and `GraphCrowd` share the same instances of the same classes (`PlazaBehavior`,
`plan_path`, `GraphMotionSource`, `BufferVelocity`), so a behavior rule cannot reach one
without the other. What differs is outside the loop: mjlab's clock, interpolation to its
50 Hz step, and mocap writes.

## Motion sources

`GraphMotionSource` and `OnlineMotionSource` give the graph and the online generators
(ARDY, Kimodo) one interface (`reset`, `step`, `commit`, `path_frames`). The online
source owns the buffer of *executed* features fed back as history, which the throughput
and quality harnesses each rebuilt before. This keeps the comparison of a library
generator against online generation, and of one generator against another
([library generator](../../decisions/library-generator.md)), a matter of swapping one
object.

## Bit-exact by construction

The reorganization was behavior-preserving, proven by golden outputs captured from the
pre-migration code (`tests/golden/`; run `runs/20261001T225208Z_pre_migration_snapshot`).
That forced some deliberate oddities, each kept so that saved baselines reproduce:

- **Two velocity estimators.** `WindowVelocity` (displacement over the last
  `min(5, n − 1)` executed frames; library generation and the quality benchmark) and
  `BufferVelocity` (a six-frame buffer seeded with the start position, divided by 5; the
  evaluator and mjlab) differ on the first edges after a reset. Unifying them is a
  behavior change.
- **Cost terms summed in a fixed order.** The graph runtime's monolithic step became
  select → correct → inertialize, with candidate scoring split into `CostTerm`s
  (trajectory, pose distance, end velocity). The weighted sum runs left to right in
  `TERMS` order, which reproduces the old expression exactly. New terms (facing, turn
  commitment) are appended with a zero default weight that skips them, so adding a term
  changes nothing until it is used.
- **`PlazaBehavior` keeps one shared generator.** Plaza steering was split into
  `GoalIntent` (goals, waits) and `LocalSteering` (avoidance, walls, obstacles), per
  ledger D05, but both still draw from one torch generator in the old call order. This
  violates the per-human stream invariant below and is the first deferred change.
- **Named seed formulas.** The four seed formulas in use (quality, library and
  throughput streams, evaluator, `GraphCrowd`) are kept as named schemes of
  `core.rng.human_seeds`.
- **Lenient `GraphConfig.from_dict`.** It still ignores unknown keys, so saved graphs and
  old configs load; only the evaluator's `graph_overrides` are checked strictly.

## Deferred behavior changes

These are known problems whose fixes change outputs. Each needs a new baseline and a
paired comparison, not a golden match:

1. **Per-human steering streams** for `PlazaBehavior` (required for training; changes
   every plaza trajectory).
2. **Strict `GraphConfig`**: reject unknown keys everywhere, so a misspelled override
   cannot silently run the baseline.
3. **One velocity estimator**: choose between the two window rules.
4. **Obstacles in the mjlab scene**: the replay draws steering obstacles, but the mjlab
   room draws only the scanner mast, so any other obstacle would be invisible to the
   LiDAR.
5. **Seed formulas**: move library generation, benchmarks and the evaluator to one hashed
   `human_seeds` default.

The behavior work queued after them (committed turns biased toward 90°, a facing cost)
is in [crowd architecture](crowd-architecture.md#behavior-review).

## Invariants

- **Per-human randomness is independent of batch order and composition** (`AGENTS.md`):
  every draw is a function of the human's stream id and counter. Exception until the
  first deferred change: `PlazaBehavior`.
- **Layering** (`tests/unit/test_layering.py`): imports point to the same or a lower
  layer; only `sim.mjlab` imports mjlab at module level and only `backends` imports
  ARDY or Kimodo, so the behavior and graph code runs, and is tested, with neither.
- **Goldens are evidence, not targets.** A golden mismatch means behavior changed.
  Expected files are re-captured only for a deliberate, recorded behavior change.
- **Saved artifacts keep loading.** `library.pt` and `graph.pt` from earlier runs load
  without regeneration; the `graph_build` and saved-graph goldens check it.
- **Generators stay offline** (ledger D10): no motion-source call to ARDY or Kimodo inside
  training; `OnlineMotionSource` serves library generation, benchmarks and evaluation.
