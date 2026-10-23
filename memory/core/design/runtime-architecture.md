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

## Training lineages and W&B

Ledger D20 (user, 2026-10-18); code in `crowdmotion.tracking` and `sim.mjlab.wandb_log`,
usage in [CLI](../../../docs/cli.md#train) and [Slurm](../../../docs/slurm.md).

- **Local files are the record; W&B mirrors them.** A lineage is one run directory that
  its attempts resume in place, and one W&B run whose id is the directory's name.
- **Rows go out only through the last saved checkpoint.** mjlab does not checkpoint env
  state, so a resumed attempt reruns the iterations after its checkpoint with different
  rollouts. Publishing live and skipping replayed steps (hrc) is sound only for
  bit-deterministic training. The cost is a lag of one save interval (100 iterations,
  about 5.5 min at 4,096 envs).
- **The writer runs inside the trainer** (RSL-RL's dict logger), not as a mirror process:
  hrc's mirror was unsignalled and killed at requeue, which is where its failures came
  from. The flush sits in the runners' `save`, because mjlab calls the writer's
  `save_model` only with `upload_model`.
- **Catch-up from TensorBoard** takes each iteration from the attempt whose trajectory
  continued: an attempt contributes rows only up to its own last checkpoint, cut where a
  later attempt starts, so one that died before its first save contributes nothing
  (review finding 3). The floor is `published_through`, never W&B's summary, which an
  evaluation row overwrites. There is no W&B offline mode: wandb 0.30 ignores `resume`
  offline.
- **Published means confirmed** (ledger D20). `log()` only queues, so the floor and the
  eval ledger advance only after `finish()` returns within the SDK's `finish_timeout`.
  A session that ends otherwise stays `unconfirmed`; the next holder reads the run's
  history back (`scan_history`, bounded) and replays only what W&B lacks. Rows above a
  gap are kept in `published_above`; committed rows that neither TensorBoard nor W&B
  holds are recorded in `lost`, and TensorBoard is flushed at every checkpoint so that
  this stays rare. wandb 0.30's default `reinit` returns a still-active run, so every
  session is finished before its lock is released and opened with
  `reinit="finish_previous"` and an identity check.
- **One writer per W&B run** is a flock on `<run>/wandb.lock`. On `/coc/flash7` (NFSv4.2,
  `local_lock=none`) it is a server-side lock that every node sees: while a compute node
  held it, an attempt from sky1 was refused (CPU job 4133247, 2026-10-18). A process
  drops such a lock when it closes any descriptor of the file, so the holder never
  reopens it.
- **Requeue.** This cluster requeues preempted overcap jobs itself (`PreemptMode=REQUEUE`,
  `GraceTime=300`), and a cancelled job must stay cancelled. So the job requeues itself
  only after a TERM near its time limit, and only once the trainer has exited
  `checkpointed`.
- **Verified** by the GPU check `runs/20261018T153936Z_train_traversal_wandb_check` (job
  4133159; requeue and in-place resume, W&B off) and by the W&B canary
  `runs/20261019T012044Z_train_traversal_TEST_canary_debug` (job 4138449, scratch project
  `crowd-control-traversal-debug`, user-approved 2026-10-19): TERM at iteration 24,
  requeue, resume in place to 39, an evaluation attached at 39; W&B holds training
  iterations 0–39 exactly once on the step metric `iteration`, one run id over three
  sessions. Only the fake SDK has exercised a finish timeout, a kill mid-session
  (reconciliation) and a partial media upload. `DEFAULT_MODE` is `online` since
  2026-10-21 (user); lineages created before keep their recorded `disabled`.

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
