# Architecture

How the code is organized and how a human's motion is produced. Usage is in the
[guides](README.md#usage-guides); the rules for changing this structure are in
[development](development.md).

## Data flow

```text
offline (GPU)                                     online (any device)
─────────────                                     ───────────────────
ARDY H40  ──crowd library generate──▶ library.pt ──crowd graph build──▶ graph.pt
(random commands, closed loop)        (streams)    (kNN candidates)      (runtime tables)
                                                                            │
          PlazaBehavior ─▶ plan_path ─▶ GraphMotionSource ─▶ velocity ◀─────┘
          (intent, steering)  (root path)   (one 0.4 s edge)   update
                         └─────────── CrowdController ───────────┘
                                  │                          │
                    crowd behavior evaluate          mjlab GraphCrowd term
                    (traces, metrics, replay)        (capsule mocap, LiDAR, viewers)
```

ARDY and Kimodo never run inside training or simulation: they generate the library
offline. The graph stitches the library for thousands of humans at run time; the
behavior layer decides what each human asks the graph for.

## Layers

`src/crowdmotion/` is split into layers. A module imports only from its own layer or
lower ones (`tests/unit/test_layering.py` parses every import and checks this).

| Layer | Package | Holds |
|---|---|---|
| L4 | `cli` | The `crowd` command table |
| L3 | `iterate/` | Behavior evaluation (`evaluate`), paired comparison (`compare`), iteration sessions and their local and Slurm executors |
| L3 | `library/` | Offline library generation (`generate`) |
| L3 | `bench/` | Throughput, continuation quality, graph speed, simulator speed, summary tables |
| L3 | `sim/mjlab/` | The mjlab integration: the `GraphCrowd` command term (`command`), the plaza environment (`plaza`), demos (`demo`), the corridor fixture (`corridor/`) |
| L3 | `viz/` | Replay videos (`replay`), the graph audit report (`report`), stick-figure clips (`stick`), HTML templates |
| L2 | `behavior/` | The crowd loop: `controller`, motion `sources`, command sources (`commands`, `plaza` = `intent` + `steering`), `planner`, velocity estimators (`kinematics`) |
| L1 | `graph/` | The motion graph: `config`, `tables` (`MotionGraph`, `graph.pt`), `build` (kNN), `costs`, `runtime`, `audit` |
| L1 | `backends/` | Online generator adapters: `base` (request/chunk contract), `registry` (typed configs, `make_backend`), `ardy`, `kimodo`, `text` |
| L1 | `metrics/` | Locomotion direction classes, foot sliding, seams, posture, response times, spacing, graph diagnostics |
| L1 | `sim/scene`, `sim/capsules` | The plaza scene builders and capsule humans, without mjlab |
| L0 | `core/` | `rng` (per-human streams), `geometry`, `skeleton`, `trace`, `config`, `paths` (artifacts), `rundir`, `provenance`, `timing` |

Two further import rules: only `sim.mjlab` imports mjlab at module level, and only
`backends` imports ARDY or Kimodo. Everything else runs with neither installed.

## The crowd loop

Every consumer of human motion drives it through one class,
`behavior.controller.CrowdController`. Each `advance(t)`:

1. asks a **command source** for each human's `(speed, heading)` from its current
   kinematics;
2. **plans** a world root path from the executed state toward that command
   (`behavior.planner.plan_path`: the velocity blends to the commanded one over 0.5 s);
3. asks a **motion source** for the frames that follow the path;
4. commits the frames that were executed and updates each human's root, heading and
   **velocity estimate**.

Humans are rows `[H]`. With `group = M`, rows are environment-major (`env * M + i`) and
`advance(t, groups=env_ids)` steps only those environments, as mjlab needs.

| Consumer | Command source | Motion source | Velocity |
|---|---|---|---|
| `crowd behavior evaluate` | `PlazaBehavior` | `GraphMotionSource` | `BufferVelocity` |
| mjlab `GraphCrowd` | `PlazaBehavior` | `GraphMotionSource` | `BufferVelocity` |
| `crowd library generate` | `RandomCommands` | `OnlineMotionSource` (ARDY) | `WindowVelocity` |
| `crowd bench quality` | `ScenarioCommands` | `GraphMotionSource` or `OnlineMotionSource` | `WindowVelocity` |
| `crowd bench throughput` | random arcs (`planner.random_arc_paths`) | `OnlineMotionSource` | (no planner) |

The evaluator and `GraphCrowd` share the same loop, so a rule added to a command source,
the planner or the graph runtime changes both the measured behavior and the simulated
crowd. They differ only in what surrounds the loop: mjlab interpolates the 20 fps frames
to its 50 Hz step and writes mocap poses.

### Command sources

A command source implements `command(t, kinematics, groups) -> (speed [H], heading [H])`.

- `RandomCommands`: per-human semi-Markov segments (stand, or walk at a sampled speed
  and heading change, optionally curving), from each human's own NumPy generator.
- `ScenarioCommands`: the quality benchmark's schedules (`cruise`, `stop_start`,
  `turn90`, `turn180`, `speed_step`), with per-human change offsets.
- `PlazaBehavior`: intent then steering, the separation the project keeps between
  deciding where to go and avoiding others:
  - `GoalIntent`: random goals inside the walls at a per-goal preferred speed; on
    arrival, sometimes wait;
  - `LocalSteering`: avoidance of neighbors by time to closest approach (with priority
    over the goal), a hard push at contact range, wall and obstacle repulsion.

### Motion sources

A motion source turns requested root paths into body motion:
`reset(seeds, root_xz, heading)`, `step(state, path) -> MotionStep`,
`commit(state, step, take)`; `path_frames` says how much path it reads.

- `GraphMotionSource` plays one graph edge (8 frames, 0.4 s at 20 fps) per step. The
  `MotionStep` carries world joints, root, heading and contacts, plus the runtime's
  per-human diagnostics in `extras`.
- `OnlineMotionSource` calls an online backend (ARDY or Kimodo) and keeps the history of
  **executed** features, so only frames that were actually played condition the next
  call.

### Velocity estimators

`WindowVelocity` (displacement over the last `min(5, n − 1)` executed frames) and
`BufferVelocity` (a fixed six-frame buffer seeded with the start position) differ only
on the first edges after a reset. Both exist because two code paths grew them; both
are kept so that saved results reproduce exactly.

## The motion graph

A library of N streams × T frames is cut into edges of `L` frames (8). A **boundary**
(node) is the frame before an edge; its descriptor is the root-relative, heading-aligned
joint positions plus `vel_weight` × joint velocities. A boundary's **candidates** are the
outgoing edges of its `neighbors` nearest boundaries; its own successor is a candidate
at distance 0, so a human that keeps following it replays the library exactly.

`graph.runtime.GraphRuntime.step` advances every human by one edge in three stages:

1. **select** (`select`): score each candidate with the cost terms, add per-human
   Gumbel noise from `(seed, counter)` at `temperature`, take the best;
2. **correct** (`correct`): optionally bend the chosen edge's root path toward the
   request, by at most `correct_yaw_deg` of yaw and `correct_speed` of speed;
3. **inertialize** (`inertialize`): decay the offset between the human's current pose
   and the edge's start pose over the edge.

**Cost terms** (`graph/costs.py`) each map a `CandidateContext` (the candidates, their
root paths placed on the human, the request) to a cost per candidate. The total is the
weighted sum in `TERMS` order:

```text
cost = trajectory + pose_weight · pose_distance + end_vel_weight · end_velocity_error
```

Each weight is a `GraphConfig` field named by the term. The fixed order keeps floating-
point results identical; a new term is appended, with a zero default weight that skips
it ([development](development.md#adding-a-graph-cost-term)).

`GraphConfig` fields `neighbors`, `vel_weight` and `hunch_ratio` are baked into the saved
tables; the others may be overridden when a saved graph loads.

## Online backends

`backends.base` defines one batched call: a `MotionRequest` (executed history, root
path, per-human seeds and counters, fresh-start state, text features) returns a
`MotionChunk` of **new frames only**. `backends.registry` holds the typed configs
(`ArdyConfig`, `KimodoConfig`) and `make_backend(config)`; both adapters expose
`generate(request)`. Adding a generator means one config class, one adapter and one
`BACKENDS` row. Details: [benchmarks](usage/benchmarks.md#backend-contract).

## Randomness

Per-human randomness is counter-based: draw `counter` of human stream `seed` depends on
nothing else (`core.rng`: SplitMix64 hashing, per-row torch generators), so results do
not depend on batch size, order or composition. Graph sampling, start boundaries,
diffusion noise and `RandomCommands` follow this rule.

Stream ids come from `core.rng.human_seeds`, whose named schemes are the formulas
existing outputs were made with (`quality`, `stream`, `evaluator`, `graphcrowd`).

**Known exception:** `PlazaBehavior` draws goals, speeds, waits and start states from
one generator shared by all environments and humans, in call order. It is kept that way
so evaluation baselines reproduce; switching it to per-human streams is a planned,
deliberate behavior change.

## Formats

- `core.trace.Trace`: executed motion of a group of humans (joints, planar root, body
  heading, contacts, fps, foot indices, extras). The evaluator's `seed_<n>.npz` is a
  saved `Trace`; `Trace.load` also reads the quality benchmark's `clips_*.npz`.
- `core.skeleton.SkeletonInfo`: joint names, parents, foot and head indices, neutral head
  height and a rest pose, from library or graph metadata or an upstream skeleton.
- `library.pt` and `graph.pt`: see [library and graph](usage/library-and-graph.md).

## mjlab integration

`sim.mjlab.plaza.make_plaza_env_cfg` builds a kinematic plaza: walls, a fixed LiDAR rig
(360 × 16 rays) and capsule humans (one mocap body per bone, `sim.capsules`). The
`GraphCrowd` command term (`sim.mjlab.command`) owns a `CrowdController`; each
environment step it advances clocks, steps the controller for environments whose edge
ran out, interpolates frames to the current time, converts them to MuJoCo Z-up and
writes mocap poses. Because the crowd is a command term, every mjlab viewer shows it.

`sim.scene` builds the plaza for both mjlab and the replay renderer from one place. The
two presets differ deliberately: the replay draws steering obstacles as cylinders,
while the mjlab room draws only the scanner mast, so steering obstacles other than the
mast are not seen by the LiDAR.

The corridor fixture (`sim.mjlab.corridor`) is separate: physical primitive humans with
a waypoint intent, used for LiDAR and contact checks ([corridor](usage/corridor.md)).
