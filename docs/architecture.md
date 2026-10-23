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
| L3 | `library/` | Offline library generation (`generate`; random or turning commands), turn checks of turning libraries (`turns`) |
| L3 | `bench/` | Throughput, continuation quality, graph speed, simulator speed, summary tables |
| L3 | `sim/mjlab/` | The mjlab integration: the `GraphCrowd` command term (`command`), queued live controls (`tuning`), the plaza environment (`plaza`), the robot training environment (`traversal`) and its training command (`train`), frozen SONIC on the G1 (`sonic_control`) and the planning arms' decision-rate env (`planning`), demos (`demo`), the corridor fixture (`corridor/`) |
| L3 | `tracking/` | W&B run continuity for training lineages: identity and the writer lock in `run.json` (`lineage`), evaluation outbox and ledger (`evals`, `attach_eval`), TensorBoard rows cut per attempt (`history`), `crowd wandb sync\|status` (`sync`). The live writer is `sim/mjlab/wandb_log` |
| L3 | `viz/` | Replay videos (`replay`), the graph audit report (`report`), sampled posture/height sheets (`edge_review`), stick-figure clips (`stick`), HTML templates |
| L2 | `behavior/` | The crowd loop: `controller`, motion `sources`, command sources (`commands`, `plaza` = `intent` + `steering`, `encounter`; `designs` builds either), `planner`, velocity estimators (`kinematics`) |
| L1 | `graph/` | The motion graph: `config`, `tables` (`MotionGraph`, `graph.pt`), `build` (kNN, multi-library loading), `costs`, `runtime`, `audit`, turning coverage `turns`, derived `annotations`, independent `posture` state, saved joint `rotations` (`rotations.pt`, for rigid mesh humans) |
| L1 | `backends/` | Online generator adapters: `base` (request/chunk contract), `registry` (typed configs, `make_backend`), `ardy`, `kimodo`, `text`; NVIDIA SONIC for the robot: the tracker, its plan buffer and the batched kinematic planner (`sonic`) over a batched ONNX executor (`onnx_port`) |
| L1 | `metrics/` | Locomotion direction classes, foot sliding, seams, posture, response times, spacing, graph diagnostics |
| L1 | `sim/scene`, `sim/capsules`, `sim/body`, `sim/rays` | The plaza scene builders, capsule humans, mesh humans (Core skin: skinning, decimation, rigid per-bone pieces, quaternion and swing helpers) and the Warp ray target for rigid humans, without mjlab |
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
| `crowd behavior evaluate` | `PlazaBehavior` or `EncounterBehavior` | `GraphMotionSource` | `BufferVelocity` |
| mjlab `GraphCrowd` | `PlazaBehavior` or `EncounterBehavior` | `GraphMotionSource` | `BufferVelocity` |
| `crowd library generate` | `RandomCommands`, or `TurnCommands` (also the planner; heading targets as `root_heading`) | `OnlineMotionSource` (ARDY) | `WindowVelocity` |
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
- `TurnCommands`: per-human semi-Markov stand / walk / in-place turn / arc segments for
  the turning library. Its `plan` method is the controller's planner (`planner.plan_arc`),
  and `motion_options()` hands per-frame body-heading targets to the motion source.
- `ScenarioCommands`: the quality benchmark's schedules (`cruise`, `stop_start`,
  `turn90`, `turn180`, `speed_step`), with per-human change offsets.
- `PlazaBehavior`: intent then steering, the separation the project keeps between
  deciding where to go and avoiding others:
  - `GoalIntent`: random goals inside the walls at a per-goal preferred speed; on
    arrival, sometimes wait;
  - `LocalSteering`: avoidance of neighbors by time to closest approach (with priority
    over the goal), a hard push at contact range, wall and obstacle repulsion.
- `EncounterBehavior` (the `encounter` design): waypoint or velocity conditioning,
  then a collision bubble with assertiveness- and distance-dependent switching into
  committed interventions (sidestep, veer, pivot, yield), then a stop rule. It plans
  with `planner.plan_arc`, so a gradual turn requests an arc of a chosen radius and a
  zero speed turns in place. Through `motion_options` it asks the graph to hold a body
  target during sidesteps and in-place turns. Its draws come from per-human counter
  streams. Usage: [behavior designs](usage/behavior-iteration.md#behavior-designs).

`behavior.designs.make_behavior` builds either design's command source and planner,
which is how the evaluator, `GraphCrowd` and the demos select a design from a config.

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
diffusion noise, `RandomCommands` and `TurnCommands` follow this rule.

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
(360 × 16 rays) and humans, by default rigid Core-skin meshes (`sim.body`), optionally
capsules (`sim.capsules`) or both. The `GraphCrowd` command term (`sim.mjlab.command`)
owns a `CrowdController`; each environment step it advances clocks, steps the
controller for environments whose edge ran out, interpolates frames to the current
time, converts them to MuJoCo Z-up and writes mocap poses. Because the crowd is a
command term, every mjlab viewer shows it.

Rigid humans pair the graph's joint positions with saved joint rotations
(`graph.rotations`, checked against the graph at load). Their pieces are visual geoms
outside the LiDAR's geom groups. The LiDAR sees them through a Warp ray target
(`sim.rays`): one mesh per piece shared by all humans, a box test per human, then
per-piece queries in bone frames. `GraphCrowd` registers it with mjlab's
`RayCastSensor.add_ray_target` (a general hook in our mjlab) and recaptures the sense
graph. The target merges by nearer hit, so `min_distance`, `max_distance` and geom ids
keep their meaning.

`sim.mjlab.traversal` is the robot training environment. It is mjlab's flat G1 velocity
task with several additions:

- The encounter plaza's walls and a `GraphCrowd` of rigid humans.
- `CrowdGoalCommand`, which keeps the velocity task's `twist` interface but recomputes
  the command every step toward a random goal in the plaza. It also places the robot,
  after the crowd resets, at the start farthest from the humans.
- A LiDAR on a provisional head site (`torso_link`). Its geom groups are only the
  floor, the walls and the humans' ray-target group, so the robot never sees itself.
- Robot–human clearance between robot spheres and human joint spheres, which drives
  the proximity and contact costs and the collision termination.
- Privileged nearest-human states for the critic.

The env has three observation groups: `actor` (the deployable LiDAR input), `teacher` (privileged nearest-human states and plaza pose in addition to the LiDAR; `--no-teacher-lidar` drops the LiDAR) and `critic` (both). Three tasks share it: PPO on `actor`, PPO on `teacher`, and a student that distills a teacher checkpoint into `actor` with RSL-RL's `Distillation`. The humans perceive the robot as an external agent of the encounter design (`GraphCrowdCfg.perceive_entity`), each oblivious, stubborn or cooperative toward it by the behavior config's `robot_*` settings; `--robot-blind` turns this off. `--crowd-curriculum RATE` (`traversal.crowd_difficulty_term`) moves those shares toward the robot-blind crowd during training, one step per iteration while the training collision rate stays at or below RATE: it edits the crowd's live encounter config, which each human reads at its reset, and its level is saved in checkpoints through mjlab's `CurriculumManager.state_dict`. `--forward-first` scales the goal command's speed by max(cos heading error, 0), and the `goal_heading` reward, (1 + cos heading error) / 2 at weight 1.0, keeps the robot facing its goal (without it the policy learns to walk backward). The goal command also logs episode means of the heading error, the backward-walking share and the speed. A collision ends the episode; the `collision_end` reward (weight 0, so off, by default) adds a terminal penalty on that step, since otherwise a collision costs only the forfeited return. `crowd train traversal` (`sim.mjlab.train`)
registers the task and runs mjlab's training inside a run directory. `crowd play traversal`
(`sim.mjlab.play`) replays a run's policy in the env its `run.json` records, in Viser, the
native viewer or an MP4. Its Viser viewer is live (`sim.mjlab.live_viewer`; what it offers
and what each switch needs is `sim.mjlab.live`): it builds the evaluator's env
(`evaluate.eval_env_cfg`) for a policy and crowd, switches either from the GUI by loading
in place, by a queued edit of the running crowd, or by rebuilding the env under the same
Viser server, and shows `TraversalMetrics` live.

### Planning arms

`--planner-rate 1|2` trains a policy over NVIDIA's released G1 stack instead of on joint
targets. `backends.sonic.SonicTracker` is SONIC's universal-token tracker (the release's
`model_encoder.onnx` and `model_decoder.onnx` as torch MLPs, equal to ONNX Runtime) and
`SonicPlanner` the kinematic planner (`planner_sonic.onnx`, batched by
`backends.onnx_port.OnnxPort` under `torch.vmap`, FP32). `PlanBuffer` keeps each env's
planned motion as the release's deploy code does (30 to 50 Hz resampling, an 8-frame
cross-fade of each new plan, context sampled from the planned motion, playback one frame per
control step). `sim.mjlab.sonic_control.SonicController` builds the tracker's observations
from mjlab's G1, which has SONIC's joint order, gains, default pose and action scale, and
returns the env's `joint_pos` actions. `sim.mjlab.planning.PlanningVecEnv` wraps the traversal
env as an RSL-RL env whose step is one decision: the action (3 numbers in [-1, 1]) is a
residual on the goal command, a planar velocity offset in the heading frame and a facing
offset, so that a zero action is plain goal following (P0). Each decision replans every env,
then the tracker runs 50 (1 Hz) or 25 (2 Hz) control steps; rewards are summed over them and
an env that ends mid-decision ends its transition there. Observations add the plan summary
(planned root positions 0.25, 0.5 and 1 s ahead, the plan's remaining time) and the previous
action. The env keeps only the task rewards (`traversal.PLANNER_REWARDS`); PPO counts in
decisions (16 s of rollout per env and iteration, gamma 0.75 and lambda 0.9 per second).
Teachers and students work as for the plain policy.

`sim.scene` builds the plaza for both mjlab and the replay renderer from one place. The
two presets differ deliberately: the replay draws steering obstacles as cylinders,
while the mjlab room draws only the scanner mast, so steering obstacles other than the
mast are not seen by the LiDAR.

The corridor fixture (`sim.mjlab.corridor`) is separate: physical primitive humans with
a waypoint intent, used for LiDAR and contact checks ([corridor](usage/corridor.md)).
