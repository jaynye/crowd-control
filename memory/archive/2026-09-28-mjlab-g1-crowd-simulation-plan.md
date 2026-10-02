# G1 crowd simulation implementation plan

> Archived from `mjlab/docs/plans/g1_crowd_simulation.md` on 2026-10-01; historical. Links were adjusted to the archive location.

> **2026-09-28:** Human embodiment (section 2, and the planar finite-mass
> humans in section 3) is superseded by
> [kimodo_stochastic_humans.md](2026-09-28-mjlab-kimodo-stochastic-humans.md): Kimodo/SOMA
> motion baked offline, motion matching online, stochastic intent in-sim.
> The uv environment now lives at the crowd-control root.

Updated: 2026-09-28. The first primitive corridor fixture is implemented in
`src/mjlab/tasks/crowd/`, with `scripts/demos/crowd.py` for interactive inspection
and bounded scan/PNG export. The rest of this document describes the broader
design; timed MID-360 emulation, G1 integration, generated indoor layouts, and
mesh humans remain future work.

**Current sequence:** build variant (a), validate primitive/room raycasting and
contacts, then evaluate Kimodo directly as the preferred motion baseline for
variants (b)/(c). Do not train a custom diffusion model at this stage. Deforming
human-surface LiDAR support remains a separate requirement for mesh variants.

The primitive stage has CPU checks for ray agreement with native MuJoCo across
all seven human parts, moving-world isolation, wall occlusion, minimum range,
floor/ceiling hits, physical contacts, and stochastic-motion reset isolation.
A four-person five-second rollout produced rendered frames, scan arrays, and
contact histories. GPU throughput and a live browser session remain unverified.
See `docs/source/debugging/crowd.rst` for runnable commands and current limits.

Build a GPU batched crowd simulation in mjlab, then train one velocity-conditioned
G1 policy from random initialization. The deployed policy consumes proprioception,
velocity commands, and LiDAR history and produces joint targets. Crowd avoidance
belongs inside that policy. Human route selection belongs to the simulator.

The user confirmed a Livox MID-360 with a provisional mounting transform and a
preference for training end to end from scratch. A curriculum can change the
training distribution while optimizing the same policy throughout. SONIC is an
optional later experiment for motion priors or distillation: its published task
is motion tracking, and using it would require a reference-motion interface and
a separate comparison against training from scratch. See the
[SONIC project](https://nvlabs.github.io/GEAR-SONIC/).

The first deliverable is a reproducible corridor scene with observable, moving
humans, correct robot-human contact attribution, and MID-360-like scans. A trained
robot is not required to validate that deliverable: sensor fixtures, contact
fixtures, and crowd-only rollouts cover its behavior independently.

**1. Reuse the existing infrastructure and address its gaps.**

| Existing component | Planned use and required extension |
| --- | --- |
| `src/mjlab/tasks/velocity/config/g1/env_cfgs.py` | Reuse G1 configuration, actuators, locomotion rewards, and task registration patterns. Create a separate crowd task. |
| `src/mjlab/sensor/raycast_sensor.py` | Reuse GPU ray intersections. Add a timed MID-360 pattern and sensor output contract. |
| `src/mjlab/sensor/sensor_context.py`, `src/mjlab/sim/sim.py` | Reuse BVH refitting for moving geometry. Add sensor scheduling without changing other sensors' timing. |
| `src/mjlab/sensor/contact_sensor.py` | Match human subtrees against the robot; aggregate events over physics substeps. |
| `src/mjlab/scene/scene.py`, `src/mjlab/terrains/` | Compile primitives and existing terrain assets; add indoor layout metadata. |
| `src/mjlab/viewer/viser/`, `offscreen_renderer.py` | Interactive inspection and deterministic image artifacts. |
| `src/mjlab/managers/recorder_manager.py` | Save terminal events before automatic reset and episode provenance. |
| `scripts/slurm/viser_play.sbatch` | Build on the existing remote viewer launcher. |

Specific findings from this checkout:

- The original raycaster supports grid, pinhole, and downward ring patterns.
  The primitive fixture adds an angular raster; timed scans are still future
  work. Patterns are currently generated once.
- `sim.sense()` refits the geometry BVH, so there is an existing path for moving
  humans. A dedicated moving-target regression must verify freshness.
- Ray geometry IDs are now exposed in `RayCastData` for validation; keep semantic
  IDs out of actor observations.
- Current velocity-task timing is 5 ms physics and four substeps per action,
  yielding a 50 Hz policy. Sensing currently runs once per policy step.
- `EventManager` step events run after physics and reward calculation. They are
  unsuitable as the sole mechanism for updating moving humans during substeps.
- A contact sensor can expand its primary match to many elements, but its
  secondary filter matches a single element. `secondary_policy="any"` removes
  the filter if multiple elements match. It does not mean a union of humans.
- G1 already enables full collision geometry. Its visual meshes use group 2 and
  collision proxies group 3. Sensor groups and collision masks are separate.
- Viser already provides pause, stepping, contacts, and reward plots. Existing
  Slurm notes describe the tunnel, but a live CUDA/EGL session remains to be
  verified. See `memory/2026-09-26-slurm-viser-and-agentic-crowds.md`.

**2. Model humans with consistent geometry for perception and contact.**

Use a validation baseline composed of a head sphere, torso and pelvis shapes,
and separate arm and leg capsules. Parameterize stature, shoulder width, limb
dimensions, mass, and clothing thickness. Use a small catalog of compiled body
variants first; per-world geometry randomization requires verifying BVH updates,
inertia recomputation, and viewer synchronization.

The baseline uses the same primitive surfaces for LiDAR and collision. A torso,
head, and separated legs produce more useful occlusion and partial visibility
than a single cylinder. Preserve stable human and body-part IDs. A social comfort
radius is metadata, not an invisible physical collider or LiDAR surface.

Use three physical degrees of freedom per human on flat ground: two planar
translations and yaw, with finite mass and inertia. A bounded force/torque
controller follows a desired velocity. Contacts can alter actual motion, and
the behavior controller reads that actual state on its next update. Upright
orientation and floor height are constrained; these humans cannot fall and do
not model injury biomechanics. MuJoCo documents the planar joint construction
in its [modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html).

Keep a kinematic replay mode for sensor fixtures and comparisons. Do not use
teleported poses to infer physically meaningful impact forces: MuJoCo treats
mocap bodies as fixed during dynamics, omits their imposed motion from relative
contact velocity, and does not generate mocap-static or mocap-mocap contacts.
See [MuJoCo's mocap explanation](https://mujoco.readthedocs.io/en/stable/modeling.html#mocap-bodies).

Initially the limbs have a fixed pose. Add limited articulated gait and arm swing
after validating silhouette coverage and throughput, using bounded actuators so
physical and sensed poses agree. Detailed meshes can then replace primitive
LiDAR surfaces with matched collision proxies. Measure the discrepancy between
the two representations instead of assuming a visual mesh implies a collider.

Preallocate a maximum number of human slots per environment. Start with capacity
for eight and vary the active count. Inactive slots must neither collide nor
appear in scans; use verified per-world masking or isolated parking outside all
scene, ray, and contact bounds. Changing topology requires a new compiled batch.

The user proposed SMPL/SMPL-X surfaces driven by a generative gait model. This
is a viable perceptual target, and the primitive baseline need not become the
final training representation. The recommended mesh design uses a shared
skeleton and body shape to drive a deforming surface for LiDAR and articulated
capsule/convex proxies for physical contacts. SMPL represents body shape and pose;
SMPL-X adds articulated hands and facial expression. For this crowd task, begin
with SMPL unless the motion data already use another compatible body model.
See the [SMPL](https://smpl.is.tue.mpg.de/) and
[SMPL-X](https://smpl-x.is.tue.mpg.de/) project descriptions.

The main integration work is batched deformation and ray intersection. In the
installed Warp backend, the mesh ray path uses mesh acceleration structures
indexed by model mesh, while per-world geometry transforms move those meshes.
This does not expose independent SMPL deformation per world. MuJoCo skins are
visualization objects, and mjlab's current ray sensor does not provide a skin
intersection interface. Flex rendering support elsewhere in the backend does
not establish that this sensor can raycast arbitrary animated skins. See
[MuJoCo's skin/flex distinction](https://mujoco.readthedocs.io/en/stable/modeling.html#deformable-objects).

Prototype a Warp mesh-ray adapter that updates human vertices and refits their
triangle acceleration structures. Cast scene/robot rays and human-surface rays,
then select the nearest hit across both before applying sensor noise or dropout.
Exclude duplicate human collision proxies from the scene ray pass. Keep strict
world isolation and use the same acquisition times for both passes. Benchmark
full and decimated surfaces; preserve shoulders, limbs, and gaps between limbs
when simplifying. Update only the selected debug world's mesh in the viewer.

Root translation and gait must be coupled. Condition the proposed gait model on
local velocity, yaw rate, acceleration or a short intended trajectory, motion
history/phase, body shape, and a persistent style latent. Predict joint pose,
pelvis-height variation, and foot-contact state. Sample style or motion-segment
variation coherently instead of injecting independent joint noise each frame.
Lock stance feet in world coordinates with contact-aware correction/IK, release
them during swing, and permit natural pelvis bob. Setting a fixed root height
or translating an independently generated walk cycle does not prevent foot slip.
Include stop/start, turning, and sidestepping data, not only straight walking.

Use a small motion library to validate grounding and the mesh-ray pipeline, then
evaluate [Kimodo](https://github.com/nv-tlabs/kimodo) directly before considering
custom training. Verify the chosen checkpoint's skeleton and trajectory controls;
retarget explicitly rather than assuming generated poses are SMPL parameters.
Measure foot slip, penetration, trajectory tracking, transition continuity, pose
diversity, and throughput against clip playback. Also measure reconditioning
latency after abrupt changes of intent: a realistic long clip is insufficient if
the human cannot react promptly. Cached/generated motion windows are acceptable
for an initial integration, with persistent history and grounded transitions.
Route/intent changes remain governed by the crowd behavior layer. AMASS and the
locomotion subset of BONES-SEED remain options for later fine-tuning or training
only if the baseline leaves a measured gap.

For collision-event rewards alone, a prescribed animated human plus correctly
matched proxies can be sufficient; a learned human balance controller is not
required. Forces and reciprocal physical reactions need additional dynamics.
For that mode, drive an articulated proxy with bounded actuators and skin the
mesh from its actual pose, so contact displacement also moves the sensed human.
Measure proxy-to-surface error across poses and body shapes. Never use one convex
hull of the full human as the collider: it bridges gaps between limbs. Keep the
kinematic and dynamic contact modes explicitly labeled in experiments.

Before committing to meshes for large training batches, compare them with the
primitive baseline using the same trajectories and scan timings. Measure ray
coverage, silhouette differences, collision-surface mismatch, VRAM, and step
time. Body deformation, acceleration-structure refitting, and ray queries all
scale with crowd and environment count. Add coarse clothing/bag geometry later
because a parametric body surface alone omits those LiDAR-visible shapes.

**3. Give humans persistent intent with bounded stochastic changes.**

Represent each person with batched tensors for actual pose/velocity, route,
destination, desired speed, intent mode, mode timer, reaction delay, preferred
passing side, and cooperation level. Save independent random streams per
environment, episode, and human so resetting one world does not change another.

Use a semi-Markov state machine: modes last for sampled durations, with changes
also permitted at meaningful locations. Proposed initial modes are walking to a
destination, turning at a junction, waiting, yielding, resuming, and changing
destination. A seeded event schedule determines random choices. Small correlated
speed variations can operate within a mode; directions do not reset randomly
every frame.

Generate routes on the scene's room/portal graph, then follow corridor waypoints.
At junctions, sample among reachable destinations. Keep intended routes distinct
from actual trajectories so contacts and local avoidance can modify movement.
Model doorway hesitation, crossing traffic, overtaking, and stopping in a lane.
Add group walking and leader-following as a later behavior family.

Implement a vectorized local controller with goal attraction, anticipatory
avoidance based on time to closest approach, wall clearance, and acceleration
and yaw-rate limits. Preserve a passing-side preference during an encounter to
reduce oscillation. Validate swept motion against wall geometry and do not
silently teleport a stuck pedestrian. Report jams and overlaps as scene failures
or intentional congestion scenarios.

Use a mixture of attentive, delayed, and non-yielding pedestrians. Occlusion and
attention determine whether a human reacts to the robot; route knowledge alone
must not give humans awareness of the robot through a wall. Behavior choices
persist long enough to infer from observations, while turns, pauses, and reaction
delays supply uncertainty. Keep physically unavoidable encounters in a separately
labeled stress suite.

Include ORCA/RVO2 as a reference controller for small evaluation scenes if useful.
ORCA's reciprocity assumptions do not automatically hold for a learned G1 or a
non-yielding person, and clamping its output acceleration changes its guarantees.
It is a comparison and possible controller adapter, not a safety guarantee. See
the [authors' ORCA description](https://gamma-web.iacs.umd.edu/ORCA/).

Start with intent updates at 2-5 Hz, steering at 10-20 Hz, and force control at
every 200 Hz physics substep. These are tunable engineering defaults. Add a
small explicit substep lifecycle hook, inactive for existing tasks, for crowd
control and contact aggregation. Keep policy actions restricted to robot joints.
Test reset and terminal-event ordering alongside this integration.

**4. Implement a timed MID-360 sensor on the native raycasting backend.**

There are useful existing implementations:

- [MuJoCo-LiDAR-cpp](https://github.com/hansolo12334/MuJoCo-LiDAR-cpp)
  documents MID-360 scanning, dynamic geometry, CPU/Taichi backends, and a G1
  ROS2 example. Evaluate its scan generator and use it as a standalone reference.
  Its README does not establish compatibility with mjlab's batched Warp worlds;
  no runtime compatibility or accuracy has been validated here. Pin the source
  and review its license and scan-pattern provenance before reusing code.
- The [MID360 mjlab locomotion fork](https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR)
  describes ideal height scans, sparse-scan augmentation, and height-map memory.
  It is relevant to later terrain perception, but does not establish a timed
  crowd LiDAR simulator. Its
  [README](https://raw.githubusercontent.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR/main/README.md)
  makes that training approximation explicit.

Recommendation: extend mjlab's existing Warp raycaster, using external scan
patterns only after validation. This avoids introducing a second GPU simulator
backend into the training loop.

Use Livox's nominal specifications as the initial sensor configuration:
360-degree azimuth, elevation from -7 to +52 degrees in the sensor frame,
0.1 m near blind zone, 200,000 first-return points per second, and typical 10 Hz
cloud framing. Nominal range is 40 m at 10% reflectivity and 70 m at 80%; near
returns from 0.1-0.2 m have reduced accuracy. These are hardware specifications,
not claims that our approximation reproduces them. See
[Livox's specifications](https://www.livoxtech.com/mid-360/specs).

Attach a configurable site to G1's torso assembly with an explicitly provisional
translation and rotation. Use full body orientation, including pitch and roll.
The physical mount determines where the asymmetric elevation field points;
do not assume the sensor frame is aligned with the torso. Draw its axes and
frustum and later replace the transform using hardware calibration.

Define sensor outputs as ranges/points, validity, ray directions, acquisition
timestamps, publication timestamp, and sensor pose. Geometry/body IDs are debug
metadata. Required sensor behavior:

- Occlusion by people, walls, ceiling, obstacles, and robot surfaces, including
  returns from moving limbs. Raycast first, then classify self returns. Removing
  a self return must not reveal the object hidden behind the robot.
- An evolving non-repeating scan pattern, independent phase per environment,
  and an explicit scan clock. A uniform angular raster is only an early fixture.
  Validate any community pattern against recorded hardware directions later.
- A frozen-frame approximation for initial smoke tests, followed by packets
  accumulated at their acquisition times. In the validation mode, cast packet
  subsets against contemporaneous geometry at physics substeps. A moving person
  must not be rendered entirely at the end-of-scan pose.
- Configurable publication interval, transport/processing delay, dropout, range
  error, and extrinsic perturbations. Mark missing rays as unknown. Dropping a
  near or low-reflectivity return must not turn the occluded region into free space.
- Bounded history and ego-motion compensation using the same estimates intended
  for deployment. Do not use true human motion to deskew dynamic objects.

Offer a reduced-ray training mode and a dense validation mode with the same
interface. A 10 Hz scan at nominal point rate contains about 20,000 points.
Prototype 2,048-4,096 sampled rays per published scan for training, with correct
timing and coverage, then quantify the loss in human detection and reaction
distance relative to the dense mode. These ray budgets are hypotheses to profile.

The current sensor shares directions across worlds and computes every sensing
call. Extend it for per-world phase, timestamp buffers, and a clock/gate that
actually skips ray work. A cached output alone does not save computation.
Retain fixed tensor sizes for CUDA graphs. Add packet sensing through the
substep hook without forcing all other sensors to run at packet frequency.

Start without synthetic intensity as an actor feature. Reflectivity-dependent
dropout can be randomized, but material appearance alone does not calibrate
optical return strength. Glass and reflective materials require later validation.

**5. Attribute and penalize robot-human contacts explicitly.**

For the initial implementation, create one contact sensor per human slot, with
that human's subtree as primary and the G1 pelvis subtree as the single secondary.
Use `secondary_policy="error"`. This gives a stable human index and excludes
floor, walls, and other people. For detailed debugging, use raw contact geom
pairs to identify robot links and human body parts.

Configure collision masks for robot-human, robot-scene, human-wall/obstacle, and
human-human interactions. Human floor support is provided by the planar joints
in the first version. Check the resulting compiled masks and exclusions,
including the existing G1 self-collision configuration and contact dimensions.

Collect contact presence over every physics substep, event onset, duration,
peak compressive force, and impulse. Summing normal-force magnitudes over actual
contacts and time avoids cancellation; net force or only the strongest retained
contact is not total impulse. Begin with presence/duration and peak force; add
an exact raw-contact reduction before using total impulse in rewards.

Use a dominant contact cost, an additional severity/duration cost, a smooth
clearance penalty, and optionally an approach-speed/TTC penalty. Evaluate
clearance against the robot's body extent, including arms and feet, rather than
only pelvis-to-human center distance. Privileged geometry is permitted for
training rewards and critic inputs. Distinguish contact margins from active
physical contact; test grazing cases instead of relying on an arbitrary force
threshold alone.

Audit both pushing people aside to gain progress and freezing indefinitely to
avoid contact. Include humans colliding with a stationary robot. Preserve the
contact outcome before reset. Report all contacts even if training uses a
severity threshold for termination. Impulses from a simplified upright proxy
are simulation diagnostics, not injury estimates.

Increase and profile `nconmax`, `njmax`, and `contact_sensor_maxmatch` using dense
encounters. Treat contact-buffer overflow, missing matches, and tunneling as
invalid rollouts. Verify CPU MuJoCo and Warp contact behavior on a small fixture
before scaling.

**6. Generate indoor scenes through one reusable scene description.**

Define a versioned `IndoorSceneSpec` with metric geometry, coordinate convention,
collision and sensing assets, surface properties, walkable regions, connected
rooms and portals, spawn/exit regions, and route metadata. Both procedural
generation and future real2sim import produce this description.

Generate a connected floor plan first, then instantiate boxes and other
primitives for floors, ceilings, walls, doors, columns, and furnishings. Initial
families are a straight corridor, L corner, T junction, crossing, doorway, and
room connected to a corridor. Sample dimensions and obstacle placement subject
to route connectivity and body clearance. Validate spawn separation, reachable
exits, ceiling clearance, and boundaries. Rejection sampling has a bounded
number of attempts and reports failure explicitly.

Render ceilings and walls in physics and sensing. A viewer cutaway can hide the
ceiling for inspection without changing physical or LiDAR geometry. Maintain
separate IDs for walkable ground and walls so foot-ground reward filtering does
not silently count feet pressed against a wall as floor support.

MuJoCo-Warp batches share compiled topology. Begin with cached scene families
and fixed-capacity obstacle/human slots; vary safe per-world parameters and
spawns at reset. Recompile only between batches or runs. Avoid an unbounded
scene atlas: other tiles in the same world can appear within LiDAR range.
Enclosed bounds and explicit ray isolation tests must verify scene separation.

Reuse existing box stair/ramp generators later. Add friction patches, step
height, landing width, slope, and headroom to the same scene description.
Use floor/level-specific walkability; a global heightfield cannot represent a
ceiling over a floor or several stacked levels. The first planar humans stay
on level surfaces. Human stair traversal needs a later support-height/articulation
controller and explicit stair links; simply moving the planar proxy upward
would not validate walking or contacts on stairs.

For real2sim, reserve an importer for a registered mesh or point-cloud survey:
establish units and gravity, clean dynamic objects, reconstruct surfaces, derive
walkable layers/portals, and build collision proxies. Preserve detailed geometry
for sensing where affordable. Arbitrary point clouds or Gaussian splats are not
automatically collision surfaces. A concave room mesh needs decomposition or
primitive colliders because normal MuJoCo mesh collisions use convex geometry;
otherwise doorways can be sealed by the collision hull. See
[MuJoCo collision documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection).

Validate imported collision clearance against the sensing mesh and overlay real
and simulated scans at registered poses. Friction remains an explicit calibrated
or randomized property; it cannot be inferred reliably from geometry alone.

**7. Make visual and numerical inspection available from the first milestone.**

I will own implementation and iteration of human behavior as the main agent.
The simulator runs ordinary batched controllers; model interaction occurs during
development, scenario authoring, and failure investigation.

Build a small session API used by both a CLI and an optional local MCP adapter.
Codex supports local STDIO and Streamable HTTP MCP servers; see the
[official MCP documentation](https://developers.openai.com/codex/mcp).
Implement the CLI and artifact path first so inspection also works without an
MCP connection or attached browser.

| Proposed operation | Result |
| --- | --- |
| `load_scenario(name, seed)` / `reset()` | Scene/config hash, seed, session and episode IDs. |
| `advance(steps)` / `pause()` | Bounded stepping with simulation time and metrics. |
| `set_human_intent(id, goal, speed, mode, duration)` | Logged behavior override at a known step. |
| `inspect_human(id)` | Actual/desired velocity, mode, timer, route, neighbors, controller forces. |
| `capture_views()` | Synchronized overhead, oblique, and robot-view images with timestamps. |
| `capture_lidar()` | Sensor-frame cloud/range view, age/validity, and debug hit IDs. |
| `inspect_contacts()` | Geom pairs, points, normals, force, duration, and substep history. |
| `save_episode()` / `replay()` | Episode manifest, state/action/event traces, and visual artifacts. |

Queue mutations onto the simulation thread; HTTP/MCP callbacks must not step the
GPU concurrently. Every operation returns session ID, episode ID, step count,
and simulation time. Keep interactive edits isolated from evaluation rollouts.
Use a bounded debug session on a Slurm GPU and the existing tunnel approach.
Verify CUDA, EGL, and compute-node reachability with a short smoke job during
implementation.

Use Viser for live human inspection. Add routes, target velocities, actual
velocities, intent labels, social radii, clearance, LiDAR points, and contact
vectors. Use point-cloud drawing with subsampling rather than creating one
viewer object per LiDAR point. Highlight a selected person and allow a replay
to stop immediately before first contact.

Use `OffscreenRenderer` for agent images independent of browser clients. Export
PNGs/contact sheets that I can inspect with the local image tool, plus short
videos for human review. Show the same timestamp across RGB, LiDAR, and contact
plots. A saved contact must come from the live physics state: recomputing forward
dynamics for a screenshot does not reproduce the original solved contact forces.

Save scene spec and asset hashes, code/config versions, policy checkpoint/hash
when applicable, RNG states, human intents and overrides, physics state,
actions, sensor clocks/history, recurrent policy state, contacts, rewards, and
reset reasons. Distinguish state replay for visualization from resimulation.
Expect numerical tolerance in GPU resimulation rather than promising bitwise
identity across hardware or library versions.

The agent iteration loop is: run seeded scenarios, inspect images and traces,
identify a concrete failure, change one behavior parameter or algorithm, and
rerun the held-out scenario suite. Useful failures include doorway oscillation,
wall clipping, humans yielding through occluders, stale LiDAR hits, and missing
short contacts. Turn each confirmed defect into a focused regression.

**8. Integrate training from scratch after simulator validation.**

Create a proposed `Mjlab-Velocity-Crowd-Unitree-G1` task. Reuse the 50 Hz joint
target interface and PPO infrastructure. Initialize the entire actor and critic
randomly. Keep all actor components trainable as curriculum difficulty grows.

The actor receives velocity command, joint state, IMU-derived orientation and
angular velocity, previous action, and a compact temporal LiDAR representation.
Prototype an angular range image with validity and age plus a small recurrent
encoder; later compare a point encoder. Preserve elevation information for
terrain and overhangs. Audit the existing `imu_lin_vel` observation: a simulator
body-velocity sensor is not automatically a deployable IMU velocity estimate.
Choose an estimator or history-based substitute and train with its limitations.

The critic can additionally observe true human positions/velocities, contact
state, terrain, and friction. The actor receives no human IDs, destinations,
future intentions, routes, map, or ground-truth dynamic-object velocity.

Begin with standing and simple velocity tracking in an empty level scene, then
mix in stationary people, single predictable crossings, bidirectional traffic,
and stochastic/occluded encounters. Retain easier scenes to preserve locomotion.
Schedule density, cooperation, sensor degradation, and terrain difficulty
separately, advancing on locomotion and encounter metrics. Early severe collision
termination can suppress learning when balance is poor; introduce termination
rules with the curriculum and always log the collisions.

Velocity commands express desired local motion. During encounters, the reward
must allow temporary slowing and sidestepping while still discouraging permanent
stopping. Measure contact rate jointly with commanded-direction progress and
unobstructed tracking error. Retain explicit stand commands. A constant command
pointing into a wall may appropriately produce a stop; reaching an arbitrary
destination through a building requires a separate source of directional intent.
For corridor benchmarks, supply known command sequences from the scenario and
avoid quietly turning them into an obstacle-avoiding planner.

**9. Deliver in small stages with measurable acceptance criteria.**

| Stage | Deliverable | Acceptance gate |
| --- | --- | --- |
| 0 | One indoor fixture, one human, capture/trace CLI. | Headless images and state share timestamps; save/reload preserves geometry and IDs; no checkpoint needed. |
| 1 | Procedural humans, planar dynamics, collision instrumentation. | Stationary and moving contact fixtures identify the right person; substep contacts survive aggregation; bounded motion and finite forces; CPU/Warp comparison. |
| 2 | Timed MID-360 sensor and provisional mount. | Analytic distance checks, occluded-person and self-occlusion tests, moving-target freshness, packet timing, reset isolation, and dense/sparse comparison. |
| 3 | Seeded behavior and indoor scene families. | Replayable event choices; reachable routes; no invalid spawns or unexplained wall clipping; attentive/non-yielding mixtures; batch reset independence. |
| 4 | Viser overlays, episode replay, optional MCP adapter. | A failed crossing can be replayed, viewed from several cameras, and diagnosed from LiDAR/contact traces through tools. |
| 5 | Throughput benchmark and crowd task. | Memory/time breakdown for physics, steering, raycasting, and policy; no buffer overflows; actor observations audited for privileged information. |
| 6 | Training from scratch and held-out evaluation. | Compare contact, clearance, progress, falls, and response latency across multiple training seeds and unseen layouts/behaviors. |
| Later | Rough/slippery terrain, stairs, articulated humans, real2sim. | Each extension preserves sensing/contact alignment and passes its own traversal and performance checks. |

Stages 1 and 2 both build on stage 0. The minimal capture path is available at
stage 0; the richer interface at stage 4 must not delay basic visual debugging.

Profile batches of 1, 64, and 256 environments first, then scale if capacity
allows. Sweep 0/4/8 humans and reduced/dense LiDAR; record compilation time,
VRAM, environment steps/s, simulated seconds/s, sensor age, and p95 step time.
A nominal dense MID-360 across 4,096 worlds implies about 819 million ray samples
per simulated second. Hardware-scale scans should not be assumed affordable
without measurement. Keep rendering out of the bulk training loop.

The evaluation suite includes empty corridors, standing blockers, head-on
passing, perpendicular crossings, overtaking, doorway contention, blind corners,
sudden stops/turns, non-yielding people, sensor delay/dropout, and unseen scene
layouts. Split by scene families and behavior parameters, not just random seeds.
Measure contacts per encounter and per distance, peak force/contact duration,
minimum body clearance, progress, tracking error, waiting/deadlock time, falls,
and time from first observable conflict to robot response. Report uncertainty
over multiple seeds and separate physically unavoidable stress cases.

For fair comparisons, hold initial conditions and latent random events fixed.
Reactive pedestrians can legitimately follow different trajectories when the
robot changes behavior. Use a separate fixed-trajectory replay suite to isolate
robot reactions from that feedback. Compare against a velocity-only policy
trained from scratch under a matched budget and ablate LiDAR history and human
cooperation. Treat any future SONIC initialization as a separate experimental arm.

Proposed source layout:

```text
src/mjlab/asset_zoo/humans/        # Geometry and physical proxy configurations.
src/mjlab/tasks/crowd/            # Behavior, scenes, task terms, G1 configuration.
src/mjlab/sensor/lidar_sensor.py  # Timed LiDAR interface and MID-360 pattern.
src/mjlab/viewer/crowd_debug.py   # Shared overlays and captures.
scripts/demos/crowd.py           # Seeded, checkpoint-free simulator inspection.
scripts/tools/crowd_session.py   # CLI session/capture/replay interface.
scripts/tools/crowd_mcp.py       # Optional adapter over the same session API.
tests/test_crowd_*.py            # Focused behavior, contact, and layout fixtures.
tests/test_lidar_sensor.py       # Geometry, occlusion, timing, and reset fixtures.
```

Use `uv run` for Python tooling and tests. Run focused tests during development,
`make check` before any commit, and `make test` before a PR. Add user-facing
features to `docs/source/changelog.rst` under Upcoming with Added/Changed/Fixed
entries. This planning draft changes no simulator behavior and requires no
training run or code test suite.
