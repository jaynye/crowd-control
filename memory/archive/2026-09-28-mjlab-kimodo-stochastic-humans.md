# Stochastic, reactive humans from an offline Kimodo model

> Archived from `mjlab/docs/plans/kimodo_stochastic_humans.md` on 2026-10-01; historical. Links were adjusted to the archive location.

Drafted 2026-09-28. This note supersedes the human-embodiment parts of
[g1_crowd_simulation.md](2026-09-28-mjlab-g1-crowd-simulation-plan.md) (section 2, and the planar
finite-mass humans in section 3). It keeps that plan's intent layer, LiDAR,
contact, scene, and training sections. The decision it proposes: **decisions
online, bodies offline**. Kimodo writes a motion database ahead of time. At
run time, a batched motion-matching layer inside mjlab turns online,
stochastic, robot-aware intent into SOMA body motion. Kimodo never runs in the
training loop.

## 1. The problem

Kimodo (Kimodo-SOMA-RP-v1.1) is a full-sequence diffusion model:

- 30 fps output, at most 10 s per prompt.
- A 100-step DDIM loop per sample.
- An 8B LLM2Vec text encoder (about 17 GB of VRAM when resident).
- Conditioning on text, dense or sparse 2D root paths, full-body keyframes, and
  end effectors.
- Continuation from a prefix of frames. This is how its multi-prompt mode
  stitches segments.

A training batch has thousands of worlds with up to eight people each, and the
simulation runs much faster than real time. People must make decisions
mid-episode: stop, turn, yield, change destination, react to the robot. Two
naive options fail:

- **Clip playback only.** Pre-generate each episode's motion and play it back.
  Every decision is fixed at reset, so people can never react to the robot or
  to each other.
- **Kimodo in the loop.** Regenerate on every decision. The measured numbers
  in section 7 rule this out for training. One L40S produces 7-28 seconds of
  motion per wall-clock second. A 4,096-world batch with 8 people consumes
  about 32,768 person-seconds per simulated second.

The only thing that must be online is the *decision*. The body motion that
realizes a decision can be looked up, provided the library covers everything
the decision layer can ask for.

## 2. Architecture

```text
offline (Kimodo + SOMA env, GPU batch jobs)         online (mjlab process, pure torch)
------------------------------------------------    ---------------------------------------------
intent-layer rollouts + scripted primitives         Intent layer (per env/human RNG streams)
   -> dense root2d paths (+ headings)                 semi-Markov modes, routes, TTC steering,
   -> Kimodo batches, cached text embeddings          robot awareness gated by attention/occlusion
   -> post-process, quality filter                      | desired trajectory: 3 future points
   -> bake: Z-up, features, bone transforms,            | + heading + action tag, 10-50 Hz
      capsule fits, tags, provenance manifest           v
               |                                    Motion runtime (motion matching)
               +--> motion_db.pt ------------------>  nearest-neighbor query, inertialized blend,
                                                      animation-owned root + bounded correction
                                                        | bone world transforms, contacts, velocity
                                                        v
                                                    Embodiment adapter
                                                      mocap capsule proxies (contact + LiDAR);
                                                      later skinned SOMA xlo mesh for LiDAR
```

Every source of stochasticity lives in the intent layer. It is ordinary
batched torch with private per-env, per-human RNG streams, as `CorridorWalk`
already implements. The motion runtime is deterministic given its inputs, and
it answers any intent change within one query interval.

### 2.1 Intent layer (online; this is where stochasticity lives)

This grows out of `tasks/crowd/mdp/walking.py`. Keep everything in
g1_crowd_simulation.md section 3: semi-Markov modes with sampled durations,
route graphs, TTC-based local avoidance, passing-side persistence, a mix of
attentive, delayed, and non-yielding people, and occlusion-gated awareness of
the robot. The one interface change is the output. Instead of a velocity
target for a planar actuator, each person emits:

- **A desired root trajectory** in their current root frame: 2D positions
  and facing directions at +0.33, +0.67, and +1.0 s. This is the standard
  motion-matching query, and it can express speed changes, stops, turns in
  place, sidesteps, and backsteps.
- **An action tag**: `locomote`, `idle`, `look_around`, `phone`, `startle`,
  and so on. Tags filter which database segments the runtime may match, so
  discrete non-locomotion behaviors are triggered by the decision layer too.
- **An optional style ID**, fixed per person per episode (brisk, elderly,
  distracted). This restricts matching to that style's slice of the library.
  Style persists for the whole episode, so the robot can infer it from
  observations.

The intent layer reads the **actual** root state every update, as
`CorridorWalk` does today. Divergence between intended and realized paths
therefore never accumulates.

### 2.2 Motion runtime (online; motion matching)

The database holds every baked frame `i` with a feature vector `x_i`:

- Left and right foot positions and velocities in the root frame.
- Hip velocity.
- The clip's own future root positions and directions at +0.33, +0.67, and
  +1.0 s.

This is about 27 dimensions, normalized per feature group, with tunable
weights. Each person keeps a playback cursor `(clip, frame)`. On each query:

1. Build the query vector: current pose features from the cursor, plus the
   intent layer's desired trajectory.
2. Take the nearest neighbor among frames allowed by the person's tag and
   style mask. The search is a batched `cdist` over a coarse cluster
   prefilter.
3. If the best frame improves the cost enough over continuing playback, jump
   to it, and start an **inertialization** blend: store the pose and velocity
   offset at the jump and decay it with a critically damped spring over about
   0.2 s.
4. Otherwise keep playing.

Queries run every 0.1-0.2 simulated seconds and immediately on any intent
event such as a mode change or a stop. Response latency is therefore one
query interval plus blend time, independent of clip length.

**The animation owns the root.** Root displacement per frame comes from the
matched Kimodo motion, which keeps stance feet planted. Two bounded
corrections pull it toward the intent:

- Yaw warping of at most ~30 deg/s.
- Speed scaling within ±10%.

Correction beyond those bounds is not applied. The intent layer sees the
residual on its next update and replans. The alternative, where steering owns
the root and animation is laid on top, produces foot skate. Astra's plan
already rejects that.

Output per person per policy step:

- World transforms of the bones used by the proxies: about 15-20 body
  joints of `somaskel77`, with no fingers.
- Root pose and velocity.
- Foot-contact labels.

Everything is batched as `[num_envs, max_humans, ...]`. The runtime should
be a few hundred lines of torch: database loading, feature normalization, cluster
prefilter, matching, inertialization, and root correction. It does not import
`kimodo` or `soma`. The database is a plain tensor file.

**Why motion matching and not a motion graph.** A graph of Kimodo clips joined
at canonical "hub" poses is simpler, and every transition in it is a genuine
Kimodo inbetween. Its drawbacks:

- Decisions are quantized to branch points, which are often 0.5-1 s apart.
- Continuous steering (arbitrary headings and curvatures around a moving
  robot) needs combinatorially many clips.
- A policy trained against fixed branch points can learn their timing.

Motion matching reacts within about 0.1-0.2 s to any continuous trajectory.
Discrete, clip-like actions still fit, as tag-filtered segments in the same
database. The graph remains a fallback if matching quality proves hard to
tune.

### 2.3 Offline library builder (Kimodo; never in the training loop)

The key idea is **coverage by construction**. Root paths for generation are
sampled from the same distribution the runtime will query:

- Roll out the intent layer offline in empty and cluttered layouts, without
  the robot. Also roll it out with a scripted "robot" obstacle, so yield and
  sidestep paths appear.
- Add scripted primitives:
  - starts and stops (0 to 1.6 m/s and back);
  - turns in place (±45, 90, 135, 180 deg);
  - arcs of 1-6 m radius;
  - speed ramps, sidesteps, backsteps, and S-curves;
  - standing idles.
- Cut these into 5-8 s windows and canonicalize: root at the XZ origin at
  frame 0, heading 0 facing +Z. These are Kimodo's conventions (Y-up, meters).

Generate with dense `Root2DConstraintSet` paths, plus headings where the path
implies facing (sidesteps and backsteps). Prompts come from a small style
vocabulary: "A person walks briskly", "An elderly person walks slowly", "A
person walks while looking at their phone", "A person stops and looks
around", and so on. Settings:

- Batch size 64-256.
- Post-processing on, for foot-skate cleanup and constraint enforcement.
- Text embeddings cached per prompt. There are only dozens of prompts, so the
  8B encoder runs once and is unloaded before bulk generation.
- Several seeds per path, for body-motion diversity.

Quality filters, with thresholds set against a held-out visual review:

- Root-path error.
- Foot slide during labeled contact.
- Floor penetration and floating.
- Jerk.
- Self-intersection of the xlo mesh (optional).

Bake to `motion_db.pt`:

- Z-up conversion (done once, here).
- Per-frame `somaskel77` local rotations and root. This is exactly what
  `Kimodo.__call__` returns; it converts from its internal `somaskel30`, and
  the result drives `SOMALayer` directly.
- Precomputed bone world transforms for the proxy bones.
- Matching features, foot contacts, tags, and style IDs.
- SOMA-derived capsule radii and lengths per bone, fitted to xlo mesh vertices
  by skinning weight.
- The xlo mesh plus skinning weights, reserved for mesh LiDAR.
- A manifest: Kimodo model and commit, SOMA version, prompts, seeds, source
  paths, filter statistics, and a content hash. Episode provenance records
  the database hash.

The library is not frozen. Offline generation is cheap (section 7): about
26 minutes for a 3-hour library. During training, log queries whose best
match cost exceeds a threshold, turn their trajectories into new root paths,
generate, and re-bake between runs. Coverage then follows the distribution
the policy actually induces.

The runtime interpolates the 30 fps database to the 50 Hz policy rate. Mocap
targets are interpolated per physics substep, so a person never teleports 3 cm
into the robot within one step.

**Body shape.** Kimodo generates on one canonical skeleton. Different SOMA
identities change bone lengths, which breaks foot contact if motion is simply
reapplied. The v1 approach:

- Vary **stature** by uniform scaling of the whole motion, including root
  translation. This preserves contacts exactly. Scale speeds by the same
  factor, which is also realistic.
- Vary **girth** through capsule radii only.

Full retargeting for non-uniform proportions is deferred.

### 2.4 Embodiment adapter (mjlab)

- A single `CrowdBodies` entity holds `max_humans` × about 15 capsule bodies,
  all mocap, with human and body-part IDs preserved. mjlab's
  `write_mocap_pose` writes one mocap root per entity. Add a small batched
  writer for `data.mocap_pos` and `data.mocap_quat` at a fixed index tensor.
- Inactive slots are parked outside every scene, ray, and contact bound. This
  is the plan's existing rule.
- Astra's geom-group LiDAR filtering, `min_distance` invalidation, and
  `geom_ids` debug output carry over unchanged.
- Per-human contact sensors need one change. MuJoCo requires mocap bodies to
  be direct children of the world body, so one person's capsules are
  siblings with no shared subtree, and `mode="subtree"` no longer isolates a
  person. Name bodies `human_{k}_{bone}` and use
  `ContactMatch(mode="body", pattern=r"human_3_.*", entity="crowd")`. The
  regex expands to one primary column per capsule; reduce over those columns
  for per-person presence and peak force. Keep the robot subtree as the
  single secondary.

Consequences of kinematic humans. These are stated explicitly and replace the
planar-dynamics design:

- People have infinite effective mass. The robot cannot push them aside, and
  impulses are diagnostics of the robot driving into a moving obstacle, not
  estimates of human reaction. Contact presence, duration, and peak force
  remain usable for rewards.
- MuJoCo creates no contacts between mocap bodies and static geometry, or
  between two mocap bodies. The intent layer must keep people out of walls
  and out of each other. Measure person-wall and person-person penetration as
  a scene-validity metric instead of relying on physics.
- If force-accurate interactions are later needed, drive free capsule bodies
  with soft weld constraints to the mocap targets, for evaluation only, and
  label that mode. It costs about 6 DoF per proxy body.

LiDAR on capsules is the v1 surface. It matches collisions exactly. A skinned
xlo SOMA mesh (612 vertices) is the planned high-fidelity LiDAR surface. It
needs a custom Warp ray kernel that culls rays with each person's bounding
capsule before testing triangles, because MuJoCo-Warp meshes are shared
across worlds and cannot deform per world. That work is independent of the
stochasticity design.

## 3. Backends behind one interface

```python
class HumanMotionBackend(Protocol):
  def reset(self, env_ids: Tensor, spawn: SpawnState) -> None: ...
  def step(self, intent: IntentBatch, dt: float) -> BodyState: ...
```

| Backend | Used for | Reactive? | Notes |
| --- | --- | --- | --- |
| `MotionMatchingBackend` | Training (default), most eval | Yes, ~0.1-0.2 s | Pure torch over `motion_db.pt`. |
| `ClipReplayBackend` | Fixed-trajectory eval suite | No | Per-episode Kimodo sequences baked offline. This is the plan's "hold latent events fixed" suite, and every frame is a genuine Kimodo sample. |
| `RecedingHorizonKimodo` | Small eval and demos | Yes, ~seconds | Separate GPU server process. It keeps a committed buffer longer than generation latency and requests continuations conditioned on the last 5 frames (full-body prefix constraint, as in Kimodo's multi-prompt transition) plus a new root path. Batches requests across people. |
| ARDY / MotionBricks (future) | Eval and demos | Yes | ARDY is NVIDIA's autoregressive, real-time Kimodo sibling with streaming text and constraint updates. As of 2026-09 it ships core (20 fps) and G1 (25 fps) skeletons only; SOMA is "coming soon". Adopt it when a SOMA checkpoint exists. |

A branching-future variant was considered: per episode, pre-generate a tree of
continuations at fixed decision times and pick branches online. Every frame
would be a genuine Kimodo sample, but storage grows exponentially, decision
times are fixed, and people cannot react to the robot's continuous position.
It is not recommended except as a curated evaluation set.

## 4. What changes in the existing crowd work

| Existing piece | Status |
| --- | --- |
| `asset_zoo/humans/primitives.py` (rigid primitive human on planar joints) | Superseded by SOMA-fitted capsule proxies. Keep it as a fixture for sensor tests until the adapter lands. |
| SMPL layer plus a custom generative gait model (plan section 2) | Superseded by Kimodo plus SOMA. No custom motion-model training. |
| `tasks/crowd/mdp/walking.py` (`CorridorWalk`) | Kept. It becomes the intent layer's seed. Its output changes from a velocity target to a trajectory query. Its per-env Park-Miller streams are the right pattern. |
| `LidarPatternCfg`, `min_distance`, `geom_ids`, viz stride | Kept unchanged. |
| Room fixture, per-human `ContactSensorCfg`, crowd demo and export script | Kept. The demo gains `--backend {primitive,mm}`. |
| Planar finite-mass humans with velocity actuators | Replaced by kinematic humans. Soft-weld dynamic proxies are an opt-in evaluation mode. |

## 5. Layout

```text
crowd-control/
  pyproject.toml, uv.lock, .venv/      # one env: mjlab + kimodo + py-soma-x
  crowdgen/                            # offline only; imports kimodo and soma
    paths.py      # intent rollouts + primitives -> canonical root paths
    generate.py   # batched Kimodo generation, embedding cache, seeds
    filters.py    # quality metrics and thresholds
    bake.py       # Z-up, features, bone transforms, capsule fits, manifest
  mjlab/src/mjlab/tasks/crowd/
    motion/database.py        # load motion_db.pt, normalization, clusters
    motion/matching.py        # batched query, inertialization, root correction
    motion/backend.py         # HumanMotionBackend + the three backends
    mdp/intent.py             # grows from walking.py; emits IntentBatch
    bodies.py                 # CrowdBodies entity + batched mocap writer
```

## 6. Milestones and acceptance gates

| Stage | Deliverable | Gate |
| --- | --- | --- |
| M0 | Unified env; Kimodo and SOMA smoke test on an L40S | Partly done 2026-09-28 (section 7). Done: weight loading, latency, dense-path tracking, and SOMA xlo skinning. Not done: text-conditioned generation (blocked on gated Llama-3 access) and any visual review of samples. |
| M1 | Library v0: about 30 min of locomotion and idles from scripted primitives | Root-path error, foot slide during contact, and penetration under set thresholds; visual review of a random 5% in Viser. |
| M2 | Standalone motion-matching runtime (no mjlab), driven by random intents | p95 intent-to-response latency under 0.3 s; transition foot slide; 32k people per step within budget on an L40S. |
| M3 | mjlab adapter: capsule proxies, substep interpolation | Astra's LiDAR and contact fixtures pass against MM humans; limbs are visible in scans; no mocap penetration spikes. |
| M4 | Intent layer v1 (modes, routes, robot-aware yielding) | Seeded replay; behavior statistics; wall and person penetration below threshold. |
| M5 | Library v1 from intent-layer rollouts, plus styles | Match-cost distribution on held-out intents shows no uncovered regions. |
| M6 | Eval backends: clip replay and receding-horizon Kimodo | Identical scenarios run under all three backends. |

The biggest unvalidated assumption is the motion-matching runtime itself.
Nothing measured so far covers its throughput, its intent-to-response
latency, or its transition quality at 32k people per step. M2 exists to test
this before any mjlab integration. If matching cannot be made clean, fall
back to the hub-pose motion graph described in section 2.2.

## 7. Measurements (M0, 2026-09-28)

These runs used `scripts/kimodo_smoke.py` (via `scripts/kimodo_smoke.sbatch`)
on one L40S (46 GB), with Kimodo-SOMA-RP-v1.1 and 5 s clips (150 frames). Text
features were zero (constraint-only sampling), because the gated Llama-3
encoder is not yet accessible. Denoiser cost does not depend on the prompt.
Raw results are in `crowd-control/logs/kimodo_smoke*.json` and
`logs/slurm/kimodo-smoke-*.out`.

| Steps | Batch | fp32 wall time | fp32 clip-s per wall-s | TF32 wall time | TF32 clip-s per wall-s | Peak GB |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 1 | 1.70 s | 2.9 | | | 1.2 |
| 100 | 16 | 21.2 s | 3.8 | | | 1.7 |
| 100 | 64 | 93.2 s | 3.4 | 45.8 s | 7.0 | 3.3 |
| 100 | 256 | 376 s | 3.4 | | | 9.5 |
| 25 | 1 | 0.45 s | 11.2 | | | 1.6 |
| 25 | 64 | 23.5 s | 13.6 | 11.6 s | 27.7 | 3.3 |
| 25 | 256 | 94.1 s | 13.6 | | | 9.5 |

- **Throughput is compute-bound.** It stops improving past batch 16, and TF32
  doubles it. Mixed precision is untested.
- **Offline cost.** A 3-hour library (10,800 clip-seconds) takes about 26 min
  on one L40S at 100 steps with TF32, before filtering losses. Offline
  generation is cheap.
- **Online cost.** Kimodo supplies 7-28 clip-seconds per wall-second per
  GPU. A 4,096-world × 8-person batch consumes about 32,768 human-seconds of
  motion per simulated second. Even at real-time simulation speed, that is
  3-4 orders of magnitude beyond one GPU. RL runs faster than real time, so
  the gap is larger.
- **Receding-horizon latency.** One 5 s continuation takes 0.45 s at 25 steps
  (fp32). A committed buffer of about 1 s makes the evaluation backend viable
  for a handful of people.
- **Dense root path.** A 90-degree, 3 m-radius arc at 1.2 m/s, batch 8,
  constraint-only, tracked with 1.5 cm mean and 8.5 cm max root error.
  Post-processing added almost no time (4.81 s vs 4.90 s) and did not reduce
  the max error (11 cm). The library filter should check root error directly.
  Labeled foot contact was about 40% of frames per channel. SOMA output has
  **6** contact channels, not the 4 the README lists; verify their order
  before baking.
- **Multi-prompt continuation** (2 × 5 s, batch 8, prefix-conditioned):
  9.7 s.
- **SOMA xlo skinning** (Warp 1.14, GPU, local `soma-x/` assets): 2,400 poses
  × 612 vertices in 24 ms, all finite. The mesh has 1,220 faces.
