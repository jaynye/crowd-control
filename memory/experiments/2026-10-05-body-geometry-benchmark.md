# Human body geometry for LiDAR: cost and fidelity at training scale

Question (user, 2026-10-05): which geometry should be computed from graph joint
positions for LiDAR and robot contact, at 4,096 environments × 4–16 humans? The user's
rule: if a deforming (skinned) surface is efficient at runtime, mesh simplification
follows; otherwise use rigid mesh pieces, with about 32 morphologies loaded at sim init
and the graph changing only the pose. Feeds the
[human body model](../decisions/human-body-model.md) decision. Follow-up question
(user, 2026-10-06): a memory, step-cost and accuracy matrix over the number of humans,
rigid against skinned with the SAH tree.

## Outcome

Matrix on the A40 at 4,096 environments (G1 flat step 37.1 ms on the same GPU):
- Each cell is cost in ms per step, then × G1, then device memory in GB.
- Rigid pieces use the per-human kernel with 32 shapes.
- Skinned surfaces use an SAH-built tree; per-human shapes add nothing to their cost.
- "Off" is the share of human-touching rays whose range is more than 5 cm from
  full-resolution skinning. It was measured at 16 humans with one shape and is assumed
  independent of the number of humans.

| Geometry | Off | 4 humans | 8 humans | 16 humans |
|---|---|---|---|---|
| Capsules (one shape) | 41% | 16.3 · 0.44 · 1.9 | 22.1 · 0.60 · 2.2 | 32.6 · 0.88 · 3.0 |
| Rigid pieces, 1,600 faces | 9.0% | 18.0 · 0.48 · 1.8 | 26.1 · 0.70 · 1.9 | 41.5 · 1.12 · 2.3 |
| Rigid pieces, 19,384 faces (full) | 2.9% | 19.6 · 0.53 · 1.8 | 29.0 · 0.78 · 2.0 | 46.4 · 1.25 · 2.4 |
| Skinned, 600 faces | 11.1% | 17.1 · 0.46 · 3.1 | — | 45.3 · 1.22 · 7.5 |
| Skinned, 1,200 faces | 6.0% | 22.5 · 0.61 · 4.3 | 37.4 · 1.01 · 7.1 | 66.0 · 1.78 · 12.6 |
| Skinned, 2,500 faces | 2.4% | — | — | 108.7 · 2.93 · 23.7 |
| Skinned, 4,500 faces | 1.5% | 49.1 · 1.32 · 11.4 | 89.9 · 2.42 · 21.2 | out of memory (LBVH build; SAH not tried) |

- **Rigid pieces** are the cheapest geometry for their accuracy at every number of
  humans. Their memory stays flat (1.8–2.4 GB for the whole process), because the
  meshes are shared and only per-human poses grow. Full resolution costs 1.6 / 2.9 /
  4.9 ms more than 1,600 faces at 4 / 8 / 16 humans, so simplification is optional.
  Shape variety is limited to the loaded set, and 32 sets cost 2.8 ms (4 humans) to
  8–10 ms (16 humans) over one. The pieces leave seams at the joints.
- **Skinned surfaces** cost and hold memory in proportion to humans × faces: about
  130 bytes per triangle, Warp's build scratch included. Matching rigid's 2.9% takes
  2,500 faces (2.4%, 2.3× rigid's cost at 16 humans). The 4,500-face mesh (1.5%) costs
  2.5× rigid at 4 humans and 3.1× at 8. Per-human shape is free (continuous variety),
  and there are no seams. Simplification is required.
- **At 4 humans the costs converge.** Every geometry lies within 16–23 ms except
  skinned 4,500 faces, so the choice there rests on the other tradeoffs.
- **SAH tree for skinning:** it saves 19–25% against LBVH (16 humans: 56.1 → 45.3,
  86.4 → 66.0, 145.6 → 108.7 ms; 4 humans: 27.7 → 22.5, 64.9 → 49.1 ms). It is built
  at every sim init in about 0.41 s per million triangles (3.9 s for 600 faces at
  4 humans, 70.5 s for 2,500 faces at 16). The LBVH builds in under a second.

Rigid pieces at full resolution have exactly the reference's vertices. Their whole
error comes from vertices weighted to several bones, near the joints, plus the cap
faces. Median range error is 0.04 cm, 90th percentile 0.78 cm, 99th 7.7 cm. Skinning
at 2,500 faces gives 0.17 / 0.75 / 3.9 cm: the difference is in the tail, at the
joints. The reference is LBS skinning, which has its own joint artifacts.

No threshold is set here. Whether the skinned cost counts as "efficient" is the user's
call.

Rigid pieces as mujoco_warp mesh geoms cost 88–94 ms (2.4–2.5×) with one shape and
101 ms with 32 shapes at 16 humans, at any face count (38.0 ms at 4 humans). That
route needs no new code and gives convex-hull robot contact. The per-human kernels
need a raycast extension in our mjlab and separate contact geometry.

Every cost assumes a full LiDAR scan every 20 ms step. A 10 Hz scan would divide these
costs by about 5.

## Matrix follow-up run

- `runs/20261006T143625Z_body_geometry_matrix/`, from
  `configs/bench/body_geometry_matrix_a40.json`.
- A40 node spd-13, driver 580.178.04, job 4022257, 14:36–14:46 UTC.
- Same source fingerprint as the record run (`db339928…`). Three repeated record cases
  reproduce it within 0.04 ms: capsules 32.62, 32-shape full-resolution pieces 46.44,
  skinned 1,200 with SAH 65.98.
- All 17 cases ran.

Fidelity added in that run: skinned 2,500 faces gives 2.4% off (recall 0.989,
precision 0.994). The rigid 19,384 anchor reproduced 2.95%.

Skinned stage medians (refit / human ray queries, ms) and SAH build time (s):

| Humans | Faces | Refit | Ray queries | Build |
|---|---|---|---|---|
| 4 | 600 / 1,200 / 4,500 | 4.3 / 8.6 / 32.3 | 4.8 / 5.8 / 8.0 | 3.9 / 8.1 / 32.1 |
| 8 | 1,200 / 4,500 | 17.5 / 64.2 | 10.0 / 14.2 | 16.1 / 65.1 |
| 16 | 600 / 1,200 / 2,500 | 17.6 / 34.4 / 71.1 | 14.7 / 18.1 / 22.9 | 15.4 / 32.3 / 70.5 |

Rigid pieces' human ray queries take 10.6 / 17.7 / 30.9 ms at 4 / 8 / 16 humans with
1,600 faces, and 12.3 / 20.6 / 35.9 ms at full resolution.

## Adoption and integration timing (2026-10-06)

User decision (ledger D15): full-resolution rigid pieces, with no simplification,
are the plaza default. They are implemented as a per-human Warp ray target, posed with
bone rotations from artifact `graph-rotations`. The plaza environment at 4,096 envs on
an A40 (node nestor, where G1 takes 45.0 ms) gives these medians in ms per step, from
`runs/20261006T164941Z_sim_speed/`:

| Humans | Geometry | Graph stepping and posing | Sensing | Whole step |
|---|---|---|---|---|
| 16 | Capsules | 3.8 | 40.3 | 53.3 |
| 16 | Rigid pieces | 2.3 | 45.5 | 57.3 |
| 4 | Capsules | 1.3 | 25.4 | 29.4 |
| 4 | Rigid pieces | 0.8 | 25.8 | 29.3 |

- At 16 humans, rigid sensing costs 5.2 ms more than capsules. The benchmark measured
  +3.4 ms on another A40 node. At 4 humans the difference is 0.4 ms.
- Rigid posing runs as one fused Warp pass and is cheaper than capsule posing in torch.
  An earlier torch version cost 34.8 ms per step at 16 humans
  (`runs/20261006T164021Z_sim_speed/`).
- Checks:
  - the ray target's hits equal CPU MuJoCo `mj_ray` on the visual pieces (geom ids
    exact, ranges within 1 mm), on CPU and on an A40 with the captured sense graph;
  - the rotations file matches the graph's bone directions within 1 degree;
  - the swing correction stays near zero when inertialization is off.

## Results on the A40 (record)

Hardware and run settings:
- NVIDIA A40 (48 GB, node chappie), driver 580.178.04; torch 2.9.0+cu128, warp 1.14.0,
  mujoco_warp 3.11.0.
- Commit `5c1ef9c` plus the working tree. The source fingerprint is identical to the
  A5000 like-for-like run (`db339928…`).
- Job 4019717 ran 05:30–05:41 UTC. Its directory stamp, 03:02, is the submission time.
- 4,096 envs; 5,760 rays per env, 23.6M per step.
- All numbers are medians in ms per batched step from
  `runs/20261006T030259Z_body_geometry/`.
- "Cost" is sensing (CUDA graph) plus geometry inputs (torch pose, boxes, kinematics);
  the harness's own pose source is excluded.
- × G1 is the summary's "total / ref" column: the G1 flat step at 4,096 envs on an A40,
  37.09 ms (`runs/20260930T215734Z_sim_speed/rows.json`).

At 16 humans:

| Geometry | Faces per human | Shapes | Sense | Cost | × G1 | Range off by > 5 cm |
|---|---|---|---|---|---|---|
| Static scene only | — | — | 6.3 | — | — | — |
| Capsules (mujoco_warp) | 25 capsules | 1 | 29.7 | 32.6 | 0.88 | 41% (recall 0.89) |
| Rigid pieces, custom kernel | 892 (600 decimated) | 1 | 30.0 | 34.3 | 0.92 | 14.5% |
| Rigid pieces, custom kernel | 1,600 (1,200) | 1 | 29.5 | 33.7 | 0.91 | 9.0% |
| Rigid pieces, custom kernel | 5,260 (4,500) | 1 | 31.6 | 35.9 | 0.97 | 4.4% |
| Rigid pieces, custom kernel | 19,384 (full) | 1 | 31.8 | 36.0 | 0.97 | 2.9% |
| Rigid pieces, custom kernel | 1,600 | 32 | 37.2 | 41.5 | 1.12 | 9.0% (one shape) |
| Rigid pieces, custom kernel | 5,260 | 32 | 41.5 | 45.8 | 1.23 | 4.4% (one shape) |
| Rigid pieces, custom kernel | 19,384 | 32 | 42.2 | 46.5 | 1.25 | 2.9% (one shape) |
| Rigid pieces, mujoco_warp mesh geoms | 892–19,384 | 1 | 79.9–86.2 | 88.0–94.3 | 2.37–2.54 | same geometry |
| Rigid pieces, mujoco_warp mesh geoms | 1,600 | 32 | 92.4 | 100.6 | 2.71 | same geometry |
| Skinned (LBS + refit) | 600 | 1 / 32 | 50.2 / 50.2 | 56.1 / 55.9 | 1.51 | 11.1% |
| Skinned (LBS + refit) | 1,200 | 1 / 32 | 80.4 / 80.2 | 86.4 / 85.9 | 2.33 / 2.32 | 6.0% |
| Skinned, SAH-built tree | 1,200 | 1 | 60.0 | 66.0 | 1.78 | 6.0% |
| Skinned (LBS + refit) | 2,500 | 1 | 139.7 | 145.7 | 3.93 | not measured |
| Skinned (LBS + refit) | 4,500 / 18,152 | 1 | out of memory (48 GB) | — | — | 1.5% / reference |

The 1,600-face pieces' fidelity was measured through the custom kernel. The other piece
rows take it from the same pieces through mujoco_warp; the two routes differ by 0.2
points at 1,600 faces.

At 4 humans:

| Geometry | Faces per human | Sense | Cost | × G1 |
|---|---|---|---|---|
| Capsules | — | 15.5 | 16.4 | 0.44 |
| Rigid pieces, custom kernel | 1,600 | 14.2 | 15.2 | 0.41 |
| Rigid pieces, mujoco_warp | 1,600 | 36.1 | 38.0 | 1.02 |
| Skinned | 1,200 | 26.2 | 27.7 | 0.75 |
| Skinned | 4,500 | 63.4 | 64.9 | 1.75 |

Stage medians at 16 humans:
- **Skinned refit:** 26.2 / 50.9 / 101.7 ms at 600 / 1,200 / 2,500 faces, so linear in
  faces; 34.4 ms with the SAH tree at 1,200.
- **Skinned:** LBS takes 0.8–2.5 ms; human ray queries take 16.9 / 21.7 / 29.1 ms
  (18.1 with SAH).
- **Rigid pieces, human ray queries:**
  - 23.2 ms with one shape at 1,600 faces;
  - 30.9 ms with 32 shapes;
  - 35.9 ms with 32 shapes at 19,384 faces.
- **Device memory:** 7.5 / 12.6 / 23.7 GB for skinned at 600 / 1,200 / 2,500 faces;
  2.3 GB for pieces; 3.0 GB for capsules.
- **Per-piece mesh trees:** SAH instead of cubql made pieces slower (35.2 ms against
  33.7).

## Results on the A5000 (earlier set)

These were measured before the A40 run, on the same code. Do not compare their absolute
speeds with the A40 table: refit is about 1.45× slower on the A40, while ray queries
are faster. The ordering is the same on both GPUs.

- RTX A5000 (24 GB), driver 595.84; same versions as above.
- Numbers are from `runs/20261006T033355Z_body_geometry_a5000_like2/` unless marked.
- Denominator on that GPU: the G1 flat step is 42.9 ms
  (`runs/20261006T031153Z_sim_speed/rows.json`). The runs' `summary.md` "total / ref"
  column divides by the A40 G1 step and is invalid on this GPU.

| Geometry | Faces per human | Shapes | Sense | Cost | Range off by > 5 cm |
|---|---|---|---|---|---|
| Static scene only (build run) | — | — | 7.6 | — | — |
| Capsules (mujoco_warp) | 25 capsules | 1 | 35.6 | 37.9 | 41% |
| Rigid pieces, custom kernel | 892 / 1,600 / 5,260 / 19,384 | 1 | 37.6 / 36.1 / 39.2 / 39.2 | 41.7 / 40.2 / 43.3 / 43.3 | 14.5% / 9.2% / 4.4% / 2.9% |
| Rigid pieces, custom kernel | 1,600 / 5,260 / 19,384 | 32 | 45.6 / 50.7 / 51.5 | 49.7 / 54.8 / 55.6 | as one shape |
| Rigid pieces, mujoco_warp mesh geoms (build run) | 892–19,384 | 1 | 103–112 | +5.6 | same geometry |
| Skinned (LBS + refit) | 600 | 1 / 32 | 48.7 / 48.9 | 56.1 / 56.4 | 11.1% |
| Skinned (LBS + refit) | 1,200 | 1 / 32 | 73.7 / 74.3 | 81.2 / 81.7 | 6.0% |
| Skinned (LBS + refit) | 4,500 | 1 | out of memory (24 GB) | — | 1.5% |

Skinned refit on the A5000 took 17.8 ms at 600 faces and 35.9 ms at 1,200.

## Fidelity

The A40 run's `fidelity.json` reproduces
`runs/20261006T031133Z_body_geometry_a5000_fidelity/` to four digits:
- 128 envs × 16 humans, 8 frames, one shape;
- the A40 run adds the 1,600-face pieces through the custom kernel.

The fidelity column is the share of rays touching a human, in either set, whose range
differs by more than 5 cm from full-resolution skinning.
- Median range errors: rigid 18,152 0.04 cm; rigid 1,200 0.31 cm; skinned 1,200
  0.34 cm; capsules 2.3 cm (90th percentile 10.7 cm).
- Rigid ray hits sit slightly nearer than skinned ones (mean −0.2 to −1.5 cm), from the
  cap faces at the joints.
- Fidelity of the 32 scaled shapes was not measured; they are timing stand-ins, not
  morphologies.

## Findings

1. **Rays dominate.** At 23.6M rays per step, even capsules sense in 29.7 ms on the
   A40, close to a whole G1 step (37.1 ms); the static room alone costs 6.3 ms. On the
   A5000 these figures were 35.6 against 42.9 ms. The ray budget and scan rate are
   levers as large as the geometry: sensing cost is linear in rays.
2. **The scene-BVH build pose is not a drift risk.** mjlab's compile-pose SAH tree
   puts each human's bones at one spawn point, so it groups bones per human and stays
   compact as humans walk. Warp's GPU `rebuild()` only builds LBVH, and its trees were
   slower: on the A40, capsules cost 38.0 ms from the current placement and 44.4 ms
   from an independent one, against 32.6 ms at the compile pose. A true SAH rebuild
   from a spread pose was not tested.
3. **mujoco_warp's mesh path is the slow part, not the triangles.** Native rigid
   pieces cost 88–94 ms on the A40 at 892 or 19,384 faces alike. On the A5000 they
   cost 103–112 ms, and 95 ms in the scale run before the mesh-bounds fix. The same
   pieces through a per-human kernel cost 34–36 ms on the A40, with matching ranges
   (9.0% against 9.2% off at 1,600 faces; recall 0.990 against 0.989). That kernel
   tests a box per human, then queries each piece's mesh in its bone frame.
4. **Skinning cost is refit, and memory limits its resolution.**
   - One grouped Warp mesh (one subtree per human, built from a walking pose) refits
     in time linear in triangles: 26.2 / 50.9 / 101.7 ms at 600 / 1,200 / 2,500 faces
     on the A40.
   - An SAH-built tree (`constructor="sah"`, built once at init) refits in 34.4 ms
     instead of 50.9 at 1,200 faces, and queries slightly faster. The total drops from
     86.4 to 66.0 ms (−24%), against about 12% in the 32-env A5000 smoke. Across the
     matrix follow-up the SAH saving is 19–25%, at a build cost of about 0.41 s per
     million triangles at every sim init.
   - At 16 humans, 4,500 faces does not fit in 48 GB (A40) or 24 GB (A5000).
5. **Shape variety prices differently.** On the A40, 32 scaled piece sets (704 meshes)
   add 8–10 ms to rigid (9–12 ms on the A5000). Per-human skinned shapes add nothing
   (55.9 against 56.1 ms at 600 faces, 85.9 against 86.4 at 1,200), because only the
   rest vertices differ.
6. **The custom kernels' human boxes are conservative.** Each box bounds every joint
   segment grown by the farthest surface point that bone moves (`box_bounds`). It
   held every posed vertex across 205 library poses at both variant-scale extremes,
   and in a unit test. An earlier nearest-joint margin was also lossless, with
   identical skinned hit fractions with and without boxes.
7. **Cost scales sublinearly with the number of humans.** From 16 to 4 humans, the
   human part of sensing (sense minus the 6.3 ms static room) shrinks by 2.5× for
   capsules and mujoco_warp rigid, 3.0× for rigid pieces and 3.7× for skinned 1,200.

Bug found: mujoco_warp bounds a mesh geom by a box centred on the geom origin with
half the vertex extent. MuJoCo centres meshes on their centre of mass, so body pieces
poke out by up to 3 cm and grazing rays are culled. This affects any off-centre mesh
seen by mjlab's LiDAR, including robot meshes. The harness widens the bound to
`max(|min|, |max|)`; an mjlab fix in the sensor context is a candidate.

## Setup

- Body: ARDY's Core skin (`cskel27/skin_standard.npz`: 9,084 vertices, 18,152 faces,
  5 weights per vertex), on the library's own skeleton, so no retargeting.
  Login-node checks:
  - Bind transforms equal `joints.p` plus the 0.97 m hip height; bone lengths agree
    within 1.2 cm.
  - FK of the fp16 `local_rot` (re-orthonormalized) reproduces `posed_joints` to
    within 1.3 mm.
  - On frames the library flags as foot contact, the lowest skinned foot vertex sits
    at a median of −0.9 cm (95th percentile +1.5 cm); without contact it lifts to
    +1.8 cm.
- Code: `crowdmotion.sim.body` covers skinning, quadric decimation
  (`fast-simplification`, newly added at the root; torch 2.9.0 and warp 1.14.0
  unchanged) and rigid per-bone pieces: each face goes to the bone with the most
  weight, giving 22 pieces per human with holes capped. A CPU test shows MuJoCo-posed
  pieces match LBS for single-bone vertices. Harness: `crowd bench body-geometry`.
- Scene: plaza with walls and a scanner mast; 360 × 16 LiDAR rays from the mast
  (mjlab pattern). Humans play 64-frame library clips in place at random positions
  and headings, at least 0.8 m apart.
- Custom kernels (`pieces`, `skinned`) cull humans with per-human boxes (finding 6).
  Their human hits are merged with mujoco_warp's static-scene hits. Shape variants
  scale bone-local coordinates by 0.85–1.15, one per human from a seeded draw, for both
  routes.
- Timing: CUDA-event medians of 50 steps after 10 warm-up steps. Each stage is timed;
  the sense stages are also timed as one CUDA graph, as mjlab runs them. `synth` is
  the harness's own pose source (bank gather and placement), standing in for the
  graph, and is excluded from totals.
- Denominator, not a threshold: the G1 flat step on the same GPU, 37.1 ms on an A40
  and 42.9 ms on an A5000.

## Runs

- Record: `runs/20261006T030259Z_body_geometry/`, from
  `configs/bench/body_geometry_a40.json` (A40, job 4019717). 33 of 35 cases ran;
  `skinned4500_m16` and `skinned18152_m16` ran out of device memory.
- Matrix follow-up: `runs/20261006T143625Z_body_geometry_matrix/`, from
  `configs/bench/body_geometry_matrix_a40.json` (A40, job 4022257); all 17 cases ran.
- A5000 set:
  - `runs/20261006T033355Z_body_geometry_a5000_like2/` (like-for-like);
  - `runs/20261006T030545Z_body_geometry_a5000_build/` (build pose, native rigid);
  - `runs/20261006T031133Z_body_geometry_a5000_fidelity/`.
- Superseded: `runs/20261006T031749Z_body_geometry_a5000_like/` and
  `runs/20261006T030317Z_body_geometry_a5000_scale/`.
- Smoke: `configs/bench/body_geometry_smoke.json`. Job 4019649 (A40) was cancelled
  before it started.
- Smoke runs and failures are listed in the [run log](../log/runs.md).

## Known limits

- In-place playback: roots stay put, so drift is represented only by the
  independent-placement build.
- Skinned 600 faces at 8 humans, 2,500 faces at 4 and 8 humans, and 4,500 faces with
  SAH at 16 humans were not run. Fidelity was measured only at 16 humans, one shape.
- The skinned tree's topology is fixed from a walking pose and then refit; motions far
  from walking (sitting, bending) were not tested.
- Native MuJoCo flex was not tested; it would be a per-world triangle soup with a
  refit per flex.
- The saved graph carries no joint rotations, which both mesh routes need; the
  library stores them (fp16).
- Robot contact was not measured. MuJoCo collides mesh geoms as convex hulls.
  The custom LiDAR routes leave contact geometry to a separate set of geoms.
