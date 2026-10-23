# Human body model: which skeleton and mesh do simulated humans use?

Opened 2026-09-30. **Whose call:** empirical, then the user.

**Partly settled (2026-10-06, ledger D15):** LiDAR geometry is full-resolution rigid
Core-skin pieces, one per bone, with no simplification (user). They are cast through a
per-human Warp ray target (`sim.rays`) and posed with saved graph joint rotations
(artifact `graph-rotations`). They are the plaza default. Still open: the morphology
set, robot contact geometry, and the ray budget and scan rate (below).

## Question

The two backends use different skeletons. What body model and geometry do the humans
use for LiDAR and contact?

- **Kimodo** outputs SOMA. Internally it uses `somaskel30`; the API converts to
  `somaskel77`, which drives `SOMALayer` directly. The SOMA-X `xlo` mesh has 612 vertices
  and 1,220 faces (2026-09-28 smoke).
- **ARDY** released weights use the 27-joint Core skeleton (`cskel27`). The repository
  ships a Core skin (`ardy/ardy/assets/skeletons/cskel27/skin_standard.npz`) with 9,084
  vertices, 18,152 faces and 5 LBS weights per vertex (A40 smoke, 2026-09-30). That is 15×
  the `xlo` vertex count. The repository also carries a `somaskel77` skin and a
  `somaskel30` fit file, for the unreleased SOMA model.

## Options

1. SOMA `xlo` or `lo` mesh (Kimodo, or a future ARDY-SOMA).
2. ARDY Core skin, decimated to a LiDAR budget.
3. Retarget Core → SOMA and skin with SOMA. No retargeting path has been checked.
4. Capsule proxies fitted per bone for LiDAR and contact (the 2026-09-28 proposal's v1),
   with a mesh only for high-fidelity LiDAR later.

Keep LiDAR geometry distinct from the physics collision representation (handoff §6).
Validate simplified meshes against sensor error and missed collisions.

## Scale (user, 2026-09-30)

4,096 environments with m = 4–16 humans is 16,384–65,536 humans (ledger D09). At that
scale the 9,084-vertex Core skin is 149M–595M vertices per posed step, against 10M–40M
for SOMA `xlo`. For RL raycasting this favours capsule proxies or `xlo`-class meshes. An
offline TANGO-style renderer can afford full meshes
([learning paradigm](../archive/2026-10-12-decision-learning-paradigm.md)).

## What would decide it

The library generator (ARDY Core vs Kimodo SOMA; [library generator](library-generator.md)),
the learning paradigm, the LiDAR ray budget per environment, and a sensor-error
comparison of mesh vs capsules.

## Evidence (2026-10-06)

User direction (2026-10-05): the user wants varied morphology. If a deforming surface
is efficient at runtime, it gets simplified; otherwise rigid meshes are used, with
about 32 morphologies loaded at sim init and the graph changing only the pose.

The [benchmark](../experiments/2026-10-05-body-geometry-benchmark.md) compared
geometries at 4,096 environments, ray cast by per-human kernels, on an A40 (G1 step
37.1 ms; an earlier A5000 set gives the same ordering).
- Each cell is sensing plus geometry inputs in ms per step, then device memory in GB.
- Rigid pieces have 32 shapes; skinned surfaces use an SAH tree and per-human shapes
  for free.
- "Off" is the share of human-touching rays more than 5 cm from full-resolution
  skinning.

| Geometry | Off | 4 humans | 8 humans | 16 humans |
|---|---|---|---|---|
| Capsules (one shape) | 41% | 16.3 · 1.9 | 22.1 · 2.2 | 32.6 · 3.0 |
| Rigid Core-skin pieces, 1,600 faces | 9.0% | 18.0 · 1.8 | 26.1 · 1.9 | 41.5 · 2.3 |
| Rigid Core-skin pieces, 19,384 faces | 2.9% | 19.6 · 1.8 | 29.0 · 2.0 | 46.4 · 2.4 |
| Skinned, 600 faces | 11.1% | 17.1 · 3.1 | — | 45.3 · 7.5 |
| Skinned, 1,200 faces | 6.0% | 22.5 · 4.3 | 37.4 · 7.1 | 66.0 · 12.6 |
| Skinned, 2,500 faces | 2.4% | — | — | 108.7 · 23.7 |
| Skinned, 4,500 faces | 1.5% | 49.1 · 11.4 | 89.9 · 21.2 | out of memory |

The same rigid pieces through mujoco_warp's mesh path cost 88–101 ms at 16 humans.

Rigid pieces are the cheapest geometry for their accuracy at every number of humans.
Their memory stays flat, because the meshes are shared. Skinned cost and memory grow
with humans × faces (about 130 bytes per triangle), and the SAH tree takes about
0.41 s per million triangles to build at every sim init (up to 70 s).

Matching rigid's 2.9% floor takes skinning at 2,500 faces: 2.3× rigid's cost at 16
humans. The floor comes from vertices weighted to several bones, near the joints, and
from the cap faces. At 4 humans every geometry except skinned 4,500 costs 16–23 ms, so
the choice there rests on non-cost tradeoffs:
- continuous per-human shape and no seams, for skinning;
- flat memory, no build time, and contact hulls that match the LiDAR surface, for
  rigid pieces.

The user's rule decides whether skinning is efficient enough. With rigid pieces,
simplification is optional (about 5 ms at 16 humans).

Settled 2026-10-06 (D15):
- **Rigid or skinned:** rigid, full resolution (user).
- **Integration route:** the per-human ray target, the route whose costs the user
  accepted (agent, implementing the choice). Native mesh geoms (2.7–3.1× capsules)
  were not used.
- **Joint rotations:** saved per graph frame by `crowd graph rotations`, checked
  against the graph at load.

Still open for the user:
- **Morphology source.** The Core skin has one body shape. The benchmark's 32 variants
  were timing-only scaled copies and are not used. A real set needs SOMA-X identities
  refit to the Core skeleton, or retargeting. The ray target already takes per-piece
  meshes; shape variants would add one mesh set per shape (measured at +8–10 ms for 32
  at 16 humans).
- **Contact geometry.** The rigid pieces are visual only (`contype` 0), and the plaza
  disables contacts. Convex hulls of the same pieces are a candidate robot-contact
  proxy that matches what the LiDAR sees; contact cost is unmeasured.
- **Ray budget and scan rate.** At 23.6M rays per step, even capsules cost 0.88 of a
  G1 step. All costs assume a scan every 20 ms step; 10 Hz would divide them by about
  5.
