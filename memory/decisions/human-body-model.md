# Human body model: which skeleton and mesh do simulated humans use?

Opened 2026-09-30. **Whose call:** empirical, then the user.

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
([learning paradigm](learning-paradigm.md)).

## What would decide it

The library generator (ARDY Core vs Kimodo SOMA; [library generator](library-generator.md)),
the learning paradigm, the LiDAR ray budget per environment, and a sensor-error
comparison of mesh vs capsules.
