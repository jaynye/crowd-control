# External knowledge

References that inform the project without being part of it: upstream repositories and
models, their APIs and traps, and papers. Project design, results and status live in
[core](../core/README.md); undecided questions in [decisions](../decisions/README.md).

| Note | Scope |
|---|---|
| [ARDY](ardy.md) | Released checkpoints, `autoregressive_step` semantics (history returned, FSQ re-encoding), window budget, guidance passes, Core skin |
| [Kimodo and SOMA-X](kimodo.md) | Batching vs multi-prompt semantics, continuation via transition constraints, cached text features, SOMA skeletons and meshes |
| [mjlab (our copy)](mjlab.md) | First-party simulator (ledger D13): our LiDAR sensor changes, the inherited corridor fixture, Viser over Slurm, traps |
| [TANGO](tango.md) | arXiv:2609.09158: offline plan-edit-track expert data and a whole-body VLA trained by imitation; static clutter, RGB, G1 (from a summarizing fetch) |

## Related work not yet reviewed

- MotionBricks (NVIDIA): real-time motion in-betweening. Kept out of the initial
  comparison by the user (ledger D03).
- GEAR SONIC (NVIDIA): a humanoid behavior foundation model; ARDY can plan for it.
  TANGO uses it as gait source and tracker.
- Bones Rigplay 1 / SEED: the training data for ARDY and Kimodo; the SEED G1 CSVs drive
  ARDY's constraint demo.

Source policy: state the local commit or HF revision inspected, and link upstream pages
for claims about unreleased models.
