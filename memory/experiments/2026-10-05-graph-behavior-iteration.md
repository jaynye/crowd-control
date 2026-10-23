# Saved-graph crowd behavior iteration

Status: historical, unfinished

Outer run: `runs/20261005T020459Z_behavior_iteration/`. Existing local session in
`session/`, launcher allocation 4014910 on voltron (NVIDIA A40, driver 610.57.04,
torch 2.9.0+cu128). No Slurm submissions. Deadline 2026-10-05 05:50 UTC;
stop experiments by 05:10 for validation, report and memory.

Plan: freeze seeds 0/1/2 and the complete baseline configuration, evaluate before
runtime edits, replay overview and follow views, extract and inspect stills. Then
change one shared mechanism at a time: committed sampled turns biased to 90 degrees,
and explicit body-facing cost. Compare each on matching seeds, including direction,
spacing, acceleration, foot sliding, seams and response delay. Audit turn coverage
when it limits behavior. Reserve seeds 11/23/47 and a small mjlab check for a ready
candidate. Preserve graph, library, dependencies, plaza RNG scheme and goldens.

All requests use the existing session, unique names, explicit rule notes and bounded
timeouts. Candidate configs and outputs stay in the outer run. Session notes record
plans, observations and decisions. No acceptance thresholds are invented.

Initial access: bubblewrap loopback setup failed; reviewed shell execution recovered
access before experiments. Initial tree contains only pre-existing launcher and docs
work, recorded in `initial_worktree.txt` and the launcher source snapshot.


## Follow-up inspection (2026-10-05 evening)

The earlier run and session manifests still say `running` after their deadline; no
final report was found. The preserved session journal records completed requests
through `055_heading100_probe` at 02:51 UTC, including several behavior candidates.
Its unfinished code and all outputs were preserved. The
[evening session](2026-10-05-evening-graph-behavior-iteration.md) independently froze
and evaluated the original baseline, repeated the mechanisms, added a heading
deadband, and completed held-out, visual and mjlab checks. This note preserves the
original plan; it is not a completed result report for the earlier allocation.
