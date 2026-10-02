# Motion graph behavior audit and prepared review loop

The saved graph contains substantial sideways/backward travel despite only walk/stand
command labels. This read-only audit precedes behavior tuning. **User instruction:**
stop after the HTML report and loop setup for review; no real evaluation/render loop
or Slurm job was launched.

## Evidence

- Final local report and machine-readable data:
  `runs/20261001T055543Z_behavior_report/index.html` and
  `runs/20261001T055543Z_behavior_report/audit.json`.
- Saved graph: `runs/20261001T013254Z_motion_graph/graph.pt`; library:
  `runs/20261001T004918Z_ardy_library/library.pt`. SHA256 and source snapshots are in
  the report. A preliminary report remains in `runs/20261001T054243Z_behavior_report/`.
- Supplied historical video: `runs/20261001T012334Z_mjlab_graph_demo/crowd.mp4`.
  A six-frame contact sheet was inspected; no new crowd render was made.
- Audit host: sky1, CPU only, PyTorch one thread, 2.9.0+cu128, CUDA wheel 12.8,
  CPython 3.13.15. No Slurm job or GPU; no timing comparison was collected.
- Original library/graph were generated on A40, driver 595.84, ARDY Core H40,
  TF32/no guidance/5 steps. Graph k=128, velocity descriptor weight 0.1 s,
  temperature 0.01 m, ±5° yaw/±10% speed correction, end-velocity weight 0.25 s,
  head-height filter 0.85, no lookahead. Exact upstream commits and checkpoint
  revision are preserved in linked artifact provenance.

## Measured graph composition

163,840 boundaries in 2,048 streams; 161,792 possible continuation edges;
26,972 dropped by posture curation; 134,820 valid edges = 14.98 h.

| Frame label | Count | Share of valid frames |
|---|---:|---:|
| Still / slow | 126,438 | 11.723% |
| Forward | 463,772 | 42.999% |
| Oblique | 146,074 | 13.543% |
| Sideways | 154,952 | 14.367% |
| Backward | 187,324 | 17.368% |

Forward is 48.709% of moving frames. Median/p95 body-facing error among moving
frames is 31.67°/167.40°. Majority-class edges: 15,437 still/slow; 5 turning while
slow; 53,652 forward straight; 5,338 forward turning; 17,828 oblique; 19,332 sideways;
23,228 backward. These are availability figures, not online visitation rates.

Definitions: root frame difference relative to midpoint body yaw, including each
edge predecessor; still below 0.15 m/s; forward within 30°, oblique 30–60°,
sideways 60–120°, backward ≥120°. Dominant frame label classifies each edge, with
low-index tie breaking; still/forward edges split at 10° net yaw per 0.4 s. These
are analysis cutoffs, not calibrated population statistics. “Turning while slow”
does not establish a deliberately authored turn-in-place motion.

## Connections and behavior

- 20,971,520 candidate slots across all boundaries. After valid edges, 95.423% of
  connections jump streams; 134,760 end boundaries have candidate incoming edges.
  Raw class-to-class mixes and availability matrices are in the audit.
- A candidate clip starts at its neighbor boundary and ends at clip id + 1.
  Connectivity uses a single-frame pose and finite-difference joint velocity;
  stored contacts are not explicitly used for matching/scoring.
- The runtime scores root trajectory, pose similarity and end velocity; there is
  no explicit forward-facing penalty or turn-before-walk rule. Steering can request
  abrupt changes from repulsion or random goals. This explains an available failure
  mechanism, not a measured causal decomposition of the historical video.
- The prior report's claim that the library contains no sidestepping/backward motion
  was incorrect: there are no deliberate labels/commands for these motions, but
  root-only constraints allow substantial body-relative sideways/backward output.

## Prepared tools and validation

Usage (now `crowd graph report` and `crowd behavior evaluate|render|compare`):
[behavior iteration](../../docs/usage/behavior-iteration.md).
The config-driven evaluator loads the graph once per seed bank, saves native-rate
executed joints/body yaw/contacts/commands/goals/edge selections, and records
direction, turn, root-gap and planted-foot metrics. MuJoCo replay is separate from
graph stepping and mjlab/LiDAR; paired HTML compares matching seeds and discovers
saved overview/follow videos. Every output is fresh; graph/source hashes and source
snapshots preserve input identity.

14 CPU synthetic tests passed (existing graph tests plus new direction, topology,
trace and comparison tests). Static checks and Slurm shell syntax passed. The real
replay scene builds 400 mocap capsule bodies for 16 people; installed MuJoCo camera
and arrow APIs were checked without rendering. Actual EGL replay and full mjlab
behavior integration remain unexecuted, as requested. The original plaza steering
still uses a shared RNG and is not training-ready.
