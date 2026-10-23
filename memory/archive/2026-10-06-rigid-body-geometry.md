# Rigid mesh humans in the mjlab crowd (2026-10-06)

Request (user, 2026-10-06): adopt full-resolution rigid Core-skin pieces, with no
simplification, as the human geometry in the main training loop. Add a Viser toggle in
the motion graph viewer that switches both the visualized humans and the LiDAR between
the rigid mesh and capsules. Evidence:
[body geometry benchmark](../experiments/2026-10-05-body-geometry-benchmark.md).

No training env exists yet, so "adopt" means rigid becomes the plaza env-config default
that a training env will pick up. Route: the per-human ray kernel (the costs the user
accepted), not mujoco_warp mesh geoms. One body shape (the Core skin); the 32 scaled
copies were timing stand-ins.

## Plan

1. Rotation sidecar. `crowd graph rotations` writes body-local joint rotations for
   every graph edge frame and boundary, from the library, through `RunDir`.
   - Add an artifact entry, a data doc line and a run-log row.
   - At load, validate against `edge_pose` with parent rotation times rest offset
     against the child direction; raise on a mismatch.
2. L1 geometry (`sim.body`, new `sim.rays`):
   - rotation helpers and the swing-to-child correction;
   - per-piece bounding spheres for conservative human boxes;
   - Warp per-human box plus per-piece mesh query kernel, writing piece geom ids and
     world normals.
3. mjlab hook: `RayCastSensor.add_ray_target`, run in-graph after `rays()`, nearer hit
   wins. Add a changelog entry and an mjlab test.
4. `GraphCrowd`:
   - rotation buffers beside `frames` (reset from `pred_rot`, hold keeps the last
     rotation, nlerp between frames);
   - piece mocap poses and kernel transforms;
   - a device-side enable flag;
   - a queued geometry toggle.
5. Plaza config `human_geometry` (default rigid only). The Viser demo builds rigid and
   capsules with a toggle; the video demo gets `--geometry`. Pieces go in geom group 1
   (visible, not traced by mujoco_warp); capsules stay in group 2.
6. The golden harness pins capsules; no expected file may change. Add unit tests:
   - the rotation sidecar against the graph;
   - kernel against CPU `mj_ray` on the visual pieces;
   - rigid mode never returns a capsule id;
   - the toggle;
   - boxes contain the pieces over a rollout.
7. Run an A40 timing of the plaza env at 4,096 × 16, rigid against capsules, and a CPU
   Viser smoke test.
8. Update docs, the ledger entry, the decision note, status and the architecture note.

## Progress

- [x] 1 sidecar: `runs/20261006T162450Z_graph_rotations/` (job 4022604, A40). The bone
  direction error is 0.044 deg median, 0.50 deg p99 and 0.97 deg max. Artifact
  `graph-rotations` is in `artifacts.toml`. Job 4022602 failed at `source` under
  `sh` (`--wrap` needs `bash -c`) and made no run dir.
- [x] 2 L1 geometry: `sim/body.py` (quaternions, swing, `core_pieces`,
  `piece_spheres`, a body-name tag) and `sim/rays.py`.
- [x] 3 mjlab hook: `add_ray_target`, a changelog entry and a test (36 passed,
  5 CUDA-only skipped on CPU).
- [x] 4 `GraphCrowd`: `human_geometry`, `rotations_path`, `lidar_group`,
  `set_geometry` and the Viser dropdown.
- [x] 5 plaza (`human_geometry`, default rigid) and demos (`--geometry`; Viser builds
  both).
- [x] 6 tests:
  - `tests/unit/test_rigid_crowd.py` (5): kernel ids exact against CPU `mj_ray`, ranges
    within 1 mm, swing near 0 without inertialization;
  - the golden pins capsules, with no expected file changed (fast and full
    `mjlab_plaza` MATCH);
  - 196 unit tests plus fast goldens pass.
- [x] 7 A40 timing. The first run had torch posing at 34.8 ms per step
  (`runs/20261006T164021Z_sim_speed/`); a fused Warp pose pass replaced it. Final run
  `runs/20261006T164941Z_sim_speed/`: at 16 humans, rigid 57.3 ms against capsules
  53.3 ms; at 4 humans, 29.3 against 29.4. GPU tests: job 4022640, 45 passed. The CPU
  Viser smoke test started cleanly.
- [x] 8 docs and memory: ledger D15, decision note (partly settled), status, crowd
  architecture, experiment note (adoption section), run log, and an AGENTS rule. Full
  CPU suite: 197 passed, 3 CUDA-only skipped.

Retired 2026-10-06: results promoted to the experiment note, ledger D15, status and
the crowd architecture note.

Note: `command.py` and `demo.py` carry another session's uncommitted edits (last
touched 2026-10-05 ~19:35 local). Check mtimes before editing; keep changes contained.
