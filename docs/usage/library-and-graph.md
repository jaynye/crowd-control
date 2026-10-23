# Motion library and motion graph

An offline ARDY motion library, generated under random locomotion commands, and the
stochastic motion graph that steps many humans through it at run time. The graph is
scored with the same quality harness as the online backends ([benchmarks](benchmarks.md)).
Library generation and graph builds want a GPU; a saved graph runs anywhere.

## Library generation

```bash
source scripts/env.sh
crowd library generate --config configs/library/ardy_a40.json --out runs/<UTC>_ardy_library
```

Every human streams closed-loop ARDY motion through a `CrowdController` with
`RandomCommands` and an `OnlineMotionSource`. At every edge boundary (`edge_frames`, a
multiple of ARDY's 4-frame token):

1. the human's command comes from `RandomCommands` (`behavior/commands.py`);
2. a root path is planned from its **executed** state (`behavior.planner.plan_path`);
3. ARDY generates from the last `history` executed frames;
4. only the first `edge_frames` new frames are kept.

Each edge is therefore a genuine ARDY continuation of the frames before it, under a
known command.

**Commands** are a per-human semi-Markov process: stand, or walk at a speed drawn from
`speed_bands`, with a heading change (none, up to ±90°, or up to ±180°) and an optional
constant turn rate. Segment lengths come from `stand_s` and `walk_s`. Draws come from
each human's own generator, so they do not depend on batch composition.

**Config keys:** `model`, `steps`, `cfg_type`, `precision`, `history`, `edge_frames`,
`humans`, `batch` (humans per generation batch), `duration` (seconds per stream),
`seed`, `command_source` (`"random"`, the default, or `"turns"`), and `commands` (fields
of `CommandRanges`, or of `TurnRanges` for turning commands).
`configs/library/ardy_a40.json` is the setting of the current library (H40, TF32, no
guidance, 5 steps, 2,048 × 32 s); `ardy_smoke.json` is a small pipeline check.
`ardy_turns_a40.json` is the supplementary turning library: the same model setting,
turning commands with heading and path constrained on every 10th planned frame,
1,024 × 32 s. `ardy_turns_smoke.json` is its 256-stream check with every frame
constrained.

### Turning commands and body-heading targets

Root-path conditioning alone never asks ARDY to rotate a body that stays in place, so a
random-command library holds almost no in-place or slow turns. With
`"command_source": "turns"`, `TurnCommands` (`behavior/commands.py`) drives the
generation instead, and the request also carries per-frame **body-heading targets**:

- **Segments** form a per-human semi-Markov chain over walk, stand, in-place turn and
  arc. The next kind is drawn from `start`, `next_walk`, `next_stand`,
  `next_in_place` and `next_arc`, which list probabilities in the order walk, stand,
  in_place, arc. The defaults give sequences such as stand → in-place turn → walk off,
  and walk → arc → walk.
- **In-place turns** have speed 0. Each turns through a signed angle (`in_place_deg`)
  at a rate (`in_place_rate_deg`), then settles for `settle_s`.
- **Arcs** follow a circle of radius `arc_radius_m` through a signed angle (`arc_deg`).
  Speed comes from `speed_bands`; after walking, the arc keeps the walking speed
  (`arc_keep_speed`). Tighter arcs are walked slower, capped at `arc_max_rate_deg`.
- **Draws** come from each human's own generator, never from the executed motion, so
  segment sequences do not depend on batch composition, nor on whether targets are
  passed. A segment starts from the executed heading at the edge boundary that enters it.
- **Planning.** `TurnCommands.plan` is the controller's planner. It traces the rest of
  the segment with `planner.plan_arc` and stores per-frame targets. These targets
  start at the executed heading and blend to the segment schedule over `blend_s`. While
  walking or on an arc, they turn toward the planned path's tangent, fully at
  `forward_speed`, so the body faces where it walks. `motion_options()` hands them to
  the motion source as `root_heading`.
- **Switches.** `heading_targets` (off: path only, with the same segments),
  `walk_heading` and `stand_heading` (constrain straight walking and standing too),
  `heading_every` and `path_every` (constrain every k-th planned frame only; the other
  frames are free).

`MotionRequest.root_heading` (`[B, F]`, radians, NaN = unconstrained frame) is
implemented for ARDY only. It constrains ARDY's `global_root_heading` feature to
`[cos h, sin h]`, normalized like the root path, in both condition modes. A NaN path
frame is supported by the vectorized conditions only. Kimodo rejects heading targets.
`scripts/smoke/ardy_heading.py` checks that the two condition modes agree, and checks
the angle convention from standing.

The returned `heading` is decoded from that same feature. Under a target it can drift
away from where the body actually faces. Turn checks therefore measure the hip line
from the posed joints (`crowd library turns`).

Some in-place turns are **twists**: the pelvis turns while the feet stay planted.
`crowd library turns` reports the twist rate: the share of in-place turns whose feet
have turned less than 40% of the command 1.5 s after the segment. `crowd graph turns`
counts edges whose feet point away from the body.

### `library.pt`

A `torch.save` dict:

| Key | Shape | Notes |
|---|---|---|
| `posed_joints` | `[N, T, J, 3]` | World joint positions, float32 |
| `root_pos` | `[N, T, 3]` | |
| `heading` | `[N, T]` | Radians; 0 faces +Z |
| `contacts` | `[N, T, C]` | The model's foot-contact labels |
| `local_rot` | `[N, T, J, 3, 3]` | float16 |
| `features` | `[N, T, D]` | ARDY normalized features, float16 (to re-seed ARDY from a library state) |
| `cmd_speed`, `cmd_heading`, `cmd_mode` | `[N, T / edge_frames]` | The command each edge was generated under (mode 0 walk, 1 stand; turning commands add 2 in-place turn, 3 arc) |
| `heading_target` | `[N, T]` | Turning commands only: the dense planned heading reference of each executed frame (also where it was not constrained) |
| `cmd_rate`, `cmd_angle`, `cmd_radius`, `cmd_segment` | `[N, T / edge_frames]` | Turning commands only: signed turn rate (rad/s), signed segment turn (rad), arc radius (m, NaN otherwise), segment counter |
| `seeds` | `[N]` | Per-stream seed |
| `meta` | dict | `fps`, `edge_frames`, `history`, `joint_names`, `parents`, `foot_idx`, `head_idx`, `neutral_head_height`, generator settings (`command_source` when not random) |

## The motion graph

```python
from crowdmotion.graph import GraphConfig, GraphRuntime, MotionGraph

graph = MotionGraph.load(path, device="cpu")          # or MotionGraph(library, GraphConfig(...))
runtime = GraphRuntime(graph)
state = runtime.reset(seeds, root_xz, heading)        # per-human stream ids, world start
out = runtime.step(state, path)                       # path: [H, runtime.path_frames, 2] world XZ
```

- **Boundaries** are the frames before each edge, and serve as graph nodes. Stream
  frames `[0, edge_frames)` come from a fresh start and are not an edge.
- **Candidates** of a boundary are the outgoing edges of its `neighbors` nearest
  boundaries. Distance uses a descriptor: root-relative, heading-aligned joint positions
  plus `vel_weight` × joint velocities. A boundary's own successor is a candidate at
  distance 0, so a human that keeps following it replays the library exactly.
- **`step`** advances every human by one edge:
  1. **Select.** Score each candidate with the cost terms (`graph/costs.py`): the RMS
     distance between its root path, placed on the human's root and heading, and
     `path`; plus `pose_weight` × the descriptor distance; plus `end_vel_weight` × the
     error between the candidate's end-of-edge root velocity and the path's. With
     `lookahead`, the candidate's own successor is also scored against a two-edge path.
     Sample with Gumbel noise from `(seed, counter)` at `temperature` (0 is greedy).
  2. **Correct.** Optionally bend the edge's root path toward the request: up to
     `correct_yaw_deg` of yaw and `correct_speed` fractional speed, ramped over the edge.
  3. **Inertialize.** Add the offset between the current pose and the edge's start pose,
     decayed to zero over the edge with a smoothstep (`inertialize`).
- **Outputs**: world `joints`, `root`, `heading` and `contacts` for the edge's frames,
  plus per-human diagnostics:
  - `traj_err_m`, `best_traj_err_m`: path error of the chosen and the best candidate;
  - `end_vel_err_mps`, `best_end_vel_err_mps`: the same for end-of-edge velocity;
  - `desc_dist_m`, `pose_offset_m`, `vel_jump_mps`;
  - `natural`: whether the human followed its own successor;
  - `yaw_corr_rad`, `speed_scale`.
- **Curation.** `hunch_ratio > 0` drops edges whose mean head height is below that
  fraction of the skeleton's neutral standing head height.
- **Randomness** is batch-independent: Gumbel noise and start boundaries hash
  `(seed, counter, slot)` with SplitMix64.
- **Not yet output:** local rotations, so mesh skinning is not supported. Positions only.

`configs/graph/default.json` holds the setting of the saved graph (k = 128, correction
5° / 10%, end-velocity weight 0.25, hunch ratio 0.85). The `GraphConfig` dataclass
defaults keep the newer scoring terms off, so older configs reproduce.

## Saved graphs

```bash
crowd graph build --library <library.pt> --config configs/graph/default.json \
    --out runs/<UTC>_motion_graph/graph.pt
```

- `MotionGraph.save` stores the runtime tables plus one rest pose (for capsule bone
  lengths), and `graph.json` records the build. Positions are float16 and candidate ids
  int32: about 440 MB for an 18 h library.
- `MotionGraph.load(path, device, cfg_overrides)` restores the graph on any device
  without the library or the kNN build.
- Overrides may change scoring and correction, but not `neighbors`, `vel_weight` or
  `hunch_ratio`, which are fixed when the graph is built.
- The build takes a few seconds on an A40 and is very slow on CPU.
- **Several libraries** build one graph:
  `crowd graph build --library base.pt turns.pt ...`.
  - The libraries must agree on `fps`, `edge_frames`, joint names, parents, feet, head
    and neutral head height.
  - Streams are concatenated in the order given. Every stream is truncated to the
    shortest library's whole edges (a warning names the truncated libraries), so
    generate supplements at the base library's duration.
  - The merged `meta` lists each library under `sources` (path, streams, frames,
    command source), in stream order.
  - `crowd graph rotations` takes the same libraries in the same order.

`graph.pt` is a dict: `tables` (`pred_pose`, `pred_vel`, `edge_root`, `edge_yaw`,
`edge_pose`, `edge_contacts`, `edge_end_vel`, `valid`, `root_speed`, `cand`,
`cand_dist`), `cfg`, `meta`, `fps`, `L`, `streams`, `K`, `joints_n`, `stats` and
`rest_joints`. `crowd graph report` audits it on CPU
([behavior iteration](behavior-iteration.md#1-audit-the-graph)).

The graph keeps joint positions only. Rigid mesh humans in the mjlab plaza also need
bone rotations, saved beside it by

```bash
crowd graph rotations --graph <graph.pt> --library <library.pt> [<library.pt> ...] --device cuda \
    --out runs/<UTC>_graph_rotations
```

`rotations.pt` holds `edge_rot [B, L, J, 4]` and `pred_rot [B, J, 4]`, aligned with
`edge_pose` and `pred_pose`. These are float16 wxyz quaternions of the library's
global joint rotations relative to each frame's heading. It also holds `offsets [J, 3]`,
each joint's rest offset in its parent's frame. Loading
(`graph.rotations.GraphRotations.load`) samples edges and checks that each parent's
rotation carries the rest offset onto the graph's bone vector; a file from another
graph or library fails. For `graph-default` the file is the `graph-rotations`
artifact: 0.32 GB, built in about a minute on an A40, with bone directions within
1 degree of the graph.

## Evaluating a library and its graph

**Quality.** Graph cases in a quality config use `"backend": "graph"`, a `"graph"` dict
of `GraphConfig` fields and `"periods": ["edge"]`:

```bash
crowd bench quality --config configs/bench/graph_a40_v2.json --library <library.pt> --out runs/<UTC>_graph_quality
```

- A graph case uses the same start states, command schedules and change offsets as an
  online case with the same seed, so an online case replanned every `edge_frames / fps`
  seconds is a like-for-like reference.
- All rows add posture: `frac_frames_hunched` (head below 85% of the neutral standing
  height) and `head_height_ratio_median`.
- Graph rows add `g_*` diagnostics: means and p95 of the per-human diagnostics above,
  excluding the first step from the start pose; plus `g_natural_frac`,
  `g_unique_edges_frac`, `g_traj_err_frac_gt_5cm` (also for the best candidate) and the
  build statistics.
- For single-change scenarios, `gw_*` averages the diagnostics over the first 1.2 s
  after each human's change, split into humans who respond (`_responders`) and who never
  do (`_never`). If even the best candidate is far from the request, the gap is
  coverage; if the best is close but the chosen one is not, it is selection.

**Turning.** `crowd library turns --library <library.pt ...> --overrides <json>`
checks turning libraries per segment kind:

- **Turns.** Executed against commanded in-place and arc turns, measured on the hip
  line from the frame before the segment to `--lag-s` after it.
- **Heading.** Returned heading against the reference and the hip line.
- **In place.** Root drift (split by whether the human was at rest before), foot turn
  and lift-offs during in-place turns.
- **Arcs.** Executed arc radius and forward travel.
- **Feet and ground.** Planted-foot speed, pelvis height, and per-edge validity and
  flat-ground admissibility under the given graph overrides. The admissibility uses the
  same annotations and mask as the graph.

A random-command library in the same call gives walk and stand reference rows.
`--clips N` writes in-place turn clips for `crowd bench clips`.

`crowd graph turns --graph <graph.pt ...> --overrides <json>` counts a saved graph's
admissible edges by turning class, upright or hunched:

- in place;
- forward turns, by radius bin;
- non-forward turns;
- 1.2 s windows turning at least 45°.

A merged graph is also split per source library, and every count is also given per
motion hour.

**Runtime cost.**

```bash
crowd graph speed --library <library.pt> --out runs/<UTC>_graph_speed
```

Builds the graph (timed), then steps 4,096, 16,384 and 65,536 humans one edge per call
toward random straight paths and reports the synchronized wall time per step and
`motion_s_per_wall_s = humans × edge_s / wall` (GPU).

**On the cluster,** `scripts/slurm/library_pipeline.sbatch` generates a library and runs
both evaluations in one job, and `scripts/slurm/graph_eval.sbatch` evaluates an existing
library ([Slurm](../slurm.md)).

Unit tests of the graph run on CPU with a synthetic library:
`pytest tests/unit/test_graph.py tests/unit/test_graph_costs.py`.
