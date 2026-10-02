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
`seed`, and `commands` (fields of `CommandRanges`). `configs/library/ardy_a40.json` is
the setting of the current library (H40, TF32, no guidance, 5 steps, 2,048 × 32 s);
`ardy_smoke.json` is a small pipeline check.

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
| `cmd_speed`, `cmd_heading`, `cmd_mode` | `[N, T / edge_frames]` | The command each edge was generated under (mode 0 walk, 1 stand) |
| `seeds` | `[N]` | Per-stream seed |
| `meta` | dict | `fps`, `edge_frames`, `history`, `joint_names`, `parents`, `foot_idx`, `head_idx`, `neutral_head_height`, generator settings |

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

`graph.pt` is a dict: `tables` (`pred_pose`, `pred_vel`, `edge_root`, `edge_yaw`,
`edge_pose`, `edge_contacts`, `edge_end_vel`, `valid`, `root_speed`, `cand`,
`cand_dist`), `cfg`, `meta`, `fps`, `L`, `streams`, `K`, `joints_n`, `stats` and
`rest_joints`. `crowd graph report` audits it on CPU
([behavior iteration](behavior-iteration.md#1-audit-the-graph)).

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
