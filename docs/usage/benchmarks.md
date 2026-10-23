# Benchmarks: online ARDY and Kimodo

Measures how much **usable** human motion ARDY and Kimodo generate online, and how well
they continue from executed motion. These harnesses decided that the generators run
offline only; they stay for comparing future library generators. They need a GPU and
the checkpoints ([data](../data.md)).

**Configs.** `configs/bench/*.json` are the historical configs of the A40 benchmark runs
(their names end in `_a40`; `smoke.json`, `quality_smoke.json` and `graph_smoke.json`
are small checks). Keep them unchanged so earlier numbers stay reproducible, and write a
new config for new measurements. Compare speeds only within one GPU type.

## Backend contract

`crowdmotion.backends.base` defines one batched call:

```text
MotionRequest(text_feat, seeds, counters, history=None, root_path=None,
              start_xz=None, start_heading=None)   ->   backend.generate(request) -> MotionChunk
```

- **`history`**: native normalized features of frames that were actually executed,
  oldest first, as returned in a previous chunk's `features`. `None` means a fresh start
  at `start_xz` / `start_heading`.
- **`root_path`** `[B, F, 2]`: world XZ targets for the frames after the history, at the
  native frame rate. Desired velocity is expressed as a path.
- **`seeds`, `counters`**: each human's random stream and draw. The initial diffusion
  noise of human `b` depends only on `(seeds[b], counters[b])`. Both samplers are
  deterministic DDIM, so outputs do not depend on batch order or composition.
- **`MotionChunk`** holds **new frames only**: `features`, `root_pos`, `local_rot`,
  `posed_joints`, `contacts`, `heading`, `fps`. `stage_seconds()` gives per-stage GPU
  time once the caller has synchronized.

Conventions are native to each model: Y-up, meters, heading 0 facing +Z.

| Backend | Class and config | Skeleton | FPS | New frames per call |
|---|---|---|---|---|
| ARDY | `ArdyBackend(ArdyConfig)` | Core, 27 joints | 20 | the model horizon (8 or 40) |
| Kimodo | `KimodoBackend(KimodoConfig)` | SOMA `somaskel30` (optionally `somaskel77`) | 30 | `new_frames` |

Build either with `crowdmotion.backends.registry.make_backend(config)`; benchmark cases
are flat JSON dicts mapped onto the typed configs by `config_from_case`.

### ARDY (`ArdyConfig`)

Fields: `model`, `device`, `steps` (default: the checkpoint's), `cfg_type`, `cfg_weight`,
`future_frames`, `precision`, `conditions`, `timing`.

- Reproduces `Ardy.autoregressive_step` with ARDY's own window, encode and decode
  methods, and strips the history that upstream returns.
- The window is `history + horizon + future_frames` frames, each a multiple of 4. The
  root path covers the horizon plus the future context.
- **`cfg_type`**: denoiser passes per step: `separated` 3, `regular` 2, `nocfg` 1.
- **`conditions`**: `vectorized` fills the root-position slices directly; `library`
  builds one `Root2DConstraintSet` per human, as the ARDY demo does
  (`tests/gpu/test_ardy_adapter.py` checks that they agree). Use `vectorized` at
  population scale: `library` costs about 2 s per call at batch 4,096.
- **`precision`**: `fp32`, `tf32` (TF32 matmuls) or `bf16` (autocast around the denoiser
  loop only).

### Kimodo (`KimodoConfig`)

Fields: `model`, `device`, `steps` (100), `cfg_weight`, `transition_frames` (5),
`new_frames`, `precision`, `post_processing`, `to_soma77`, `seam_blend_frames`, `timing`.

- **Continuation** matches Kimodo's multi-prompt transition: the last
  `transition_frames` executed frames become full-body and end-effector constraints at
  the start of the window; the window is translated so its first frame is at the
  origin; the root path constrains the following `new_frames`.
- Only frames after the transition are returned. Kimodo's own multi-prompt mode
  cross-fades the previous segment's last frames, which is impossible once they were
  executed; `seam_blend_frames > 0` instead decays the offset between the last executed
  frame and the regenerated transition over the first new frames. With 0, the
  continuation equals `_multiprompt`'s second segment (`tests/gpu/test_kimodo_adapter.py`).
- Constraints are built per human with Kimodo's classes (a Python loop); their cost
  appears as the `conditions` stage.
- Text features are empty prompts until the gated encoder is available
  ([data](../data.md#text-encoder)).

## Throughput

```bash
crowd bench throughput --config configs/bench/ardy_a40.json --out runs/<UTC>_ardy_throughput
```

**Config.** `defaults` (`reps`, `warmup`, `time_budget_s`) and `cases`. List-valued keys
in a case expand to a Cartesian product.

| Key | Values |
|---|---|
| `backend`, `model` | Which model to run |
| `batch`, `microbatch` | Population per refresh; microbatch size (an out-of-memory error halves it) |
| `precision` | `fp32`, `tf32` or `bf16` |
| `steps` | Denoising steps |
| `history` | ARDY history frames, or Kimodo transition frames |
| `future` | ARDY future context frames |
| `new_frames` | Kimodo new frames per call |
| `cfg_type` | ARDY guidance mode |
| `conditions` | ARDY condition builder: `vectorized` or `library` |
| `post_processing`, `to_soma77` | Kimodo output processing |
| `workload` | `continue` (from executed history) or `fresh` |

**Procedure.** Each configuration builds a population of `batch` humans with random
starts and per-human arcs (0.5–1.6 m/s, within ±60°/s), warms up (filling the history
buffer for continuation), then times `reps` synchronized population refreshes, stopping
early once `time_budget_s` is spent. Model loading is timed separately
(`run.json: load_seconds`) and excluded.

**Outputs:** `run.json` (config, environment, load times), `rows.csv` and `rows.jsonl`
(one row per configuration).

| Column | Meaning |
|---|---|
| `new_s` | Newly generated seconds per human per call; history and transition frames excluded |
| `wall_p50`, `wall_p95` | Synchronized wall time of one population refresh, all microbatches included |
| `R_new` | `batch · new_s / wall_p50`: new-motion seconds per wall-clock second |
| `u_<policy>` | Consumed fraction. `refill` = 1. `P<p>` = replan every `p` simulated seconds: `p / new_s` if `p < new_s`, else 1 |
| `R_usable_<policy>` | `u · R_new` |
| `S_star_<policy>_m<m>` | `R_usable / (4096 · m)`: the highest simulator speed (simulated seconds per wall second per environment) this GPU could feed, in isolation, for 4,096 environments with `m` humans each |
| `stage_<name>` | Mean GPU seconds per refresh in each stage: `conditions`, `noise`, `encode_history`, `denoise`, `decode`, `kinematics`, `postprocess`, `to_soma77` |
| `peak_gb` | Peak allocated CUDA memory, warm-up included |

`S*` is capacity in isolation. End-to-end speed also depends on sharing the GPU with the
simulator, sensors and policy. Measurement rules: count only newly generated frames,
apply the consumed fraction, exclude loading and export from steady-state timing, and
record GPU, driver, CUDA and PyTorch versions, commits, checkpoint revisions, precision
and acceleration settings with every number (`run.json` does this).

## Continuation quality

```bash
crowd bench quality --config configs/bench/quality_a40.json --group ardy --out runs/<UTC>_quality
```

**Rollout.** Humans follow command schedules (speed, heading) in simulated time
(`ScenarioCommands`):

- Single-change scenarios switch at 4 s (and restart at 7 s), plus a per-human offset
  drawn uniformly over one replan interval, so changes do not coincide with replan
  boundaries. `"jitter_changes": false` in a case disables the offset.
- At each replan, a root path is planned from the **executed** root state; velocity
  blends linearly to the command over 0.5 s.
- The backend generates, and the first `P` seconds are executed (a `periods` entry), or
  the whole chunk for `"refill"`. Graph cases execute one edge (`"edge"`).

| Scenario | Command schedule |
|---|---|
| `cruise` | New random speed (0.5–1.6 m/s) and heading change (±90°) every 4 s |
| `stop_start` | 1.2 m/s; stop at 4 s; restart at 7 s |
| `turn90`, `turn180` | 1.2 m/s; heading +90° or +180° at 4 s |
| `speed_step` | 0.6 → 1.6 m/s at 4 s |

**Metrics** (one row per case × scenario × period, in `rows.jsonl`):

| Metric | Definition |
|---|---|
| `path_err_*` | Executed root XZ against the path in force when the frame was generated |
| `speed_err_mean_mps` | Executed speed against planned speed |
| `foot_skate_*` | Horizontal speed of the foot joints on frames the model labels in contact twice in a row |
| `hgate_skate_*` | The same, with "planted" meaning within 2.5 cm of the joint's own 5th-percentile height, independent of the model's labels. It has a floor of about 0.1–0.15 m/s from swing-phase frames, so compare within one skeleton only |
| `frac_frames_hunched`, `head_height_ratio_median` | Head below 85% of the neutral standing height; median head height ratio |
| `toe_min_height_p05_m`, `frac_frames_penetration_gt_2cm`, `frac_frames_no_contact`, `frac_frames_lowest_foot_gt_15cm` | Ground contact checks |
| `seam_*` vs `interior_*` | Joint displacement and root acceleration across call boundaries against within chunks |
| `stop_response_s`, `start_response_s`, `turn_response_s`, `speed_response_s` | Simulated seconds from each human's command change until the executed motion satisfies it for 0.25 s, including the wait for the next replan. `_never_frac`: humans that never did |
| `replan_wait_s` | Simulated seconds from the command change to the next generation call |
| `failed_humans_*` | Non-finite output, or path error above 1 m |
| `branch_check` | From one executed history: whether the history is unchanged, the maximum difference when a seed repeats, and the final-pose spread across 4 seeds |

Each case also saves `clips_<case>_<scenario>_<period>.npz` (joints, planned path,
contacts, call boundaries, skeleton) for the first six humans. Graph-case diagnostics
(`g_*`, `gw_*`) are described in [library and graph](library-and-graph.md#evaluating-a-library-and-its-graph).

## Reports, videos and probes

```bash
crowd bench tables --throughput runs/<a> --quality runs/<b> > summary.md   # Markdown tables
crowd bench clips runs/<quality run>/clips_<case>_<scenario>_<period>.npz  # stick-figure MP4 next to the clips
crowd bench sim-speed --num-envs 1024 4096 --out runs/<UTC>_sim_speed     # simulator alone (GPU)
```

- `crowd bench clips` draws six panels; a panel's title turns red on frames where a new
  generation call starts, so seams can be inspected.
- `crowd bench sim-speed` measures `S` for the G1 velocity task (robot only, an upper
  bound) and the corridor fixture with 4 and 8 humans and LiDAR, with zero actions and
  no policy. `--scenes plaza-rigid-16 plaza-capsules-16 ...` adds the graph-driven
  plaza with rigid mesh or capsule humans, and times graph stepping with pose writes
  (`command_ms`) and sensing (`sense_ms`).
- `crowd bench body-geometry --config configs/bench/body_geometry_a40.json` times LiDAR
  sensing for human body geometries at training scale. Geometries: capsules, rigid
  Core-skin pieces per bone (mesh geoms on mocap bodies, optionally 32 per-world
  variants) and a Core skin deformed by LBS every step. It also measures per-ray
  range fidelity against full-resolution skinning. Each case runs in its own process
  and appends to `rows.jsonl`; `summary.md` tabulates the rows. The geometries come
  from `crowdmotion.sim.body`; the module docstring lists the stages it times.
  `configs/bench/body_geometry_matrix_a40.json` covers 4, 8 and 16 humans for 32-shape
  rigid pieces and SAH-built skinned surfaces.
- The ARDY profiler and `torch.compile` comparison, and the model smoke tests, are
  scripts: `scripts/smoke/profile_ardy.py`, `scripts/smoke/ardy_smoke.py`,
  `scripts/smoke/kimodo_smoke.py` (GPU; on the cluster through `scripts/slurm/`).
- Adapter parity tests: `pytest tests/gpu` (GPU, checkpoints).
