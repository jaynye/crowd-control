# ARDY (nv-tlabs)

Autoregressive diffusion with a hybrid representation for interactive human motion:
explicit root features plus FSQ-quantized latent body tokens, with streaming text and
kinematic constraints. SIGGRAPH 2026 (Zhao et al., TOG 45(4) 86). Checked 2026-09-30.

- **Checkout:** `ardy/`, commit `693f74d1` ("Initial commit", 2026-07-09), unmodified.
- **Weights:** `~/flash/checkpoints/ARDY/core-rp-fps-20-horizon-{8,40}`, with alias
  symlinks `ARDY-Core-RP-20FPS-Horizon{8,40}`. HF revisions: H8 `257a0843`,
  H40 `abe6c43b`.

## Released models

| Model | Skeleton | FPS | Horizon | State here |
|---|---|---|---|---|
| ARDY-Core-RP-20FPS-Horizon8 | Core (27 joints) | 20 | 8 frames (0.4 s) | Downloaded |
| ARDY-Core-RP-20FPS-Horizon40 | Core | 20 | 40 frames (2 s) | Downloaded |
| ARDY-G1-RP-25FPS-Horizon{8,52} | Unitree G1 | 25 | 8 / 52 | Not downloaded |
| SOMA | SOMA | (30) | (60) | **Not released**; the `soma` nickname exists in the registry but has no weights |

## API facts (verified on A40, 2026-09-30)

- `load_model(name, device, text_encoder=False)` loads without a text encoder. It reads
  `CHECKPOINTS_DIR/<full name>`; without that variable it downloads from HF.
- `Ardy.autoregressive_step(num_frames, num_denoising_steps, motion_mask, observed_motion,
  cfg_weight, text_feat, text_pad_mask, init_history_sequence, init_global_translation,
  init_first_heading_angle)` generates **one** window and returns normalized explicit
  features of **history + horizon** frames. It returned `[4, 12, 330]` for H8 and
  `[4, 44, 330]` for H40 with 4 history frames. Slice off the history.
- **History is re-encoded:** the history passes through the FSQ tokenizer and is decoded
  again together with the new frames. The returned history differs from the frames
  played by 2 mm (H8) and 4 mm (H40) mean joint error. The seam that matters is last
  played frame → first new frame.
- The window is `history + horizon + future` frames, each a multiple of 4
  (`num_frames_per_token`). Constraints can cover the future context. The demo's
  defaults are a history crop of 4 frames and future crop = max (window budget 10 s =
  200 frames).
- Sampling uses deterministic DDIM (eta = 0) with `num_base_steps: 10`, and the demo uses
  all 10 steps. The initial noise, one `torch.randn` call in `_generate_window`, is the
  only randomness.
- Guidance is `cfg_type: separated`: three denoiser passes per step (text, constraint,
  unconditional) batched as 3B. Alternatives: `regular` (2 passes) and `nocfg` (1).
- The motion representation is 330-d: root position, heading (cos, sin), local joint
  positions, 6-D global rotations, velocities and 4 foot contacts.
  `motion_rep.inverse` gives `local_rot_mats [B,T,27,3,3]`, `posed_joints`,
  `root_positions`, `foot_contacts [B,T,4]` and `global_root_heading [B,T,2]`.
- **Root constraints:** `Root2DConstraintSet` sets `root_pos` x and z (and optionally
  the heading as cos/sin). A vectorized fill of those slices, then `normalize × mask`,
  matches the library exactly (`tests/gpu/test_ardy_adapter.py`).
- Velocity control in the demo is not a tensor input. It converts the target velocity
  into 2-D root waypoints every few frames (upstream
  `third_party/ardy/scripts/interactive_demo/gen_constraints.py`).
- **Post-processing** (foot skate and constraint cleanup) uses the `motion_correction`
  C++ extension. It is identical to Kimodo's, apart from a license header and one
  compiler guard.
- **Skin:** `ardy/ardy/assets/skeletons/cskel27/skin_standard.npz` has 9,084 vertices,
  18,152 faces and 5 LBS weights per vertex (`ardy.viz.core_skin.CoreSkin`).
- **Dependencies:** ARDY pins `numpy<2` and `transformers==5.8.1`, but it runs on this
  env's numpy 2.5.1 and transformers 5.1.0 for everything except its text encoder, which
  is untested here.
- **Acceleration:** the demo supports TensorRT (`[trt]` extra, not installed) and
  `torch.compile`, via `Ardy.compile_denoiser`.

## Measured behavior (A40, 2026-09-30)

Numbers are in the benchmark note, linked from the core
[motion backends](../core/design/motion-backends.md) conclusions. Facts about the model
itself:

- **Compute.** The denoiser is ≥ 97% of a step and compute-bound past batch ~256. The
  profile splits CUDA time into matmuls 38%, attention 27% and LayerNorm plus
  elementwise ops about 20%. `torch.compile` (inductor) gives ≤ 1.2×.
- **Guidance and steps.**
  - With zero text, `nocfg` (one pass) followed root paths at least as well as
    `separated` (three passes), at a third of the cost.
  - 5 steps is close to the default 10.
  - 2 steps produces foot sliding late in 2 s windows, while the predicted contact
    labels switch off, so the labels hide it.
- **Precision.** bf16 autocast adds visible root jitter (about 3× interior
  step-change). TF32 does not.
- **Window.** Future-context frames are costly: 0 → 160 frames cuts H40 throughput about
  3.4×. H8 needs future context to follow a path, which removes its speed advantage.
- **Short replans hurt.** Replanning every 0.2 s from a 4-frame history makes responses
  slower: the window's early frames mostly continue the history.
- **Constraint loops.** One `Root2DConstraintSet` per sample costs 2 s at batch 4,096.
  Fill the `root_pos` slices directly instead (`src/crowdmotion/backends/ardy.py`).
