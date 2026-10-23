# GEAR SONIC (NVIDIA): released tracker and kinematic planner

Checked 2026-10-18 from the repository README and two pages of the docs site; nothing
was downloaded or installed at that initial check. The release was subsequently
fetched/pinned and its graph inspected; see the
[core batching conclusion](../core/design/motion-backends.md#sonic-robot-references-2026-10-18). Sources: <https://github.com/NVlabs/GR00T-WholeBodyControl>,
<https://nvlabs.github.io/GR00T-WholeBodyControl/references/planner_onnx.html>,
<https://huggingface.co/nvidia/GEAR-SONIC> (license "other"; whether access needs
acceptance was not checked). Paper: arXiv:2511.07820.

## Controller (tracker)

- **Variants:** three released, Default, Low-latency and v1.1. Each comes in two
  forms:
  - for deployment: ONNX `model_encoder.onnx` and `model_decoder.onnx` plus
    `observation_config.yaml`;
  - for evaluation and continued training: a PyTorch checkpoint and config, such as
    `sonic_release/last.pt` and `config.yaml`.
- **Interface:**
  - a universal-token controller with a 64-d token, at 50 Hz;
  - reference lookahead: 10 future frames (about 200 ms) for Default and v1.1, 4
    frames for Low-latency;
  - v1.1 uses a target orientation normalized to the robot's heading.
- **Training:** in Isaac Lab (v2.3.2; `gear_sonic/train_agent_trl.py`), from scratch on
  Bones-SEED or as a fine-tune of the release.
- **MuJoCo sim2sim:** a loop exists, `gear_sonic/scripts/run_sim_loop.py` together with
  the C++ deploy stack (`gear_sonic_deploy/`, `deploy.sh sim`). The README does not
  name the G1 model file it uses. Neither mjlab nor MuJoCo Warp is mentioned anywhere.

## Kinematic planner (`planner_sonic.onnx`, ONNX only)

- **Inputs:**
  - `context_mujoco_qpos` `[1, 4, 36]`: four recent G1 qpos frames (root position,
    wxyz quaternion, 29 joints), world frame;
  - `target_vel` `[1]` m/s (a value ≤ 0 uses the mode's default);
  - `mode` `[1]` int64: 27 styles, for example 2 = walk;
  - `movement_direction` `[1, 3]` and `facing_direction` `[1, 3]`, independent of each
    other;
  - `height` `[1]` (negative turns height control off).
- **Advanced inputs:**
  - `has_specific_target` with four waypoints `[1, 4, 3]` and their headings `[1, 4]`,
    which override the internal spring target;
  - `random_seed`;
  - `allowed_pred_num_tokens`: a mask over horizons of 6–16 tokens.
- **Outputs:** actual released graph: `mujoco_qpos` `[1, 64, 36]` float32 at
  30 Hz, valid up to `num_pred_frames` (`[1]` int32, despite the docs' scalar/int64 description).
  The horizon is 4 frames per token: 24–64 frames, 0.8–2.1 s. Full-body G1 qpos, so it
  can be tracked directly.
- **Batch fixed at 1.** The exported graph has no Loop/Scan/If: a fixed sequence
  of shared, root-token and pose-token transformer stages uses masks/argmax. Frames are MuJoCo Z-up (X forward, Y left), and the model
  canonicalizes internally.
- **Context:** after initialization, the released C++ caller samples history from
  the current planned animation (with lookahead), rather than measured robot qpos.
- **Replanning:**
  - a 10 Hz planner thread that does not replan on every tick;
  - mode, facing and height changes always replan;
  - for moving modes, speed and direction changes replan, as does a timer (0.1 s for
    running);
  - output is resampled from 30 to 50 Hz, and plans cross-fade over 8 frames.

## How the planner is built (read 2026-10-18)

The planner is **MotionBricks** (`third_party/GR00T-WholeBodyControl/motionbricks/`,
PyTorch source and training scripts; checkpoints are Git LFS objects under `out/`).
It is the robot planner's source here, not a human motion backend: the AGENTS.md line
that keeps MotionBricks out of the initial comparison concerns the human backends.

- **The planner is keyframe in-betweening behind a hand-written front end**
  (`motion_backbone/demo/full_agent.py`; the ONNX bakes the same steps):
  1. canonicalize the 4 context frames to the first frame's heading;
  2. a **critically damped spring** turns the command into a target root position 1 s
     ahead: current position + speed × movement direction, where the speed is the
     mode's average or 2 × `target_vel` (the code notes the actual speed is usually
     half the target). The target heading follows `facing_direction` with a 0.17 s
     half-life. Waypoints (`has_specific_target`) replace this target;
  3. the target's **body pose comes from the mode's style clip** (a frame chosen by
     `random_seed`), rotated to the target heading. So `mode` only selects a target
     clip: neither network has a mode input, and the released configs have no text
     conditioning (`text_embeddings: null`);
  4. **MotionBricks in-betweens** from the 4 context frames to the 4 target frames.
- **Root model** (34 M parameters; 3 shared + 3 root-token transformer layers, width 512,
  16 heads): classifies the number of tokens (6–16; 4 frames each at 30 Hz), then a
  conv decoder regresses a continuous root path conditioned on the start and end frames
  only (no middle keyframes). Walking masks the count to 6–11 tokens (0.8–1.47 s).
- **Pose model** (16 layers, width 1,024, 16 heads): a masked-token (MaskGIT-style)
  transformer conditioned on the root path, run for one step. It outputs logits over
  16 positions × 8 heads × 10 codes; deployment takes the argmax, and training-time
  sampling is Gumbel.
- **VQ-VAE** (8 heads, 10 codes each, 256-d codes): a conv decoder turns tokens plus
  the root path into poses.
- **Weights:** the root checkpoint (`motionbricks_root/version_1`, LFS oid
  `d7299a9b…69c7`, 409,551,754 bytes) matches `planner_sonic.onnx` exactly for 146 of
  its 150 tensors (34,115,852 of 34,122,833 parameters). The 4 unmatched are an
  embedding and three small vectors consistent with constant folding at export
  (`scripts/smoke/motionbricks_weights_match.py`, CPU job 4137474). The pose and
  VQ-VAE checkpoints (1.64 GB and 286 MB) are not yet compared.
- **Training code:** the scripts in `scripts/train_{vqvae,pose,root}.py` default to
  synthetic data; the full robot training pipeline is "targeted for approximately one
  month out" (README, 2026-04-27). Training data: Bones-SEED, retargeted to G1.
- **Deploy replanning** (`gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp`, `Planner()`
  at 10 Hz): a timer replans walking every 1.0 s (running 0.1 s, crawling 0.2 s), and
  any change of mode, facing, height, speed or direction replans at once. Context comes
  from the current planned motion 2 steps (0.04 s) ahead.

## Consequences for this project

- **Batching is the gate's first question.** Online use at 4,096 envs and 10 Hz means
  about 41,000 planner calls a second. That needs a dynamic batch (ONNX graph surgery
  or a torch port) or offline generation.
- **The planner's inputs make a ready action space** for a high-level policy: movement
  direction, facing, speed, and optionally waypoints.
- How this shapes option (c):
  [robot policy architecture](../decisions/robot-policy-architecture.md).
