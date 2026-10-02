# Kimodo (nv-tlabs) and SOMA-X

Kimodo is a full-sequence text- and constraint-conditioned diffusion model for
controllable offline motion authoring. Checked 2026-09-28 and 2026-09-30.

- **Checkout:** `kimodo/`, commit `58e78189` (2026-09-22). It is installed
  non-editable into the root venv as `kimodo 1.0.0`.
- **Weights:** `~/flash/checkpoints/kimodo/soma-rp-v1.1`, alias
  `Kimodo-SOMA-RP-v1.1`, HF revision `6c9233af`.
- **SOMA-X:** `soma-x/` holds the HF assets for `py-soma-x==0.3.2`, release tag v0.3.2.

## API facts

- `load_model(name, device, text_encoder=...)` requires some text encoder. Pass a stub
  that returns zeros (`crowdmotion.backends.text.ZeroTextEncoder`) while Llama is
  gated. It reads `CHECKPOINT_DIR/<display name>` (note: no S; ARDY uses
  `CHECKPOINTS_DIR`).
- `Kimodo.__call__(prompts, num_frames, num_denoising_steps, multi_prompt=False,
  constraint_lst, cfg_weight=[2, 2], ..., post_processing=False)`. A list of prompts is a
  batch of **independent** samples. `multi_prompt=True` treats the list as sequential
  segments of one motion, not independent humans.
- `_generate(texts, max_frames, ..., text_feat, text_pad_mask)` accepts cached text
  features; `__call__` does not expose them.
- 30 fps, SOMA skeleton. Internally it uses `somaskel30`; `__call__` converts the output
  to `somaskel77`, which drives `SOMALayer` directly. Windows are at most 10 s.
- `num_base_steps: 1000`, respaced to `num_denoising_steps`. The README default is 100,
  and 25 or fewer are "reduced". DDIM is deterministic (eta = 0), and the initial
  `torch.randn` in `_generate` is the only randomness.
- Guidance is `separated`: three passes per step.
- **Continuation** (`_multiprompt`): the last 5 frames of the previous segment become
  `FullBodyConstraintSet` plus `EndEffectorConstraintSet` (hands and feet) on frames 0–4
  of the next window. The window is translated so frame 0's smooth root is at the
  origin, and the first heading is taken from those frames. The 5 transition frames are
  regenerated, so they are not new motion.
  - Without post-processing, `_multiprompt` then **cross-fades the previous segment's
    last 5 frames** into the regenerated ones. Online, those frames have already
    played, so it cannot be reproduced there.
  - Without it, root seams jump about 7× the interior motion (2026-09-30).
  - `crowdmotion`'s adapter matches `_multiprompt`'s second segment to 1e-3
    (`tests/gpu/test_kimodo_adapter.py`). It offers a forward blend instead, which cuts the
    jump to 5.5×.
- **Mask semantics.** The denoiser computes `x·(1 − mask) + observed·mask`, so a zero
  mask and `None` are equivalent.
- **Constraints:** `create_conditions_from_constraints_batched` loops over the batch in
  Python whenever each sample has its own constraint list.
- **Root representation:** `smooth_root_pos` plus a hips offset. `Root2DConstraintSet`
  constrains the smooth root (x, z).
- **Foot contacts:** 6 SOMA channels (2026-09-28 smoke), not the 4 in the docstring.
- **Upstream quirk:** `Root2DConstraintSet` given a 3-D root keeps columns `[0, 1]` (x
  and y), not x and z. Always pass 2-D (x, z).

## Upstream guidance

- Post-processing (foot-skate cleanup and constraint enforcement) is recommended when
  constraint or contact accuracy matters
  ([limitations](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html)).
- Sequential generation conditions on preceding frames, but offline stitching is not
  validated online control (handoff §1).
