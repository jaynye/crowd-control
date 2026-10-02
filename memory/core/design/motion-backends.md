# Motion backends

What a body-motion backend must provide, how its capacity is measured, and what has
been concluded about each candidate. **Outcome (user, 2026-09-30):** at 4,096
environments with m = 4–16, both generators run offline only
([ledger](decision-ledger.md) D09–D10). The runtime is the selected motion graph (user, 2026-10-01; ledger D11);
[current design](motion-library.md).

## Contract

`src/crowdmotion/backends/base.py` (typed configs and `make_backend` in `registry.py`;
both adapters expose `generate(request)`), from handoff §1:

```text
Generate(executed_history, prompt_embeddings, future_constraints, seeds)
    -> new_root_motion, joint_rotations, contacts, frame_rate
```

- **Batched.** One call serves any subset of the population that needs a refill or a new
  plan.
- **Executed history in, new frames out.** History is what was actually played, and it
  is never returned or modified. A backend that regenerates overlap frames (Kimodo's
  transition, ARDY's re-encoded history) does not count them as new.
- **Future constraints** are world-frame root paths, and optionally headings, at the
  native frame rate. Desired velocity becomes a path; neither model has a velocity
  tensor input.
- **Per-human random streams.** Initial diffusion noise for human `b` is drawn from
  `(seed_b, counter_b)`. With deterministic DDIM, a human's motion depends on its own
  stream and inputs, not on batch order or composition (verified to 1e-4 for ARDY).
- **Native skeletons and conventions** are preserved: Y-up, meters, heading 0 facing
  +Z. Kimodo is SOMA (`somaskel30` internally, `somaskel77` out) at 30 fps. ARDY is Core
  27 joints at 20 fps.

## Capacity metrics

For batch `B`, new duration per human per call `H` (seconds), synchronized wall time
`T_B`, and consumed fraction `u`:

- `R_new = B·H / T_B`: new-motion seconds per wall-second.
- `R_usable = u · R_new`.
  - Buffer refill consumes everything (`u = 1`).
  - Replanning every `P` simulated seconds with `P < H` discards the rest: `u = P / H`.
- Demand is `R_required = 4096 · m · S`. With `S` unspecified (ledger D09), each result
  is reported as the break-even `S* = R_usable / (4096 · m)`.
- Also reported: p50/p95 latency (a synchronous stall), peak VRAM, the per-stage split
  (conditions, noise, history encoding, denoising, decoding, kinematics,
  post-processing), and model load time (separate).

## Conclusions

As of 2026-09-30, on one A40 in isolation
([benchmark](../../experiments/2026-09-30-motion-backend-benchmark.md)):

- **Kimodo is not viable online.**
  - At most 60 new-motion seconds per wall-second per A40 (bf16, 10 steps, 5 s
    windows); 26 at 25 steps. That is 27–220 A40s of generation for 4,096 environments
    at `S = 0.4` (m = 1–8).
  - Guidance-off is untried. Even an optimistic 3× would leave it more than 15× below
    ARDY.
  - Online continuation has seam jumps about 7× the interior root motion. Upstream
    hides them by cross-fading frames that would already have played.
  - It remains the SOMA-native option for **offline** generation.
- **ARDY H40 is the online candidate** at TF32, no guidance, 5 steps.
  - **Throughput:** 2,800 new-motion seconds per wall-second, flat from batch 256.
    That is `S* = 0.69 / 0.17 / 0.086` for m = 1 / 4 / 8.
  - **Tracking and seams:** 0.5–0.9 cm path error, seams no rougher than the interior,
    stable over 60 s rollouts.
  - **Quality vs defaults** (3-pass guidance, 10 steps, 5.7× slower): near parity.
    Height-gated foot sliding is about 10–15% higher on turns. This holds with zero
    text only.
- **Rejected ARDY settings:**
  - 2 steps: severe foot sliding in 2 s windows, hidden by the model's own contact
    labels;
  - H8 without future context: does not follow its path;
  - bf16: 1.46× faster, but triples root jitter; needs visual review before use;
  - `torch.compile`: ≤ 1.2×. TensorRT is untried.
- **Response delay is replan wait plus about 0.5 s.** Median response after a command
  change is:
  - refill with 2 s chunks: 1.3–1.6 s;
  - replan every 1 s: 1.0 s;
  - replan every 0.5 s: 0.8 s.

  Replanning every 0.2 s is *slower* to respond: windows are dominated by the history.
- **Online is infeasible at the user's scale** (m = 4–16, ledger D09). At m = 16 and
  `S = 0.4`, ARDY would need 9.3 A40s at u = 1 and 18.6 at u = 0.5. The finer
  breakdown, kept for reference:
  - The simulator alone runs at `S = 0.41–0.68` on an A40 (G1 only 0.54).
  - Online ARDY fits m = 1 on a dedicated A40 **only if** event-driven replanning
    keeps u ≳ 0.6: refill, plus an immediate replan on behavior switches. Its
    zero-wait response measured about 0.4–0.45 s.
  - Periodic replanning every 0.5–1 s needs 1.2–2.3 A40s even for m = 1.
  - m = 4 needs ≥ 2.3 generation A40s, or an offline library/graph.
  - m = 8 is out of reach on one GPU.
  - Sharing one A40 synchronously would slow training by an estimated 1.6–1.8× (m = 1),
    3.4–4.1× (m = 4) and 5.8–7.3× (m = 8).
- **Without text, body style is uncontrolled and drifts.**
  - Both models sometimes produce hunched or crouched walking (head below 85% of
    neutral standing height).
  - For ARDY at the recommended setting, this rises from 3–6% of frames in 12 s
    rollouts to 24% over 60 s. Posture drifts toward stooping on long horizons.
  - Persistent walking styles need prompts
    ([text encoder access](../../decisions/text-encoder-access.md)), offline curation,
    or periodic resets.
- **Offline use** of ARDY (library plus motion graph) is in
  [motion library](motion-library.md).
- **Historical:** Kimodo on an L40S gave 7–28 clip-seconds per wall-second
  ([Kimodo smoke](../../experiments/2026-09-28-kimodo-smoke.md)).

## Caveats

- **Isolated capacity** is necessary, not sufficient. End-to-end runs must share the GPU
  with mjlab-warp, LiDAR and the policy, and must be compared against the same scene
  replaying cached motion (handoff §2).
- **Zero text features** measure the constraint-only path. Guidance runs the same number
  of passes as with real prompts.
