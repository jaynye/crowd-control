# Motion-backend benchmark: ARDY vs Kimodo for online crowds

Run 2026-09-30 on A40s. Handoff stages 1–2.

> Code paths and commands below are as run. Since the 2026-10-01 reorganization the
> code lives in `src/crowdmotion/` and runs through the `crowd` command line
> ([architecture](../../docs/architecture.md), [CLI](../../docs/cli.md)).

## Outcome

1. **Kimodo cannot drive the crowd online.**
   - Best case: 60 new-motion seconds per wall-second per A40 (bf16, 10 steps, 5 s
     windows, batch 64); 26 at 25 steps.
   - Guidance-off Kimodo was not tried. Even an optimistic 3× would leave it about 15×
     below ARDY.
   - Online continuation leaves visible seams: the root's per-frame travel jumps 7× more
     at call boundaries than inside a chunk.
   - Path error is 5 cm, 6× ARDY's.
   - It remains a candidate for **offline** SOMA-mesh motion generation.
2. **ARDY H40 is the online candidate.** At TF32, with no guidance and 5 denoising steps:
   - 2,800 new-motion seconds per wall-second per A40 from batch 256 up;
   - 0.5–0.9 cm path error, 0.04 m/s speed error, seams indistinguishable from the
     interior;
   - about 0.5 s from a replan to the new behavior;
   - stable over 60 s rollouts.
3. **Feasibility depends on `m`** (humans per environment) and `S` (simulator speed),
   which the user had not fixed at the time (since settled: [ledger](../core/design/decision-ledger.md) D09).
   - One A40 of isolated generation feeds 4,096 environments at `S* ≈ 0.69` (m = 1),
     `0.17` (m = 4) and `0.086` (m = 8).
   - Robot-only G1 already runs at `S ≈ 0.54` on one A40, and the crowd fixture at
     0.41–0.68.
   - These `S*` values assume u ≈ 1: buffer refill, or event-driven refill that also
     replans immediately on behavior switches. Event-driven refill was not run as a
     policy. Its response is bounded by the boundary-aligned runs (jobs 3991920,
     3991972), where the replan wait was zero and response took about 0.4–0.45 s.
   - So online ARDY fits m = 1 on its own A40 **if** event-driven replanning keeps
     u ≳ 0.6.
   - Periodic replanning every 1 s (u = 0.5) or 0.5 s (u = 0.25) needs 1.2 or 2.3 A40s
     even for m = 1.
   - m = 4 needs at least 2.3 A40s, or a library/graph. m = 8 is out of reach on one
     GPU.
4. **Response delay is dominated by the replan wait.** With changes spread uniformly in
   time, the median response is:
   - buffer refill (2 s chunks): 1.3–1.6 s;
   - replan every 1 s: 1.0 s;
   - replan every 0.5 s: 0.75–0.85 s.

   Replanning every 0.2 s is worse: 1.8 s to stop, and 22% never stop within the
   rollout. The first frames of each window mostly continue the history, so the change
   only appears in frames that get discarded.

## Setup

- **Hardware:** one A40 (46 GB) per job. The driver is 595.84 on ig-88, qt-1 and
  deebot, and 580.178.04 on perseverance and conroy; each `run.json` records its own.
  Nothing else ran on the GPU, so every number is isolated capacity.
- **Software:** torch 2.9.0+cu128, CUDA 12.8, warp 1.14.0, Python 3.13.
  - ARDY commit `693f74d1`; H8 `257a0843`, H40 `abe6c43b`.
  - Kimodo commit `58e78189`; weights `6c9233af`.
- **Adapters:** `crowdmotion/backends/{ardy,kimodo}.py`.
  - Initial noise comes from per-human (seed, counter) streams, and only new frames are
    counted.
  - Every call carries a root-path constraint.
  - Text features are zero (Llama is gated).
- **Parity tests:** `tests/test_{ardy,kimodo}_adapter.py` (now `tests/gpu/`), 5/5 on A40 (jobs 3991757 and
  3992023).
  - The ARDY adapter equals upstream `autoregressive_step` under identical noise, and
    its vectorized root conditions equal the library's.
  - The Kimodo continuation equals `_multiprompt`'s second segment (after its
    transition frames) to 1e-3.
  - Noise is independent of batch order and composition (within 1e-4).
- **Branch checks:** from a shared executed history, the history is unchanged
  bitwise and the same seed reproduces bitwise. Different seeds spread final poses by
  1.8 cm (2 steps) to 6–8 cm (5–10 steps).

## Throughput (A40, continuation, root-path constraints)

`R_new` is new-motion seconds per wall-second. `S*` assumes buffer refill (u = 1) and the
whole GPU; m is humans per environment.

| Setting | Batch | p50 wall | R_new | S* m=1 / 4 / 8 | Peak GB |
|---|---|---|---|---|---|
| ARDY H8 default (TF32, 3-pass, 10 steps) | 4096 | 8.59 s | 191 | 0.047 / 0.012 / 0.006 | 6.6 |
| ARDY H40 default (TF32, 3-pass, 10 steps) | 4096 | 16.4 s | 499 | 0.12 / 0.030 / 0.015 | 17.7 |
| ARDY H40 fp32 default | 4096 | 33.9 s | 242 | 0.059 / 0.015 / 0.007 | 17.7 |
| **ARDY H40 TF32, no guidance, 5 steps** | 4096 | 2.90 s | **2,824** | **0.69 / 0.17 / 0.086** | 7.2 |
| same | 16,384 | 11.7 s | 2,810 | same | 26.1 |
| ARDY H40 TF32, 2-pass guidance, 5 steps | 4096 | 5.56 s | 1,472 | 0.36 / 0.090 / 0.045 | 12.3 |
| ARDY H40 TF32, 3-pass guidance, 5 steps | 4096 | 8.23 s | 996 | 0.24 / 0.061 / 0.030 | 16.8 |
| ARDY H40 bf16, no guidance, 5 steps | 4096 | 1.99 s | 4,118 | 1.0 / 0.25 / 0.13 | 7.5 |
| ARDY H40 TF32, no guidance, 2 steps | 4096 | 1.32 s | 6,202 | 1.5 / 0.38 / 0.19 | 7.2 |
| ARDY H8 + 40 future frames, no guidance, 5 steps | 4096 | 3.09 s | 530 | 0.13 / 0.032 / 0.016 | 8.0 |
| Kimodo tf32, 25 steps, 1 / 2 / 5 s windows | 256 | 30 / 41 / 74 s | 8.5 / 12.6 / 17.3 | ≤ 0.004 | 5.0 / 6.3 / 10.4 |
| Kimodo bf16, 25 steps, 5 s windows | 256 | 49.8 s | 25.7 | 0.0063 / 0.0016 / 0.0008 | 8.4 |
| Kimodo tf32, 100 steps, 5 s windows | 64 | 74.2 s | 4.3 | 0.001 | 3.5 |
| Kimodo tf32 / bf16, 10 steps, 5 s windows | 64 | 7.6 / 5.3 s | 42 / 60 | 0.010 / 0.015 (m=1) | |
| Kimodo tf32, 25 steps, 2 s / 5 s windows, + post-processing | 64 | 14.3 / 24.3 s | 8.9 / 13.2 | | |

- **ARDY plateaus from batch ~256.** More batch buys no throughput, only latency.
- **Denoising is ≥ 97% of the time.** The profile at batch 1,024 splits CUDA time into
  matmuls 38%, attention 27%, and LayerNorm plus elementwise ops about 20%.
- **`torch.compile` (inductor) gives ≤ 1.2×** at batch ≤ 1,024 and 1.02× at 4,096, after
  40–170 s of compilation. TensorRT is ARDY's supported acceleration path and was
  **not tried**: it needs `tensorrt-cu12==10.13.0.35` in the env.
- **The window is a cost lever.**
  - H40 at batch 1,024: future context 0 → 40 → 160 frames gives 498 → 313 → 148.
  - History 4 → 20 → 40 frames gives 498 → 400 → 324.
  - H8 needs future context to follow paths at all (see quality), which makes it slower
    than H40.
- **Per-human constraint loops.**
  - ARDY's library path (one `Root2DConstraintSet` per human) costs 0.032 s at batch
    64, 0.53 s at 1,024 and 2.06 s at 4,096. The vectorized fill costs 0.004 s at
    4,096. The library path would add about 70% to the recommended setting's 2.9 s
    refresh.
  - Kimodo's full-body transition constraints cost 0.75 s at batch 256 (about 3 ms per
    human).
- **Fresh starts** cost about the same as continuations. H40 at batch 4,096: 530 vs 499
  at defaults.
- **Burst** (every human replans at once, m = 4, 16,384 humans): 11.7 s and 26 GB
  (TF32), or 7.9 s and 22.5 GB (bf16).

## Continuation quality (closed loop, 32 humans, 12 s; 60 s for the long cruise)

Commands change at a time drawn uniformly over one replan interval per human. Full
tables: `crowdmotion.bench.report` over the quality runs (now `crowd bench tables`).

| Setting, policy | Path err | Speed err | Skate (label) | Skate (height-gated) | Root-step change p95, seam / interior | Stop / turn response median (p95) |
|---|---|---|---|---|---|---|
| ARDY H40 default, refill | 1.8–2.1 cm | 0.05 | 0.03–0.04 | 0.24–0.31 | 0.9–1.3 / 0.6 cm | 1.6 (2.4) / 1.5 (2.3) s |
| ARDY H40 default, P 0.5 | 0.8–0.9 cm | 0.04 | 0.03–0.04 | 0.21–0.26 | 0.8–1.0 / 0.6 cm | 0.85 (1.2) / 0.65 (0.9) s |
| **ARDY H40 no guidance, 5 steps, refill** | 0.75–0.9 cm | 0.04 | 0.03–0.04 | 0.27–0.35 | 1.0–1.1 / 0.6 cm | 1.6 (2.4) / 1.5 (2.3) s |
| **same, P 1.0** | 0.5–0.7 cm | 0.03–0.05 | 0.02–0.04 | 0.22–0.31 | 0.9–1.1 / 0.6 cm | 1.0 (1.4) / 1.0 (1.4) s |
| **same, P 0.5** | 0.55–0.7 cm | 0.04–0.05 | 0.03–0.04 | 0.23–0.30 | 0.7–0.9 / 0.6 cm | 0.8 (1.1) / 0.75 (0.9) s |
| ARDY H40 no guidance, 2 steps, refill | 1.1–1.9 cm | 0.03 | 0.03–0.16 | **0.48–1.1** | | label-hidden sliding |
| ARDY H40 bf16, no guidance, 5 steps, refill | 0.8–1.0 cm | 0.05 | 0.04–0.06 | 0.25–0.36 | 1.6 / **1.7** cm (jitter) | as TF32 |
| ARDY H8 default, no future | 7.5–11 cm | **0.5–0.7** | | | | commands mostly never met |
| ARDY H8 + 40 future frames, no guidance, 5 steps | 1.2–1.9 cm | 0.07–0.11 | 0.03–0.04 | | 0.6 / 0.8 cm | 0.75 / 0.45 s (non-jittered) |
| Kimodo 2 s, 25 steps, refill | 4.6–5.1 cm | 0.15–0.17 | 0.05–0.06 | 0.17–0.23 | **3.3–3.5 / 0.45 cm** | 1.8 (2.3) / 1.6 (2.4) s |
| Kimodo + 5-frame forward blend | 4.5–5.0 cm | 0.15–0.17 | 0.05–0.06 | 0.16–0.23 | 2.5–2.7 / 0.45 cm | same |

- **No guidance at 5 steps is near parity with the defaults** (TF32, 3-pass, 10 steps).
  - Path tracking is better; response is equal.
  - Height-gated sliding is about 10–15% higher on turns.
  - This holds for zero text only; prompts may need guidance back
    ([text encoder access](../decisions/text-encoder-access.md)).
- **2 steps fails.**
  - In 2 s refill mode, 55–99% of height-gated planted frames slide faster than
    0.1 m/s.
  - The model's own contact labels hide this: 60–97% of frames carry no contact.
  - It is only acceptable when just the first 0.5 s is executed, and at u = 0.25 that
    gives no throughput gain.
- **bf16** triples within-chunk root jitter in refill mode (interior step-change p95
  1.6–1.8 cm vs 0.6 cm). It needs visual review before use.
- **ARDY H8 without future context** does not follow its 0.4 s path: humans lag and
  stall. With 40 future frames it tracks, but it is slower than H40.
- **Kimodo's seam jumps are inherent to online use.** Upstream `_multiprompt`
  cross-fades the *previous* segment's last frames, which is not possible once they
  have played. The parity test confirms the adapter matches upstream. A forward blend
  reduces the jump from 7× to 5.5× the interior.
- **Contact channels.** Kimodo's post-processing *raised* height-gated sliding (0.33 vs
  0.16 m/s on cruise, 6-human clips), and contact semantics differ between Core and SOMA.
  Do not rank the backends on the sliding metrics; compare within one skeleton.
- **Metric floor.** The height-gated metric reads about 0.14 m/s even in good settings,
  because swing-phase frames near the ground count as planted. Use it for relative
  comparisons only.
- **Style is uncontrolled without text.** Both models sample stylized postures from
  their training data. Measured from the saved 6-human clips; "hunched" means head below
  85% of the skeleton's neutral standing head height (Core 1.68 m, SOMA 1.59 m).
  - **ARDY recommended setting:** 3–6% of frames in 12 s rollouts, but **24% over 60 s
    cruise**. Posture drifts toward stooping on long horizons.
  - **ARDY at other settings:** defaults about 7–20%; history 20: 47%; H8 with future
    context: 22–40%.
  - **Kimodo:** 0% at 25 steps, 17% at 100 steps.
  - **Not comparable across backends:** ARDY's median head height is 89–93% of neutral
    and Kimodo's 96–98%, possibly partly a skeleton convention.
  - Hands above head: ≤ 0.3% of frames everywhere except H8 with future context during
    stop/start (3%).
  - Persistent per-person styles need prompts (or another style control) and a visual
    review. The metrics above do not measure upper-body naturalness.

## Demand reference (simulator alone, one A40, zero actions)

From `scripts/sim_speed.py`, job 3991937, `runs/20260930T215734Z_sim_speed`:

| Scene | 4,096 envs: env steps/s | S |
|---|---|---|
| G1 flat velocity (robot only) | 110,440 | 0.54 |
| Crowd fixture, 4 primitive humans + 180×16 LiDAR (no robot) | 139,629 | 0.68 |
| Crowd fixture, 8 humans | 83,670 | 0.41 |

Real crowd training adds G1, LiDAR, humans, policy inference and PPO updates, so its `S`
is below every row here.

**Estimated slowdown** if generation and simulation share one A40 synchronously:
`1 + S_sim / S*_gen`, taking `S_sim` = 0.41–0.54. This is an estimate from isolated
measurements, not an end-to-end run.

| Setting | m = 1 | m = 4 | m = 8 | A40s of generation needed at S = 0.4 (m = 1 / 4 / 8) |
|---|---|---|---|---|
| ARDY recommended (TF32, no guidance, 5 steps) | 1.6–1.8× | 3.4–4.1× | 5.8–7.3× | 0.6 / 2.3 / 4.6 |
| same, periodic replan every 1 s (u = 0.5) | 2.2–2.6× | 5.8–7.3× | 10–14× | 1.2 / 4.6 / 9.3 |
| ARDY with 2-pass text guidance | 2.1–2.5× | 5.6–7.0× | 10–13× | 1.1 / 4.5 / 8.9 |
| ARDY default settings | 4.4–5.4× | 14–19× | 28–36× | 3.3 / 13 / 26 |
| Kimodo best (bf16, 10 steps) | 29–38× | 110–150× | 220–300× | 27 / 109 / 218 |

The replan policy scales these through `u`:
- event-driven refill (refill, plus an immediate replan on a behavior switch) gives
  `u ≈ 1 − H / (2 T_switch)`, where `T_switch` is the mean time between switches;
  for example 0.8 when `T_switch` = 5 s;
- periodic replanning every 1 s gives u = 0.5; every 0.5 s gives u = 0.25.

## Runs

| Job | What | Output |
|---|---|---|
| 3991567 | ARDY smoke (`scripts/ardy_smoke.py`) | `runs/20260928T000000Z_legacy_logs/ardy_smoke.json` |
| 3991757, 3992023 | Adapter parity tests | `runs/20260928T000000Z_legacy_logs/slurm/crowd-pytest-*.out` |
| 3991782, 3991801 | Harness smoke | `runs/20260930T214718Z_bench_smoke`, `runs/20260930T214913Z_bench_smoke2` |
| 3991802 | ARDY sweep (`ardy_a40.json`) | `runs/20260930T214918Z_ardy_throughput` |
| 3991807 | Kimodo sweep (`kimodo_a40.json`) | `runs/20260930T215159Z_kimodo_throughput` |
| 3991808 | Quality smoke | `runs/20260930T215159Z_quality_smoke` |
| 3991920, 3991921 | Quality, command changes on replan boundaries (response times biased; other metrics valid) | `runs/20260930T215438Z_quality_{ardy,kimodo}` |
| 3991926 | ARDY profile and `torch.compile` | `runs/20260930T215533Z_ardy_profile` |
| 3991937 | Simulator speed | `runs/20260930T215734Z_sim_speed` |
| 3991970 | Fast-ARDY throughput (`ardy_fast_a40.json`) | `runs/20260930T220248Z_ardy_fast_throughput` |
| 3991972 | Fast-ARDY quality (group `ardy_fast`, non-jittered) | `runs/20260930T220258Z_quality_ardy_fast` |
| 3992024 | Jittered quality (group `jitter`) | `runs/20260930T221945Z_quality_jitter` |
| 3992025 | Guidance bracket and burst (`ardy_guidance_a40.json`) | `runs/20260930T221945Z_ardy_guidance_throughput` |

Rollout videos (`crowdmotion.bench.render`, now `crowd bench clips`) are next to their
clips as `clips_*.mp4`.
They show 6 humans per clip; call boundaries are marked in red.

## Caveats

- **Isolated capacity only.** No end-to-end run shares the GPU with mjlab-warp, LiDAR and
  a policy. The slowdown table is an estimate.
- **Zero text.** The no-guidance result may not survive style prompts; the 2- and
  3-pass guidance rows bracket the cost.
- **A40 only.** Other GPUs need their own measurements.
- **Early response times are biased.** In the non-jittered runs (jobs 3991920, 3991921,
  3991972), command changes fell on replan boundaries. Use only job 3992024 for response
  times.
- **Scope.** Quality uses synthetic arc and straight-line paths. There are no obstacles,
  no neighbours, and no visual review beyond spot checks of the videos.
