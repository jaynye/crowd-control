# Backend choice: which motion backend drives the humans?

> Archived from `runs/20261001T003520Z_memory_archive/backend-choice.md` on 2026-10-01; historical. Links were adjusted to the archive location.

Opened 2026-09-30. **Whose call:** empirical (the benchmark), then the user. The
benchmark is done; the choice now waits on the user's budget
([demand budget](2026-09-30-decision-demand-budget.md)).

## Question

Which backend realizes human body motion in training?

- ARDY online;
- Kimodo online;
- an offline-generated library stitched online;
- a hybrid: library locomotion plus online generation for unusual transitions.

The handoff (§3) prefers online generation if it meets the measured budget. It asks
for a choice on end-to-end throughput, response delay, continuation quality, VRAM and
existing mesh compatibility.

## Evidence (2026-09-30, one A40, isolated)

Full numbers: [benchmark](../experiments/2026-09-30-motion-backend-benchmark.md).
Conclusions: [motion backends](../core/design/motion-backends.md#conclusions).

| Option | Capacity for 4,096 envs at S = 0.4 | Quality | Mesh |
|---|---|---|---|
| **ARDY H40 online** (TF32, no guidance, 5 steps) | At u = 1: m = 1: 0.6 A40; m = 4: 2.3; m = 8: 4.6. With 1 s periodic replans (u = 0.5): 1.2 / 4.6 / 9.3 | 0.5–0.9 cm path error, clean seams, ~0.5 s response after a replan | Core skin, 9k vertices (not SOMA) |
| ARDY H40 online with text guidance (2-pass / 3-pass) | m = 1: 1.1 / 1.6 A40 | Not measured with real prompts | Core |
| **Kimodo online** | m = 1: 27 A40s (bf16, 10 steps) | 5 cm path error, 7× seam jumps | SOMA (612-vertex `xlo`) |
| **Offline library + stitching** | Library cost is tiny: ARDY makes 10 h of motion in about 13 s of A40 time, Kimodo in 10–23 min (10–25 steps) | Unmeasured: depends on the stitching runtime | Either |

Two further levers were not tried:
- TensorRT, ARDY's supported acceleration;
- bf16 after a visual review: 1.46× faster, but it adds root jitter.

## Recommendation (agent, pending the user's budget)

- **If m = 1, a dedicated generation A40 is acceptable, and behavior switches are
  infrequent enough for u ≳ 0.6:** run ARDY H40 online.
  - Settings: TF32, no guidance, 5 steps.
  - Replanning: event-driven refill (refill, plus an immediate replan on behavior
    switches). Zero-wait response is about 0.4–0.45 s.
  - Periodic replanning every 0.5–1 s responds within about 1 s even without switch
    events, but needs 1.2–2.3 A40s for m = 1.
- **If m ≥ 4, or generation must share the training GPU:** use an offline library with
  an online stochastic graph ([offline runtime](2026-10-01-decision-offline-runtime.md)).
  - Generate the library with ARDY; it is cheap, and ARDY's continuation quality is
    better.
  - Or generate with Kimodo, if a SOMA mesh matters more than cost.
  - Keep online ARDY for evaluation and rare transitions (the hybrid).
- **Either way:** do not use Kimodo online.

## What still decides it

- `m`, target `S`, the training GPU, and whether generation may have its own GPU
  ([demand budget](2026-09-30-decision-demand-budget.md)).
- Whether style prompts need guidance ([text encoder access](../decisions/text-encoder-access.md)).
- The body model for LiDAR ([human body model](../decisions/human-body-model.md)).
