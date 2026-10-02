# Demand budget: what throughput must a motion backend sustain?

> Archived from `runs/20261001T003520Z_memory_archive/demand-budget.md` on 2026-10-01; historical. Links were adjusted to the archive location.

Opened 2026-09-30. **Whose call:** the user. It now blocks the backend choice.

## Question

The handoff defines demand as `R_required = 4096 · m · S`, where `m` is humans per
environment and `S` is simulated seconds per wall-clock second per environment. It does
not fix `S`, `m`, the training GPU, or whether the 4,096 environments share one GPU. It
also leaves the acceptable training slowdown and simulated-time response delay open,
and says not to invent pass thresholds.

## What the user needs to decide

1. **`m`:** 1 first, then 4 and 8 (handoff). Which is the training target?
2. **Target `S`,** or accept the achieved speed of the full scene with cached motion.
   That end-to-end baseline does not exist yet.
3. **Hardware:** the training GPU type and count, and whether motion generation may have
   its own GPU(s), for example an asynchronous generation server.
4. **Limits:** acceptable wall-clock slowdown, and acceptable simulated-time response
   delay (how long a human may take to react to a behavior switch).

## Evidence (2026-09-30, A40)

- **Simulator alone** at 4,096 environments, zero actions, no policy
  ([benchmark](../experiments/2026-09-30-motion-backend-benchmark.md#demand-reference-simulator-alone-one-a40-zero-actions)):

  | Scene | S |
  |---|---|
  | G1 flat velocity | 0.54 |
  | Crowd fixture, 4 humans + LiDAR | 0.68 |
  | Crowd fixture, 8 humans + LiDAR | 0.41 |

  Real crowd training (G1 + humans + LiDAR + policy + PPO) is slower.
- **Best acceptable online generator** (ARDY H40, TF32, no guidance, 5 steps):
  break-even `S* = 0.69 / 0.17 / 0.086` for m = 1 / 4 / 8 on one A40 in isolation.
- **Estimated synchronous slowdown** on one shared A40: 1.6–1.8× (m = 1), 3.4–4.1×
  (m = 4), 5.8–7.3× (m = 8).
- **Burst:** all 16,384 humans (m = 4) replanning at once take 11.7 s and 26 GB (TF32).

## What would close it

The user's answers to 1–4. Then an end-to-end run: the same scene with cached motion vs
the chosen backend, sharing the GPU with mjlab-warp, LiDAR and the policy.
