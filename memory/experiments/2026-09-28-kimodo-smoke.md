# Kimodo smoke and latency (L40S), historical

Run 2026-09-28 by an earlier session and reported in section 7 of the
[Kimodo stochastic humans proposal](../archive/2026-09-28-mjlab-kimodo-stochastic-humans.md). Recorded here on 2026-09-30 as the
earliest evidence. **Not comparable with A40 numbers.** The 2026-09-30
[benchmark](2026-09-30-motion-backend-benchmark.md) remeasures Kimodo on A40 with
continuation and per-stage timing.

## Setup

- One L40S (46 GB). Kimodo-SOMA-RP-v1.1 (HF revision `6c9233af`), 30 fps, 5 s clips
  (150 frames), fresh generation (no history).
- Zero text features: the gated Llama-3 encoder was unavailable.
- Code: `scripts/kimodo_smoke.py`, `scripts/kimodo_smoke.sbatch` (now
  `scripts/smoke/kimodo_smoke.py`, `scripts/slurm/kimodo_smoke.sbatch`).
- Outputs: `runs/20260928T000000Z_legacy_logs/kimodo_smoke_tf32.json`, `runs/20260928T000000Z_legacy_logs/slurm/kimodo-smoke-{3978179,3978200}.out`.
  The fp32 JSON was overwritten; its numbers survive only in the Slurm log and the design
  doc.

## Results

| Steps | Batch | fp32 wall | fp32 clip-s / wall-s | TF32 wall | TF32 clip-s / wall-s | Peak GB |
|---|---|---|---|---|---|---|
| 100 | 1 | 1.70 s | 2.9 | | | 1.2 |
| 100 | 16 | 21.2 s | 3.8 | | | 1.7 |
| 100 | 64 | 93.2 s | 3.4 | 45.8 s | 7.0 | 3.3 |
| 100 | 256 | 376 s | 3.4 | | | 9.5 |
| 25 | 1 | 0.45 s | 11.2 | | | 1.6 |
| 25 | 64 | 23.5 s | 13.6 | 11.6 s | 27.7 | 3.3 |
| 25 | 256 | 94.1 s | 13.6 | | | 9.5 |

- Throughput is compute-bound: it stops improving past batch 16, and TF32 doubles it.
- **Dense root path:** a 90° arc (3 m radius, 1.2 m/s), batch 8, constraint-only. Root
  error was 1.5 cm mean and 8.5 cm max. Post-processing added almost no time (4.81 s vs
  4.90 s) and did not reduce the max error (11 cm).
- **Contacts:** SOMA output has 6 foot-contact channels, not the 4 the README lists.
- **Multi-prompt continuation**, 2 × 5 s, batch 8: 9.7 s.
- **SOMA `xlo` skinning** (Warp): 2,400 poses × 612 vertices in 24 ms, all finite.
  The mesh has 1,220 faces.
