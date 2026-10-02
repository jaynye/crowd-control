# 2026-09-30: memory system and motion-backend benchmark

> Archived from `runs/20261001T003520Z_memory_archive/2026-09-30-memory-and-benchmark.md` on 2026-10-01; historical. Links were adjusted to the archive location.

## Request (user, 2026-09-30)

1. Create an agentic memory system like `~/flash/hrc` that organizes knowledge,
   decisions and proposals.
2. Start the motion benchmark from the handoff, using the repos here and the checkpoints
   in `~/flash/checkpoints`, to assess online generation. If it is infeasible, move to
   offline generation and a stochastic motion graph.

## Plan and progress

- [x] Orientation: handoff, hrc memory layout and checker, ARDY and Kimodo code, prior
  smoke results, cluster state.
- [x] Environment:
  - ARDY checkpoint alias symlinks;
  - `vector-quantize-pytorch` added to the root pyproject (lock diff was additions only;
    torch 2.9.0 and warp 1.14.0 kept);
  - `scripts/env.sh`.
- [x] Memory skeleton: `AGENTS.md`, `memory/{core,decisions,experiments,knowledge,journal}`
  and `scripts/check_memory.py`.
- [x] ARDY smoke on A40 (job 3991567).
- [x] Adapters and ARDY parity tests (job 3991757, 4/4).
- [x] Throughput harness and smoke (jobs 3991782, 3991801).
- [x] Full sweeps: ARDY job 3991802, Kimodo job 3991807 (supplementary rows still
  running), fast ARDY job 3991970, guidance bracket job 3992025.
- [x] Quality: jobs 3991920, 3991921 and 3991972 (commands on replan boundaries), and
  job 3992024 (jittered commands).
- [x] Profile and `torch.compile` (job 3991926); simulator speed (job 3991937).
- [x] Kimodo parity test (job 3992023); rollout videos; readout promoted to
  [motion backends](../core/design/motion-backends.md#conclusions). Decisions updated:
  backend choice, demand budget, offline runtime (with the ARDY-tabulated graph
  proposal), text encoder access.
- [x] Job 3991807 finished; benchmark note closed (no longer running).
- [ ] After the user answers the demand budget: an end-to-end mjlab run, or the graph
  prototype.

## Resume

```bash
cd ~/flash/crowd-control && source scripts/env.sh
squeue -u $USER
ls runs/
```

Results land in `runs/<UTC>_{ardy,kimodo}_throughput/rows.csv` and
`runs/<UTC>_quality*/rows.jsonl`.
