# SONIC released planner batching and reuse

The release is fetched and pinned. **Dynamic batching is reachable through a hand
port**, with closely matching FP32 outputs at batch 4,096, but one A40 serves only
**239 plans/s** against 40,960 plans/s requested at 10 Hz. Boundary-only ONNX surgery
fails; IO binding does not help materially; stock onnx2torch does not convert the
model. Use the released planner as an offline prior/teacher for now. No training or
SONIC tracker integration was performed.

User authorized 2026-10-18: fetch NVIDIA GEAR-SONIC and GR00T-WholeBodyControl,
inspect and test batching, measure 4,096 plans on one A40, and cost reuse options.
No training. Implementation stays in `scripts/smoke/sonic_planner_batching.py`.

## Plan

1. Pin and download release/source into `checkpoints/GEAR-SONIC` and
   `third_party/GR00T-WholeBodyControl`; preserve root lock and torch/warp.
2. Inspect shapes, random sampling and token generation. Establish deterministic
   batch-1 reference cases. Test graph surgery/ORT IO binding and a torch candidate.
3. Compare outputs to references, then time synchronized full plans at batch 4,096
   (microbatch if necessary) on one A40 through Slurm. Record startup separately.
4. Cost direct command conditioning, distillation and offline G1 graph references.
   Promote durable findings to core/architecture decision; run memory checks.

## Environment and provenance

- Login host: `sky1.cc.gatech.edu`; inference and builds use compute allocations.
- Existing working tree is dirty, including `uv.lock`; preserve its starting bytes.
- Sandbox shell fails with `bwrap: loopback: Failed RTM_NEWADDR`; commands use reviewed
  execution fallback.
- Skill: `/nethome/jni66/.agents/skills/hf-cli/SKILL.md`.
- Sources: <https://huggingface.co/nvidia/GEAR-SONIC>,
  <https://github.com/NVlabs/GR00T-WholeBodyControl>.

## Commands and jobs

- Source: `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git third_party/GR00T-WholeBodyControl`.
- HF info: `HF_HUB_OFFLINE=0 hf models info nvidia/GEAR-SONIC --format json`.
- Download: `HF_HUB_OFFLINE=0 .venv/bin/hf download nvidia/GEAR-SONIC --revision 6733128a3d8a523b1418b06bca3cdf61c8b0987f --exclude 'bones_seed_smpl/*' --exclude 'media/*' --exclude '*.gif' --local-dir checkpoints/GEAR-SONIC`.
  An earlier CLI invocation used variadic excludes and downloaded some media instead;
  the corrected repeated-option command fetched the model release (4.0 GiB total).
- Source commit `45c03facbf848e0c743091b7d9c17952096ff908`;
  HF commit `6733128a3d8a523b1418b06bca3cdf61c8b0987f` (not gated).
- Probe Python 3.13.15, NumPy 2.5.1, ONNX 1.22.0; existing hub library 1.33.0.
- Initial lock SHA256 `c98d19b20a71abdf6903b2a02618300502ae490df148013a6d2a10dc777dc2b8`;
  torch 2.9.0, warp-lang 1.14.0. No uv sync.
- CPU job 4133507 installed `onnxruntime-gpu==1.24.4 onnx2torch==1.5.15`
  with `uv pip install --python .venv/bin/python --target .cache/sonic-probe --no-deps`.
- CPU inspection 4133510 failed because download was still in flight;
  `runs/20261018T164634Z_sonic_inspect`. Retry 4133513 succeeded:
  `runs/20261018T164722Z_sonic_inspect`.
- CPU converter job 4133519 first found missing optional torchvision import.
- A40 ORT job 4133520 on gundam: batch-1 references, IO binding, 4,096 serial full
  plans, then boundary-dynamic graph test. Command:
  `PYTHONPATH=.cache/sonic-probe .venv/bin/python scripts/smoke/sonic_planner_batching.py ort`.
- Optional converter dependency job 4133524: `uv pip install --python .venv/bin/python --target .cache/sonic-probe --no-deps torchvision==0.24.0`.
- All jobs: `sbatch --partition=rl2-lab,overcap --qos=short --exclude=bishop --nodelist=gundam --cpus-per-task=4 --time=00:30:00 --output=logs/slurm/%x-%j.out --wrap='<command>'`; GPU job adds `--gres=gpu:a40:1`.
  Node checked before submission: FreeMem 155,964 MB, idle.


## Inspection and current results

- Planner SHA256 `39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea`;
  773,952,989 bytes. Opset 17.
- One flat graph, 7,932 nodes; no Loop, Scan, If, RandomUniform or Multinomial.
  486 Reshape, 203 Expand, 49 ScatterND, 14 ScatterElements, 272 Gather.
- Transformer stages: 3 shared layers + 3 root-token layers (width 512), then
  16 pose-token layers (width 1,024), 16 attention heads. The exported generation
  schedule is fixed and mask-based, not a runtime autoregressive token loop.
  Token count and pose tokens use ArgMax, with a padded 16-token buffer and
  `num_pred_frames = num_tokens * 4`. Random selection is seed-fed arithmetic;
  no provider RNG operator.
- Shape examples: first `/Reshape` consumes `[1,3,3]`; attention exports hard-code
  `[9,16,32]`, `[25,16,32]`, `[16,16,64]`. Blindly replacing every 1 would corrupt
  scalar, time, joint, head and broadcast dimensions.
- Actual outputs: qpos `[1,64,36]` float32; count `[1]` **int32**, unlike the
  docs' scalar/int64 description. Four reference cases yield 44/44/44/64 valid frames.
  No planner reference `.pt` is present in this HF revision; references come from
  unchanged release inference, not a claimed upstream golden.
- Batch-1 repeat and CUDA IO binding are bit exact on the saved inputs. Ordinary
  `.run`: 56.2192 s per 4,096 plans, 72.8576 plans/s, p50 13.7477 ms / p95 13.8160 ms.
  CUDA IO binding: 55.9162 s, 73.2525 plans/s, p50 13.6744 / p95 13.7189 ms.
  These are actual full population refreshes, not a latency extrapolation.
- ORT startup 2.300 s plus first inference 0.445 s; later first-case calls 13.9–14.2 ms.
  A40 on gundam, one visible GPU, driver 595.84, CUDA 12.8 / torch 2.9.0+cu128,
  ORT GPU 1.24.4, FP32, TF32 disabled in provider, ORT graph optimization enabled,
  CPU intra/inter-op threads 1.
- Boundary-only dynamic ONNX retains exact batch-1 parity; B2 fails `/Reshape`:
  input `[2,9]`, requested `[1,3,3]`. IO binding cannot change these graph semantics.
- `onnx2torch==1.5.15` lacks OneHot, ScatterElements, GreaterOrEqual, LessOrEqual;
  conversion stops at OneHot. There are no candidate outputs to compare or time.
- The throwaway operator hand port constant-folds shape subgraphs and preserves
  original logical batch-1 dimensions under an independent `torch.vmap` axis.
  3,428 runtime operations / 4,922 static values. Four batch-1 comparisons have
  maximum qpos errors 7.93e-6 / 1.65e-5 / 1.73e-5 / 5.26e-6; frame counts exact.
  Its raw vmap output is `[B,1,64,36]` plus `[B,1]`; the singleton logical
  ONNX axis can be squeezed by a future adapter. B2 works and agrees closely. Moving Slice indices and Clip scalar limits out of
  capture enables full CUDA graph replay. Full B4096 works with maximum reference
  errors 9.78e-6 / 1.70e-5 / 8.46e-6 / 7.51e-6 and exact frame counts. Repeated
  equivalent rows vary by at most 1.22e-5 (floating-point batching differences);
  random seeds remain per request, with no shared provider RNG.

### A40 population throughput (job 4133685)

All timings are synchronized full forward plans including preprocessing, networks,
decoding and postprocessing. Inputs stay on GPU; initialization, GUI, serialization,
30→50 Hz resampling, tracker and simulator are excluded. Six synchronized latency
samples follow warmup; a separate measured population of 4,096 requests is timed.
For microbatches, this is the sum of sequential CUDA graph replays, not multiplied
batch-1 latency. Four heterogeneous command/history/seed cases repeat across rows.

| Microbatch / batch | 4,096-plan refresh, s | Plans/s | Batch p50 / p95, ms | Peak allocated GPU memory, GB (cumulative) |
|---|---:|---:|---:|---:|
| 2 | 26.9968 | 151.72 | 13.256 / 13.987 | 0.809 |
| 16 | 18.6963 | 219.08 | 73.300 / 74.188 | 0.809 |
| 64 | 17.5503 | 233.39 | 271.067 / 274.785 | 0.809 |
| 256 | 17.5006 | 234.05 | 1,090.723 / 1,093.473 | 0.936 |
| **4,096** | **17.1216** | **239.23** | **17,130.733 / 17,149.095** | **3.515** |

The 10 Hz requirement is 40,960 plans/s (0.1 s per population refresh), **171.2×**
above full-batch FP32 capacity. Its isolated break-even simulator speed is
`239.23 / 40960 = 0.00584` simulated-seconds/wall-second. This is a prototype
measurement, not a ceiling on an optimized native implementation. Static-shape
constant folding and CUDA capture do not remove the dense network's work.

### Reduced precision and placement

- Mixed FP16 (autocast for all eligible operations), job 4133696,
  `runs/20261018T165911Z_sonic_torch_amp16`: batch-1 max errors
  0.06831 / 0.68691 / 0.06998 / 0.00911, counts unchanged. It is not a
  reference-preserving candidate; qpos mixes meters, quaternion entries and radians,
  so these errors must not be described as centimeters or gait quality scores.
  Microbatch 16 serves an actual 4,096 requests in 43.8001 s (93.52 plans/s),
  p50/p95 171.111/171.483 ms, peak allocated 2.481 GB. Cancelled after this
  measured population and divergent outputs; no need to finish its full sweep.
  Autocast casts and unfused vmap operator kernels make this a poor FP16 port,
  not an optimized mixed-precision limit.
- ORT profiling job 4133697, `runs/20261018T165911Z_sonic_profile_fp32`:
  8,823 CUDA node events versus 66 CPU events over three profiled forwards;
  summed node durations 220,434 versus 187 us, including startup. Do not use this
  profiled aggregate as steady-state latency. CPU fallback is minor here; IO
  binding cannot turn batch-1 internal shapes into vectorized shapes.
- Dense multiply/add count from actual model shapes: MatMul 4.587323188 GF,
  Gemm 0.590360576 GF, Conv 1.428553728 GF: **6.606237492 GF/plan**,
  excluding elementwise work. At 40,960 plans/s that is 270.59 TF/s. Even ignoring
  convolutions, matrix products alone require 212.08 TF/s. NVIDIA lists an
  optimistic dense FP16 tensor peak of 149.7 TF/s for A40
  ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/proviz-print-nvidia-a40-datasheet-us-nvidia-1469711-r8-web.pdf)).
  This arithmetic comparison is an inference, not a hardware measurement: a dense
  evaluation of the released full model cannot reach the requested rate merely
  by changing batch metadata. Compression, reuse, reduced work or multiple GPUs
  would be needed, before simulator contention.
- TF32 (matmul and cuDNN enabled), job 4133727,
  `runs/20261018T170104Z_sonic_torch_tf32`: batch-1 max errors
  0.08678 / 0.19590 / 0.00450 / 0.00892, frame counts exact. Microbatch 16 serves
  an actual 4,096 requests in 17.5084 s (233.95 plans/s), p50/p95
  67.853/68.939 ms. Cancelled after observing these errors and measuring that
  population; no reference-preserving speed gain. No threshold for downstream
  gait quality was invented: both precision candidates change the reference
  substantially relative to the FP32 port; physical consequences were not tested.

### Candidate decision

| Candidate | Reference check | Dynamic batch outcome | Decision |
|---|---|---|---|
| Original ONNX + ORT GPU IO binding | Exact batch-1 qpos/counts | Original graph rejects B>1 | Useful for offline generation; 73.25 plans/s |
| Boundary-dynamic ONNX surgery | Exact batch-1 qpos/counts | B2 fails first fixed `[1,3,3]` reshape | Metadata editing is insufficient; semantic lifting/re-export would need every internal reshape/index/attention axis handled |
| Stock onnx2torch | No output: conversion fails at OneHot | Not reached | Repairing operators alone would still leave baked singleton semantics; hand port is more direct |
| Operator hand port + vmap + CUDA graphs, FP32 | Four cases and B2/16/64/256/4096 closely match; counts exact | **B4096 executes** | Technically viable batching, insufficient measured online capacity |
| Same hand port, autocast FP16 / TF32 | Max output changes 0.687 / 0.196 | B16 executes, 4,096 requests measured | Not adopted as a reference-preserving candidate |

A dynamic ONNX export of the hand port was not attempted. The boundary surgery failure
does not prove all graph surgery impossible; vmap's independent leading axis shows
how a semantic lifting could work. Even if that export removed dispatch overhead,
it would retain the dense arithmetic budget above. No TensorRT, quantization,
network pruning, student training or end-to-end simulator measurement is claimed.



### Captured replay correctness

Final A40 job 4133778 (`runs/20261018T170552Z_sonic_torch_fp32_validate`), command
`PYTHONPATH=.cache/sonic-probe .venv/bin/python scripts/smoke/sonic_planner_batching.py torch --validate-only`,
checks the actual captured outputs at B4096, then changes all GPU input buffers in
place by rotating the request order and replays the same graph. Both sets match their
corresponding batch-1 references to a maximum 1.7047e-5, with exact frame counts.
This verifies the captured graph computes from current inputs. No new timing claim
comes from this validation run; throughput remains job 4133685's measured result.
Node FreeMem was rechecked at 150,129 MB before submission. Job completed.

## Reuse costs and engineering estimates

These are planning estimates, not measured implementation/training durations. GPU
figures are isolated planner cost; there is no dollar estimate without an A40 hourly
rate. Multiply GPU-hours below by the relevant rate. All options also need the
still-unmeasured frozen tracker in mjlab: G1 model/joint order, PD gains, action
scales, observation histories, 30→50 Hz interpolation and blending. Budget roughly
3–7 engineer-days for that shared integration and stability/throughput checks;
training or sim2sim quality is not demonstrated by this experiment.

| Reuse option | What the policy emits | Additional engineering estimate | Compute and limitations |
|---|---|---|---|
| Frozen planner, velocity/facing | Planar movement angle, positive speed, independent facing angle; idle as an explicit mode | 2–4 days after tracker integration for batching, buffers, reset/seed handling and RL action wrapper | Zero planner training. Pays full measured planning cost at every replan. `target_vel <= 0` means **mode default**, not stop; use idle for stop. Planner has no LiDAR/collision awareness; the RL teacher supplies avoidance. |
| Frozen planner, waypoints | Four planar positions and four headings (12 outputs), or full xyz+heading (16) | Add 2–4 days to velocity wrapper for frames, reachability, interpolation and tests | Same neural planner cost, not a speed remedy. The four targets are one token's four frames (0.133 s at 30 Hz), not four evenly spaced waypoints over the full 0.8–2.1 s output. |
| Distill released planner | Small conditional sequence model taking history and the same commands, generating G1 qpos/tokens | 1–3 weeks for dataset, model, losses, deployment and closed-loop validation | Reuses the prior as teacher, but still trains a new student. At stock 73.25 plans/s, 1M labels = 3.79 A40-h, 10M = 37.92 h, before filtering or tracking. At 44 frames/plan, qpos labels alone are 6.34 GB/M examples (9.22 GB with all 64 padded frames); history+commands add more. Student training hours/accuracy are unmeasured. Averaging multimodal motions, contact/rotation consistency and distribution shift require care. No distillation was run. |
| Offline G1 references + graph | Policy supplies commands to graph selection, while frozen SONIC tracks graph frames | 1–2 weeks for G1 payload/contacts/FK, coordinate conversion, graph selection and seam validation | Generate once, then amortize over every RL iteration. Existing graph stores Core-style joint positions in Y-up; it cannot feed G1 qpos directly. Preserve 29 joint angles and root quaternions in a parallel payload, use G1 FK descriptors/contact/support-foot velocities, convert Z-up→Y-up for graph features, and re-root only SE(2) without losing ground height. New G1 adapter/payload must not alter existing human outputs or saved validity. |

**Offline arithmetic (stock measured rate):** with 44 valid frames, each plan produces
1.467 s of references. Using every frame would supply 107.4 reference-seconds per
wall-second: 100 reference-hours in 0.931 A40-h. Consuming 1 s per replan instead
costs 1.365 A40-h per 100 reference-hours; consuming only 0.1 s costs 13.65 h.
These are generation-only estimates: independent clips have no validated seams, and
open-loop chained plans may drift. Count consumed frames and filter contacts/seams.
100 reference-hours of qpos alone at 30 Hz and float32 require 1.56 GB; FK, velocities,
contacts and graph candidates are additional storage. Existing exact kNN build is
quadratic (`torch.cdist` in chunks); choose subsampling/approximate search for millions
of boundaries, rather than assuming build cost is free. No library was generated.

**Tracker context correction:** the release's `UpdateContextFromMotion` samples the
current planned animation, with lookahead; it does not normally replan from the
robot's measured recent qpos. Feeding actual executed qpos is an available input
choice for our wrapper, but its tracking/stability behavior needs testing. This
corrects the earlier architecture table's claimed advantage over an offline graph.

The existing human graph timing (65,536 streams in 47 ms) is a historical Core result,
not a measurement of G1 reference generation, seam tracking, or SONIC control.
No task here authorizes replacing D18's RL teacher with a designed navigation teacher.



## Attempt records

| Jobs | Run record | Outcome |
|---|---|---|
| 4133510 | `runs/20261018T164634Z_sonic_inspect` | Failed before model download completed |
| 4133513 | `runs/20261018T164722Z_sonic_inspect` | Inspection, node dump and shape constants |
| 4133519 / 4133531 | `runs/20261018T164913Z_sonic_convert`, `runs/20261018T165137Z_sonic_convert` | Missing torchvision, then unsupported ONNX operators |
| 4133520 | `runs/20261018T164913Z_sonic_ort` | Reference bank, GPU run/IO-binding timing, boundary surgery failure |
| 4133530 / 4133533 | `runs/20261018T165137Z_sonic_torch`, `runs/20261018T165159Z_sonic_torch` | Draft fixes: Expand broadcasting, optional Pad scalar |
| 4133568 / 4133681 | `runs/20261018T165234Z_sonic_torch`, `runs/20261018T165350Z_sonic_torch` | Parity/vmap succeed; capture fixes for Slice indices and Clip limits |
| 4133685 | `runs/20261018T165436Z_sonic_torch` | Completed FP32 B4096 benchmark and reference checks |
| 4133696 | `runs/20261018T165911Z_sonic_torch_amp16` | Divergent outputs; measured population, then cancelled |
| 4133697 | `runs/20261018T165911Z_sonic_profile_fp32` | ORT provider profile |
| 4133727 | `runs/20261018T170104Z_sonic_torch_tf32` | Divergent outputs; measured population, then cancelled |
| 4133778 | `runs/20261018T170552Z_sonic_torch_fp32_validate` | Captured B4096 and changed-input replay match batch-1 references |
| 4137474 (CPU; 4137472 failed: script on login-node `/tmp`) | no run directory (`scripts/smoke/motionbricks_weights_match.py`) | 2026-10-18 addendum: the MotionBricks root checkpoint matches the ONNX planner for 146 of 150 tensors, 34,115,852 of 34,122,833 parameters |

Reproduce the final script with `... sonic_planner_batching.py inspect|convert|ort|torch|profile`;
set `PYTHONPATH=.cache/sonic-probe` for ORT/conversion/torch, and optionally
`SONIC_PRECISION=amp16` or `SONIC_PRECISION=tf32` for the rejected precision candidates.
Use the `sbatch` wrapper above; omit `--gres` for inspection/conversion.
The script creates a new timestamped RunDir and saves config/command, environment,
git diff and its own script snapshot. The runs retain raw `result.json` metrics.
An attempted interactive `srun` allocation was refused by cluster policy (only
batch jobs with short QoS); all actual computation used `sbatch`.

## Final scope and recommendation

*Correction (2026-10-18, after the user's question about training-only use):* the
171× shortfall applies to synchronous 10 Hz replanning. For training, the measured
PPO iteration (3.27 s collection + 0.20 s learning at 4,096 envs) gives 3.4× / 5.7×
training time at 1 / 2 Hz replans with this port; see the
[architecture note](../decisions/robot-policy-architecture.md#training-cost-of-the-frozen-planner-online-2026-10-18).

Start with reuse of the released motion prior, not a planner trained from scratch.
At this scale the most economical no-training route is offline G1 qpos/FK clips plus
our graph selection pattern, **conditional on** frozen SONIC tracking G1 seams.
Velocity/facing is the smallest navigation action space, whether backed by an offline
graph or a future compressed planner; four waypoints add control but do not lower cost.
If online history conditioning proves necessary, distill/compress the released planner
rather than learning its motion prior anew; this would require a separate training
request. At the measured FP32 batched rate, 1M/10M labels would take 1.16/11.61 A40-h,
and 100 reference-hours using 44 frames per plan would cost 0.285 A40-h. These are
arithmetic projections from the synthetic-command benchmark, not generated datasets.

Deployment defaults replan ordinary walking periodically at 1 Hz, with immediate
command-change replans; that does not meet the requested 10 Hz changing-policy workload
without further evidence on response and reuse. For the current 76 ms simulator step
at 50 Hz, synchronously adding this 17.12 s planner every five steps would give about
46× slowdown if costs simply added; GPU contention could change that estimate.
The reported break-even speed is the more direct isolated result.

Four synthetic standing-history cases exercise walking, translation/yaw, independent
facing, explicit waypoints, idle and the 64-frame horizon; this is not coverage of all
27 modes, random contexts, long continuations, contact fidelity or tracker stability.
Reference `.npz` inputs/outputs are in the ORT run. The successful final FP32 probe
snapshot is `runs/20261018T165436Z_sonic_torch/probe.py`; later precision/profile
runs also retain the exact script snapshot. Initial draft failures predate snapshots
and are described above/logged, not promoted as candidates.

Only the throwaway smoke script and source-pin manifest implement this experiment;
production code/configs and golden baselines are unchanged. Model files and the
source clone are untracked. Root lock checksum and torch/warp versions are checked
at completion. No training, commit, push, W&B write or tracker integration.

Final validation records the unchanged lock checksum above, torch 2.9.0 and
warp-lang 1.14.0; `ruff check`, Python syntax compilation and
`scripts/check_memory.py --check` pass (no broken links, unlogged runs or
outward experiment references).
