# Goal-reaching planner objective and non-reactive graph-crowd ablations

Status: running (arrays submitted 2026-10-21 20:23 UTC).

User authorized implementation and a dense ablation set on intermediate rewards,
credit horizon and related confounders, in parallel with non-reactive graph humans
that can overlap one another. SONIC remains frozen. No commits or pushes.

Plan:
1. Add a persistent, observable one-goal task with terminal success/time rewards,
   collision/fall/deadline failure, opt-in intermediate terms and correct terminal shaping.
2. Add a shared non-reactive graph command path: persistent roaming, optional coherent
   flow, no interhuman/robot avoidance or separation; retain boundaries and motion quality.
3. Preserve legacy defaults/replay, add unit checks and compute-node GPU preflight,
   including resume and evaluation.
4. Freeze a matched factorial matrix separating reward terms, discount and crowd
   interactions; use multiple seeds, W&B and requeue-safe A40 jobs.
5. Queue condition-wise safety and goal-completion evaluations for every lineage;
   record run IDs/jobs/provenance and update core before handoff.

All heavy work runs through Slurm. Historical context and rationale:
[proposal](../decisions/planner-objective-and-crowd.md),
[previous curriculum results](2026-10-21-blind-crowd-arms.md).

## Implemented task and matched design

`--navigation` embeds a strict NavigationCfg in each lineage. All arms have the same
4–8 m persistent goals, 0.8 m/s nominal speed, 30 s deadline, goal-relative/time
observations and frozen SONIC. Arrival earns 10 + 2/(1 + elapsed/nominal time), failure
earns zero unless explicitly ablated. Collision, fall and deadline are task terminals;
failure wins simultaneous arrival. Collision geometry/threshold is unchanged (5 cm
penetration). Legacy task defaults and saved graph formats are unchanged.

Potential shaping acts once per planner decision with PPO's discount and zero
terminal potentials. Pure sparse/potential arms have no residual or raw-env intermediate
costs. Independent walkers bypass every neighbor intervention, stopping, separation and
graph clearance in EncounterBehavior; boundary handling/ordinary turns and graph
quality preferences persist. Configured flow membership reflects at boundaries, so
actual velocity order is measured rather than equated with its nominal fraction.

48 arms, 4,096 environments, 1 Hz, fixed training distributions:

| Group | Factors | Arms |
|---|---|---:|
| Core factorial | sparse / potential / all intermediate terms × gamma 0.75 / 0.99 per second × reactive-blind / independent × seeds 42 / 43 | 24 |
| Middle discount | sparse gamma 0.95 × both crowds, seed 42 | 2 |
| Intermediate mechanisms | tracking / proximity / residual regularization / centered full dense × both crowds at gamma 0.99, seed 42 | 8 |
| Other credit/reward controls | GAE 0.90 / rollout 32 s / no terminal speed bonus / failure reward −10 × both crowds, seed 42 | 8 |
| Independent traffic | density 4 / density 16 / 90% shared initial heading × seeds 42 / 43 | 6 |

The default is GAE 0.95, 16 s rollouts and 120 iterations (1,920 s/environment).
The 32 s rollout probe uses 60 iterations for matched simulated exposure; its number
of optimizer updates is necessarily different. Single-seed probes are diagnostic,
not a replicated interaction study. No adaptive difficulty controller changes exposure.

The centered dense control subtracts the tracking terms' positive ceiling (5/s at
the fixed weights). Positive intermediate rewards can otherwise reward delaying a
terminal arrival. Centering intentionally changes variable-duration episode returns;
it is not a policy-invariant transformation, and its negative costs with free failure
can create an early-failure incentive. Read success, collision/fall and deadline outcomes
together, not reward alone. The pure sparse failure-cost probe tests another part of
this incentive question; the full centering × failure-cost interaction is not covered.

## Frozen campaign and evaluation

Final plan: `runs/20261021T201831Z_navigation_ablations`. It stores every argv/config,
projected run/evaluation ID, seven fixed evaluation conditions, copied package/mjlab/scripts
and SHA256 for every copied source. The earlier unsubmitted plan
`runs/20261021T201456Z_navigation_ablations` is marked superseded before submission:
two middle-discount potential arms were replaced by centered dense controls; no jobs
were submitted from it. No measured campaign results informed this design change.

Training uses at most four A40 array elements, evaluations at most two. All lineages
use W&B online in the main project, with labeled navigation-ablation/group/crowd tags.
Each training lineage requeues in place; the condition-grid evaluation resumes partial
results without rerunning completed checkpoint/condition pairs. Frozen source is used
for both, so later workspace edits cannot change an arm on requeue.

Every arm's last three checkpoints receive the same paired evaluation, seed 101,
1,024 worlds × 60 s per condition, end-on-collision:
- independent traffic at 4, 8 and 16 humans;
- nominal 90% coherent traffic at 8 and 16 (fixed initial world heading, with initial
  goal/flow-angle outcome splits and measured velocity order);
- original reactive aware and reactive blind traffic at 8.

Primary comparisons are collision rate and collision-free goal completion within
each condition, including easy-condition retention. Also report successful goals per
robot-minute, deadline-penalized completion time (failed trips cost 30 s), falls,
encounter exposure and censored counts. Successful-only completion time is secondary.
No readiness threshold or winning objective has been selected.

## Validation

- CPU unit suite: 342 passed, 3 skipped, one existing two-second wall-clock deadline
  test failed under load; the failing test passed on rerun. Final focused objective,
  matrix, behavior, layering and CLI checks passed (48 tests), plus the final centered
  config/matrix subset (5 tests).
- Fast legacy goldens on a CPU compute allocation, job **4178595**: **12 passed**,
  2 slow cases deselected; expected files unchanged.
- Initial GPU capture check **4178561** exposed the evaluator's unconditional lookup
  of legacy `time_out`; fixed to support finite task-deadline termination.
- GPU preflight **4178570** (A40, omgwth): arrival/auto-reset capture passed; sparse
  PPO at 4,096 envs × 3 iterations completed (about 226–238 s/iteration), then a
  256-env resume and 35 s checkpoint evaluation. All check lineages stay off W&B.
  Runs: `runs/20261021T200341Z_train_traversal_check_navigation` and
  `runs/20261021T200341Z_train_traversal_check_navigation_resume`.
- GPU potential/grid check **4178613** (A40, voltron): potential-shaped PPO, three
  iterations at 256 envs, plus actor checkpoint loading at different human counts and
  a two-condition 35 s grid; verifies completed evaluation retry is idempotent.
  Training: `runs/20261021T201403Z_train_traversal_check_navigation_potential`.

Both GPU jobs finished with exit 0. Sparse/resume evaluation:
`runs/20261021T201934Z_eval_traversal_check_navigation`; potential/grid evaluation:
`runs/20261021T201915Z_eval_navigation_check`. The latter completed both the aware
4-human and independent 16-human cases, then a second call returned without duplicate
results. All verification lineages were off W&B. Startup/driver/source metadata live in
each run.json; these are execution/semantic checks, not learned-performance results.

## Submitted jobs and handoff

- **Training array 4178665**, elements 0–47, initially concurrency 4, now 48. First four elements are
  sparse / potential, gamma 0.99, seed 42, each in reactive-blind and independent
  crowds; all four started (voltron / spd-13). At the user's request the throttle
  was lifted: all 48 elements are now running, 45 on A40 and elements 45–47 on
  Quadro RTX 6000. Arrays were submitted at the end of preflight; the attempted environment
  dependency was not applied by Slurm, but both GPU preflights completed successfully
  before this handoff. No model result depends on that scheduling detail.
- **Evaluation array 4178666**, matching elements 0–47, initially concurrency 2,
  raised to 8, explicit
  `aftercorr:4178665` verified in scontrol. Every element waits for its own training
  lineage to finish, then runs the same seven-condition grid. Pending-invalid-dependency
  elements are canceled if their training fails.
- Rough training budget is about 350–380 A40-hours, plus evaluations, based on earlier
  1 Hz runs and the new 226–238 s/iteration check; densities and node load change cost.
  Queue/wall-clock completion depends on shared availability and preemptions.
- Every projected training/evaluation path is already indexed in the run log, so
  later array starts do not create unlogged directories. The campaign manifest is the
  canonical arm-to-path/job-index mapping. Do not edit its source/config snapshots.

Start checks: all four lineages have target 120, the expected sparse/potential objective,
gamma 0.99, W&B online in the main project (SDK run URLs confirmed), and the identical
source fingerprint `99c535ea2d93...`. All 408 copied source files match their recorded
SHA256 values. Shared-node FreeMem was checked (about 78 GB voltron / 143 GB spd-13).

Next: inspect first full-iteration rows, monitor failed array elements, requeues and
evaluation attachments, then compare both primary outcomes per condition and seed.
Do not promote an objective based on training reward curves.


## Resource expansion and GPU memory checks (user, 2026-10-21)

The user authorized RTX 6000 and 2080 Ti when memory permits and asked to claim
idle A40s immediately. Raised training cap 4 → 25 → 48 and evaluation cap 2 → 8.
All 48 training elements reached RUNNING: 45 A40, three RTX 6000. Updated pending
jobs only; retained running lineages, their W&B writers, aftercorr evaluation
mapping and the frozen 408-file source/config snapshot (all hashes rechecked).
Slurm GPU changes were in place on existing elements; no duplicate training jobs.

The lab partition QoS is eight A40s / eight L40Ss per user and zero RTX 6000s /
2080 Tis. A multi-partition RTX submission was rejected at submission with
QOSMaxGRESPerUser. `overcap` alone permits 48 GPUs per user and accepts these cards;
remaining pending training/evaluation elements were moved there. Unallocated GPUs
also need free CPUs: the first snapshot had 18 unallocated A40s but no slots with
eight free CPUs. Subsequent scheduler cycles admitted the additional A40 jobs.
Shared-node FreeMem was checked before/after expansion. This is legitimate preemptible
capacity; requeue/checkpoint rules remain in force.

Verification uses the frozen campaign source, FP32, unchanged 4,096 training envs,
one full PPO update and no W&B. Instrumentation/specs/results are saved under the
campaign's `hardware_checks/`; each training/evaluation lineage is indexed separately.

| Check | Job | Result |
|---|---:|---|
| Quadro RTX 6000, 24 GB, reactive 16 humans + 32 s rollout | 4179732 | Passed complete PPO update. Peak PyTorch allocated 15177 MiB / reserved 15922 MiB; sampled total process peak 14260 MiB misses short allocation spikes (sampling is not an exact peak). 12:16 total wall time, including startup. |
| RTX 2080 Ti, 11 GB, same combined stress case | 4179733 | OOM at first graph neighbor-cost reset, requesting 1.46 GiB with 1.40 GiB free. |
| RTX 2080 Ti, actual reactive 8 humans + 32 s rollout | 4179790 | OOM in planner Mul, requesting 2.39 GiB with 2.24 GiB free; sampled process peak 10272 MiB, PyTorch reserved peak 8790 MiB. |
| RTX 2080 Ti, actual independent 16 humans + 16 s rollout | 4179791 | OOM creating Warp CUDA graph; sampled process peak 10472 MiB, PyTorch reserved peak 8726 MiB. |
| RTX 2080 Ti, evaluation 1024 envs, reactive 16, blind, 35 s | 4179925 | Running; tests evaluation capacity beyond the 30 s task deadline. |

The combined stress case exceeds the largest crowd and rollout combination in any
single training arm; both actual 2080 Ti workload checks also failed. Do not reduce
the matched batch size to fit a training arm. Device memory limits, rather than CUDA
compatibility, block these 11 GB training cases. Use RTX 6000 for training after this
24 GB check; test 2080 Ti for evaluation. Mixed-device training is labeled in each
lineage's provenance; compare speed only within a GPU type.

Future launcher defaults now permit 48 training / eight evaluation elements on
`overcap`, with explicit `train_gpu_type` and `eval_gpu_type` fields instead of a
hardcoded A40 resource request. The running campaign keeps its original source;
scheduling history is stored separately in its manifest.

## 2026-10-22 snapshot: stopped lineages, crowd-blind baseline, report

- **Training:** 39 of 48 lineages reached target. Nine stopped early and were not
  requeued, so their `aftercorr` evaluations cannot run:
  - seven requeued at about 06:22 UTC and started on `rl2-lab` without a GPU allocation
    ("No devices were found", exit 6; see below): `all_g75_reactive_s42` (last model 109),
    `all_g75_reactive_s43` (109), `potential_g75_independent_s42` (100),
    `potential_g99_independent_s42` (118), `potential_g99_reactive_s43` (117),
    `sparse_g75_independent_s42` (115), `sparse_g95_reactive_s42` (117);
  - `no_time_bonus_independent_s42` (61): CUDA OOM, another process held 36.5 GB of the
    shared cheetah A40;
  - `coherent90_independent_s43` (55, RTX 6000 glados): exit 75 after its checkpoint,
    restart count 0, never requeued.

  At the user's request all nine resume in place from their newest checkpoints with the
  frozen source: training array **4191658** (elements 3, 7, 9, 10, 14, 22, 24, 39, 47; A40,
  excluding bishop/crushinator/heistotron) and evaluation array **4191659** (`aftercorr`).
  All nine 4191658 elements started on `rl2-lab` with **no GPU allocated** (`AllocTRES`
  lacked the requested `gres/gpu:a40`), so `nvidia-smi` found no device and the worker
  exited 6 before training; the lineages were untouched. The original restart failures
  had the same signature (the "GPU-less nodes" were a misdiagnosis: voltron ran an
  evaluation on its GPU at the same time). Every campaign start on `overcap` got its GPU.
  Resubmitted on `overcap` only: training **4192670**, evaluations **4192671**.
- **Crowd-blind baseline on the seven-condition grid:** the zero-residual planner
  (frozen planner + SONIC walking straight at the goal) had only been measured on the
  random-goal task. Job **4191359** runs it on the campaign grid from the frozen source,
  with the env options of `sparse_g99_reactive_s42`, seed 101, 1,024 envs × 60 s, end mode:
  `runs/20261022T154711Z_eval_navigation_crowd_blind_baseline`.
  **Result (complete, A40 spot, 67 min):** arrival-only reward (mean of the fully evaluated
  8-human, gamma 0.99 arms, both training crowds and seeds) beats the baseline on goal success
  **and** collisions in all 7 conditions: success +7 to +16 points (mean +12), collisions per
  robot-minute −16% to −31% (mean −25%), and more successful goals per robot-minute
  everywhere. Baseline / sparse success: independent 4 .717/.825, 8 .509/.657, 16 .250/.378;
  flow 8 .565/.682, 16 .304/.413; reactive aware .797/.871, reactive blind .519/.675. The
  all-intermediate-terms reward is below the baseline (independent 8: 24% vs 51%).
- **Partial results (13 arms fully evaluated, 8 humans, 100 s horizon):** with 8
  independent walkers, arrival-only reward reaches 66% goal success (4 arms) against 63%
  with potential shaping and 22% with all intermediate terms. The all-terms arms take
  16.7 s per success against 6.8 s and end 57% of attempts in collisions against 32%.
  Sparse success falls from 66% to 38% at 16 walkers. A 4 s horizon gives 59% success and
  4.9 collisions / robot-min against 66% and 3.2 at 100 s. Training crowd (reactive or
  independent) changes sparse success at 8 independent walkers by less than one point.
  No crowd-blind comparison yet; no objective promoted.
- **Overview report** (artifact "Crowd Traversal Progress", built from `runs/` by a
  scratch aggregation script) covers the curriculum, sparse-reward and independent-crowd
  results against the crowd-blind baseline.
