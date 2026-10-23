# Goal navigation over frozen SONIC

`crowd train traversal --planner-rate 1 --navigation configs/navigation/sparse.json`
enables a persistent-goal task. Without `--navigation`, the existing velocity task
and its replay remain unchanged. SONIC stays frozen in both.

Each episode has one goal 4–8 m from its spawn, nominal speed 0.8 m/s and a 30 s
deadline. Observations add goal displacement in the robot heading frame, remaining
time and initial goal distance. Goals persist; no command timers replace them or
request standing. Arrival earns 10 plus `2 / (1 + elapsed / nominal_trip_time)`.
Collision, fall and deadline are true terminal failures, zero reward by default;
failure wins an arrival tie. Reward events are independent of the control timestep.
PPO rollout boundaries bootstrap, task deadlines do not.

The strict JSON config is embedded in the lineage record and inherited by resume,
branching, replay and evaluation. Its `reward_mode` selects:

| Mode | Intermediate reward, in addition to the common arrival reward |
|---|---|
| `sparse` | None |
| `potential` | Discount-correct distance potential at planner decisions |
| `tracking` | Linear/angular command tracking and goal facing |
| `proximity` | Human/wall proximity and contact costs |
| `regularized` | Residual magnitude/change costs |
| `all` | Tracking, proximity and residual costs |
| `centered` | Same as `all`, subtracting the positive tracking ceiling per second |

`gamma_per_s` and `gae_lambda_per_s` are converted to the decision rate; `rollout_s`
controls simulated time per PPO iteration. Set discounts in this JSON so shaping and
PPO agree. Potential shaping is `gamma * Phi(next) - Phi(current)`, `Phi = -distance`,
with zero potential at every terminal state. Negative progress is retained. It is
not summed undiscounted across control substeps. Other fields set the goal/deadline,
terminal rewards, potential scale and residual costs; see the example config.

`--behavior configs/behavior/nonreactive.json --robot-blind` selects independent graph
walkers through the shared crowd controller. `encounter.reactive: false` bypasses all
neighbor decisions, stopping, separation and graph clearance against humans/robot.
Humans can overlap. Ordinary turns, boundaries, graph motion quality, rigid LiDAR
geometry and robot–human collision detection stay active.

`flow_fraction` assigns persistent shared-heading membership; `flow_heading_deg: null`
samples a common heading per world/reset, while a number fixes it in the library frame
(90° is simulator +x). `flow_spread_deg` adds independent variation. Members reflect
near boundaries; this is exogenous traffic, not flocking. Report measured flow order,
because boundary reflections can reduce realized coherence.

```bash
# Fresh campaign plan/configs/source snapshot; no Slurm jobs yet.
.venv/bin/crowd train navigation-ablations
# Fresh campaign and training/evaluation array submission.
.venv/bin/crowd train navigation-ablations --submit
```

Defaults: 48 lineages, 4,096 environments, 1 Hz, 120 iterations. The core is sparse/
potential/all × discounts 0.75/0.99 × reactive-blind/independent crowds × seeds 42/43
(24 arms). Probes add sparse discount 0.95, isolated intermediate terms, centered dense
rewards (to test incentives to delay arrival), GAE 0.90, 32 s rollouts,
no time bonus, −10 failure reward, density 4/16 and 90% initial flow coherence. Core and
density/coherence arms have two seeds; other probes use seed 42. The 32 s rollout arm
uses 60 iterations for equal simulated exposure, with fewer optimizer updates.

The campaign snapshots and hashes package, mjlab and scripts. Requeues and evaluation
use that source. Defaults allow up to 48 training and eight evaluation elements,
subject to scheduler availability and the shared per-user GPU quota. Campaigns use
the preemptible `overcap` partition, avoiding the lab partition's smaller GPU quota.
The strict campaign JSON accepts `train_parallel`, `eval_parallel`, `partition`,
`train_gpu_type` and `eval_gpu_type` (both default to `a40`). GPU alternatives are
`l40s`, `a5000`, `rtx_6000` and `2080_ti`; verify the actual workload fits before
switching and retain 4,096 environments for matched training comparisons.
Training uses existing lineage/W&B/requeue rules; test lineages stay off W&B.

Every model gets the same grid: independent traffic with 4/8/16 humans; coherent traffic
with 8/16; original reactive aware/blind traffic with 8. The last three checkpoints play
1,024 worlds × 60 s, paired goals, seed 101, end-on-collision. Coherent results also split
by initial route/flow angle. Grid evaluation appends results across preemptions, skips
completed condition/checkpoint pairs and attaches each condition separately to W&B.

Compare collisions per robot-minute and goal success within each condition, alongside
falls, encounters, measured human speed and flow order. `goal/success_fraction` excludes
rollout-censored episodes and reports their count; `goal/success_per_robot_min` measures
throughput. `goal/deadline_penalized_time_s` assigns failed trips the deadline, preventing
successful-trip speed alone from hiding failures. Inspect `metrics.json`, `tables/`
and `run.json`; never average away an easy-condition regression.
