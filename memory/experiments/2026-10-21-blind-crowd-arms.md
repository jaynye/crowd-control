# 2026-10-21: teaching the planning arm to evade robot-blind humans

Completed 2026-10-21; analyzed from saved training logs and all ten completed evaluations.

**Result.** The curriculum improves blind-crowd collisions by 8% and goal throughput by
11% relative to P1, but increases aware-crowd collisions by 18% while improving goals by
10%. Fixed-blind training reduces blind collisions further (15%) but loses more aware
safety (+51% collisions). This is a safety/progress tradeoff across conditions, not
evidence that curriculum training beats both baselines on both metrics everywhere.
The two-stage penalty recipe does not close the gap. One training seed per arm.

User (2026-10-21), after the [planning arms](2026-10-19-sonic-planning-arms.md): "So what we
can try is a learning curriculum in which we increase probability of sampling parameters
which increase the 'difficulty' of the simulated humans as the robot gets better at
avoiding. We can also try just strictly increasing the number of humans which don't see
the robot without implementing a curriculum, as a baseline. Yes, all runs should use wandb
by default on. I think a2pf -> planning is also a very reasonable idea."

## Question

The planning arms slow down near people rather than steer clear. In the robot-blind crowd
they equal a2 (3.6–3.7 collisions per robot-minute) and stay above a2pf (3.17).
- Does training against humans who do not see the robot teach evasion, either with an
  adaptive curriculum or with a fixed fully blind crowd?
- Does a2pf's recipe close the gap to a2pf on a planning arm? The recipe is to train
  without the terminal penalties first, then add both at −1,000.

## Arms

All arms are P1's setup unless stated otherwise:
- 1 Hz residual over frozen SONIC (`--planner-rate 1`);
- 4,096 envs, 8 humans, graph-turns, `encounter_decisive_robot.json`, seed 42;
- `collision_end` and `fall_end` at −1,000;
- A40, `--signal=B:TERM@1000`, `CROWD_SIGNAL_LEAD_S=1000`, requeued in place. The fixed-blind
  arm and stage 1 run on rl2-lab. The control, stage 2 and the curriculum also run on the
  preemptible overcap partition, because the lab's A40 group limit was full; a preemption
  requeues in place.
- The fixed-blind arm and stage 1 began attempt 1 while the curriculum and W&B edits were
  landing in `src/` and `mjlab/`. Their env is the same either way (no curriculum), and
  later attempts and the evaluations run the final code; their `run.json` diffs may differ
  between attempts.

They are the first lineages on W&B by default, in the main project.
The curriculum arm was submitted after its check job, at 04:53 UTC.

| Arm | Run | Job | Change from P1 | Iterations |
|---|---|---|---|---|
| Curriculum | `runs/20261021T045347Z_train_traversal_plan1hz_curr` | 4172896 | `--crowd-curriculum 2`, adaptive robot-blind share (below) | 120 from scratch |
| Fixed blind | `runs/20261021T043526Z_train_traversal_plan1hz_blind` | 4172720 | `--robot-blind`: every human oblivious to the robot, the curriculum's top level | 120 from scratch |
| a2-style, stage 1 | `runs/20261021T043526Z_train_traversal_plan1hz_a2` | 4172721 | no terminal penalties, as a2 | 120 from scratch |
| a2pf-style, stage 2 | `runs/20261021T043526Z_train_traversal_plan1hz_a2pf` | 4172723 (afterok 4172721) | `--resume` stage 1 with both penalties at −1,000, as a2pf | +30, to 150 |
| Control: P1 +30 | `runs/20261021T043526Z_train_traversal_plan1hz_ext30` | 4172722 | `--resume` P1 unchanged | +30, to 150 |

The +30 iterations are a2pf's ratio: 1,000 on a2's 4,000, the same simulated time per env.
The control separates the a2pf recipe from the extra iterations, so stage 2 is compared
with it, not with P1.

Both +30 branches actually train labels 119–148 (`target_iterations=149`), because
the saved parent iteration is 119. Their evaluated checkpoints are 140, 145 and 148;
both branches use the same convention. Curriculum checkpoints are 115, 117 and 119;
the fixed-blind and stage-1 checkpoints are 110, 115 and 119.

The baseline reads "without implementing a curriculum" as a fixed crowd. Its share of
humans that do not see the robot is the curriculum's top end, the fully blind crowd. A
time-based ramp would itself be a schedule; it was not run.

## The curriculum (fixed before launch)

`traversal.crowd_difficulty_term`, training only, never in play or evaluation:
- **Level.** d ∈ [0, 1] is the chance that a human is drawn from the robot-blind crowd
  instead of the training crowd. The oblivious share is 0.05 + 0.95 d, the stubborn share
  0.10 (1 − d) and the cooperative share 0.85 (1 − d). d = 0 is the training crowd and
  d = 1 the robot-blind one. Humans draw their traits when they reset, from the crowd's
  own live encounter config.
- **Rule.** Once per PPO iteration (16 decisions × 50 env steps), the trainer measures
  training collisions per robot-minute: `crowd_collision` ends over the iteration's
  robot time.
  - If the rate is at most the target, d rises by 0.05.
  - If it is above 1.5 × the target, d falls by 0.05.
  - Otherwise d holds.
  - The full range takes at least 20 iterations.
- **Target, 2 per robot-minute.** At that rate about half of the 20 s episodes end
  without a collision (e^(−2/3) = 0.51): the frontier where the robot fails about half
  the time.
  - In training units: P1 started at 84 collision ends per decision across 4,096 envs
    (1.23 per robot-minute) and ended at 22 (0.32).
  - The fixed-blind arm starts at 230 (3.4).
  - So from P1's start the level should rise at once to about 0.25. At P1's final skill
    it should hold near 0.45–0.7, and go higher only as the robot learns to evade.
- **Requeues.** The level is saved in checkpoints (mjlab's new
  `CurriculumManager.state_dict`), and a resumed attempt measures afresh from its first
  step.
- **Logged** as `Curriculum/crowd_difficulty/{level, oblivious, stubborn, collision_rate,
  blind_share}`. The blind share is read back from the humans in the plaza.

## Results: safety and progress under each human condition

User evaluation requirement (2026-10-21): compare **both collision rate and goal
completion rate under different human behaviors**, since the curriculum should retain
easy-condition skill while improving the hard conditions. Do not hide regressions in
one condition by averaging across conditions.

Protocol: A40, 4,096 envs × 60 s, 8 humans, paired goal chains, evaluation seed 0,
mean actions, mean of the last three saved checkpoints. Aware means the original
5% oblivious / 10% stubborn / 85% cooperative crowd; blind means 100% oblivious.
The curriculum is disabled during evaluation: every policy faces the same fixed
condition. Goals/min below means arrivals per minute with a moving **command**, not
per minute the robot physically moves, so standing still cannot improve that denominator.
Checkpoint ranges measure checkpoint variation, not independent training-seed variance.

End mode, where a collision resets the episode:

| Policy | Aware collisions / robot-min | Aware goals / moving-command min | Blind collisions / robot-min | Blind goals / moving-command min |
|---|---:|---:|---:|---:|
| Original P1 | 0.273 | 3.350 | 3.680 | 2.424 |
| P1 +30 control | 0.264 | 3.306 | 3.680 | 2.405 |
| Adaptive curriculum | 0.322 [0.308–0.330] | 3.677 | 3.385 [3.375–3.396] | 2.687 |
| Fixed blind | 0.412 [0.403–0.426] | 3.490 | 3.127 [3.120–3.140] | 2.556 |
| No terminal penalties, stage 1 | 0.528 | 3.573 | 3.843 | 2.638 |
| Stage 2: add both penalties, +30 | 0.476 | 3.553 | 3.767 | 2.610 |

- Against iteration-matched P1, curriculum gives **8.0% fewer blind collisions and
  10.8% more blind goals**, but **17.9% more aware collisions and 9.8% more aware goals**.
  It does not dominate P1 in both conditions. Against the +30 control, its aware
  collision regression is 21.8%; the control itself barely changes P1.
- Fixed-blind training gives **15.0% fewer blind collisions and 5.4% more blind goals**
  than P1, but **50.8% more aware collisions**. Curriculum retains more aware safety
  and has higher goal throughput in both conditions than this baseline, while colliding
  8.3% more in the blind condition. Neither dominates the other on all outcomes.
- The fraction of assigned goal legs completed agrees with throughput: P1 → curriculum
  is **38.4% → 40.5% aware** and **24.7% → 27.2% blind**. Fixed blind is 39.0% / 26.7%.
- Stage 2 improves its own stage 1 only modestly: aware collisions −9.8%, blind −2.0%,
  with slightly fewer goals in both. Against the +30 control it collides 80% more aware
  and 2% more blind, with 7–9% more goals. The successful low-level a2pf result does not
  transfer to this planning-arm recipe in this training budget.
- Falls stay rare (about 0.0005–0.0011 / robot-min for these arms in end mode); collision
  gains are not purchased with frequent falls. The old low-level a2pf remains at 0.18
  aware / 3.17 blind collisions per robot-minute; fixed-blind planning only matches its
  blind safety, while aware safety is substantially worse. This is contextual, since
  architectures and training histories differ.

Continue mode checks goal throughput without resetting on contact (post-contact
states are outside training, so it is a diagnostic, not the sole evaluation):

| Policy | Aware collisions / robot-min | Aware goals / moving-command min | Blind collisions / robot-min | Blind goals / moving-command min |
|---|---:|---:|---:|---:|
| Original P1 | 0.336 | 3.455 | 4.393 | 3.531 |
| P1 +30 control | 0.327 | 3.392 | 4.365 | 3.485 |
| Adaptive curriculum | 0.387 | 3.814 | 4.160 | 3.865 |
| Fixed blind | 0.500 | 3.738 | 3.883 | 3.777 |
| Stage 2: add both penalties | 0.551 | 3.761 | 4.526 | 3.795 |

The same tradeoff survives: curriculum has 5.3% fewer blind collisions and 9.5% more
blind goals than P1, with 15.4% more aware collisions. Its blind goal advantage is not
solely the consequence of fewer collision resets in end mode.

### Does it actually avoid humans?

There is a small real gain under blind humans, where the humans cannot yield to the robot.
It is not explained by parking or by falling, but the available metrics do not establish
reliable, threat-directed dodging.

| Blind, end mode | P1 | Curriculum | Fixed blind |
|---|---:|---:|---:|
| Encounters / robot-min | 42.08 | 41.55 | 39.54 |
| Collisions / encounter | 0.08745 | 0.08147 | 0.07907 |
| Moving-command time with TTC <1 s | 17.2% | 16.3% | 15.2% |
| Median time to first collision (s) | 11.83 | 12.97 | 13.96 |
| 60 s collision survival estimate | 1.18% | 1.70% | 2.58% |
| Cross-track deviation, free / encounter (m) | 0.130 / 0.158 | 0.181 / 0.234 | 0.243 / 0.304 |
| Robot speed at 0–0.25 m clearance ahead (m/s) | 0.392 | 0.443 | 0.447 |
| Robot velocity toward that human (m/s) | 0.272 | 0.295 | 0.296 |
| Collisions with robot speed <0.15 m/s | 17.2% | 13.5% | 12.7% |

- Curriculum cuts collisions per encounter **6.8%**, with only 1.3% fewer encounters;
  the gain is more than reduced encounter exposure. Fixed blind cuts per-encounter
  collisions 9.6%, with 6.0% fewer encounters. Encounters are policy-dependent, so
  this decomposition is descriptive, not a controlled causal attribution.
- Both policies move more at close approaches and complete more goals. They have
  less reliance on stopping to let a human pass, but still move toward close humans;
  this is not a clean signature of timely evasion.
- Larger deviations occur even in free space. Extra path deviation alone cannot
  distinguish deliberate dodging from a generally different path/heading strategy.
  Near-person deviation grows more than free-space deviation, consistent with some
  adaptation, but trajectories or matched threat interventions are needed for attribution.
- About 84–85% of remaining blind collisions are human-led by the velocity definition.
  The corresponding robot-led rates are about 0.49 P1 / 0.51 curriculum / 0.50 fixed.
  The total improves; the attribution does not prove who was responsible, since slowing
  itself changes which side is labeled human-led.
- Aware-crowd collisions with cooperative humans rise from **0.088 P1 → 0.143 curriculum
  → 0.231 fixed blind per robot-min**. Collisions with oblivious humans in that condition
  fall slightly, 0.174 → 0.161 → 0.150. These are per-total-time contributions, not
  trait-normalized risks. They support a loss of coordination with yielding humans,
  rather than general improvement across behavior types.
- The built-in reaction metric is invalid for these planning arms: P0's replanning
  changes velocity even without awareness. No claim about reaction latency follows.

### What the level curve says

The text logs average curriculum values over reset events, so transition iterations
can contain fractional levels; read stable levels and saved boundaries rather than
rounding every logged number into a separate update.

| Training point | Stable difficulty d | Interpretation |
|---|---:|---|
| Iteration 7 | 0.35 | Early ascent while collision rate is below 2 / robot-min |
| Iteration 20 | 0.50 | About 52.5% configured oblivious humans |
| Iteration 35 through first requeue at 61 | 0.55 | Rate settles just above the upward gate |
| After first requeue, about iteration 63–117 | 0.60 | About 61% observed blind share; mostly 2.1–2.2 collisions / robot-min |
| After second requeue, final iteration 119 | 0.65 | 66.75% configured / 64.4% observed oblivious share |

Both requeues are followed by temporarily low first-window rates (about 0.96 and
1.03 / robot-min), then an extra +0.05, while the steady-state rate returns above 2.
The level is restored correctly, but worlds are reconstructed and traits apply only
at reset; this pattern is consistent with startup exposure/trait transients. **The
last two level increases are not evidence of newly learned avoidance.** The controller
never reaches a fully blind crowd. For most of training it holds in the 2–3 collision
band, with no progress or easy-condition-retention criterion.

## Implications for the robot-prefixed parameters

User (2026-10-21) recalls robot-prefixed parameters for the proposed behavior-curriculum
Stage 2. This is separate from the already-run **penalty-recipe stage 2** above.
`EncounterCfg` and the live controls expose exactly these four robot-prefixed fields:

| Parameter | Current role | Implication for a follow-up |
|---|---|---|
| `robot_oblivious_fraction` | No robot perception or robot stop rule; already increases as 0.05 + 0.95d | Retain as the hard endpoint; increasing it alone has shown only modest evasion gains |
| `robot_stubborn_fraction` | Sees robot and stops/separates, but does not initiate avoidance; already decreases as 0.10(1−d) | Vary independently to teach passing stationary/non-yielding people; this is a different challenge from oblivious walkers |
| `robot_assertiveness_range` | Cooperative humans' intervention hazard is proportional to 1−assertiveness (with blocked-time decay); currently fixed at [0, 0.35] | Best next smooth axis: later/less likely yielding while preserving perception and the stop rule; no benefit has yet been measured |
| `robot_radius_m` | Robot proxy radius in the humans' threat/stop/separation/avoidance/graph-clearance calculations, currently 0.3 m | Larger is generally easier, smaller permits closer approach; keep tied to plausible perceived geometry and evaluate separately, because it does not change robot collision spheres or LiDAR geometry |

At d=0.60 only 34% are cooperative; at d=1 none are. Assertiveness then affects only a
minority or has no effect, so blindly ramping every parameter with d cannot exercise all
axes. Stubbornness and obliviousness also obey a shared sum ≤1. They should be sampled
as distinct coherent behavior regimes, with persistent reset traits, not independent
unconstrained fractions.

Recommended next comparison (proposed, not launched or promoted as a result):
- Keep easy, middle and fully blind conditions in evaluation for **every** policy and
  checkpoint, with the same goals, population and geometry. Add controlled stubborn and
  high-assertiveness conditions to test distinct interaction types. Existing evaluations
  cover only aware and fully blind endpoints; middle-condition generalization is unknown.
- Report collision rate, goals per moving-command minute and completed-leg fraction
  within each condition, plus falls and encounters. Keep both end and continue diagnostics.
  User acceptance should be joint safety/progress improvement or an explicitly accepted
  tradeoff, not low aggregate collision rate or a higher curriculum level alone.
- Train with retained exposure to the easy and middle regimes, rather than replacing
  the whole crowd distribution as difficulty rises. Compare against fixed aware, fixed
  blind and a fixed mixture with matched average exposure; the current experiment has
  no fixed-mixture control and cannot isolate the value of adaptivity from exposure.
- Handle restart warm-up before using a window for advancement; use stable recent
  evidence and monitor progress/retention as well as training collisions. Exact gates
  require a user-approved criterion; no new numerical pass threshold is inferred here.
- Run more than one training seed and evaluate targeted head-on/crossing threats before
  claiming reactive dodging. The 1 Hz action cadence and short-horizon goal-tracking
  incentives are plausible remaining limits, not mechanisms isolated by these arms.

No new training/evaluation jobs or behavior changes were made for this analysis.

## Start checks

- **Fixed blind:** `run.json` records `perceive_robot` false. Iteration 0 had 229.8
  collision ends per decision (P1: 84).
- **Stage 1:** `Episode_Reward/collision_end` and `fall_end` read 0 at iteration 0.
- **W&B:** the lineages are `online` and live in `rl2-group/crowd-control-traversal`. A
  read-back at 04:55 UTC found iteration 0 for the blind arm and stage 1, and 119–120 for
  the control.
- **Control:** loaded P1's `model_119.pt`. Both penalties are active (`collision_end` −0.09
  to −0.10, at P1's end-of-training level of 18.6–21.8 collision ends per decision).
- **Curriculum, full scale** (4172896, cheetah): iteration 0 had 90.7 collision ends per
  decision (1.33 per robot-minute), so the level stepped up at its end. 211 s per
  iteration.
- **Curriculum check** (job 4172817, A40, 256 envs; runs
  `runs/20261021T044358Z_train_traversal_check_curriculum` and its branch
  `runs/20261021T044358Z_train_traversal_check_curriculum_resume`, off W&B as check runs):
  - mjlab's runner tests pass, including the new curriculum-state round trip.
  - The level rose 0.05 per iteration at measured rates of 1.48 and 1.70 per robot-minute.
    That matches the logged `crowd_collision` ends: 7.3 per decision at 256 envs is 1.71.
  - Each update fell on an iteration boundary (800 env steps).
  - Checkpoints stored the level beside `common_step_counter` (model_2: 0.15). The branch
    resumed at 0.15 and continued to 0.20.
  - Two logging blemishes were fixed before launch: the blind share read 1.0 before the
    humans' first reset, and a resumed attempt restarted the logged rate at 0.

## Completed evaluation protocol and provenance

`crowd eval traversal` as before: last 3 checkpoints, aware and blind crowds, both modes,
4,096 envs × 60 s, paired goals, seed 0. All ten afterok evaluations completed
(`eval_traversal.sbatch`, `--partition=rl2-lab,overcap --time=03:00:00`):

| Arm | Aware | Blind |
|---|---|---|
| Fixed blind | 4172926 | 4172927 |
| Stage 1 | 4172928 | 4172929 |
| Control | 4172930 | 4172931 |
| Stage 2 | 4172932 | 4172933 |
| Curriculum | 4172934 | 4172935 |

Saved evaluation directories (each contains both modes and three checkpoints):

| Arm | Aware | Blind |
|---|---|---|
| Fixed blind | `runs/20261021T112758Z_eval_traversal_plan1hz_blind_2` | `runs/20261021T112758Z_eval_traversal_plan1hz_blind` |
| Stage 1 | `runs/20261021T114607Z_eval_traversal_plan1hz_a2` | `runs/20261021T114607Z_eval_traversal_plan1hz_a2_2` |
| P1 +30 control | `runs/20261021T070152Z_eval_traversal_plan1hz_ext30` | `runs/20261021T070709Z_eval_traversal_plan1hz_ext30` |
| Penalty stage 2 | `runs/20261021T133108Z_eval_traversal_plan1hz_a2pf` | `runs/20261021T133112Z_eval_traversal_plan1hz_a2pf` |
| Curriculum | `runs/20261021T124200Z_eval_traversal_plan1hz_curr` | `runs/20261021T124207Z_eval_traversal_plan1hz_curr` |

Hardware/software: NVIDIA A40, driver 580.178.04, torch 2.9.0+cu128 (CUDA 12.8),
Warp 1.14.0, FP32 SONIC planner/tracker, TF32 matmul off. Repository base
`5c1ef9cb87f5d7c15174ef0af8a2dd002182df45` plus each run's saved source diff/fingerprint.
Graph/rotation artifacts and frozen SONIC checkpoints are the same as the
[planning-arm experiment](2026-10-19-sonic-planning-arms.md). All per-run environment,
attempt/checkpoint and attachment records remain in `run.json` / `metrics.json`.

Checks done:
- **The curriculum's first in-place requeue:** attempt 2 resumed from `model_61.pt` at
  iteration 61 (W&B online). The level was 0.55 before and after the boundary, held at 0.55
  for the last 10 iterations of attempt 1 (rate in the hold band) and reached 0.60 at
  iteration 62. A second requeue resumed model 117 and finished model 119. Remote W&B
  single-row-per-iteration across the boundaries was not independently read back here.
- **Control (P1 +30):** training 4172722 and both evaluations (4172930, 4172931) completed
  with exit 0; results are included above.

- **Stage 2 at start:** log confirms stage 1 model 119 loaded and both penalty weights
  −1,000. Stage 1's terminal weights were zero; all comparisons have their intended setup.
- **Local tracking metadata:** all five training lineages are done, W&B online and
  published through their newest checkpoint (119 or 148). This analysis did not mutate
  remote W&B state or independently audit its history.

The predeclared criterion was fewer blind-crowd collisions without more aware-crowd
collisions or fewer goals. **It is not met**: both harder-crowd arms improve blind
collisions and progress but regress on aware collisions.

Expected cost is about 7 A40-hours for each 120-iteration arm and 1.8 for each +30 arm,
about 25 in all, plus about 1 A40-hour for each of the 10 evaluations.
