# 2026-10-12: robot-aware crowd for the traversal teacher

Status: complete. Every run ended by 01:50 UTC on 2026-10-13: a2 at 4,000 iterations,
s7 at 1,500, and t7 and a2s7 at the 4 h limit (`model_3700.pt`). The collision-penalty
arms, if the user approves them, get their own note.

## Conclusions

- **Robot-aware crowd and command rewards.** The first robot-blind runs ended about 70%
  of episodes in a collision. With humans that perceive the robot, a turn-rate weight
  of 2.0 and the `goal_heading` reward, every arm plateaus at 17–21%. The original
  command walked the robot backward; `--forward-first` made it park and stand, and was
  abandoned.
- **The teacher needs the LiDAR, and with it equals PPO on LiDAR.** Without it, t5
  collided twice as often as a2 in replay. t7 is level with a2 in training and replay;
  a2's seed replicate agrees within 0.01. Privileged states add nothing under the
  current rewards.
- **Distillation works on collisions, but the students fall more.** The student s7
  replays level with t7 and a2 (0.16, 0.21, 0.17 collisions per episode end; replay
  noise ±0.05). Both students fall more than their teachers in replay (s7 23 of 585
  ends, t7 0; s5 14 of 715, t5 1; a2 0–7 per about 590), and s7 reaches fewer goals in
  training (0.93–0.97 against 0.99).
- **The reward sets the plateau.** A collision forfeits a return of only about 4, and
  the robot barely slows for a human ahead. `collision_end` (weight 0 by default) adds
  a terminal penalty; the proposed arms and their decision rule are below.

## Question

The first traversal runs plateaued from about iteration 900 with about 70% of
episodes ending in a collision, and the privileged teacher was no better than the
LiDAR policy:
- `runs/20261012T153938Z_train_traversal_a0`: LiDAR policy, PPO;
- `runs/20261012T155620Z_train_traversal_teacher_t0`: privileged teacher.

Both used `encounter_turns.json`, with humans that do not perceive the robot. Does a
crowd that perceives the robot make the teacher learn avoidance? Is the goal command,
which can ask for fast backward walking, a second cause?

## Baseline evidence (diagnostic jobs 4058804 and 4058826, A40)

`scripts/smoke/policy_check.py` plays a checkpoint with 512 envs
for 1,500 steps (30 s). At each `crowd_collision` it reads the pre-step state of the
human with the least clearance.
- **Observation sanity:** the teacher's nearest-human entries match recomputed
  geometry. Their velocities match the hip finite difference: median cosine 1.00,
  speed ratio 0.99–1.00. The teacher was not blind.
- **Collision attribution, policy trained in the robot-blind crowd:**

| Policy, crowd | Collisions | Time-outs | Human closing faster | Human behind the robot |
|---|---|---|---|---|
| t0 (`model_1600`), turns, robot-blind | 968 | 181 | 70% | 54% |
| a0 (`model_1900`), turns, robot-blind | 885 | 219 | 75% | 49% |
| t0, decisive, robot-blind | 1,030 | 179 | 67% | 57% |
| t0, decisive, robot-aware (`encounter_decisive_robot.json`) | 228 | 443 | 14% | 65% |
| t0, turns, robot-aware with every human cooperative | 286 | 407 | 7% | 70% |

"Human closing faster" means that, along the line between the two bodies, the
human's velocity toward the robot exceeds the robot's velocity toward the human.
"Behind" means the human's bearing in the robot's yaw frame is beyond 120°.

- In the robot-blind crowd most collisions are humans walking into the robot, often
  from behind.
- With robot perception, the remaining collisions are the robot moving into nearly
  still humans: robot 0.53 m/s toward the human, human 0.02 m/s away from it (medians).
  Most of these humans are behind the robot.
- The command's component along the line to the human is about 0 (median): the
  robot is not steered at the human, but it walks backward into people it is not
  facing (next section).
- t0 never needed to steer around still humans, because the crowd ran into it.

## The robot walks backward (jobs 4058852, 4058864, 4058870)

Policies trained with the original goal command walk backward. The command is a
world-frame velocity toward the goal, expressed in the body frame, with a turn rate of
1.0 × the heading error clipped to ±1 rad/s. In the robot-aware decisive crowd:
- **Backward share:** t0 commands and executes body vx < −0.1 m/s in 90–91% of
  moving-env steps; a0 does the same.
- **Collisions:** 97% of collisions happen while walking backward, and 98% of those
  are commanded backward.
- **Turn rate:**
  - the command is saturated (median |wz| 1.0 rad/s);
  - the robot turns at a median of 0.5 rad/s;
  - the signs agree only 39–41% of the time, and the correlation is −0.27 to −0.31;
  - the heading error has p10/p50/p90 of 91–106° / 166–167° / 178°.
- **Sign conventions agree:** d(`heading_w`)/dt against the body yaw rate has
  correlation 0.993 and a ratio of 1.02.

So the policy keeps its back to the goal: it counter-steers as the error crosses 180°,
then tracks the linear command backward. This is a **local optimum**, not a cheaper
solution. With linear std² 0.25 and turn-rate std² 0.5, at a turn-rate weight of 1.0:
- facing away and tracking backward earns about 2.0 (linear 2, turn rate about 0.01);
- facing the goal and walking forward earns 3.0.

Near the ±180° wrap the turn command flips sign while the body-frame linear command
stays the same, which can hold a policy there. t0 to t2 used a turn-rate weight of 1.0,
half of mjlab's default of 2.0. That is a plausible lever, not an established cause.

A forward-first command has the same kind of trap: when the robot faces away the
linear command is zero, so standing still also earns the linear reward. That is still
worse than turning and walking, so t2 and t3 are tests, not fixes.

## Runs

| Run | Job | Arm |
|---|---|---|
| `runs/20261012T175650Z_train_traversal_teacher_t1` | 4058853 (crushinator) | Planned control (original command). Cancelled at about iteration 110 once the backward walking was found |
| `runs/20261012T175651Z_train_traversal_teacher_t2` | 4058854 (baymax) | `--forward-first`: linear command scaled by max(cos heading error, 0); turn-rate weight 1.0 |
| `runs/20261012T180329Z_train_traversal_teacher_t3` | 4058873 (ig-88) | t2 with turn-rate weight 2.0 (`--env.rewards.track-angular-velocity.weight 2.0`) |
| `runs/20261012T184541Z_train_traversal_teacher_t4` | 4059091 (cheetah) | t3 plus the opt-in `goal_heading` reward, (1 + cos heading error) / 2, weight 1.0 |
| `runs/20261012T192733Z_train_traversal_teacher_t5` | 4059515 (chappie) | Original command (no forward-first), turn-rate weight 2.0, `goal_heading` 1.0 |
| `runs/20261012T192724Z_train_traversal_teacher_t6` | 4059516 (ig-88) | t5 without `goal_heading`. Cancelled at about iteration 820 |
| `runs/20261012T201213Z_train_traversal_a2` | 4059720 (voltron) | PPO on LiDAR (no teacher) with t5's env and rewards: the baseline for the distilled student |
| `runs/20261012T213649Z_train_traversal_student_s5` | 4060025 (sonny) | `--distill` of t5's `model_2000.pt` into the LiDAR student, 1,500 iterations |
| `runs/20261012T214348Z_train_traversal_teacher_t7` | 4060075 (qt-1) | t5 with `--teacher-lidar`: the teacher reads the LiDAR as well as the privileged terms |
| `runs/20261012T215129Z_train_traversal_a2s7` | 4060119 (chappie) | a2 with `--agent.seed 7` (every other arm uses seed 42): the seed replicate of the baseline |
| `runs/20261012T234846Z_train_traversal_student_s7` | 4061509 (major) | `--distill` of t7's `model_2000.pt` (teacher with LiDAR), 1,500 iterations: the deployable student |

Shared settings:
- `crowd train traversal --teacher --humans 8`, 4,096 envs, 4,000 iterations, seed 42
  (the PPO config's);
- behavior `configs/behavior/encounter_decisive_robot.json`;
- graph-turns with its rotations, on A40s.

## What to compare

Compare the share of episode ends that are collisions, `goals_reached`,
`goal_distance`, `crowd_contact`, `fell_over` and
`Episode_Reward/track_angular_velocity`. Then rerun `scripts/smoke/policy_check.py`
with `--forward-first` for the backward share, heading error and collision attribution.
Mean reward is not comparable with t0, because forward-first changes what velocity
tracking measures.

**Decision rule at iteration 700, written before the curves:**
- **Healthy arm:** all three of these hold:
  - `goal_distance` falls and `goals_reached` heads toward t0's 0.63;
  - `track_angular_velocity` is well above t0's 0.02;
  - the collision share of episode ends is below t0's.
- **Stuck:** both arms have low goal arrivals and a flat turn-rate reward. Stop them
  and launch the fallback: a heading term with a smooth gradient through 180°,
  (1 + cos heading error) / 2, or turn-direction hysteresis near ±180° in
  `CrowdGoalCommand`. Either is a reward or command change for the user to approve.

## Iteration 700: forward-first is stuck (job 4059072)

| | t2 (iteration 743) | t3 (iteration 711) | t0, for reference |
|---|---|---|---|
| Collision ends / time-out ends per step | 0.17 / 4.6 | 0.13 / 3.8 | 5.2 / 1.9 |
| `goals_reached` | 0.006 | 0 | 0.63 |
| `goal_distance`, m | 6.5, rising | 6.3 | 4.2 |
| `track_angular_velocity` | 0.08 | 0.19, rising | 0.02 |

Playing the iteration-700 checkpoints with `--forward-first` (512 envs, 12 s):
- the median heading error is 90–97°, exactly where the forward-first command reaches zero;
- the robot hardly turns (median 0.14–0.16 rad/s against a saturated 1 rad/s command;
  correlation −0.05 to −0.07);
- nothing is commanded backward.

The policy parks with the goal to its side and stands. The few collisions are mostly
humans closing on a robot that is standing still. The 5% oblivious share remains.

Per the rule above, t2 was cancelled at about iteration 820. t3 continues as the
control for t4, which adds the fallback heading term (the agent's call; the user can
veto it).

## Forward-first abandoned; t5 and t6 (19:27 UTC)

t4 did not escape: at iteration 711 `goal_heading` is 0.48, which is a 90° park, and
`goals_reached` is 0.014. At iteration 1,452, t3 still has `goals_reached` 0.008. Both
were cancelled.

**Conclusion:** a command that allows zero speed loses here. In a robot-aware crowd,
standing still is both rewarded (the linear command is tracked exactly) and safe. In
t3's replay, 16 collisions in 512 envs × 12 s, almost all oblivious humans walking
into a still robot. The policy learns to stand before it learns to walk, and never
has to walk. The original command always asks for motion, which is how t0 learned to
walk. `--forward-first` stays as an opt-in flag, off by default.

t5 and t6 use the original command with mjlab's turn-rate weight of 2.0; t5 adds
`goal_heading` at weight 1.0.
- **t6** tests whether the weight alone stops the backward walking. Confound: it also
  differs from t0 in the crowd (robot-aware).
- **t5 against t6** tests the heading term.

From these runs on, `CrowdGoalCommand` logs episode means over moving steps:
`heading_error_deg`, `backward_share` (body vx < −0.1 m/s) and `speed`.

**Decision rule, written before the curves:**
- **Early signal, about iteration 700:** `heading_error_deg` clearly below 90° and
  falling, and `backward_share` falling. t0 was already walking by then.
- **Gate, about iteration 1,000:** `goals_reached` at least near t0's 0.34 at
  iteration 933, with a low backward share.
- Judge the collision share only together with goal arrivals; the forward-first runs
  looked good on collisions because they stood still.

If t5 also settles into backward walking, the next levers are a goal-progress reward
or starting from a checkpoint that already walks. Neither is built.

## Iteration 700: the heading term works (20:10 UTC)

| | t5 (`goal_heading` 1.0), iteration 715 | t6 (no heading term), iteration 805 | t0 at its plateau |
|---|---|---|---|
| `heading_error_deg` | 17.5, falling (64 → 24 → 17.5) | 110, rising | about 167 (replay) |
| `backward_share` | 0.025 | 0.56, rising | about 0.90 (replay) |
| `speed`, m/s | 0.55 | 0.42 | — |
| `goals_reached` | 0.67 | 0.25 | 0.63 |
| `goal_distance`, m | 3.9 | 3.3 | 4.2 |
| Collision ends / time-out ends per step | 1.7 / 3.4 | 1.1 / 3.4 | 5.2 / 1.9 |

**Conclusion:** the turn-rate weight alone does not stop the backward walking (t6);
the smooth goal-heading reward does (t5). By iteration 700, t5 walks forward, faces
its goals and reaches as many as t0 did at its plateau. About a third of its episode
ends are collisions, against t0's 73%. t6 was cancelled; its GPU runs the LiDAR
baseline a2 with t5's settings. Gate at iteration 1,000 as written above.

## Iteration 1,000 gate: t5 passes (20:30 UTC)

At iteration 1,013, t5 has:
- `goals_reached` 0.90 (t0: 0.34 at iteration 933, 0.63 at its plateau);
- heading error 15.5° and backward share 0.012;
- speed 0.58 m/s;
- 1.1 collision ends against 3.0 time-out ends per step, so 27% of episode ends are
  collisions (t0: 73%);
- falls 0.1 per step.

t5 keeps training. Its turn-rate weight of 2.0 and `goal_heading` 1.0 became the code
defaults at 20:30 UTC; t5 and a2 pass them as overrides. Distillation starts from a
mature t5 checkpoint (around iteration 2,000, or once it plateaus).

## Iteration 2,000: the teacher plateaus above the LiDAR baseline's collisions (21:35 UTC)

Means over 100-iteration windows (single-iteration values are noisy):

| | t5 1,250–1,350 | t5 2,000–2,026 | a2 (LiDAR PPO) 1,250–1,350 | t0 2,000–2,063 |
|---|---|---|---|---|
| `goals_reached` | 0.89 | 0.94 | 0.90 | 0.59 |
| `goal_distance`, m | 3.72 | 3.67 | 3.75 | 4.06 |
| `heading_error_deg` | 16.1 | 16.3 | 15.6 | — |
| `backward_share` | 0.014 | 0.014 | 0.020 | — |
| `speed`, m/s | 0.60 | 0.61 | 0.58 | — |
| Collision share of episode ends | 0.31 | 0.29 | 0.19 (rising: 0.16, 0.17, 0.19) | 0.78 |
| `Policy/mean_std` | 0.56 | — | 0.54 | — |

t5 has plateaued since about iteration 1,250. At matched iteration and goal rate the
LiDAR-only baseline a2 collides less than the privileged teacher, with similar action
noise. The teacher (`teacher` group) sees proprioception, the command, the 6 nearest
humans' hips (position, velocity, clearance) and its plaza pose; it does **not** see the
LiDAR, so it is not a superset of the student's input. Distilling t5 can then at best
copy a policy that collides more than a2, which bears on ledger D18 (teacher first).

**Replay (job 4060024):** `policy_check.py` plays t5 `model_2000` and a2's newest
checkpoint with mean actions, the same behavior and the same 512 envs × 1,500 steps,
and splits collisions by the human's trait toward the robot.

**Decision rule, written before the replay:**
- **t5 at or below a2 in collisions per episode end:** the training gap came from
  exploration noise or the crowd's draw; distil t5 as planned.
- **t5 still above a2, the excess mostly with cooperative humans:** the teacher's input is
  the limit. Next arm: a teacher with the LiDAR plus the privileged terms (a superset
  of the student), launched as an agent's call that the user can veto.
- **The excess mostly with oblivious or robot-stubborn humans:** neither policy can see
  the trait; compare approach speed and margin (robot velocity toward the human).
- Judge a2 again at its plateau (its job ends about 23:55 UTC), not only at matched
  iteration.

**s5 (job 4060025):** `--distill` of t5's newest checkpoint into the LiDAR student,
4,096 envs, 1,500 iterations. It measures the distillation gap (student against
teacher) and exercises the pipeline at scale. It is not yet the route to the final
policy: the replay decides which teacher to distil.

t5 was cancelled at iteration about 2,300 (plateaued; `model_2300.pt` kept) and its GPU
runs a2s7.

## Replay: the teacher drives into cooperative humans (job 4060024, 21:40 UTC)

Mean actions, `encounter_decisive_robot.json`, 512 envs × 1,500 steps (30 s):

| | t5 `model_2000` | a2 `model_1500` |
|---|---|---|
| Collisions / episode ends | 251 / 682 (0.37 per end) | 108 / 588 (0.18 per end) |
| With oblivious humans | 38 | 32 |
| With robot-stubborn humans | 21 | 5 |
| With cooperative humans | 192 | 71 |
| Robot speed at collision, p50 | 0.79 m/s | 0.56 m/s |
| Robot velocity toward the human, p50 | 0.48 m/s | 0.27 m/s |
| Human in front (bearing within 60°) | 63% | 56% |
| Human closing faster than the robot | 13% | 27% |
| Heading error p50 / backward share | 6.2° / 0.009 | 6.3° / 0.011 |

The observation checks still pass for both (`crowd_nearest` matches the recomputed geometry;
velocities match the hip finite difference).

The replay widens the training gap (1.6× at matched iteration) to 2× per episode end. Of
its 143 extra collisions, 121 are with cooperative humans: t5 walks at about 0.5 m/s
into a human in front of it who is moving slightly away or is stopped (a third are
stopped). Collisions with oblivious humans, which neither policy can foresee from the
trait, are about equal (38 against 32).

The rule above reads this as the teacher's input being the limit. At the time the
replay seemed to support that only weakly (superseded: t7 below shows the missing LiDAR
was the main cause): the teacher saw these humans (the observation checks pass;
most are in front of it, a third stopped) and hit them faster (0.79 against 0.56 m/s,
closing at 0.48 against 0.27). That also fits a teacher that keeps a smaller margin and
approaches faster, trusting exact positions and velocities that the stop rule then
contradicts. **t7** adds the LiDAR to the teacher group (`--teacher-lidar`, 421 inputs against the student's 387),
so the teacher sees everything the student sees. This is the agent's call; the user can
veto it. Same settings as t5 otherwise (code-default rewards, seed 42).

**Decision rule for t7, written before the curves:**
- **About iteration 1,300 (matched with the windows above):** t7's collision share of
  episode ends at or below a2's 0.19, with `goals_reached` near 0.9. Then t7 is the
  teacher to distil, and s5 only measures the distillation gap.
- **t7 like t5 (about 0.3):** the missing LiDAR was not the cause. Two explanations
  remain: seed variance (every arm so far is seed 42, so each comparison is n = 1) and
  a speed-for-margin trade that the privileged inputs allow and the student cannot copy.
  The second argues for fixing the teacher (for example a margin or approach-speed
  term), not for dropping the teacher stage. Check it with the replay's approach speed
  in clearance bins (`policy_check.py` section (f)), t5 against a2.
- **No D18 conclusion goes to the user without a second seed** of the arms that decide
  it. a2s7 is the baseline's replicate; a teacher replicate follows from t7's result.
- **In between:** replay t7 against a2 with `policy_check.py --teacher-lidar` before
  choosing.

## Approach speed by clearance (job 4060111, 21:58 UTC)

Second replay on newer checkpoints (t5 `model_2200`, a2 `model_1700`), same settings.
The gap holds: 0.37 against 0.18 collisions per episode end, again mostly cooperative
humans (175 against 61). New section (f): over moving env-steps with the nearest human
within 60° of the robot's heading, the robot's velocity toward that human by clearance:

| Clearance | t5 toward, p50 / p90 (m/s) | a2 toward, p50 / p90 (m/s) |
|---|---|---|
| 1.00–1.50 m | 0.42 / 0.85 | 0.39 / 0.81 |
| 0.75–1.00 m | 0.40 / 0.84 | 0.36 / 0.76 |
| 0.50–0.75 m | 0.38 / 0.81 | 0.32 / 0.71 |
| 0.25–0.50 m | 0.38 / 0.78 | 0.29 / 0.65 |
| 0.00–0.25 m | 0.44 / 0.82 | 0.31 / 0.64 |

t5 hardly slows for a human ahead and speeds up inside 0.25 m; a2 slows by about a
quarter. Both spend about 2–3% of moving time within 0.25 m of a human ahead. This
seemed to support the speed-for-margin reading (superseded by t7). It also shows that neither
policy decelerates much: the rewards make a collision cheap relative to tracking. The
termination ends the reward stream, which at γ = 0.99 discounts after about 2 s. The
contact penalty is small, and there is no terminal penalty. **Candidate next lever (not
launched):** a terminal penalty on `crowd_collision`, or a clearance-scaled speed limit
in the proximity term, tested on the teacher and on the LiDAR policy alike. This is a
reward change for the user to see first, after t7 and a2s7.

## t7 at iteration 1,350, s5 done, a2's replicate (23:08 UTC)

Window means (100 iterations):

| | t7 (teacher + LiDAR) 1,250–1,350 | t5 1,250–1,350 | a2 1,250–1,350 | a2 plateau 2,000–3,100 | a2s7 1,000–1,100 (a2: same window) |
|---|---|---|---|---|---|
| `goals_reached` | 0.93 | 0.89 | 0.90 | 0.98–1.03 | 0.84 (0.81) |
| Collision share of ends | 0.21 (rising: 0.17, 0.18, 0.21) | 0.31 | 0.19 | 0.18–0.19 | 0.18 (0.17) |
| `speed`, m/s | 0.59 | 0.60 | 0.58 | 0.59–0.60 | 0.57 (0.56) |

- **The LiDAR closes most of the teacher's gap.** t7 is near a2 and far below t5.
  So the input did matter, more than the speed-for-margin reading allowed. t7 is not
  better than a2, though: privileged states on top of the LiDAR bring no collision
  advantage so far.
- **a2 replicates:** seed 7 tracks seed 42 within 0.01 in collision share at the same
  iterations, so the t5–a2 gap (0.12) is not seed noise.
- **a2 plateaus** from about iteration 2,000 at 18–19% collision ends with
  `goals_reached` about 1.0.
- **s5 (distillation of t5) matches its teacher:** iterations 1,250–1,350 give goals
  0.90 and a 0.29 collision share (t5: 0.91, 0.31); behavior loss 0.086 → 0.027. The
  distillation gap is negligible, so a student inherits its teacher's collision rate.

Per the rule (t7 between a2 and t5, slightly above a2), job 4061374 replays t7 and a2
at `model_1300` with mean actions, plus a2's newest checkpoint.

## Replay: t7 is level with a2 (job 4061374, 23:15 UTC)

Mean actions, same settings as the earlier replays:

| | t7 `model_1300` | a2 `model_1300` | a2 `model_3200` |
|---|---|---|---|
| Collisions per episode end | 0.238 | 0.224 | 0.239 |
| Oblivious / robot-stubborn / cooperative | 46 / 12 / 84 | 47 / 10 / 76 | 44 / 10 / 87 |
| Robot speed at collision, p50 | 0.63 m/s | 0.56 m/s | 0.60 m/s |
| Velocity toward the human ahead, 1.0–1.5 m → 0–0.25 m | 0.37 → 0.33 | 0.36 → 0.33 | 0.42 → 0.35 |

a2's replays range 0.18–0.24 per episode end across checkpoints (iterations 1,300,
1,500, 1,700, 3,200) while its training curve is flat, so single-replay differences
under about 0.05 are noise. **t7 and a2 are level within checkpoint noise; t5 (0.37
twice) is clearly worse.**

**Conclusions:**
- **The LiDAR was the teacher's missing input.** The speed-for-margin reading above
  was wrong.
- With it, the teacher equals PPO on LiDAR: it does not learn faster or collide less.
  For option (a) the teacher stage adds nothing under the current rewards.
- Distillation's replay gap is within checkpoint noise (s5 +0.06, s7 −0.05 collisions
  per episode end against their teachers, below). `--teacher-lidar` became the default for
  fresh teacher runs at 23:20 UTC; recorded runs without the key stay without LiDAR.
- No teacher replicate is planned: t7 ≈ a2 is a null result, and a2's seed spread
  (0.01) cannot change it.

## The reward makes collisions cheap; a terminal penalty, off by default

- **Calibration** (a2, iterations 2,500–3,100):
  - the net reward is 39.3 per episode of 908 steps, about 0.043 per step;
  - at γ = 0.99 (horizon about 100 steps, 2 s) a collision forfeits a return of about 4;
  - contact and proximity penalties add well under 1.
  - mjlab scales every reward by the step duration (0.02 s), so a terminal penalty of
    weight w costs 0.02 w.
- **`collision_end`** (weight 0, skipped) is 1 on the step `crowd_collision` ends the
  episode; `--env.rewards.collision-end.weight` enables it. Job 4061406 played a2 with
  weight −250 for 600 steps: the term fired on all 48 collision steps and on no others,
  at −250 per step (−5 in return). Run 20261012T231909Z: a fresh `--teacher` run with
  the term trains.
- **Proposed arms** (for the user; not launched):
  - LiDAR PPO at w = −250 (a collision costs about 5 more, roughly double);
  - LiDAR PPO at w = −1,000 (about 20 more, roughly 5×);
  - otherwise a2's settings; 4,096 envs, 4,000 iterations. The teacher pipeline follows
    whichever size wins.
- **Floor:** in the t7 and a2 replays about a third of the collisions are oblivious
  humans walking into the robot (`human_toward` 0.5–0.74 m/s, `robot_toward` about
  0.1), which a penalty can reduce only by dodging. About 0.15 of the 0.23 collisions
  per episode end are with humans who stop or give way. About a third of those humans
  are standing still, the clearest sign that the reward, not perception, is the limit.
- **Decision rule for the arms, written before any run:**
  - success is replay collisions per episode end at or below about 0.12 (half the
    avoidable share removed), with `goals_reached` at least 0.9 and speed at least
    0.5 m/s (these two from the training curves at the plateau: the replay reports
    collisions and approach speeds, not goals);
  - fewer collisions bought by standing or hesitating (falling goals or speed, as in
    the forward-first runs) is a failure;
  - pick the smaller weight that succeeds.

## s5 replay: a small distillation gap (job 4061638, 00:05 UTC)

`policy_check.py` now plays a distilled student (its `student_state_dict` loads as the
LiDAR actor). s5 `model_1499`, same settings:
- 0.43 collisions per episode end (t5's teacher replays: 0.37 twice), 237 of 308 with
  cooperative humans;
- the student approaches faster than its teacher: toward the human ahead 0.47 → 0.57 m/s
  from 1.0–1.5 m to 0–0.25 m (t5: 0.42 → 0.44).

In training the two were equal (0.29 against 0.31 collision share), so the replay gap,
about one unit of checkpoint noise, is the student's mean action being bolder than the
teacher's. Distillation inherits the teacher's collisions and adds a little. (s7, below,
does not repeat the gap.)

## s7 replay: the student is level with t7 and a2 (job 4062740, 01:18 UTC)

s7 finished 1,500 iterations in 1 h 25 min. Mean-action replays, same settings as
before:

| | s7 `model_1499` (student) | t7 `model_2000` (its teacher) | a2 `model_3999` |
|---|---|---|---|
| Collisions per episode end | 0.164 (96 / 585) | 0.210 (123 / 586) | 0.174 (101 / 581) |
| Oblivious / robot-stubborn / cooperative | 35 / 7 / 54 | 50 / 10 / 63 | 37 / 10 / 54 |
| Collisions with humans who perceive the robot, per end | 0.10 | 0.12 | 0.11 |
| Moving env-steps with the human ahead within 0.25 m | 1.8% | 2.3% | 2.1% |
| Velocity toward the human ahead, 1.0–1.5 m → 0–0.25 m | 0.34 → 0.27 | 0.40 → 0.32 | 0.43 → 0.30 |
| Robot speed at 0–0.25 m, p50 | 0.50 m/s | 0.53 m/s | 0.51 m/s |

Training windows (100 iterations; the student's rollouts use its own actions):

| | s7 500–1,500 | t7 2,000–2,100 | t7 3,000–3,350 | a2s7 3,000–3,350 |
|---|---|---|---|---|
| Collision share of ends | 0.185–0.189 | 0.18 | 0.18 | 0.18 |
| `goals_reached` | 0.93–0.97 | 0.99 | 1.03 | 1.03 |
| `speed`, m/s | 0.58–0.59 | 0.60 | 0.60 | 0.60–0.61 |

- **The teacher-then-distill pipeline gives a working LiDAR policy for option (a).**
  s7 replays level with its teacher and with direct PPO (all three within the ±0.05
  checkpoint noise). Its one measurable cost is goals: 0.93–0.97 in all four training
  windows against the teacher's 0.99 at the distilled checkpoint, at 0.58 m/s against
  0.60. (The student's rollouts carry different action noise from the teacher's PPO
  rollouts, so even this comparison is approximate.)
- **Correction (2026-10-14): s7 also falls.** The same replay ends 23 of s7's 585
  episodes in a fall (t7: 0 of 586, a2: 1 of 581); the table above omitted falls. s5
  did the same (14 of 715 against t5's 1 of 682, job 4061638), so falls are a second
  measurable cost of distillation; the 60 s render of s7 shows one.
- **A distillation gap in collisions is not resolvable at ±0.05.** s5 replayed 0.06
  above its teacher, s7 0.05 below; s5's bolder approach does not recur (s7 approaches
  slower than t7 and a2). Both students match their teachers' training collision share.
- **Every arm sits at 17–21% collision ends**, teacher, student and PPO alike: the
  reward, not the pipeline, sets the plateau. At a2's final checkpoint about 0.11 of
  the 0.17 collisions per end are avoidable (humans who perceive the robot), so the
  penalty arms' bar of 0.12 still means removing about half of them.
- t7 and a2s7 have plateaued (collision share 0.18, goals 1.03). Both reached the 4 h
  limit near iteration 3,800 (below), which changes nothing here.

## Final state (01:50 UTC)

Slurm's SIGTERM before the 4 h limit stopped t7 (job 4060075) at iteration 3,780 and
a2s7 (job 4060119) at 3,792; both run records say `interrupted`, and sacct reports
FAILED from the non-zero exit. The last checkpoint of each, `model_3700.pt`, loads.
Windows 3,500–3,780: t7 collision share 0.18–0.19, `goals_reached` 1.03–1.05, speed
0.61 m/s; a2s7 0.17–0.18, 1.03, 0.60 m/s. Neither moved after iteration 3,000.
