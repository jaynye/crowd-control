# 2026-10-13: a terminal collision penalty for the LiDAR policy

Status: complete. Neither arm meets the rule. p250 trims collisions modestly (0.15 per
episode end in replay against a2's 0.17); p1000 stops colliding by falling.

## Conclusions

- **−250 helps a little.** Three p250 checkpoints replay at 0.147–0.157 collisions per
  episode end, all below a2's lowest replay (0.174, range 0.17–0.24), with goals, speed
  and falls unchanged. The rule's bar, 0.12, is not met. The avoidable share (humans who
  perceive the robot) drops from 0.11 to 0.08–0.09; the oblivious humans walking into the
  robot (0.06–0.08 per end) remain.
- **−1,000 trades collisions for falls.** 85% of p1000's replay episodes end in a fall
  (0.027 collisions per end), including 150 of 162 commanded-standing episodes; it
  reaches 0.35 goals per episode in training. Falling even when told to stand means
  its locomotion collapsed, not that it falls to dodge people. The likely reason, not
  isolated: no reward term penalizes a fall, which costs only the forfeited return
  (about 4) against about 24 for a collision at −1,000, and its falls were high from
  the first windows.
- **Next arms (proposed, not launched):** a fall penalty at least as large as the
  collision penalty; the penalty applied by resuming a2 (`--resume`) rather than from
  scratch; and, for the oblivious floor, shaping that rewards dodging a human closing
  in, which a terminal penalty does not teach.

## Question

Every LiDAR arm of [robot-aware teachers](2026-10-12-robot-aware-teachers.md) (teacher,
student and PPO) plateaus near 18% collision ends, and the robot barely slows for a
human ahead. A collision costs only the forfeited return, about 4 at γ = 0.99. Does a
terminal penalty (`collision_end`, off by default) buy fewer collisions without
hesitation? The user approved the two proposed arms on 2026-10-13.

## Runs

| Run | Job | Arm |
|---|---|---|
| `runs/20261013T184201Z_train_traversal_p250` | 4071004 (chappie) | `--env.rewards.collision-end.weight -250`: a collision costs about 5 more (roughly double) |
| `runs/20261013T184200Z_train_traversal_p1000` | 4071005 (samantha) | `--env.rewards.collision-end.weight -1000`: about 20 more (roughly 5×) |

Otherwise a2's settings (the code defaults since 2026-10-12): LiDAR PPO,
`encounter_decisive_robot.json`, 8 humans, graph-turns with its rotations, 4,096 envs,
seed 42, A40 on `rl2-lab,overcap`. **3,500 iterations** rather than the proposed 4,000:
at about 16 iterations a minute 4,000 overruns the 4 h limit (t7 and a2s7 were stopped
near 3,800), and a2 plateaued by iteration 2,000.

## Decision rule (written 2026-10-12, before any run)

- Success: mean-action replay at or below about 0.12 collisions per episode end, with
  `goals_reached` at least 0.9 and speed at least 0.5 m/s from the training curves at
  the plateau. Baselines: a2 `model_3999` 0.174 (0.11 of it with humans who perceive
  the robot), t7 0.21, s7 0.16; replay noise ±0.05.
- Fewer collisions bought by standing or hesitating (falling goals or speed) is a
  failure.
- Pick the smaller weight that succeeds; the teacher-then-distill pipeline follows it.
- Clarification (2026-10-14, after p1000): episode ends count falls, so fewer collisions
  bought by falling is a failure of the same kind as hesitating. Replays report ends by
  cause.

## Iteration 1,000–1,250 (19:50 UTC)

Window means (100 iterations), a2 at the same windows in parentheses:

| | p250 1,000–1,100 | p1000 1,000–1,100 | a2 1,000–1,100 |
|---|---|---|---|
| Collision share of ends | 0.16 | 0.013 | 0.17 |
| `goals_reached` | 0.80 | 0.09 | 0.81 |
| `speed`, m/s | 0.56 | 0.45 | 0.56 |

- The penalty is live in both runs (`Episode_Reward/collision_end` fires with each
  collision, about 4× larger in p1000).
- p250 tracks a2 so far. p1000 has nearly stopped colliding but barely reaches goals:
  the hesitation the rule counts as failure, unless it recovers. Too early to judge
  either.

## Final training windows (100 iterations)

| | p250 3,250–3,350 | p1000 3,250–3,350 | a2 plateau 2,000–3,100 |
|---|---|---|---|
| Collision share of ends | 0.15 | 0.03 (most ends are falls) | 0.18–0.19 |
| `goals_reached` | 1.02 | 0.35 | 0.98–1.03 |
| `speed`, m/s | 0.60 | 0.51 | 0.59–0.60 |

Both finished 3,500 iterations (`model_3499.pt`) in 3 h 46–48 min on A40s (chappie,
samantha, `rl2-lab`). p250's collision share fell to 0.15 from iteration 2,000 on;
p1000's goals were still rising slowly (0.23 at 2,000, 0.35 at 3,300).

## Replays (job 4080895, A40, 2026-10-14)

`scripts/smoke/policy_check.py`, mean actions, 512 envs × 1,500 steps,
`encounter_decisive_robot.json`; same settings as the a2, t7 and s7 replays.

| | p250 `model_3499` | p250 `model_3000` | p250 `model_2500` | p1000 `model_3499` | a2 `model_3999` |
|---|---|---|---|---|---|
| Collisions per episode end | 0.147 (82 / 557) | 0.157 (89 / 567) | 0.153 (86 / 561) | 0.027 (34 / 1,279) | 0.174 (101 / 581) |
| Falls among ends | 0 | 1 | 2 | 1,081 | 1 |
| Oblivious / robot-stubborn / cooperative | 35 / 10 / 37 | 45 / 5 / 39 | 34 / 7 / 45 | 16 / 3 / 15 | 37 / 10 / 54 |
| Avoidable collisions per end | 0.084 | 0.078 | 0.093 | 0.014 | 0.110 |
| Moving steps with the human ahead within 0.25 m | 2.2% | 2.3% | 2.1% | 1.0% | 2.1% |
| Velocity toward the human ahead, 1.0–1.5 m → 0–0.25 m | 0.43 → 0.28 | 0.40 → 0.25 | 0.39 → 0.28 | 0.40 → 0.31 | 0.43 → 0.30 |

p250's three checkpoints agree within 0.01, tighter than a2's spread across
checkpoints. It does not approach people more slowly than a2; the reduction is in
collisions with humans who give way or stop.

## Videos

`crowd play traversal --video` (seed 0, 60 s, follow camera) renders p250 and p1000
(`runs/20261014T151629Z_play_traversal_p250`, `runs/20261014T151817Z_play_traversal_p1000`),
tiled 2×2 with the earlier a2 and s7 renders into
`logs/play_compare/compare_a2_p250_p1000_s7.mp4` (untracked). p250 walks the whole
minute without a collision or fall; p1000 stands in place for its first 9 s and lets
the humans pass around it, and falls twice.
