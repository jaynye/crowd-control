# 2026-10-18: the fall fix and the non-avoidant baseline b0

**Result (2026-10-19):** the fall fix works from a walking policy, not from scratch.
a2 resumed with `collision_end` and `fall_end` at -1,000 (a2pf) halves a2's collisions with
no falls at 12% fewer goals; the same penalties from scratch (pf1000) learn to stand still,
and ramping them in (pf1000r1500) walks but reaches 40% fewer goals than a2. b0, which
learns nothing about the humans, collides 2.3× as often as a2 in the robot-aware crowd and
1.2× in the robot-blind crowd: most of the training crowd's safety is the humans' own
avoidance.

User (2026-10-18): "After you complete the implementation, we can launch the fall fix and
non-avoidant baseline b0", then "If we are ready, we should ... launch our baselines."
Arms and rationale: [traversal evaluation](../decisions/traversal-evaluation.md)
(sections "Non-avoidant baseline (b0)" and "The fall fix").

## Question

- **Fall fix:** p1000 (`collision_end` −1,000) falls in 85% of its episodes, and no term
  prices a fall. Does a `fall_end` penalty at parity remove the falls (the user's
  hypothesis)? Does ramping the penalties in, or adding them to a policy that already
  walks, avoid the trap that forms at iterations 400–800?
- **b0:** how much of a policy's collision record comes from the humans' own avoidance?

## Arms

All arms use the env a2 and p1000 used: 4,096 envs, 8 humans, the robot-aware crowd
(`encounter_decisive_robot.json`), graph-turns (`CROWD_GRAPH`, `CROWD_GRAPH_ROTATIONS`),
seed 42, code-default rewards (equal to a2's overrides), A40 through
`train_traversal.sbatch` on `--partition=rl2-lab,overcap --qos=short`. W&B disabled
(ledger D20, no canary yet). A weight of −1,000 costs 20 per event (dt 0.02 s).

| Arm | Run | Job | Change | Iterations |
|---|---|---|---|---|
| b0 | `runs/20261018T225615Z_train_traversal_b0` | 4137004 | `--no-avoidance`: no LiDAR (actor 99 inputs against a2's 387), no crowd rewards, no collision termination | 4,000 from scratch |
| pf1000 | `runs/20261018T225717Z_train_traversal_pf1000` | 4137005 | p1000 plus `fall_end` −1,000 | 3,500 from scratch |
| pf1000r1500 | `runs/20261018T225717Z_train_traversal_pf1000r1500` | 4137006 | pf1000 with `--penalty-ramp 1500` (both penalties linear from 0) | 3,500 from scratch |
| a2pf | `runs/20261018T225717Z_train_traversal_a2pf` | 4137007 | `--resume` a2 `model_3999.pt` with both penalties at −1,000 | +1,000, to 4,999 |

Start checks (first 25–60 iterations): every `run.json` records graph-turns,
`encounter_decisive_robot.json`, 8 humans, seed 42 and W&B `disabled`;
pf1000's `Episode_Reward/fall_end` is −1.0 from iteration 0; pf1000r1500's
`Curriculum/penalty_ramp/*` read −19.7 at iteration 29 (−1,000 × 29 / 1,500 = −19.3);
b0's early fall and length curves match a2's first iterations.

## Plan when they finish

1. Training curves against a2, p250 and p1000 (falls, episode length, goals, collision
   ends), at matched iterations.
2. With the user's go-ahead: `crowd eval traversal` (last three checkpoints, both crowds,
   both modes) for these arms and a2, p250, p1000, t7, s7. Report collisions and falls
   per robot-minute side by side, b0 as the floor, and p1000's fall attribution.
3. After any requeue, confirm pf1000r1500's ramp continued rather than restarting
   (mjlab restores `common_step_counter` from the checkpoint).

## Results

Evaluations (`crowd eval traversal`, last 3 checkpoints, 4,096 envs × 60 s, paired goals,
A40; end mode; collisions and falls per robot-minute; run directories in the run log):

| Policy | Aware: collisions | Aware: falls | Aware: goals per moving minute | Aware: human-led | Blind: collisions |
|---|---:|---:|---:|---:|---:|
| b0 (no avoidance) | 0.95 [0.93–0.96] | 0.000 | 4.27 | 19% | 4.23 |
| a2 | 0.42 [0.38–0.47] | 0.000 | 4.06 | 42% | 3.65 |
| t7 | 0.42 | 0.001 | 4.11 | 42% | 3.59 |
| s7 | 0.49 | 0.024 | 3.93 | 36% | 3.51 |
| p250 | 0.31 | 0.000 | 3.78 | 53% | 3.47 |
| p1000 | 0.17 | 3.25 | 2.47 | 76% | 2.19 |
| pf1000 | 0.27 | 0.002 | **0.03** (stands) | 74% | 2.04 |
| pf1000r1500 | 0.19 | 0.000 | 2.43 | 84% | 3.18 |
| **a2pf** | **0.18** | **0.000** | 3.57 | 80% | 3.17 |

- **What the policies learn:** against b0, a2 removes 56% of collisions in the aware crowd
  but 14% in the blind crowd. b0's collisions are mostly its own (19% human-led), the
  avoiding policies' increasingly the humans' (42–84%). Every policy collides 4–9× more in
  the blind crowd.
- **The fall fix:** a fall penalty at parity stops the falls in every arm. From scratch it
  ends in standing still (pf1000: 0.03 goals per moving minute, half the encounters); ramped
  in over 1,500 iterations it walks at 40% fewer goals; from a2 it keeps 88% of a2's goals.
- **p1000's falls are partly collision escapes:** a time to contact under 1 s preceded
  49–55% of its falls in the 2 s before, against 27–30% of matched controls (closest
  clearance median 0.50–0.56 m against 0.80–0.84 m); attributable about 30% in the aware
  crowd and about 50% in the blind crowd (62–74% against 29–36%). An association: losing
  balance while dodging looks the same.

## Log

- 2026-10-18 22:56–22:57 UTC: all four started (b0 on synapse, pf1000 on tachikoma,
  pf1000r1500 on dendrite, a2pf on synapse).
- 2026-10-19 03:50 UTC: all evaluations done; s7's first evaluation (4137186) was preempted
  on overcap and rerun on rl2-lab (4137777).
- 2026-10-18 23:17 UTC: evaluations submitted (user go-ahead): `eval_traversal.sbatch <run>
  --last 3 --crowd both --mode both` for a2 4137182, p250 4137183, p1000 4137184, t7
  4137185, s7 4137186; for b0 4137187, pf1000 4137188, pf1000r1500 4137189 and a2pf
  4137190 with `--dependency=afterok` on their training jobs.
