# Traversal evaluation: command-following and collision metrics, a non-avoidant baseline, and the fall fix

Opened 2026-10-18. Status: open; harness built and used. b0 and the fall arms are trained
and evaluated ([result](../experiments/2026-10-18-fall-fix-and-b0.md)): a2pf (a2 resumed
with both penalties at -1,000) halves a2's collisions without falls; b0 collides 2.3× as
often as a2. Still open: the scripted encounter probes and whether to adopt a2pf's penalties
for every arm. **Whose
call:** the user. They decide the metric definitions and what gets built and run.

**User requirement, 2026-10-21:** compare collision rate **and goal completion rate**
under different fixed human behaviors for every policy/baseline, including easy and
hard conditions. Curriculum should retain easy-condition performance while improving
the hard cases; averaging conditions can hide the relevant regression. Existing
[blind-crowd evaluations](../experiments/2026-10-21-blind-crowd-arms.md) cover aware and
fully blind endpoints and show a tradeoff; intermediate mixtures and separately varied
robot interaction traits still need evaluation. Report arrivals per moving-command
minute and goal-leg completion fraction, with end/continue mode context.

## Built (user, 2026-10-18: "build the metric suite and evaluation harness")

- **`crowd eval traversal`** (`src/crowdmotion/sim/mjlab/evaluate.py`, metrics in
  `src/crowdmotion/metrics/traversal.py`, guide `docs/usage/traversal-evaluation.md`).
  Envs, seconds, episode time-out, humans, behavior, crowd, mode, seed and the scales are
  options. Rows go to the training lineage's W&B run through `queue_eval` and
  `flush_outbox` (ledger D20; `disabled` until the canary).
- **Paired goals** (`CrowdGoalCommandCfg.paired_seed`, seed scheme `traversal_goals`):
  goals and speeds are keyed by the goal count, timers and standing flags by the
  scheduled-resample count, and each goal is drawn from the previous goal.
- **Departures from the proposal, found on the first a2 smoke and in the Astra review:**
  - TTC divides by the line-of-sight closing speed of root and hips. The clearance's
    own finite difference swings with the gait: 267 encounters per robot-minute and a
    0.14 s "reaction" in the smoke, against 31 and 0.6 s afterwards.
  - Encounters end only after 0.5 s out of the state.
  - The reaction velocity is smoothed (0.3 s), measured against itself at the threat,
    and restarted by a new goal.
  - Fall controls are weighted to the falls' standing × episode-age strata.
  - One continuous contact is one collision.
- **Check:** a2 `model_3999`, 4,096 envs × 60 s, aware crowd
  (`runs/20261018T160808Z_eval_traversal_a2`): 0.390 collisions per robot-minute in end
  mode, equal to the 0.39 derived from its earlier replay. About 90 ms per step against
  training's 76 ms.
- **Arms launched (user, 2026-10-18):** b0 (`--no-avoidance`), pf1000 (p1000 plus
  `fall_end` −1,000), pf1000r1500 (both penalties ramped in over 1,500 iterations) and
  a2pf (a2 resumed with both penalties); jobs 4137004–4137007,
  [experiment](../experiments/2026-10-18-fall-fix-and-b0.md).

## Original proposal (2026-10-18, before the build)

The user's first instruction was to evaluate what exists and propose what to build,
without launching training runs.

## Question (user, 2026-10-18)

- Evaluate every policy on:
  - (a) **command following**, where small deviations should score better than large
    ones; for example, the average deviation from the desired trajectory over time, plus
    other measures alongside it;
  - (b) **average time to collision**, and the collision rate over a full episode.
- Train a baseline that learns no collision avoidance.
- The from-scratch penalty runs fall. The user suspects that a fall costs less than a
  collision.

## What exists (checked 2026-10-18)

| Tool | What it measures | Limits |
|---|---|---|
| Training curves | Per-episode means: `goals_reached`, `goal_distance`, `heading_error_deg`, `backward_share` and `speed` (`CrowdGoalCommand`); `error_vel_xy` and `error_vel_yaw` (mjlab's velocity command); ends by cause | Different goals in every run; training noise; means over all steps, encounters included |
| `scripts/smoke/policy_check.py` | Mean-action replay, 512 envs × 30 s: collisions per episode end, attribution (trait, who closes faster, bearing), approach speed by clearance, backward share, yaw tracking | Not a `crowd` command. It ignores `run.json`: its default behavior is `encounter_turns.json`, but the training default is `encounter_decisive_robot.json` |
| `crowd play traversal --video` | `summary.json`: env 0's collisions, their times and ends by cause | One env; meant for viewing |
| Collision rule | Clearance between the robot spheres and the human joint spheres: `crowd_contact` (any overlap), `crowd_collision` (termination at 5 cm penetration) | Play mode ends on a collision only with `--end-on-collision` (play.py's default) |

**Gaps:**
- **No trajectory-level command-following measure.** Velocity error under-reports
  detours. The command is re-aimed at the goal from the robot's own pose every step, so
  once a detour is under way the velocity error is small again.
- **"Collisions per episode end" is a ratio among terminations,** and falls inflate its
  denominator: p1000 reads 0.027 (34 / 1,279) while falling 1,081 times. Converted to
  rates from the [collision-penalty](../experiments/2026-10-13-collision-penalty.md)
  replays (512 envs × 30 s = 256 robot-minutes):

  | | Collisions per robot-minute | Mean robot time between collisions | Falls per robot-minute |
  |---|---|---|---|
  | a2 `model_3999` | 0.39 | about 2.5 min | 0.004 |
  | p250 `model_3499` | 0.32 | about 3.1 min | 0 |
  | p1000 `model_3499` | 0.13 | about 7.5 min | 4.2 |

- **No exposure normalization.** A policy that stands, or keeps falling near its spawn
  (spawned as far as possible from every human), meets fewer humans and looks safe for
  free.
- **Reactivity** is measured only by the approach-speed bins. a2 slows only from 0.43 to
  0.30 m/s between 1.0–1.5 m and 0–0.25 m of clearance to the human ahead.
- **Comparisons are not paired.** `_sample_goals` and the robot spawn draw from the
  global torch generator, so two policies see different goal sequences. The replay
  noise (±0.05 per end) exceeds the 0.02–0.03 effects being compared.

## Proposed metric suite (for the user to agree)

Every metric is computed from the root trajectory, the command and the clearance, so
the from-scratch policies and the SONIC hierarchy share the same code. Values are
reported with confidence intervals over envs and with no pass thresholds (AGENTS.md).
The scales (σ, encounter radius, TTC cutoffs) are provisional.

### (a) Command following

- **Reference trajectory:**
  - goal mode: for each goal leg, the straight segment from the robot's position when
    the goal is assigned to the goal, timed at the commanded speed (with its ramp-down
    within `slow_m`);
  - velocity mode: the command integrated from the leg start, with drift measured over
    1 s and 3 s windows.
- **Measures per leg:**
  - cross-track deviation d(t) from the segment: mean, RMS (which weighs large
    deviations more) and p95;
  - **tracking score:** mean exp(−(d/σ)²) with σ = 0.5 m, like the reward kernels.
    It is 1 on the line, falls off smoothly for small deviations and saturates for
    large detours: small deviations score better than large ones;
  - along-track lag at arrival (seconds behind the timed reference), and the
    time-to-goal ratio (actual time over ideal time);
  - detour ratio: path length over straight-line distance;
  - SPL: success × straight / max(path, straight), with success meaning arrival
    within the leg's time limit;
  - goals per moving robot-minute. The per-episode count used now depends on how
    early terminations cut episodes short.
- **Free space versus encounters:** split every deviation measure into free-space
  steps and encounter steps, with an encounter as defined under (b) below.
  - Free-space numbers measure tracking.
  - Encounter numbers measure the price of avoidance: the minimal-detour property that
    the user also asked of the crowd.
- Keep `error_vel_xy`, `error_vel_yaw`, the heading error and the backward share as
  secondary measures.

### (b) Collisions, time to collision and reactivity

"Time to collision" has two readings. Both are reported, under different names.

- **Time to first collision (episode level).** Use the play env (no 20 s time-out),
  with collisions ending the episode as in training. Treat falls and the run's end as
  censoring. Report the Kaplan–Meier median and the restricted mean to 60 s. Also
  report the hazard: collisions per moving robot-minute, which is 1 / mean time to
  collision under a constant hazard.
- **Collision rate over a full episode.** Use fixed 60 s episodes with
  `--no-end-on-collision`. Report collision onsets per episode and per minute, and the
  time in contact. States after a contact are outside the training distribution, so
  read these numbers as pessimistic.
- **Falls per robot-minute,** always reported next to collisions.
- **Fall attribution.** At each fall, record the nearest human's clearance and TTC over
  the preceding 1–2 s, and compare them with the same measures at matched random
  times. Split falls by whether any human was within 2 m. This tests the user's
  hypothesis with inference only: falls that cluster at low TTC mean the policy falls
  to escape collisions, and falls that do not mean locomotion collapse. p1000's 1,081
  replay falls are enough.
- **Exposure:** an encounter is a human within 1.0 m of clearance or with a TTC under
  3 s. This one definition is used everywhere. Report:
  - encounters per minute;
  - collisions per encounter;
  - minimum clearance per encounter.
- **Geometric TTC (time to contact).** Computed per step and per human as clearance
  divided by the clearance's closing rate (finite difference, only while closing), so
  it is consistent with the collision rule. Report each encounter's minimum TTC and
  the share of moving steps with a TTC under 1 s and under 2 s.
- **Reactivity:**
  - the approach-speed bins (kept);
  - response latency: from a TTC under 2 s to the first avoidance, meaning a speed drop
    of more than 0.1 m/s or a heading change of more than 10° away from the command.
- **Attribution (kept):**
  - by trait (oblivious, robot-stubborn, cooperative), so the floor of oblivious humans
    walking into the robot (0.06–0.08 per end) stays visible;
  - human-led against robot-led.
- **Two crowds:** the robot-aware crowd (the training default) and the robot-blind
  crowd (`--robot-blind`) as a stress test. Cooperative humans hide part of the
  difference between policies.

### Controlled encounter probes (second step)

- **Scenarios:** scripted single-human encounters: head-on, 90° crossing, overtaking,
  and a human emerging beside a wall.
- **Setup:**
  - the human is oblivious, with a fixed start and route (`EncounterCfg` `waypoints`
    conditioning and `set_waypoints`);
  - the robot gets a fixed start and goal. This needs a scenario hook in
    `CrowdGoalCommand`.
- **Report:** reaction latency, minimum clearance and TTC, detour and time lost.

### Protocol

- **The command:** `crowd eval traversal <run> [--checkpoint] [--crowd aware|blind]
  [--envs 4096] [--seconds 60] [--seed]`.
  - It writes `runs/<UTC>_eval_traversal_<label>/` with `metrics.json` and tables per
    leg and per encounter.
  - It reads `run.json` as `crowd play traversal` does, and plays mean actions.
  - It replaces `policy_check.py` for comparisons.
- **Paired goals.** Per-env `core.rng` streams (a named scheme) are not enough by
  themselves. `_sample_goals` keeps the first of 16 candidates at least 3 m from the
  robot, so equal draws give different goals once two robots stand in different
  places.
  - The fix is an eval-only goal chain drawn per env independently of the robot: each
    goal at least 3 m from the previous goal. Speeds, spawns and standing flags come
    from the same streams.
  - The option is off by default, so the training env's randomness is unchanged.
  - In the robot-blind crowd, human motion is then paired too, up to each env's first
    episode end, which resets the crowd. The robot-aware crowd reacts to the robot, so
    there only goals and spawns are paired.
- **Size:** 4,096 envs × 60 s, which is 16 times the robot time of today's replays.
  Inference alone takes an estimated 4 min per policy and crowd on an A40, from the
  76 ms training step time; the estimate is unmeasured. Evaluate the last three saved
  checkpoints to show checkpoint noise.

## Non-avoidant baseline (b0)

- **Env:** the same env, rewards and goal command, with the crowd removed from learning:
  - no LiDAR in the actor (proprioception and command only), and no privileged crowd
    terms in the critic;
  - no `crowd_collision` termination, since the forfeited return alone teaches
    avoidance;
  - no `crowd_proximity`, `crowd_contact` or `collision_end` terms.
- **Why this is enough:** the humans have no physical contact with the robot (the rigid
  pieces have `contype` 0), so this policy cannot learn anything about them. Training
  with or without humans gives the same experience. Dropping the crowd and the LiDAR
  would make it cheaper, but `CrowdGoalCommand.reset` reads the humans for the spawn,
  which needs a small code path.
- **Evaluation:** in both crowds. In the robot-aware crowd, b0 measures what the
  humans' own avoidance achieves, and a policy's contribution is the difference from
  b0. b0 is also the low-level velocity tracker for the hierarchical pre-test in
  [robot policy architecture](robot-policy-architecture.md).
- **Build and cost:** an env option recorded in `run.json` (`ENV_OPTIONS`; name to
  decide, for example `--no-avoidance`). One 4,096-env run, at most about 3.5 h on an
  A40.

## The fall fix

The training curves were read from TensorBoard. p250 and p1000 are a2 plus
`collision_end` at −250 and −1,000, with the same seed (42) and env:

| Iteration | `fell_over` (a2 / p250 / p1000) | Episode length, steps | `goals_reached` |
|---|---|---|---|
| 200 | 37.8 / 37.7 / 36.6 | 106 / 107 / 107 | 0.02 / 0.01 / 0.02 |
| 400 | 20.9 / 19.8 / 24.8 | 201 / 197 / 169 | 0.01 / 0.01 / 0.01 |
| 800 | 0.8 / 1.8 / 20.3 | 789 / 703 / 240 | 0.61 / 0.56 / 0.06 |
| last (3,999 / 3,499 / 3,499) | 0.0 / 0.0 / 9.8 | 950 / 954 / 359 | 0.89 / 0.97 / 0.30 |

**Conclusion: the user's suspicion is supported. The mechanism is sharper, but not
isolated.**
- **No term penalizes a fall.** mjlab's velocity task only terminates on `fell_over`.
  A fall costs only the forfeited return: about 2 early in training, when the step
  reward is about 0.02, and about 4 at the plateau. A collision at −1,000 costs 20 on
  top of that.
- **The runs are identical through iteration 200.** p1000 splits off between iterations
  400 and 800, exactly when a2 starts walking and meeting humans. Walking brought
  collisions at −20 each before it paid anything back.
- **p1000 walks, but its locomotion never stabilized.** At the end it moves at
  0.51 m/s and reaches 0.30 goals per episode. Yet falls stayed at 10–20 per logged
  iteration against about 0 for a2, and episodes at 230–360 steps against about 950.
  That fits a trap in learning.
- **Whether it also falls on purpose when a collision looms is not separated.** 150 of
  162 commanded-standing replay episodes ended in a fall. That fits locomotion
  collapse, but also a robot that falls to dodge an approaching human. The fall
  attribution under (b) tells them apart with inference only.
- The students' replay falls (s5, s7) come from distillation, not from this reward.

**Proposed (not built):**
1. **`fall_end`:** a terminal penalty on `fell_over` that mirrors `collision_end`
   (scaled by dt), weighted at least as heavily as the collision. Not mjlab's
   `is_terminated`: it also fires on `crowd_collision`, so the two weights could not
   be set separately.
2. **Ramp the collision weight in,** with mjlab's staged `reward_curriculum`
   (`mjlab.envs.mdp.curriculums`), or apply it by `--resume` from a2, which already
   walks. The trap forms early, so
   parity alone may not let the policy escape it.
3. **Optional:** no collision termination in training, with the per-step overlap
   penalty only. This removes ending the episode as an escape, but it changes the
   training env.

## What would decide it

- **For the user to agree:**
  - the metric definitions and their provisional scales;
  - building `crowd eval traversal` with the paired-goal option;
  - b0;
  - the fall arms.
- **Then:** run the fall attribution on p1000, and evaluate a2, p250, p1000, t7, s7 and
  b0 in both crowds. That needs GPU
  replays (inference only), to be launched only once the user approves.

## Links

- Env and rewards: `src/crowdmotion/sim/mjlab/traversal.py`. Replay script:
  `scripts/smoke/policy_check.py`. Player: `src/crowdmotion/sim/mjlab/play.py`.
- [Collision penalty](../experiments/2026-10-13-collision-penalty.md),
  [robot-aware teachers](../experiments/2026-10-12-robot-aware-teachers.md),
  [robot policy architecture](robot-policy-architecture.md).
- Runs: `runs/20261012T201213Z_train_traversal_a2`,
  `runs/20261013T184201Z_train_traversal_p250`,
  `runs/20261013T184200Z_train_traversal_p1000`.
