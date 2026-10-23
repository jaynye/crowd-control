# Traversal evaluation

Persistent-goal planner training and its fixed-condition grid are described in
[goal navigation](navigation-training.md). Success/deadline outcomes are captured
before auto-reset, separately from collision, fall and rollout censoring.

`crowd eval traversal <run>` measures a trained traversal policy (`crowd train
traversal`) under a fixed protocol, so that runs and checkpoints can be compared on the
same goals: how well it follows its goal command, and how often and how closely it meets
the humans. It plays mean actions only; nothing is trained.

```bash
# A quick look: the newest checkpoint, the training crowd, both collision modes.
.venv/bin/crowd eval traversal runs/<UTC>_train_traversal_<name> --envs 512 --seconds 30

# The comparison protocol: the last three checkpoints, both crowds, 4,096 envs x 60 s.
sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/eval_traversal.sbatch \
  runs/<UTC>_train_traversal_<name> --last 3 --crowd both --mode both
```

The run's `run.json` fixes the env: graph, rotations, behavior, humans and the
observation options (teacher LiDAR, the non-avoidant baseline). A teacher plays its
privileged actor; a distilled student plays as the LiDAR policy.

## Protocol

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint model_<i>.pt` | | Evaluate these checkpoints (repeatable; names in the run or paths) |
| `--last N` | 1 | Otherwise the run's last N checkpoints |
| `--crowd aware\|blind\|both` | `aware` | `aware`: humans perceive the robot (oblivious, stubborn or cooperative toward it, as in training). `blind`: every human is oblivious, so only the robot can avoid them |
| `--mode end\|continue\|both` | `both` | `end`: a collision ends the episode, as in training. `continue`: episodes run through collisions. Falls and time-outs end episodes in both |
| `--envs` | 4,096 | Worlds simulated in parallel |
| `--seconds` | 60 | Simulated seconds per rollout |
| `--episode-s` | none within the rollout | Episode time-out; episodes still running at the end are censored there |
| `--humans`, `--behavior` | the run's | Crowd size and encounter-design behavior config |
| `--seed` | 0 | Crowd, goal and env seed |
| `--paired` / `--no-paired` | on | Paired goals (below) |
| `--config` | | JSON of `TraversalEvalCfg` scales; unknown keys are rejected |
| `--attach` / `--no-attach` | on | Attach each result to the training run's W&B lineage |

**Paired goals.** With `--paired`, every random choice of the goal command comes from a
per-env stream keyed by the seed, env, episode and leg: the spawn candidates and yaw,
the goals, the leg speeds, the standing flags and the resample times. Each goal is the
first candidate at least 3 m from the previous goal, not from the robot. Two policies
then get the same goal chain in the same env and episode however they walk. In the
`blind` crowd the humans are paired too, since their streams are keyed by the env's
episode count and they ignore the robot; in the `aware` crowd they react to each robot.
The robot's joint initialization noise is not paired. Training never uses paired draws.

Each (checkpoint, crowd, mode) is one rollout. The env is built once per crowd and mode
and rewound before each checkpoint, so every checkpoint starts the same episodes.

## Outputs

A fresh `runs/<UTC>_eval_traversal_<name>/` holds:

- `metrics.json`: the protocol and one entry per rollout (`checkpoint`, `iteration`,
  `crowd`, `mode`, `metrics`, and the W&B attachment status);
- `tables/<iteration>_<crowd>_<mode>_<table>.csv`: one row per leg, encounter,
  collision, fall, fall control and episode;
- `run.json`.

Each rollout attaches to the training run's W&B lineage as `eval/traversal/<crowd>/<mode>/*`
at the checkpoint's iteration.

## Definitions

The scales below are the `TraversalEvalCfg` defaults. They label motion for analysis and
are not pass thresholds.

**Clearance** is the distance between the robot's 13 spheres and the humans' joint
spheres, the same geometry as the training costs. A **collision** is a robot–human pair
whose clearance falls below −5 cm (the `crowd_collision` rule). A pair counts again only
after its clearance has returned to 0, so one continuous contact is one collision.

**Time to contact (TTC)** is a pair's clearance divided by the closing speed of the
robot's root and the human's hips along the line between them, while they close faster
than 0.05 m/s. It is 0 in contact and infinite otherwise. (The clearance itself swings
with the gait, so dividing by its own rate of change is too noisy.) An **encounter** is a
pair within 1.0 m of clearance or under 3 s TTC; it ends once the pair has been out of
that state for 0.5 s, or with the episode. A step is an encounter step when any pair of
its env is in that state.

### Command following

A **leg** runs from a goal's assignment to its arrival, its resample (every 10–15 s) or
the episode's end. Its reference is the straight segment from the robot's position at
assignment to the goal, timed at the leg's commanded speed with the command's ramp-down
within 1 m of the goal. Standing envs have no legs.

| Metric | Meaning |
|---|---|
| `command/cross_track_mean_m`, `_rms_m`, `_p95_m` | Distance from the reference segment over moving steps; RMS weighs large deviations more |
| `command/tracking_score` | Mean of exp(−(d/σ)²), σ = 0.5 m: 1 on the path, smoothly lower for small deviations, saturating for large detours |
| `command/free/*`, `command/encounter/*` | The same, split into steps with no encounter and encounter steps. Free-space numbers measure tracking; encounter numbers measure the price of avoidance |
| `command/success` | Share of legs that reach the goal before a resample or the episode's end |
| `command/spl` | Mean of success × straight / max(path, straight), with straight = distance to the arrival radius |
| `command/lag_median_s`, `command/time_ratio_median` | Arrived legs: seconds behind the timed reference, and actual over ideal time |
| `command/detour_ratio_median` | Arrived legs: path length over straight distance |
| `command/goals_per_moving_min` | Arrivals per minute of moving robot time |
| `command/vel_error_xy`, `command/vel_error_yaw` | Body-frame velocity tracking error, moving steps |
| `command/drift_1s_mean_m`, `command/drift_3s_mean_m` | Executed displacement against the integrated command over 1 s and 3 s windows |

### Collisions, time to collision and reactions

| Metric | Meaning |
|---|---|
| `collision/per_robot_min` | Collisions per minute of robot time; `moving_per_min` and `standing_per_min` split by the robot's command |
| `collision/<trait>_per_robot_min` | By the human's trait toward the robot: `oblivious`, `stubborn`, `cooperative` |
| `collision/human_led_share`, `behind_share`, `robot_still_share` | Collisions where the human closed faster than the robot; the human was behind the robot (bearing beyond 120°); the robot moved slower than 0.15 m/s |
| `collision/per_encounter`, `collision/contact_share` | Collisions per encounter; share of robot time in contact (clearance below 0) |
| `first_collision/median_s`, `rmst_s`, `survival_<t>s` | Kaplan–Meier time to an episode's first collision, with episodes ending otherwise counted as censored: the median (null when survival stays above 0.5), the mean restricted to the longest episode, and survival at 10, 20, 30 and 60 s |
| `episode/collisions_mean` | Collisions per episode (meaningful in `continue` mode) |
| `encounter/per_robot_min`, `min_clear_median_m`, `min_ttc_median_s`, `min_ttc_lt_<c>s_share` | Encounters: their rate, closest approach and smallest TTC. Encounters that never close have an infinite TTC and count in the median, which is null when at least half never close |
| `ttc/moving_steps_lt_<c>s_share` | Share of moving steps with some pair under 1 s or 2 s TTC |
| `reaction/response_share`, `reaction/latency_median_s` | Encounters that reach a TTC under 2 s while the robot moves: the share in which it then slows by 0.1 m/s or turns its velocity 10° away from its velocity at that moment, and how long that takes. The velocity is smoothed (exponential, 0.3 s) so that the gait's speed swing does not count, and a new goal restarts the measurement so that turning toward it does not count either |
| `approach/<lo>-<hi>m/speed`, `toward` | With the nearest human ahead (within 60° of the heading): the robot's speed and its velocity toward the human, by clearance band |

### Falls

| Metric | Meaning |
|---|---|
| `fall/per_robot_min` | Falls (`fell_over`) per minute of robot time |
| `first_fall/*` | Kaplan–Meier time to a fall (a fall in the same step as a collision counts) |
| `fall/near_share`, `fall/min_clear_median_m_2s`, `fall/min_ttc_lt_<c>s_share_2s` | Over the 2 s before each fall: a human within 2 m of clearance, the closest clearance, a TTC under 1 s or 2 s |
| `fall/control/*` | The same at about 20,000 random times, weighted to the falls' mix of standing status and episode age (bins at 1, 2, 5 and 10 s), so that falls early in an episode meet controls early in an episode |
| `fall/control_coverage` | Share of falls whose stratum has controls |

Falls that cluster at low TTC against the controls mean the policy falls to escape a
collision; falls that do not mean its locomotion fails on its own.

## Planning arms

A run trained with `--planner-rate` plays through its decision-rate env: the policy acts once
per decision and the metrics are recorded at every env step in between, so `--seconds` must
be a whole number of decisions. `--planner-zero 1|2` evaluates P0, the released planner and
frozen SONIC following the goal command with a zero residual, in the env of the run given
(its graph, behavior and humans); nothing is loaded or attached, and the output directory is
`eval_traversal_p0_<rate>hz`.

The reaction metrics do not measure avoidance for these arms: each replan changes the
robot's velocity by itself, so P0, which ignores the humans, still counts as responding to
most threats within a fraction of a second. Compare the collision, encounter and time-to-
contact measures instead.

## Reading the two modes

`end` matches training: its time to first collision is the episode-level measure, and
its collision rate is a hazard (1 / mean time to collision under a constant hazard).
`continue` counts every collision over full episodes, but the states after a contact are
outside the training distribution, so its numbers are pessimistic. Report falls next to
collisions: a policy can trade one for the other.
