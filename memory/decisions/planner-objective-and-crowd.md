# Planner objective and a non-interacting crowd

Opened 2026-10-21. Status: **implementation and dense ablations authorized** by the user
on 2026-10-21, including non-reactive graph humans in parallel. The best objective and
generalization remain empirical questions; defaults have not been promoted. **Whose
call:** user, informed by the [active experiment](../experiments/2026-10-21-goal-objective-ablations.md).

## Question and user proposal

After the [curriculum results](../experiments/2026-10-21-blind-crowd-arms.md), the user
asks whether the planner should receive only sparse rewards for reaching a sufficiently
far/difficult goal quickly, with collisions ending episodes. They also propose humans
that randomly roam, ignore the robot and one another, and can overlap. Density and
directional coordination (for example 90% sharing one direction) would control difficulty.
These are candidate changes, not replacements already selected for the existing crowd.

## What the current implementation implies

- `traversal.PLANNER_REWARDS` contains linear/angular command tracking, goal heading,
  proximity, contact, wall proximity and terminal collision/fall terms; `PlanningVecEnv`
  adds action magnitude/change costs. The velocity command points straight to the goal
  at each step. Detours and looking away can lose reward even when useful for navigation.
  There is no explicit arrival reward. This creates an incentive mismatch, but the
  existing arms do not isolate its contribution to collision performance.
- The planner discount is **0.75 per simulated second**: a reward 10 s away is weighted
  0.0563, and 20 s away 0.00317. A terminal-goal objective needs a credit horizon chosen
  for the trip duration; merely deleting dense terms is not a fair test of sparse rewards.
- Robot goals currently resample every 10–15 s, on arrival, or at reset; 10% of commands
  are standing. A one-trip objective needs persistent goals and a defined success/failure
  deadline instead. Observations should expose goal-relative displacement and, for a
  time-dependent reward/deadline, elapsed or remaining time; do not rely only on a
  distance-saturated velocity command.
- `--robot-blind` removes human reaction to the robot but leaves human–human reactions.
  `--no-avoidance` is the robot b0 baseline, not a switch for a non-interacting crowd.
  Setting `switch_rate_hz=0` does not remove stopping, separation or graph clearance.
- Rigid human geoms already have `contype=0, conaffinity=0`. Their physical collisions
  are not the human–human avoidance mechanism. A non-interacting command source must
  remove neighbor steering, stopping, separation and neighbor-based graph clearance
  while retaining world-boundary handling, motion quality, LiDAR and robot–human
  clearance/termination. It belongs in the shared CrowdController path, not just mjlab.

## Recommended reward experiment

Use the frozen SONIC planner/tracker. A terminal-only candidate is:

- one persistent goal per episode, sufficiently separated from the spawn; route
  geometry should expose the robot to traffic rather than reward easy perimeter trips;
- a positive success reward, optionally with a bounded speed bonus based on time divided
  by a nominal trip time, so different goal distances can be interpreted consistently;
- zero intermediate rewards for the pure sparse arm, including proximity, command/facing
  tracking and the wrapper's residual magnitude/change costs; collision and fall are terminal
  failures and cannot receive the success reward, including simultaneous arrival;
- a finite deadline and correctly handled terminal values/time limits. Pure episodic
  success by deadline requires deadline expiry to be a task failure, not an accidental
  continuing-task bootstrap. Keep PPO rollout truncation distinct from task failure.

Earlier success can be preferred by temporal discounting or by the terminal speed bonus.
Do not unknowingly combine a very short discount with distant rewards. Use early
low-density episodes or the same existing planner checkpoint in all arms to obtain
successful examples; frozen SONIC already makes goal reaching feasible without crowds.

Collision termination already exists in the trained planning arms. Its new value would
come from losing the opportunity to earn success, not from a newly added collision rule.
Termination with zero reward is a reasonable sparse baseline, but **does not guarantee
low collision risk**: expected reward still trades success probability against speed.
If an explicit failure cost is tested, treat falls comparably and name it as a separate
objective choice. Do not add negative running time cost while making early failure free:
that can make failure preferable to accumulating costs on an unsuccessful trip.

If pure sparse PPO lacks successes, compare it with a goal-potential shaping variant,
`F(s,s') = gamma Phi(s') - Phi(s)`, e.g. a distance-based potential with the same discount
and correct terminal boundary potentials. It guides progress while preserving the base
MDP's optimal policy under the theorem's conditions; it does not guarantee PPO training
success. Do not clip away negative progress or omit the terminal correction. This is an
optional comparator, not a reason to retain direct-to-goal velocity/facing rewards.

References checked 2026-10-21:
- [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf): potential-based shaping and policy invariance.
- [Andrychowicz et al., Hindsight Experience Replay](https://arxiv.org/abs/1707.01495): sparse binary rewards have an exploration problem; their solution is off-policy. This is context, not evidence that HER can be dropped into our PPO runner or that sparse rewards improve this task.

## Recommended crowd experiment

Non-interacting humans form a controlled moving-obstacle task: they cannot yield to
the robot, and their trajectories do not change when the robot acts. This isolates the
robot's responsibility for avoidance and removes human–human standoffs, but transfer
to the original reactive crowd must still be measured.

Use temporally persistent random routes/velocities. Directional coordination can mean
a fraction p assigned a common heading with angular spread, with the rest independently
headed; shared intent does not require interactions or flocking. Boundary handling must
maintain population and avoid abrupt in-range teleports or resampled velocity jitter.

Density and coherence are **two factors**, even if a scalar curriculum maps into both.
Increasing density at fixed area/flow is a useful first axis. Coherence is not inherently
monotonic in difficulty: aligned flow can be easier than isotropic traffic, whereas
crossing or opposing flow can be harder. Sample/evaluate co-flow, crossing and opposing
directions relative to the robot's goal, not just a fixed world heading or p=0.9.
Retain easy traffic while progressing to denser cases.

Overlapping humans change occupied area and LiDAR occlusion, so human count per area
alone need not measure actual exposure. Report encounters, close clearances and TTC
alongside nominal density. At dense levels, distinguish short-notice unavoidable arrivals
from avoidable failures; do not spawn a human in contact with the robot. Preserve the
same robot collision definition across comparisons.

## What would decide it

Start with a matched 2 × 2 comparison, keeping SONIC, policy inputs, action cadence,
start checkpoint, goal/deadline protocol, discount and training exposure the same:

| Crowd, all robot-blind | Current tracking reward | Sparse successful-goal/time reward |
|---|---|---|
| Existing human–human-reactive crowd | matched control | reward effect |
| Non-interacting roaming crowd | crowd effect | combined proposal |

Use common low/moderate density settings or the same prescribed density schedule first;
independently adapting every arm's curriculum changes exposure and obscures attribution.
The current saved P1/fixed-blind runs provide context, not exact controls after changing
goal persistence, observations and discount. A potential-shaped third reward is optional
if sparse learning stalls. More than one seed is needed for an architectural conclusion.

Evaluate every policy on the same fixed grid of densities, coherence and relative-flow
angles, plus the original easy/robot-blind reactive crowds. Within each condition report
collision rate and collision-free goal completion; report completion time with failures
and timeouts accounted for, not only speed conditional on a successful trip. Retain
goals/time, falls and encounter exposure for comparability. No numerical pass thresholds
have been selected; user chooses acceptable safety/progress tradeoffs.

Agent recommendation: prioritize this objective/crowd ablation over adding more
robot-specific reaction parameters, since it tests reward alignment and the robot's own
avoidance directly. Density is the initial scalar curriculum; coherence is initially an
evaluation/domain factor. No code, existing goldens or crowd defaults changed.
