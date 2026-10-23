# Robot policy architecture: from scratch, SONIC residual, SONIC planner replacement, or velocity-conditioned SONIC?

Opened 2026-10-12. **Whose call:** the user, who asked which of four options to
prioritize. They may run all four, since (a) and (b) are baselines and (d) should
improve on (c). The agent's recommendation is below.

## Options (user, 2026-10-12)

- **(a)** A LiDAR-conditioned policy trained from scratch: proprioception, a velocity
  command and LiDAR in, joint targets out.
- **(b)** A LiDAR-conditioned residual on SONIC's kinematic planner and tracker that
  forces avoidance.
- **(c)** A LiDAR-conditioned high-level planner that replaces SONIC's kinematic planner
  and produces SONIC's targets (reference motion or tokens).
- **(d)** A hybrid: fine-tune SONIC to be both a motion tracker and a velocity-conditioned
  policy, so that a high-level policy can steer it directly.

## User decisions (2026-10-12, after the recommendation)

- **RL for all four options.** IL is used only for distillation: train a privileged
  teacher first, then distill it into the LiDAR policy by IL alone or by RL plus IL.
  The teacher is an end-to-end policy for (a), or a high-level planner for the SONIC
  options. This resolved the [learning paradigm](../archive/2026-10-12-decision-learning-paradigm.md) toward RL (ledger
  D18); the TANGO-style offline imitation branch is not pursued.
- **Humans will perceive the robot**, with a stubbornness toward it that is set
  separately from their stubbornness toward other humans. Astra iterates on the robot
  stubbornness after the user has reviewed the 2026-10-12 round 4 results.
- **Merge, then review.** Merge the training snapshot after Astra's deadline, then
  review the first run's curves.

The ordering below is unchanged; it applies to the teachers first, then to each
student.

## Facts (checked 2026-10-12)

The planner's interface was checked again on 2026-10-18 (batch 1, horizon 0.8–2.1 s,
waypoint inputs): [SONIC note](../knowledge/sonic.md).


**SONIC** (NVlabs/GR00T-WholeBodyControl; arXiv:2511.07820):
- **Tracker:** a Unitree G1 29-DoF tracker with an FSQ universal token, at 50 Hz.
  Checkpoints are released as ONNX for deployment and as PyTorch `.pt` (`sonic_release`,
  `low_latency`, `sonic_v1_1`) for evaluation and continued training.
- **Training code:** released 2026-04-10. Training runs in **Isaac Lab**, which needs a
  separate install. The repository's fine-tune example says "64+ GPUs recommended"; the
  user expects 4 GPUs × 4,096 envs to suffice.
- **Kinematic planner:** released as ONNX only. It is a masked-token generative model
  that replans every 0.1 s from velocity (0–6 m/s), direction and style commands. Each
  plan is a segment 0.8–2.4 s long.
- **Interfaces in the paper:** no perception or obstacle avoidance. A VLA steers SONIC
  by predicting the 64-d token, so a learned policy over tokens has a precedent.
- **Our stack:** everything is mjlab (MuJoCo Warp): the crowd, the rigid-mesh LiDAR ray
  target and the G1 velocity task. The SONIC source and models are now fetched/pinned (2026-10-18); tracker integration is unimplemented.

## Analysis

| | (a) from scratch | (b) residual | (c) planner replacement | (d) velocity SONIC |
|---|---|---|---|---|
| Runs in our stack today | yes | after the SONIC gate | after the SONIC gate | after the gate, plus a SONIC fine-tune in Isaac Lab |
| Learning problem | locomotion and avoidance together | small corrections around an obstacle-blind reference | navigation in a structured action space | navigation in a low-dimensional command space |
| Main risk | reward shaping; gait quality | a joint-space residual fights the tracker, and avoidance needs a different path, not a perturbation | an RL action space of full-body references or tokens needs a prior, or SONIC tracks implausible motion | the fine-tune; Isaac Lab and our mjlab G1 must agree |
| Role | baseline; the shared env | baseline | first SONIC arm | improvement on (c) |

The **SONIC gate** is shared by (b), (c) and (d). It asks whether the released tracker
and planner run in mjlab on our G1 at 4,096 environments and walk stably under
scripted velocity commands. That needs:
- matching joint order, PD gains, default pose and observation config;
- the planner's batched ONNX inference, or a torch port, at 10 Hz;
- a pin in `third_party.toml`, with torch and warp unchanged.

All four options need the same environment: G1, plaza, graph crowd, LiDAR, the goal
command, clearance-based contact and the metrics. Only the action head differs.

## Recommendation (agent, 2026-10-12)

1. **(a) first, now.** It needs nothing outside our stack. It validates the
   environment, rewards and metrics that every arm reuses, and gives the baseline
   numbers. The archived 2026-09-28 plan also recorded a preference for
   training from scratch. Started 2026-10-12:
   - `crowd train traversal`: the LiDAR policy trained directly, as a baseline;
   - `crowd train traversal --teacher`: the privileged teacher, after the user's
     teacher-first decision. The `teacher` observation group holds the 6 nearest
     humans' states and the plaza pose in place of the LiDAR. It is distilled later
     with RSL-RL's `DistillationRunner`, which reads the same env's `actor` group.
2. **The SONIC gate, in parallel**, as a short feasibility spike with no learning.
3. **(c) as the first SONIC arm**, in two steps:
   - first, the LiDAR policy emits the planner's commands (velocity, heading);
   - then, references or tokens, with the planner's or SONIC's latent as the action
     prior.

   **(b)** is a cheap ablation on the same stack.
4. **(d)** once (c) works and shows the planner's limits (latency, steerability). The
   fine-tune does not need the crowd, so it can run in Isaac Lab on SONIC's own
   pipeline. The same LiDAR policy then compares (c) against (d).

All four options are RL in mjlab; the user confirmed RL with teacher–student
distillation (ledger D18, [learning paradigm](../archive/2026-10-12-decision-learning-paradigm.md)).

## What would change it

- **The SONIC gate fails** (unstable in mjlab, or too slow at 4,096 envs): (a) becomes
  the main arm, and SONIC returns as distillation or a motion prior.
- **(a) cannot learn locomotion and avoidance together** (falls, or ignores the crowd):
  the hierarchical arms move up.
- **Robot-aware humans** (D19, implemented 2026-10-12). In the first runs (a0, t0)
  the humans ignored the robot and caused most of the collisions. Runs from t1 on use
  a robot-aware crowd with provisional oblivious and stubborn shares, so a0 and t0 are
  worst-case baselines ([robot-aware teachers](../experiments/2026-10-12-robot-aware-teachers.md)).
- **For (a) the teacher stage brings no advantage so far** (2026-10-12). Without the
  LiDAR the teacher (t5) collided twice as often as PPO on LiDAR (a2). With it (t7) the
  teacher is level with a2, in training and in replays; a2's seed replicate agrees within
  0.01. Privileged states add no collision benefit under the current rewards. The
  pipeline itself works on collisions: the distilled student s7 replays level with t7
  and a2 (0.16, 0.21, 0.17 collisions per episode end; noise ±0.05), though both
  students fall more than their teachers (2–4% of replay episodes against about 0%).
  The "easier to train
  first" reason does not hold for (a) here; it may still hold for a SONIC planner
  teacher. Same experiment note.
- **The reward, not the architecture, sets the collision plateau** (2026-10-14). A
  terminal collision penalty of −250 lowers replay collisions only from 0.17 to 0.15 per
  episode end; at −1,000 the policy falls instead, since nothing penalizes falling
  ([collision penalty](../experiments/2026-10-13-collision-penalty.md)). Reward work
  (fall penalty, dodging) comes before comparing architectures on collisions.
- **p1000 partly meets the "(a) cannot learn locomotion and avoidance together"
  trigger** (2026-10-18). Its training curves match a2's through iteration 200. From
  iteration 400 to 800, when a2 starts walking and meeting humans, it stays in the
  early high-fall regime. It ends walking at 0.51 m/s but never reaches stable
  locomotion ([traversal evaluation](traversal-evaluation.md#the-fall-fix)). The fall-penalty arm
  would show whether this is the reward rather than the architecture.
- **The SONIC planning arms train, but do not beat a2pf yet** (2026-10-19; one seed each,
  [planning arms](../experiments/2026-10-19-sonic-planning-arms.md#results)).
  In 120 iterations (a2's simulated time, no walk-first stage) a residual over frozen
  SONIC cuts plain goal following's collisions by 79–84% in the aware crowd, to 0.25–0.27
  per robot-minute (a2 0.42, a2pf 0.18) with almost no falls (0.000–0.003); per encounter
  the gap to a2pf is 3–18%. Near people it mostly slows down rather than steers clear, so in
  the blind crowd it only equals a2 (3.7; P0 4.3–4.6, a2pf 3.2). 1 Hz and 2 Hz collide equally;
  teacher, student and PPO on LiDAR are level, and the students keep their teachers' fall
  rates (s7 did not).
- **Blind-crowd follow-ups complete (2026-10-21;
  [experiment](../experiments/2026-10-21-blind-crowd-arms.md)):** curriculum improves blind
  collisions 3.68 → 3.39 and goals 2.42 → 2.69, but aware collisions regress 0.273 → 0.322
  while goals rise 3.35 → 3.68. Fixed-blind training improves blind collisions further
  (3.13) but loses more aware safety (0.412). The penalty-recipe stage 2 loses to P1 +30
  on collisions (0.476 / 3.77 aware / blind versus 0.264 / 3.68). One training seed each.
  These are behavioral tradeoffs, not uniform improvement across easy/hard conditions.
  User requires both collision and goal-completion rates within each behavior condition;
  independent robot-stubbornness/assertiveness curricula and retained easy-condition
  exposure are candidate follow-ups, not yet measured or launched.

## User direction and planner assessment (2026-10-18)

**User, verbatim:** "I would also strongly suggest pursuing the high level planning
idea, where we design a kinematic planner to supply references to NVIDIA SONIC which is
trained to maximize goal following behavior and minimize collisions."

**Agent's reading:** option (c), recorded here as:
- a kinematic planner supplies references to a frozen SONIC;
- the planner is trained to maximize goal following and minimize collisions; RL for
  every arm (D18). **Confirmed (user, 2026-10-18): "If we are to use a planner, then
  we do want to keep SONIC frozen."** No SONIC fine-tune in Isaac Lab for (c);
- the same metrics and baselines as for (a) ([traversal evaluation](traversal-evaluation.md)).

At the time of this assessment no runs were requested. The user subsequently authorized
release fetching and the batching/reuse experiment below; tracker integration and
training remain outside that authorization.

**What the release allows.** The facts are in the [SONIC note](../knowledge/sonic.md).
The planner takes:
- movement direction, facing, speed and a style;
- optionally, four waypoints with headings.

It returns 0.8–2.1 s of full-body G1 qpos at 30 Hz, which the tracker follows at 50 Hz.
Those inputs are a ready action space for the learned planner, emitted at 10 Hz:
- first step: direction, facing and speed;
- second step: waypoints;
- references or tokens only later.

The RL problem then resembles navigation over velocity commands, while SONIC owns
balance.

**The blocker: the planner ONNX has a fixed batch of 1.** At 4,096 envs and 10 Hz that
is about 41,000 calls a second. There are two ways to provide references at scale:

| | Online planner with a dynamic batch | Offline G1 library stitched by the motion graph |
|---|---|---|
| How | ONNX graph surgery or a torch port of `planner_sonic.onnx`, batched on the GPU | Run the planner offline (batch 1 is fine) over random command sequences, record G1 qpos and build a kNN graph; online, the planner policy picks graph commands. This is the project's own pattern for the humans (ledger D09–D11) |
| Fidelity | The input accepts recent qpos; released deployment normally uses its current planned motion (corrected 2026-10-18) | References come from the graph's state. Each edge is re-rooted at the robot, and the tracker closes the gap |
| Unknowns | Whether the masked-token generation and the internal canonicalization batch correctly | The graph and runtime handle Core joint positions today and need G1 qpos frames; it is unknown whether SONIC tracks seams |
| Throughput | Unmeasured | The graph runtime steps 65,536 streams in 47 ms (A40, isolated) |

**Designed planner.** A non-learned planner over the same command space would serve as
a baseline and possibly as the privileged teacher that D18 distills into the LiDAR
planner. It would be sampling-based or use velocity obstacles, reading the humans'
states with constant-velocity prediction. Using it as the teacher departs from "RL for
every arm", so that is the user's call.

**Pre-test without SONIC.** Put the same high-level planner over velocity commands on
top of b0, the velocity tracker that ignores the crowd
([traversal evaluation](traversal-evaluation.md#non-avoidant-baseline-b0)).
- It tests whether the decomposition lowers collisions before any of the gate's
  engineering.
- It does not test SONIC itself: b0's gait and tracking differ.

**Falls at the high level.** SONIC owns balance, but a planner can still learn to
command references it cannot track. Fall parity (`fall_end`) applies to every arm.

**Gate steps** (no training; steps 1–2 authorized and completed by the experiment below; steps 3–4 await the go-ahead):
1. Fetch the GEAR-SONIC weights into `$CROWD_CHECKPOINTS` and check license access.
   Pin GR00T-WholeBodyControl in `third_party.toml`, with torch and warp unchanged.
2. Inspect `planner_sonic.onnx` for fixed shapes. This chooses between the online and
   offline columns above.
3. Run the tracker in mjlab on one env, open loop on a planner reference. Use SONIC's
   G1 MJCF, or ours matched to its `config.yaml`: joint order, PD gains, armature,
   default pose and action scale.
4. Time the tracker at 4,096 envs.

## Released planner batching result (2026-10-18)

The user authorized fetch, batching probes and reuse costing, with no training or
production integration. Source and all model variants are under `third_party/` and
`checkpoints/GEAR-SONIC`, pinned in `third_party.toml`; root lock/torch/warp unchanged.
[Full experiment and commands](../experiments/2026-10-18-sonic-planner-batching.md).

- Dynamic batch is **reachable**: an operator hand port preserves the original
  batch-1 logical dimensions under `torch.vmap`; full B4096 runs on one A40 with
  qpos maximum error 1.71e-5 and exact counts on four synthetic cases.
- Capacity is **239 plans/s**, 17.12 s per population refresh, about 171× short of
  40,960 plans/s for 10 Hz (before the 2026-10-19 MatMul fold, which reached 1,829 plans/s,
  bit-equal; see the [planning arms](../experiments/2026-10-19-sonic-planning-arms.md#cost)). Boundary-only ONNX edits fail on internal fixed shapes;
  IO binding gives only 73 plans/s. Stock onnx2torch lacks four operators. FP16/TF32
  probes change output elements by up to 0.687/0.196; neither is adopted.
- The exported model has **no token-generation Loop/Scan/If**: 3 shared + 3 root
  transformer layers followed by 16 pose transformer layers, padded masks and
  argmax token selection. Its fixed work, rather than a Python token loop, is costly.
- **Reuse recommendation (agent, not an adopted architecture):** velocity/facing
  commands are the smallest RL action space. At scale, first test an offline G1
  reference library with the existing graph selection pattern and frozen tracker.
  This needs G1 qpos/quaternion payloads, FK/contact descriptors, coordinate conversion
  and seam validation. If online history conditioning is essential, consider
  distillation/compression of this released prior; training would need a new request.
- Released deployment samples planner context from the **planned animation**, not
  measured robot history after initialization. Using actual qpos remains a possible
  integration choice, with stability untested. Waypoints control one token's four
  frames; they do not lower neural inference cost.

This completes fetch/batching gate steps 1–2, not the shared SONIC-in-mjlab gate.
The four architecture choices and D18's RL teacher direction remain open/unchanged.

## Training cost of the frozen planner online (2026-10-18)

**User, 2026-10-18:** "maybe the batched kinematic planner can't run in real time, but if
we just use it for train, why is it a big issue? ... As long as the data phase (simulator
and model forwards) is comparable to the step phase (PPO backprop) then shouldn't it be
fine?" Real time is indeed not the constraint; the planner's share of the PPO iteration
is. The batching result above framed 10 Hz replanning as a hard limit, which overstated
it for training.

- **Measured PPO iteration** (pf1000 at iteration 300, A40 on tachikoma, 4,096 envs × 24
  steps, the mjlab G1 velocity PPO defaults): 3.46 s = 3.27 s collection + 0.195 s
  learning (28,400 env steps/s). The update is 6% of the iteration, so the planner's
  budget is set by the 3.27 s of simulation, not by backprop.
- **Added planner time** per iteration: 4,096 envs × 0.48 s simulated × replan rate,
  at the measured 239 plans/s (flat from microbatch 16 up, so replans staggered across
  steps cost the same per plan). Synchronous, on the training GPU, tracker excluded
  (historical: this predates the fold, which made the planner 7.3× faster):

  | Replans per env | Plans per iteration | Planner, s | Iteration, s | Slowdown | 3,500 iterations |
  |---|---:|---:|---:|---:|---:|
  | 10 Hz | 19,661 | 82.2 | 85.7 | 24.8× | 83 h |
  | 2 Hz | 3,932 | 16.4 | 19.9 | 5.7× | 19 h |
  | 1 Hz (release default for walking; deploy timer 1.0 s) | 1,966 | 8.2 | 11.7 | 3.4× | 11 h |

- **The port runs far below the hardware** (inference from the probe's arithmetic):
  6.6 GFLOP per plan × 239 plans/s ≈ 1.6 TFLOP/s, about 4% of the A40's 37 TFLOP/s FP32
  datasheet peak. A batched re-export (real batch dimensions, fused attention, validated
  reduced precision) may cut the 1–2 Hz rows substantially; unmeasured. At 10 Hz the
  dense work alone (270 TFLOP/s) still exceeds the A40.
- **Revised assessment (agent):** a frozen planner online is a viable training route at
  1–2 Hz event-triggered replans: about 3–6× today's training time with the current
  port, plus the unmeasured tracker. It costs about 2–4 days of wrapper work after the
  shared tracker integration, against 1–2 weeks for an offline G1 graph. The offline
  graph stays cheaper per iteration but needs seam validation. Open trade-off: a 1–2 Hz
  decision rate against reactivity (a2's median reaction latency is 0.62 s).

## Modifying the planner for crowd avoidance (2026-10-18)

**User, verbatim:** "Let's build the first set of planning-based crowd avoidance arms at
1 Hz and 2Hz. Before we do that, we should understand how the planner is implemented and
how we can best modify it (predict delta on top of planner? LoRA on planner? full
finetune? etc.)"

**How it is built** ([SONIC note](../knowledge/sonic.md#how-the-planner-is-built-read-2026-10-18)):
a hand-written spring turns the command into a target root position and heading 1 s
ahead, a style clip supplies the target's body pose, and MotionBricks in-betweens from
the 4 context frames to that target. Its root model chooses the duration (0.8–1.47 s
for walking) and regresses the root path; its pose model fills in discrete pose tokens
along that path. The released root weights are the planner's (146 of 150 tensors
identical), so a trainable PyTorch planner exists. **"1 Hz" is the release's own
default for walking** (deploy timer 1.0 s, plus a replan on any command change).

**Where avoidance is decided:** in the root path, which is set by the target and the
duration. The pose tokens shape the body along the path, not where it goes.

| Option | Assessment |
|---|---|
| **Residual on the target** (Δ target position in the robot's frame, Δ heading, optionally a stop and the duration mask), added to the goal-directed target at each replan | **First arms.** Planner and tracker stay frozen, so the motion prior is intact. A zero residual is plain goal following, so learning starts from a walking, goal-reaching policy and RL learns only the deviation. Small action space at 1–2 Hz. Limit: within one horizon the path is the model's own smooth in-between; bending it needs the next replan |
| LoRA on the root model, with a zero-initialized perception input (for example a LiDAR token added to its shared transformer) | **Second stage, if the residual arms plateau.** It can bend paths within the horizon and react through the duration. Needs a stochastic head for PPO (Gaussian on the root path or target), backprop through 34 M parameters at every replan sample, and care not to degrade the prior |
| LoRA or fine-tune of the pose model; full fine-tune | **Not for avoidance.** The pose model does not change the path; its 16 × 1,024 transformer is most of the planner's compute, and RL through discrete tokens risks the motion quality the tracker relies on |
| Residual on the planner's output qpos (50 Hz) | **Not first.** Off-manifold references for a frozen tracker. A root-only shift between replans may later serve as a fast dodge channel |

**Proposed first arms (agent; the user decides):**
- **P0:** zero residual: planner plus frozen SONIC following the goal command, evaluated
  only. It is the planner-level counterpart of b0.
- **P1 / P2:** the residual policy deciding at 1 Hz / 2 Hz, one replan per decision. At
  1 Hz walking is masked to at least 8 tokens (≥ 1.07 s) so a plan never runs out before
  the next decision; a replan when the buffer runs low stays as a guard and is counted.
- Observations: the LiDAR as in a2, the goal in the robot frame, the robot's velocity
  and a summary of the current plan (planned root positions ahead, time left).
  Rewards: goal progress and arrival, `collision_end` and `fall_end` at parity, the
  proximity and contact terms, and an action-change penalty; no locomotion shaping,
  since the tracker owns balance.
- **Direct PPO on the LiDAR policy,** departing from D18's teacher-then-distill: under
  the current rewards t7 (privileged + LiDAR) equals a2 (direct PPO). The user may
  prefer the teacher route.
- Evaluation with `crowd eval traversal`, against b0, a2 and the fall arms.

**Approved and built (user, 2026-10-18: "I approve of the build. Why don't we do all
options?"):** the gate passed and all six arms (P1/P2 by direct PPO, by teacher and by
student) are trained and evaluated; see the [planning arms experiment](../experiments/2026-10-19-sonic-planning-arms.md).

**Build order (estimates):**
1. Native batched PyTorch planner from the released checkpoints (fetch and compare the
   pose and VQ-VAE weights; batch the spring, clip lookup and canonicalization); measure
   plans/s at 4,096, which may beat the vmap port's 239 (2–3 days). Not built: the MatMul
  fold in the port reached 1,829 plans/s, so this step is lower priority.
2. SONIC's tracker in mjlab, open loop on planner references, timed at 4,096 envs (gate
   steps 3–4; 3–7 days).
3. A planner command term in `crowdmotion.sim.mjlab`: per-env plan buffers, 30→50 Hz,
   8-frame cross-fade, context from the planned motion, staggered replans (2–4 days).
4. P0 evaluation, then P1 and P2 training.

## Links

- Environment: `src/crowdmotion/sim/mjlab/traversal.py`; training command:
  `crowd train traversal`.
- [SONIC release facts](../knowledge/sonic.md) (2026-10-18);
  [traversal evaluation](traversal-evaluation.md).
- SONIC: <https://github.com/NVlabs/GR00T-WholeBodyControl>,
  <https://huggingface.co/nvidia/GEAR-SONIC>, <https://arxiv.org/abs/2511.07820>.
- [Learning paradigm](../archive/2026-10-12-decision-learning-paradigm.md);
  [TANGO](../knowledge/tango.md) (SONIC as the tracker of a VLA).
