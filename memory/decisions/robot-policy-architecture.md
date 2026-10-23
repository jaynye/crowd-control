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
  options. This resolves the [learning paradigm](learning-paradigm.md) toward RL; it is
  not the TANGO-style offline imitation branch.
- **Humans will perceive the robot**, with a stubbornness toward it that is set
  separately from their stubbornness toward other humans. Astra iterates on the robot
  stubbornness after the user has reviewed the 2026-10-12 round 4 results.
- **Merge, then review.** Merge the training snapshot after Astra's deadline, then
  review the first run's curves.

The ordering below is unchanged; it applies to the teachers first, then to each
student.

## Facts (checked 2026-10-12)

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
  target and the G1 velocity task. Nothing of SONIC is installed or pinned.

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

All four options are RL in mjlab, so this effectively settles the
[learning paradigm](learning-paradigm.md) toward RL, pending the user's confirmation.

## What would change it

- **The SONIC gate fails** (unstable in mjlab, or too slow at 4,096 envs): (a) becomes
  the main arm, and SONIC returns as distillation or a motion prior.
- **(a) cannot learn locomotion and avoidance together** (falls, or ignores the crowd):
  the hierarchical arms move up.
- **Robot-aware humans.** The humans do not yet perceive the robot, so every human is
  stubborn toward it. Results from (a) are worst-case until robot-aware encounter
  behavior exists, with a tunable stubborn share (the 2026-10-12 Astra round adds
  `stubborn_fraction` for human–human encounters).

## Links

- Environment: `src/crowdmotion/sim/mjlab/traversal.py`; training command:
  `crowd train traversal`.
- SONIC: <https://github.com/NVlabs/GR00T-WholeBodyControl>,
  <https://huggingface.co/nvidia/GEAR-SONIC>, <https://arxiv.org/abs/2511.07820>.
- [Learning paradigm](learning-paradigm.md);
  [TANGO](../knowledge/tango.md) (SONIC as the tracker of a VLA).
