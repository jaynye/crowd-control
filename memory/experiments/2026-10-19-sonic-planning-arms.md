# 2026-10-19: planning arms over frozen SONIC (P0, P1, P2)

Status: done (2026-10-19). User (2026-10-18): "Let's build the first set of planning-based
crowd avoidance arms at 1 Hz and 2Hz", then "I approve of the build. Why don't we do all
options? (P1, P2 with both direct PPO+distillation using a frozen SONIC as the base policy)".
Design and options: [robot policy architecture](../decisions/robot-policy-architecture.md#modifying-the-planner-for-crowd-avoidance-2026-10-18).

**Result.** A residual on the goal command over frozen SONIC learns avoidance in 120
iterations (a2's simulated time): 0.25–0.27 collisions per robot-minute in the aware crowd,
against 1.31–1.64 for plain goal following (P0), 0.42 for a2 and 0.18 for a2pf, with almost
no falls. All six arms (PPO, teacher and student at 1 Hz and 2 Hz) land within 0.25–0.27;
1 Hz and 2 Hz collide equally, and teacher-then-distill gives nothing over direct PPO. Like
every policy so far, they rely on the humans. Near people they mostly slow down rather than
steer clear: that works when the humans yield and barely helps when they do not (blind
crowd 3.6–3.7 against P0's 4.3–4.6; a2 3.65, a2pf 3.17). One training seed per arm.

## Setup

- **Base policy:** NVIDIA SONIC's released G1 tracker (frozen; `backends/sonic.SonicTracker`,
  a torch port of `model_encoder.onnx` / `model_decoder.onnx` equal to ONNX Runtime) following
  the released kinematic planner (`planner_sonic.onnx` through `backends/onnx_port.OnnxPort`,
  FP32), with the deploy's plan resampling, blending and context (`backends/sonic.PlanBuffer`,
  `sim/mjlab/sonic_control.SonicController`). mjlab's G1 matches SONIC's joint order, gains,
  default pose and action scale.
- **Arms** (`sim/mjlab/planning.PlanningVecEnv`): the policy decides at 1 Hz (P1) or 2 Hz
  (P2) a residual on the goal command (planar velocity offset up to 1 m/s in the heading
  frame, facing offset up to 90°); every decision replans; a zero action is P0, plain goal
  following. Observations: the traversal env's groups plus the plan summary and the previous
  action. Rewards: the task terms only (`traversal.PLANNER_REWARDS`), `collision_end` and
  `fall_end` at -1,000, small action and action-change penalties. PPO per decision: 16 s of
  rollout per env and iteration, gamma 0.75 and lambda 0.9 per second, init std 0.5.
  Teachers: privileged human states plus LiDAR (as t7), 120 iterations; students: each
  teacher distilled into the LiDAR policy, 60 iterations.
- Env as a2: 4,096 envs, 8 humans, robot-aware crowd, graph-turns, `encounter_decisive_robot`,
  seed 42; A40 on rl2-lab. 120 iterations (16 s each) simulate a2's 1,920 s per env.

## Gate (B1; no RL)

mjlab's flat G1 task, no crowd, A40 (runs `runs/20261019T003836Z_sonic_tracking_gate{,_2}`,
`runs/20261019T003601Z_sonic_tracking_gate`):

| Run | Falls per robot-minute | Joint error legs / upper | Orientation error | Speed robot / reference |
|---|---:|---:|---:|---:|
| Open loop, random commands, 1,024 × 20 s | 0.003 | 0.086 / 0.076 rad | 6.0° | 0.59 / 0.53 m/s |
| P0 goal following, 1,024 × 60 s | 0.004 | 0.088 / 0.078 rad | 5.4° | 0.77 / 0.69 m/s |

Tracker 8.0 ms per control step at 4,096 envs (FP32). P0 reached 8.0 goals per robot-minute.

## Cost

- **Before the fold** (check runs `runs/20261019T011558Z_train_traversal_planner_time_1hz`,
  `runs/20261019T011601Z_train_traversal_planner_time_2hz`): 477 s per iteration at 1 Hz, 756 s
  at 2 Hz; the planner was about 22 s of each decision (75–85% of the time).
- **Planner fold (2026-10-19 02:00 UTC):** a profile (`scripts/smoke/sonic_planner_profile.py`,
  `runs/20261019T015259Z_sonic_planner_profile`) put 79% of the planner's GPU time in the pose
  transformer's attention input projections: ONNX MatMuls of `[16, 1, 1024] x [1024, 3072]`,
  which torch broadcasts into a batched matmul and vmap into one matrix-vector product per
  token. `OnnxPort` now folds a MatMul with a 2-D right operand into one mm: equal outputs (max
  difference 0.0 on 64 varied requests) and 1,829 against 249 plans/s at batch 4,096 (A40;
  `scripts/smoke/sonic_planner_fold_check.py`). P1 restarted in place on it after one
  iteration; every other run started on it.
- **After the fold:** about 210 s per iteration (1 Hz) and 235 s (2 Hz) for PPO and the
  teachers, about 266 s for the students. The six runs took 40 A40-hours (7.2, 7.9, 7.8, 8.6,
  4.1 and 4.4 h), every one requeued in place at least once from the checkpoint written on
  TERM. A native batched MotionBricks planner (build step B2) is lower priority now; the
  planner's share of an iteration after the fold was not measured.

## Results

`crowd eval traversal`, 4,096 envs × 60 s, paired goals, seed 0, A40; mean of the last three
checkpoints (110, 115, 119; students 53/55/59 and 50/55/59), bracket = spread across them (not
seed variance; one training seed per arm). End mode (a collision ends the episode). Robot-led
= collisions per robot-minute × (1 − human-led share), where a collision is human-led when
the human's closing speed exceeds the robot's at contact. A robot that slows near people
therefore shifts collisions to human-led by construction, so robot-led depends on the
attribution and the totals do not. Baselines from the
[fall-fix note](2026-10-18-fall-fix-and-b0.md).

| Policy | Aware: collisions per robot-minute | Robot-led | Per encounter | Encounters per robot-minute | Falls | Goals per moving minute | Blind: collisions | Blind: robot-led |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 (no avoidance) | 0.95 | 0.77 | 0.030 | 31 | 0.000 | 4.27 | 4.23 | 1.15 |
| a2 | 0.42 | 0.25 | 0.013 | 32 | 0.000 | 4.06 | 3.65 | 0.49 |
| t7 / s7 (a2's teacher and student) | 0.42 / 0.49 | 0.24 / 0.32 | — | — | 0.001 / 0.024 | 4.11 / 3.93 | 3.59 / 3.51 | 0.46 / 0.57 |
| a2pf | **0.18** | **0.04** | **0.006** | 31 | 0.000 | 3.57 | **3.17** | **0.19** |
| P0 1 Hz (zero residual) | 1.31 | 1.11 | 0.030 | 43 | 0.004 | 4.39 | 4.32 | 1.48 |
| P0 2 Hz | 1.64 | 1.42 | 0.034 | 49 | 0.018 | 5.42 | 4.63 | 1.79 |
| P1, PPO 1 Hz | 0.27 [0.27–0.28] | 0.11 | 0.007 | 39 | 0.001 | 3.35 | 3.68 | 0.49 |
| P2, PPO 2 Hz | 0.27 [0.27–0.27] | 0.10 | 0.007 | 40 | 0.003 | 4.22 | 3.73 | 0.47 |
| Teacher 1 Hz | 0.25 [0.25–0.25] | 0.09 | 0.006 | 38 | 0.001 | 3.35 | 3.61 | 0.46 |
| Teacher 2 Hz | 0.25 [0.24–0.25] | 0.08 | 0.006 | 40 | 0.003 | 4.08 | 3.65 | 0.39 |
| Student 1 Hz | 0.25 [0.24–0.26] | 0.09 | 0.007 | 38 | 0.000 | 3.35 | 3.66 | 0.47 |
| Student 2 Hz | 0.27 [0.26–0.27] | 0.10 | 0.007 | 40 | 0.003 | 4.11 | 3.73 | 0.44 |

- **The residual removes 79–84% of P0's collisions** in the aware crowd (1.31 → 0.27 at 1 Hz,
  1.64 → 0.27 at 2 Hz), 36% fewer than a2, with the same simulated time per env as a2.
- **a2pf is still better.** Per robot-minute the planning arms collide 35–49% more than
  a2pf (0.25–0.27 against 0.18). Per encounter that shrinks to 3–18%, since they meet
  25–31% more people. By the attribution, robot-led collisions are 2–3× a2pf's (0.08–0.11
  against 0.04), at similar close-approach speeds (below). a2pf also had 1,000 more
  iterations from a walking start; the planning arms had 120 and no walk-first stage.
- **Near people they slow down rather than steer clear.** Within 0.25 m of a human, the
  robot's speed toward the human falls from P0's 0.55–0.64 m/s to 0.22–0.29 m/s (a2pf
  0.11–0.19, a2 0.28–0.33), and 12–18% of their collisions happen with the robot nearly
  still (below 0.15 m/s; P0 2–7%, a2pf 16%). Because human-led means the human closed
  faster, this alone moves collisions from robot-led to human-led: in the blind crowd the
  arms' robot-led collisions fall 67–78% from P0's (P2 1.79 → 0.47) while human-led ones
  rise (2.84 → 3.19 and 3.26). The totals carry the result. In the blind crowd they fall
  only 15–21% from P0's, to 3.6–3.7, equal to a2's 3.65 and above a2pf's 3.17. Splitting
  planning from walking does not break the pattern that most of the safety in the training
  crowd comes from the humans. The arms trained only in the robot-aware crowd.
- **2 Hz is not learning more.** P1 and P2 collide equally. P2's extra goals (4.22 against
  3.35 per moving minute) come from its base planner: P0 at 2 Hz reaches 5.42 against 4.39.
  Both arms give up 22–24% of their P0's goals. The 2 Hz arms' falls (0.003) are inherited
  from P0 at 2 Hz (0.018); training cut them 6×.
- **Teacher, student and PPO on LiDAR are level** (0.25–0.27, one seed each), as t7 and a2
  were. The teacher route costs a further 60 student iterations (about 4 A40-hours).
- **Distillation keeps the teacher's stability here.** The students fall as rarely as their
  teachers (0.000 and 0.003 against 0.001 and 0.003), unlike s7 (0.024 against t7's
  0.001). This fits the residual leaving balance to the frozen tracker; it was not tested
  separately.
- Goals in end mode are confounded by collision resets; continue mode gives the same
  aware-crowd order (P2 4.34 ≈ a2 4.29 > student 2 Hz 4.21 > a2pf 3.62; P1 3.46).
- The planning arms follow the path less tightly (aware crowd, free-walking cross-track
  0.15–0.21 m against a2's 0.08), as P0 does (0.19–0.29 m). The reaction metric reads 0.97
  for every planning arm: replanning changes the velocity by itself, so it does not measure
  avoidance.
- **Training curves** (iteration 0 → 119): collision ends per iteration fell 74–79% (P1
  84 → 22, P2 45 → 11, teachers 86 → 19 and 48 → 10), falls to about 0, goal-heading reward
  0.26 → 0.79–0.81. Training counts are not per-robot-minute rates.

## Runs

TERM 1,000 s (1 Hz) or 1,600 s (2 Hz) before the 4 h limit (`CROWD_SIGNAL_LEAD_S` to match),
checkpoints every 5 iterations, requeued in place.

| Arm | Training run (jobs) | Evaluations, aware / blind |
|---|---|---|
| P1 direct PPO, 1 Hz | `runs/20261019T014657Z_train_traversal_plan1hz` (4138688, 4138811) | `runs/20261019T104452Z_eval_traversal_plan1hz_2` / `runs/20261019T104452Z_eval_traversal_plan1hz` |
| P2 direct PPO, 2 Hz | `runs/20261019T015558Z_train_traversal_plan2hz` (4138689) | `runs/20261019T104447Z_eval_traversal_plan2hz_2` / `runs/20261019T104447Z_eval_traversal_plan2hz` |
| Teacher, 1 Hz | `runs/20261019T014602Z_train_traversal_teacher_plan1hz` (4138690) | `runs/20261019T104449Z_eval_traversal_teacher_plan1hz` / `runs/20261019T114147Z_eval_traversal_teacher_plan1hz` |
| Teacher, 2 Hz | `runs/20261019T014602Z_train_traversal_teacher_plan2hz` (4138691) | `runs/20261019T114201Z_eval_traversal_teacher_plan2hz` / `runs/20261019T115303Z_eval_traversal_teacher_plan2hz` |
| Student, 1 Hz | `runs/20261019T094935Z_train_traversal_student_plan1hz` (4138692) | `runs/20261019T135930Z_eval_traversal_student_plan1hz` / `runs/20261019T135935Z_eval_traversal_student_plan1hz` |
| Student, 2 Hz | `runs/20261019T104004Z_train_traversal_student_plan2hz` (4138693) | `runs/20261019T150624Z_eval_traversal_student_plan2hz` / `runs/20261019T150633Z_eval_traversal_student_plan2hz` |
| P0 1 Hz / 2 Hz (a2's env) | none | `runs/20261019T020033Z_eval_traversal_p0_1hz`, `runs/20261019T020813Z_eval_traversal_p0_1hz` / `runs/20261019T021416Z_eval_traversal_p0_2hz`, `runs/20261019T022437Z_eval_traversal_p0_2hz` (aware, blind for each) |

Evaluation jobs: 4138812–4138815 (P0; 4138694–4138697 cancelled for the faster planner),
4144308–4144315 (P1, P2, teachers), 4144318–4144321 (students, afterok on the training jobs).

## Review

Astra review of the build before launch: 3 findings fixed (planning checkpoints could not be
played; termination counts averaged instead of summed across a decision; evaluation dropped
a partial last decision); items 1–6 of the review (plan buffer, observation layouts, frames,
mid-option resets, evaluation paths, CLI plumbing) verified correct.
