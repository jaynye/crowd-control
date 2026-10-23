# 2026-10-18: SONIC planning arms (build) and the W&B canary

Status: in progress. User (2026-10-18): "I approve of the build. Why don't we do all
options? (P1, P2 with both direct PPO+distillation using a frozen SONIC as the base
policy). Concurrent with the build, we should launch a canary clearly labeled test and
with a debug name for wandb. You can use up to 1 Opus 5.5 xhigh subagent to help and 1
GPT-6 Astra xhigh subagent to review (or vica versa)."

Design: [robot policy architecture](../decisions/robot-policy-architecture.md#modifying-the-planner-for-crowd-avoidance-2026-10-18)
(residual on the planner's target; planner and SONIC frozen).

## Plan

- **A (Opus subagent, own journal `2026-10-18-wandb-canary.md`):** fix the 7 open W&B
  findings of ledger D20, Astra review of the fixes, then the canary: a lineage named
  `train_traversal_TEST_canary_debug` in the scratch project
  `rl2-group/crowd-control-traversal-debug`, tagged test/canary/debug, with a TERM →
  requeue → in-place resume and an attached evaluation, read back through the W&B API.
  `DEFAULT_MODE` stays `disabled`.
- **B1 (main): SONIC tracker in mjlab, the gate.** A `crowdmotion.backends` wrapper for
  the frozen tracker (PyTorch checkpoint, run from source like ARDY), the G1 model and
  PD/action configuration from SONIC's MuJoCo sim2sim path, open-loop tracking of
  planner references, tracker ms per step at 4,096 envs, P0 (zero delta) closed-loop
  falls per robot-minute. Report the numbers to the user before any RL training.
- **B2: batched MotionBricks planner** in `crowdmotion.backends` (fetch the pose and
  VQ-VAE checkpoints by LFS oid, compare with the ONNX, batch the spring, clip lookup and
  canonicalization), plans/s at 4,096.
- **B3: planning env:** a decision-rate wrapper over the 50 Hz traversal env (tracker
  stepped 25 or 50 times per decision; rewards summed; time-outs bootstrapped; a
  terminated env ends its transition at the terminating step), the delta action, plan
  summary observations, a zero-delta mode for `crowd eval traversal`. PPO rollout length
  in decisions; γ matched per second between 1 Hz and 2 Hz.
- **B4: arms:** P0 evaluation; P1 and P2 by direct PPO, and by teacher then distillation
  (6 training runs).
- Astra reviews: A's fixes (run by the subagent), then B1–B3.

## Log

- Running jobs protected: the three fall arms are on rl2-lab and finish under 4 h; the
  pending arm evaluations moved to rl2-lab; s7's evaluation (4137186) was preempted on
  overcap and rerun on rl2-lab (4137777).
- Opus subagent launched for A (W&B fixes, Astra review, canary).
- B1 facts: the tracker is two SiLU MLPs (G1 encoder 640 → 64 with finite scalar
  quantization 2 × 32; decoder 994 → 29); inputs per the deploy layout (10 reference frames
  5 apart: IsaacLab-order joint positions/velocities and root orientation relative to the
  base; 10-frame proprioceptive history, oldest first). mjlab's G1 matches SONIC's joint
  order, armature, gains, default pose and action scale; its torso is 7.82 kg against
  9.60 kg, inside SONIC's training mass randomization (×0.8–1.5).
- B1 code: `backends/sonic.py` (`SonicTracker` from the ONNX initializers; `PlanBuffer`:
  deploy resampling, blending, context, playback), `sim/mjlab/sonic_control.py`
  (`SonicController`), `core.paths.checkpoints_dir`; `tests/unit/test_sonic_tracker.py`
  4 passed (batched encode/decode equal single-row ONNX Runtime).
- Gate script `scripts/smoke/sonic_tracking_gate.py` (open / goal / time); jobs 4138032
  (time, 4,096 envs), 4138033 (open, 256 envs × 10 s).
- **B1 gate passed** (A40, mjlab's G1, flat, no crowd): open loop 1,024 × 20 s, 1 fall
  (0.003 per robot-minute), joint error 0.086 / 0.076 rad (legs / upper), orientation 6.0°;
  P0 goal following 1,024 × 60 s, 0.004 falls per robot-minute, 8.0 goals per
  robot-minute. Tracker 8.0 ms per step at 4,096 envs. Runs
  `runs/20261019T003836Z_sonic_tracking_gate{,_2}`, `runs/20261019T003601Z_sonic_tracking_gate`.
- B2 deferred: the HandPort planner (4.1 ms per plan at batch 1,024) suffices; a P1 run
  is estimated at a few hours of A40 with it.
- B3 built: `sim/mjlab/planning.py` (`PlanningVecEnv`), `backends/onnx_port.py` (the probe's
  port, moved), `backends/sonic.SonicPlanner`, `planner_rate` in `traversal.py` (task rewards
  only, per-decision PPO: 16 s rollouts, gamma 0.75 / lambda 0.9 per second, init std 0.5,
  save every 5), `train.py` (`--planner-rate`), `evaluate.py` (planner path, `--planner-zero`).
  Smokes passed: P2 PPO, P1 teacher, P1 distillation (64 envs), P0 eval, the plain eval path.
- Baseline evaluations (last 3 checkpoints, both crowds and modes) done for a2, p250, t7,
  p1000, a2pf: aware end collisions per robot-minute 0.42 / 0.31 / 0.42 / 0.17 (3.25 falls) /
  0.18 (0 falls); blind 3.2–3.7. p1000's falls: TTC < 1 s before 49–55% against 27–30% of
  controls (attributable ≈ 30% aware, ≈ 50% blind; an association).
- Next: 4,096-env timing checks (4138405/6) set the TERM lead and iteration budget; Astra
  review of B1–B3 running (scratchpad review_planner.md); then launch P1/P2 PPO and the two
  teachers with collision_end and fall_end at -1,000, students afterok on the teachers.
- Astra review of B1–B3: 3 findings fixed (play of planning checkpoints, termination-count
  aggregation, partial-decision evaluation); review items 1–6 verified correct.
- Timing at 4,096 envs before the fold: 477 s (1 Hz), 756 s (2 Hz) per iteration.
- 01:46 UTC launched P1 4138688, P2 4138689, teachers 4138690/1 (fixed `--run` dirs),
  students 4138692/3 (afterok), P0 evals; all on rl2-lab, penalties -1,000.
- Planner profile → MatMul fold in `OnnxPort` (bit-equal, 7.3× at batch 4,096). P1 stopped
  after iteration 0 and resubmitted in place (4138811, TERM lead 600 s); P0 evals rerun as
  4138812–4138815; P2 started on the fold.
- A done (subagent): the 7 W&B findings fixed, an Astra review's 4 further findings fixed,
  the labeled canary passed in `rl2-group/crowd-control-traversal-debug` (rows 0–39 exactly
  once across a requeue, eval row attached); `DEFAULT_MODE` still disabled. AGENTS.md's
  W&B rule updated: switching the default waits for the user. Full unit suite: 332 passed.
- 10:40 UTC: P1, P2 and both teachers done at 120 iterations (each requeued in place at least
  once). Students running (4138692 at about 266 s per iteration; 4138693). Evaluations of the
  last 3 checkpoints, one job per arm and crowd: 4144308–4144315; student evaluations
  4144318–4144321 afterok on the students. Run log, experiment note and status updated.
- Next: when the evaluations land, compare P1/P2/teachers with P0, b0, a2 and a2pf (aware and
  blind, end mode) in the experiment note and the artifact; then the students.
- 12:50 UTC: the eight evaluations landed (results in the experiment note): aware end 0.27
  (P1, P2) and 0.25 (teachers) collisions per robot-minute against P0 1.31/1.64, a2 0.42,
  a2pf 0.18; blind 3.6–3.7. Experiment note, decision evidence, status and paper evidence
  row updated. Students at iterations about 52 and 38; their TERM requeues are due.
- 13:00 UTC: the 1 Hz student checkpointed at 53 under TERM and resumed in place (attempt 2
  on consu). Waiting on the students and their evaluations (4144318–4144321).
- 16:40 UTC: students done (2 attempts each) and evaluated: 0.25 (1 Hz) and 0.27 (2 Hz),
  level with their teachers, falls as rare as the teachers'. Experiment note rewritten as the
  result (Status: done); decision evidence, status, paper evidence row and run log updated.
  Task complete; journal retired.
