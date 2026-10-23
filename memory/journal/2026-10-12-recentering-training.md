# 2026-10-12: recentering toward robot training

## Request (user, 2026-10-12)

1. **Astra loop.** Start an Astra iteration loop on decisive encounter behavior. The
   problem: humans wait too long, and two humans bump, stop, and neither acts. Avoidance
   should become more likely, with minimal deviation from each human's path; a step back
   or smallest-angle turns are candidate fixes. A small, easily tunable population of
   stubborn humans stays.
2. **Policy architecture.** While Astra runs, decide which robot-policy architecture
   to prioritize, (a)–(d), and kick off training. The user notes that a SONIC
   fine-tune likely needs only 4 GPUs × 4,096 envs.

## Layout

- **Astra edits the main checkout** `/coc/flash7/jni66/crowd-control`: prompt
  `configs/behavior/iterate_codex_decisive.txt`, job 4058374, run
  `runs/20261012T152601Z_encounter_decisive`, deadline 19:11 UTC.
- **Training work happens in a snapshot** `/coc/flash7/jni66/crowd-control-train`. It
  was rsynced from the main working tree before Astra started, without `.git`, `.venv`
  or `runs`. `runs`, `third_party` and `checkpoints` are symlinks to main, so its run
  directories land in main's `runs/`. It uses main's venv with
  `PYTHONPATH=<snapshot>/src:<snapshot>/mjlab/src` and `CROWD_ROOT=<snapshot>`.
  A pristine copy for diffing is at `/coc/flash7/jni66/.crowd-control-train-base`.

## Done

- **Astra round 4 launched.** `crowd iterate agent --check` passed (gpt-6-astra,
  xhigh). In main: added the new prompt, a recentering section in `status.md`, and a
  `docs/slurm.md` line.
- **Decision note** `memory/decisions/robot-policy-architecture.md` (snapshot), with a
  row in the decisions index.
- **Env and command** (snapshot):
  - `src/crowdmotion/sim/mjlab/traversal.py`: the env;
  - `src/crowdmotion/sim/mjlab/train.py`: `crowd train traversal`;
  - a `train` group in `cli.py`;
  - `scripts/slurm/train_traversal.sbatch` and `scripts/smoke/traversal_env.py`;
  - docs: `cli.md`, `slurm.md`, `architecture.md`.
- **Smoke checks** (jobs 4058442 and 4058449, A40; logs under the snapshot's
  `logs/slurm/`, no run dirs):
  - The env builds. Obs: actor 387, critic 419.
  - **LiDAR sees the humans:** a mean of 32 rays per env hit within 0.3 m of a human
    joint, and every env has some.
  - Robot and humans stay inside the plaza.
  - 4,096 envs × 8 humans: 76.4 ms per env step, 53.6k env steps/s, peak 5.3 GB.
- **Training (a), first job:** job 4058450 (A40 crushinator),
  `runs/20261012T153938Z_train_traversal_a0`, 4,096 envs, 8 humans,
  `encounter_turns.json` on graph-turns, about 3.2 s per iteration. Its 4,500
  iterations exceed four hours, so expect the time limit near iteration 4,300; it was
  launched before the SIGTERM handler existed, so its `run.json` stays "running".
- **Queued:**
  - continuation 4058454 (`afterany:4058450`, `--resume` the a0 run, 4,200 iterations);
  - a resume check, 4058455 (overcap baymax, 64 envs, 2 iterations). It **passed**:
    `runs/20261012T154330Z_train_traversal_resume_check` loaded a0's `model_0.pt`
    through the `resume_<run>` symlink, trained 2 iterations, and reported
    `status: done`.

- **Contact path check** (job 4058467, overcap, `logs/contact_check.py`, no run dir).
  With robots placed on human hips, clearance is −0.21 to −0.26 m. `crowd_contact` = 1,
  `crowd_proximity` = 1, and `crowd_collision` terminates exactly the 8 placed envs and
  none of the other 8.
- **Run-log rows** for a0 and the resume check were appended to main's `runs.md`, the
  one exception to "do not touch main", so Astra's memory check does not flag them.

- **User decisions, 2026-10-12 follow-up:**
  - RL for all four options, with a privileged teacher first, then IL or RL+IL
    distillation into the LiDAR policy;
  - humans perceive the robot, with a separate stubbornness toward it;
  - merge after Astra's deadline;
  - Astra's robot-stubbornness round only after the user has reviewed round 4.
- **a1 cancelled.** The direct-LiDAR continuation 4058454 was cancelled before it
  started. a0 finishes its four hours as the direct-LiDAR baseline, and it can still be
  resumed with `--resume`. The snapshot freeze for a1 no longer applies.
- **Teacher added** (snapshot): a `teacher` observation group, the
  `TEACHER_TASK_ID` task, `crowd train traversal --teacher`, and the plaza obstacle
  stripped. Teacher t0 is job 4058508 (it landed on rl2-lab, not preemptible; baymax),
  `runs/20261012T155620Z_train_traversal_teacher_t0`, about 3.3 s per iteration. Its
  run-log row was appended to main.
- **Distillation added:** `crowd train traversal --distill <teacher run>` uses task
  `STUDENT_TASK_ID`, `DistillationRunnerCfg` and `TraversalDistillationRunner`. The
  student reads `actor`, the teacher reads `teacher`, and the teacher is loaded from the
  PPO checkpoint's `actor_state_dict`. Check job 4058517 (overcap):
  `runs/20261012T155927Z_train_traversal_distill_check` loaded t0's `model_0.pt`,
  behavior loss 0.0029 → 0.0025 over 3 iterations, `status: done`.
  RL+IL (PPO fine-tuning of the student) is not wired yet.
- **a0 at iteration 298:** episode length 10 → 128 steps (iterations 70 → 251), and
  `crowd_contact` and `crowd_collision` now fire.

- **Astra round 4 finished** (17:24 UTC, early): `encounter_decisive.json`, a
  development candidate; see the experiment note 2026-10-12-encounter-decisive.
- **Merged** the snapshot into main (no conflicts except the decisions index). The
  learning paradigm was archived as ledger D18. Unit tests, Ruff and the memory check
  pass in main. The snapshot `/coc/flash7/jni66/crowd-control-train` and its base copy
  are no longer used.
- **Diagnosed a0 and t0** (`scripts/smoke/policy_check.py`): observations correct;
  most collisions were humans walking into the robot; the policies walk backward with
  their backs to the goal. Both runs were cancelled; a0's status was set by hand.
- **Robot perception** (D19): external agents in `behavior/encounter.py`, per-neighbor
  limits in `graph/costs.py`, `GraphCrowdCfg.perceive_entity`, live controls, and
  `tests/unit/test_external_agents.py` (7 tests). All 12 fast goldens match.
  `configs/behavior/encounter_decisive_robot.json` is the training default.
- **`--forward-first`** goal command. Teachers t2 and t3 are running (experiment note
  2026-10-12-robot-aware-teachers); t1 was cancelled.

## Remaining

- **Done (kept for reference): robot perception (after the merge, behavior layer):**
  - `EncounterBehavior` takes external agents per env (positions and velocities in
    library XZ) into the threat test and the stop rule;
  - a per-human robot stubbornness, drawn at reset from the encounter stream: a
    stubborn share that ignores the robot, and a separate assertiveness range for the
    rest;
  - a robot body radius;
  - `GraphCrowdCfg.perceive_entity` feeds the robot's root state before each edge
    decision;
  - make it consistent with Astra's `stubborn_fraction`, and live in Viser;
  - default off, so evaluator outputs stay unchanged.
- Resolve the learning-paradigm decision in main's ledger: RL plus teacher–student
  distillation (user, 2026-10-12). Archive the file.

- Reward and env changes go into fresh runs, not `--resume` chains.

- **When Astra finishes:** read its report.
- **Merge the snapshot into main.** Diff it against the base copy:
  `diff -ru /coc/flash7/jni66/.crowd-control-train-base /coc/flash7/jni66/crowd-control-train`,
  excluding `logs/`. New files copy straight across. Merge `cli.py`, the docs, the
  decisions index and `status.md` by hand against Astra's edits. Move this entry into
  main's journal. Add run-log rows:
  - `20261012T153938Z_train_traversal_a0`;
  - the resume check and the continuation;
  - Astra's run, if Astra did not add it.
- **Validate:** `.venv/bin/python scripts/check_memory.py --check`; unit tests and Ruff
  on main after the merge; add a unit test for `CrowdGoalCommand` and `LidarScan` if
  possible on CPU.
- **Training review:** TensorBoard curves (episode length, `crowd_contact`,
  `crowd_collision`, tracking), then a play video. Then decide reward changes and
  whether to switch the crowd to Astra's `encounter_decisive.json`. That switch needs
  Astra's code merged first.
- **Robot-aware humans** in the encounter design (a behavior-layer change, after the
  merge), with a stubborn share toward the robot.
- **SONIC gate spike** (decision note), if the user agrees.
