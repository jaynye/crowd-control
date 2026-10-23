# Current status

Snapshot updated 2026-10-21. Replace sections as work changes; do not
append history. Completed results move to dated
[experiment notes](../experiments/README.md), with their conclusions in core; open
questions move to [decisions](../decisions/README.md).

## Direction (user, 2026-09-30; 2026-10-12)

4,096 environments with 4–16 humans each. ARDY and Kimodo run **offline only** (ledger
D09–D10); the motion comes from an offline library stitched by the graph runtime. The
robot is trained by **RL in mjlab**: a privileged teacher first, then distillation into
the LiDAR policy (ledger D18; [learning paradigm](../archive/2026-10-12-decision-learning-paradigm.md)).

## Active work

### Robot traversal training (user, 2026-10-12)

- **Decisions:** RL for all four architectures in the
  [robot policy architecture](../decisions/robot-policy-architecture.md) note, each with
  a privileged teacher distilled into the LiDAR policy (D18). (a), a policy from
  scratch, comes first. The user strongly favors (c), a learned kinematic planner
  feeding SONIC (2026-10-18), with **SONIC frozen** (user, 2026-10-18); the user
  approved the build and all arms (P1, P2, direct PPO and teacher-then-distill).
- **Planning arms (2026-10-19, [experiment](../experiments/2026-10-19-sonic-planning-arms.md)):**
  the planner is MotionBricks (spring target + keyframe in-betweening; released root weights
  identical to the ONNX). SONIC's tracker runs in mjlab (torch port equal to the ONNX; 8 ms
  per step at 4,096 envs) and passed the gate: 0.003–0.004 falls per robot-minute following
  the planner without a crowd. `--planner-rate 1|2` trains a residual on the goal command
  over the frozen planner and tracker. A MatMul fold in `OnnxPort` made the planner 7.3×
  faster (bit-equal): about 210 s (1 Hz) and 235 s (2 Hz) per iteration at 4,096 envs.
  P0 (zero residual, plain goal following) walks into people: aware-crowd end mode 1.31
  (1 Hz) and 1.64 (2 Hz) collisions per robot-minute against a2's 0.42. **All six trained
  arms (PPO, teacher and student at 1 Hz and 2 Hz; one seed each) cut that by 79–84% to
  0.25–0.27**, with almost no falls, but a2pf stays better (0.18; per encounter 3–18%
  better). Near people the arms mostly slow down rather than steer clear, so in the blind
  crowd they only equal a2 (3.7; P0 4.3–4.6). 1 Hz equals 2 Hz, and teacher-then-distill adds nothing
  over direct PPO (its students keep the teachers' fall rates, unlike s7). The six runs took
  40 A40-hours. The reaction metric does not apply to planning arms (replanning changes the
  velocity by itself).
- **Blind-crowd arms complete (2026-10-21, [experiment](../experiments/2026-10-21-blind-crowd-arms.md)).**
  The 1 Hz curriculum improves blind collisions **3.68 → 3.39 / robot-min** and goals
  **2.42 → 2.69 / moving-command min**, but aware collisions regress **0.273 → 0.322**
  while goals improve **3.35 → 3.68**. Fixed-blind training reduces blind collisions
  further (3.13) but loses more aware safety (0.412); goals 2.56 / 3.49 blind / aware.
  Neither arm improves both metrics in every condition. Gains are not parking/falls:
  blind collisions per encounter fall 7–10%, but median first collision is still only
  13–14 s, and threat-directed dodging is not demonstrated. Difficulty holds near
  0.55–0.60 for most of training; the last increases to 0.65 follow restart transients.
  The two-stage penalty recipe (120 without, +30 with both penalties) gives 0.476 aware /
  3.77 blind collisions, worse than P1 +30 (0.264 / 3.68). All ten paired endpoint
  evaluations are complete, one training seed per arm; middle/other behavior conditions
  remain unmeasured. User requires collision **and** goal-completion comparisons within
  each easy/hard condition. Proposed next axes are independently sampled robot
  stubbornness and assertiveness, retaining easy-condition exposure; no new runs launched.
- **Goal objective / independent crowd implementation and dense ablations (user,
  2026-10-21; [experiment](../experiments/2026-10-21-goal-objective-ablations.md), D21):**
  implemented opt-in persistent 4–8 m goals, 30 s terminal deadlines, arrival/time reward,
  discount-correct decision potential shaping, goal/time observations and shared
  non-reactive graph walkers that allow human overlap. Robot collisions still terminate.
  Legacy defaults and all 12 fast goldens are unchanged. The frozen 48-arm design covers
  reward families, discount/GAE/rollout credit, individual intermediate costs, centered
  dense rewards, time bonus/failure costs and traffic density/coherence; core cells have
  seeds 42/43. Fixed per-condition evaluations report collisions and goal success on
  independent 4/8/16, coherent 8/16 and original reactive aware/blind crowds. Full-scale
  sparse and small potential/resume/grid preflights passed. Training array **4178665**
  now has **all 48 arms running: 45 A40 + three Quadro RTX 6000**, with its cap raised
  from 4 to 48 at the user's request. Pending elements use `overcap` to avoid the
  lab partition's eight-A40 quota. Paired evaluation array **4178666** retains
  `aftercorr`, with concurrency raised to 8. Source snapshot and manifest:
  `runs/20261021T201831Z_navigation_ablations`; algorithm sources/configs are unchanged.
  RTX 6000 24 GB passed a full 4096-env / reactive-16 / 32-s-rollout PPO memory check;
  2080 Ti 11 GB failed the actual reactive-8 and independent-16 training checks.
  A 1024-env evaluation fit check is running on 2080 Ti (4179925).
  **2026-10-22:** 39 of 48 lineages finished; **nine stopped early and were not
  requeued** (seven landed on GPU-less requeue nodes, one OOM on a shared GPU, one exit 75
  without requeue); at the user's request they resume in place on `overcap` (training
  4192670, evaluations 4192671 `aftercorr`; a first resubmission, 4191658, again started
  on `rl2-lab` with no GPU allocated). 13 arms are fully evaluated. Partial result: arrival-only reward beats
  potential shaping and all intermediate terms (66% / 63% / 22% success with 8
  independent walkers); training crowd barely matters; a 4 s horizon is worse than
  100 s. The crowd-blind baseline (zero residual) on the same grid (4191359): arrival-only
  reward beats it on success and collisions in all 7 crowds (mean +12 points, −25%
  collisions); the all-terms reward is below it. No objective promoted.
- **Env:** `crowd train traversal` (`src/crowdmotion/sim/mjlab/traversal.py`): the G1
  flat velocity task in the 14 × 10 m plaza with 8 graph humans (`encounter_turns.json`,
  rigid Core skin, obstacle stripped), a torso LiDAR of 144 × 12 rays binned to 72
  azimuths over 4 frames, random goals turned into the velocity command, and clearance
  between 13 robot spheres and human joint spheres. Proximity and contact penalties;
  termination at 5 cm penetration. Groups: `actor` (LiDAR), `teacher` (6 nearest
  humans and the plaza pose; LiDAR too with `--teacher-lidar`), `critic`. 4,096 envs × 8 humans: 76 ms per env
  step on an A40, about 3.3 s per PPO iteration.
- **First runs (a0 LiDAR, t0 teacher; robot-blind crowd):** both walk but plateaued
  from about iteration 900, with about 70% of episodes ending in a collision; both
  were cancelled. Diagnosis ([robot-aware teachers](../experiments/2026-10-12-robot-aware-teachers.md)):
  - the teacher's observations are correct;
  - most collisions were humans walking into the robot;
  - the policies walk backward about 90% of the time, keeping their backs to the goal.
- **Robot perception (D19)** is implemented and on by default in the traversal env
  (`encounter_decisive_robot.json`: 5% oblivious, 10% stubborn; provisional). It cuts
  t0's collisions 4.5×. Astra iterates on the robot stubbornness only after the user
  has reviewed round 4.
- **Forward-first abandoned:** with `--forward-first` (t2–t4) the policy parks with
  the goal about 90° to its side, where the command is zero, and stands. In a
  robot-aware crowd that is both rewarded and safe. The goal-heading term did not
  rescue it.
- **Working teacher setup (t5):** the original command, turn-rate weight 2.0 and the
  `goal_heading` reward at 1.0 (the code defaults since 20:30 UTC). The weight alone
  (t6) still walks backward. t5 plateaued from about iteration 1,250: goals about 0.9,
  heading error 16°, 1.4% backward, about 30% of episode ends collisions.
- **The teacher needs the LiDAR, and then equals PPO on LiDAR.** t5 (no LiDAR in the
  `teacher` group) collides twice as often as a2 (PPO on LiDAR, same env and rewards)
  in mean-action replays. t7 (`--teacher-lidar`, now the default) is level with a2
  within checkpoint noise, as in training (21% against 19% collision ends at iteration
  1,300). a2's seed replicate matches it within 0.01. Privileged states add no
  collision benefit under the current rewards.
- **The teacher-then-distill pipeline works for option (a).** s7 (t7 distilled, 1,500
  iterations) replays at 0.16 collisions per episode end against t7's 0.21 and a2's
  0.17, level within checkpoint noise (±0.05); a collision gap from distillation is
  not resolvable at that noise (s5 +0.06, s7 −0.05). Its measurable costs are goals
  (0.93–0.97 in training against the teacher's 0.99) and falls (23 of 585 replay
  episodes against t7's 0 and a2's 1; s5 14 of 715 against t5's 1).
- **Every LiDAR arm (teacher, student, PPO) plateaus near 20% collision ends**, and the
  robot barely slows for a human ahead. A collision costs only the forfeited return
  (about 4 at γ = 0.99, against a net 0.043 per step).
- **A terminal collision penalty helps a little, and a large one makes the robot fall**
  ([collision penalty](../experiments/2026-10-13-collision-penalty.md), user approved
  2026-10-13). `collision_end` at −250 (p250) replays at 0.15 collisions per episode end
  against a2's 0.17, with goals, speed and falls unchanged; the rule's 0.12 is not met.
  At −1,000 (p1000) 85% of episodes end in a fall: nothing penalizes falling, which costs
  only the forfeited return. About 0.07 per end are oblivious humans walking into the
  robot, which only dodging can avoid.
- **p1000's locomotion never stabilized** (curves read 2026-10-18). No term penalizes
  a fall. Its curves match a2's through iteration 200 and split off at iterations
  400–800, when a2 starts walking and meeting humans. It ends walking at 0.51 m/s, but
  falls stay at 10–20 per logged iteration against about 0 for a2. This supports the
  user's suspicion. Whether it falls on purpose near humans is not separated; the
  proposed fall attribution tests it with inference only.
- **Evaluation harness built (2026-10-18).** `crowd eval traversal` measures command
  following, collisions, time to contact, encounters, reactions and falls on paired goals
  ([traversal evaluation](../decisions/traversal-evaluation.md); guide
  `docs/usage/traversal-evaluation.md`). a2 `model_3999` at 4,096 envs × 60 s, aware
  crowd: 0.39 collisions per robot-minute, 0.012 per encounter, SPL 0.40 (end mode). The
  older "collisions per episode end" is retired: falls distort it (p1000: 0.027 per end
  at 4.2 falls per robot-minute).
- **b0 and the fall fix (2026-10-18/19):** b0 (`--no-avoidance`, job
  4137004) and three fall-fix arms: pf1000 (p1000 plus `fall_end` −1,000, 4137005),
  pf1000r1500 (the same ramped in over 1,500 iterations, 4137006) and a2pf (a2 resumed
  with both penalties, 4137007) ([experiment](../experiments/2026-10-18-fall-fix-and-b0.md)).
  **Result:** the fall fix works from a walking policy: **a2pf 0.18 collisions per
  robot-minute, no falls, 12% fewer goals than a2**; from scratch the same penalties learn to
  stand (pf1000) and ramped in they lose 40% of the goals (pf1000r1500). b0 collides 0.95
  against a2's 0.42 in the aware crowd and 4.23 against 3.65 in the blind one: most of the
  training crowd's safety is the humans' (aware end mode, last 3 checkpoints). Still proposed: shaping that
  rewards dodging a closing human. No `--finetune` of s7 under the current rewards.
- **W&B run continuity: on by default since 2026-10-21 (user; ledger D20).** One W&B run per lineage (`--run`, in place, `attempts[]`); rows only
  through the last checkpoint, counted as published only once W&B confirms the upload
  (an unconfirmed session is reconciled against W&B's history by the next writer);
  requeue on a `checkpointed` exit with TERM 300 s before the limit; evaluations attach
  through an outbox and one writer lock; `crowd wandb sync|status`. All 7 open review
  findings and 4 from a second Astra review of the fixes are fixed. The canary
  (`runs/20261019T012044Z_train_traversal_TEST_canary_debug`, scratch project
  `crowd-control-traversal-debug`) published iterations 0–39 exactly once across a
  TERM, requeue and in-place resume, plus an attached evaluation. `DEFAULT_MODE` is now
  `online` (main project); lineages created earlier keep their recorded `disabled`
  (backfill with `crowd wandb sync --enable` was not requested). `CROWD_WANDB_TAGS` labels
  test runs.
- **Tools:** `crowd play traversal` shows any run's policy in Viser or renders it to
  MP4; `crowd eval traversal` measures it. **Live viewer (2026-10-22):** the Viser path
  switches run, checkpoint and crowd (the run's own or the seven navigation-campaign
  conditions) from the browser without a restart and shows live collisions, falls and
  goals since the last switch (evaluator definitions). It builds envs through the
  evaluator's `eval_env_cfg` and the campaign's condition functions, so viewer and
  evaluation cannot diverge. Planning and navigation arms now play in Viser. Teachers are
  refused in 4-human crowds (fewer than 6 observed humans). GPU-checked headless on L40S
  (`runs/20261022T163632Z_live_viewer_check`: 30 switch stages, 19 rebuilds of 1–3.5 s,
  0.8–1.2× real time with one env); not yet exercised in a real browser. Launch:
  `docs/usage/demos.md`. Curves of runs through 2026-10-14 are on W&B
  (`rl2-group/crowd-control-traversal`), and so is every lineage created since
  2026-10-21; the runs in between are off it.
- **Distillation** (`--distill <teacher run>`, inheriting the teacher's env options) and
  **RL plus imitation** (`--finetune <student run>`: PPO from the student actor and the
  teacher's critic) both pass 3-iteration checks on t5.

### Encounter decisiveness, round 4 (user, 2026-10-12)

- **Direction:** the crowd is close to ready for training the crowd-avoiding robot
  policy. One behavior problem remains: humans are not decisive. Two humans often bump
  into each other and stop, and neither acts. Avoidance should become more likely, with
  minimal deviation from each human's path. Candidate fixes are a step back on a
  detected collision and smallest-angle turns. A small population of stubborn,
  non-avoidant humans stays, and it must be easily tunable.
- **Round 4 complete, development candidate only:**
  `configs/behavior/encounter_decisive.json`, on graph-turns with paired rotations.
  [Experiment](../experiments/2026-10-12-encounter-decisive.md), outer report
  `runs/20261012T152601Z_encounter_decisive/report.md`. Close-pair stalls at <=1.1 m
  fell 43→19 over three 60 s seeds; maximum 5.2→2.4 s; joint-sphere contact 1.36→0%
  of frames. Mean speed .616→.643 m/s; goals 3.354→3.250/human-minute. The small
  stubborn population is tunable and has finite stall patience.
- **Limits remain:** individual wall/obstacle stops reach 48 s, total stopped time
  is unchanged, foot speed/acceleration worsen, and 53% of redirect attempts remain
  unresolved. Held-out seeds 101/307/911 were used once after freezing: 2.8 s maximum
  close-pair stall and zero joint-sphere contacts, but a 47.2 s individual stop. They
  are no longer unseen. [Next experiment](../decisions/encounter-progress.md): isolate
  failed ordinary turns beside the wall/scanner, then test executable recovery.
- **Validation:** 254 CPU tests pass; all-off encounter and plaza traces match frozen
  references exactly. Final fast-golden arrays match the untouched same-host snapshot
  (283 arrays); five pre-existing expected-file diagnostic rounding mismatches remain.
  Expected files unchanged. Inspected before/after follow, overview, small-turn,
  stubborn-pass, blocked-probe and residual-failure stills; rigid GPU demo succeeds.
  Live controls are headless-tested; browser and training-scale timing unverified.
- **Backward step deferred:** saved coverage exists (402 admissible short edges from
  standing/very slow entry), but commanded execution from rest is unreliable. No
  backstep strategy, motion-prior load or graph/library rebuild was added.

### Repo migration to git and local iteration (Phases 0–9 done 2026-10-01)

- The repository now has commit `5c1ef9c`; the working tree includes preserved,
  uncommitted launcher and behavior work. No commit or push was made during iteration.
  Installation and migration steps remain in `docs/install.md`. Real-generator GPU
  migration checks are still outstanding; the saved graph passed a small GPU mjlab
  integration check on 2026-10-05.
- The refactor is behavior-preserving: CPU goldens from the pre-migration code
  (`tests/golden/`) match bit for bit after every phase, including real ARDY and Kimodo
  on CPU. One deliberate difference: a `history` of 0 is rejected (it used to pass the whole
  history by accident; no config used it). Design rationale:
  [runtime architecture](design/runtime-architecture.md). Plan, phase log and reviews:
  the repo-migration entry in the [journal](../journal/README.md).
- mjlab is first-party (ledger D13); its upstream history is outside the repo (plain
  tracked files, the plan's default; a fork or subtree can still be adopted).
- Next (Phase 11, user to agree): switch plaza steering to per-human streams and make
  `GraphConfig` strict and record a new baseline. The completed behavior iteration
  below kept both deferred changes out of scope.

### Encounter behavior design and turning library (2026-10-06)

- **User direction (ledger D16):** keep the random-goal plaza design and add an alternate
  `encounter` design that the iteration agent can also work on. It has:
  - waypoint or velocity conditioning, with random waypoints for comparison;
  - a configurable bubble;
  - an assertiveness- and distance-dependent switch hazard;
  - committed interventions, including gradual and in-place turns;
  - a stop rule with a 1 cm tolerance.

  Two humans may decide at once.
- **Implemented, opt-in.** The plaza path is bit-exact against the previous day's run.
  Configs: `configs/behavior/encounter*.json` and `plaza_forward_posture.json`.
  Usage: `docs/usage/behavior-iteration.md#behavior-designs`.
  [Results](../experiments/2026-10-06-encounter-design.md).
- **Results on development seeds 0/1/2:**
  - 94% forward travel against 70–83%;
  - lower foot sliding and acceleration;
  - joint-sphere contact at most about 1 cm, against plaza's −5 to −9 cm;
  - slower walking: 0.43–0.50 against 0.58–0.64 m/s at 60 s.

  The default graph realizes arcs but not in-place turns, which stand at a median
  2.8°/s, and sidesteps are mostly oblique.
- **Turning library (done, opt-in).** ARDY follows sparse body-heading targets: the hip
  line turns 0.98 of the command at the median, and arcs come out at 1.04 of the radius.
  The opt-in graph `graph-turns` (base plus a 9.1 h turning supplement; rotations in
  `graph-turns-rotations`) raises admissible in-place edges from 8 to 5,873, of which
  4,763 hold no planted twist. The supplement also adds 2,354 standing or straight edges
  that hold a twist (67 in the base), so on this graph a human can stand twisted with
  no turn requested.
  [Note](../experiments/2026-10-06-turning-library.md).
  - About 15% of generated in-place turns twist the pelvis while the feet stay planted.
  - On this graph the encounter design's in-place turns execute (median 83°/s against
    2.8°/s), and turn-then-walk resolutions roughly triple.
  - Joint-sphere contact regresses to 2–7 cm and sliding rises.
- **Round 3, Astra on graph-turns (2026-10-07; stopped after 13 of 225 min).**
  [Note](../experiments/2026-10-07-encounter-turns-iteration.md).
  - Opt-in planted-twist mask (`twist_mask`, `twist_max_foot_yaw_deg` 21.4°) removes
    3,533 graph-turns edges: 1,110 in-place and 2,421 straight or standing. Without it,
    2–4% of executed edges were planted twists.
  - `configs/behavior/encounter_turns.json` is `encounter.json` plus the mask: a
    development candidate, not frozen, held-out seeds unused. Minimum clearance
    improved in all 6 unpaired crowd runs (60 s: −0.027 / −0.010 / +0.043 m); sliding,
    acceleration and unresolved redirects did not.
  - The mask is a live Viser control. Rigid mjlab and the Viser demo run on
    graph-turns with `CROWD_GRAPH_ROTATIONS` set (launch: `docs/usage/demos.md`).
  - Comparison video (plaza and encounter on graph-default, encounter on graph-turns
    without and with the mask): `runs/20261007T165926Z_encounter_turns_video`.
  - The agent stopped on a KeyError in its own script and killed itself with
    `crowd iterate stop`. Both Codex prompts now treat own-script failures as bugs and
    forbid `crowd iterate stop`.
- **Follow-up completed 2026-10-12:** the decisive iteration above tested these
  contacts, jams, decision controls and spacing/speed trade-offs. Its selected
  configuration is still a development candidate; wall/obstacle progress and motion
  quality remain the next work. Preserve the older notes as historical evidence.

### Transition/posture iteration: development evidence, stopped before promotion (2026-10-05)

- The local A40 session completed 190 requests, then stopped after one frame-extraction
  invocation failed, following the user's explicit stop-on-failure instruction.
  Outputs: `runs/20261005T225742Z_behavior_iteration/report.md` and its session journal.
  [Results, inspected views and validation](../experiments/2026-10-05-transition-posture-iteration.md).
- **Incumbent unchanged:** the previously selected configuration remains
  `runs/20261005T201450Z_behavior_iteration/configs/selected_candidate.json`.
  Both it and the original baseline were frozen and evaluated before runtime edits.
  The new development candidate is
  `runs/20261005T225742Z_behavior_iteration/configs/forward_floor080_peak012.json`;
  it has not been promoted or frozen for held-out evaluation.
- Implemented opt-in load-time edge annotations, flat-ground masks with an explicit
  stationary-hold dead-end fallback, independent sticky per-human posture and matched
  starts, soft posture/turn/entry-velocity costs, turn-before-walk steering, executed
  height/posture/redirect metrics, and queued Viser live controls. All consumers still
  use `CrowdController`; graph/defaults/legacy plaza RNG/golden expected files unchanged.
- Three 60 s development seeds: ordinary forward travel 93.7–94.4% versus incumbent
  78.2–83.9%; posture-class crossings 4.3–5.8 versus 18.0–19.4 per human-minute;
  peak lowest-foot height .112 m versus 1.336 m. However seed-0 minimum root gap
  worsens .635→.433 m, feet/seams are mixed, and many redirect attempts remain unresolved.
  Controlled body responses are 47/48 versus incumbent 48/48 for both 90° and 180°.
- Coverage limits slow turns: after the chosen mask, only two upright and three hunched
  slow-turn edges remain. The .2 m/s explicit turn phase stalled; walking-turn phases
  improved controlled response but did not dominate crowd metrics. The main
  mostly-turn-then-walk objective is **unfinished**, not established by forward shares.
- Nine initial edge sheets plus brief-airborne examples and several seed-0 follow/
  overview variants were actually inspected. Seed-2 dense reversed follow exposed
  a raised-foot transition. Final candidate image review is unfinished: its MP4
  rendered, but extraction invoked base Python instead of the virtualenv and failed.
- **Validation:** final 165 CPU tests and Ruff pass. Earlier same-host compatibility
  matched 283 arrays/27 JSON outputs across all 12 fast golden cases and both frozen
  reference trace banks. Both trees retained historical expected-file mismatches;
  final peak-mask/diagnostic additions were not followed by another golden run.
  Viser bindings are headless-tested; browser and new GPU integration are unverified.
- **Comparison video (2026-10-06):** `runs/20261006T175357Z_behavior_video/comparison.mp4`
  shows baseline, round-1 selected and this candidate on the seen seeds 11/23/47.
  Posture and forward share improve. Seed 47 briefly exceeds the 0.12 m foot mask
  (0.138 m); the seed-11 wall excursion matches round 1. [Follow-up](../experiments/2026-10-05-transition-posture-iteration.md#follow-up-comparison-video-on-seen-seeds-2026-10-06).
- **Next:** repair the invocation by using the literal `.venv/bin/python` path, finish
  candidate image review, freeze it, evaluate still-unseen seeds 101/307/911 at 20/60 s,
  and run the prepared small mjlab/live-control check. Then test avoidance across the
  executed .4 s horizon and audit reachable upright walking-turn sequences. No new
  session or jobs were created in this run. [Calibration remains open](../decisions/posture-height-calibration.md).
- Historical preceding result: the [evening iteration](../experiments/2026-10-05-evening-graph-behavior-iteration.md)
  improved held-out forward shares but retained spacing/response limits; its GPU
  integration check applies to that earlier configuration, not the new controls.

### Rigid mesh humans in the plaza (adopted and implemented 2026-10-06, D15)

- **Decision (user):** full-resolution rigid Core-skin pieces, one per bone, with no
  simplification, are the human geometry for LiDAR and display. They are the plaza
  env-config default (`human_geometry`), which a training env will pick up; no
  training env exists yet.
  [Benchmark and timing](../experiments/2026-10-05-body-geometry-benchmark.md).
- **How it works:**
  - `crowd graph rotations` saves joint rotations per graph frame: artifact
    `graph-rotations`, 0.32 GB, checked against the graph at load.
  - `GraphCrowd` poses the pieces in one fused Warp pass (`sim.rays`).
  - A Warp ray target, registered through our mjlab's new
    `RayCastSensor.add_ray_target`, serves the LiDAR. The pieces are visual geoms in
    group 1.
  - The Viser demo builds both geometries; **Commands → Crowd body → Geometry**
    switches what is shown and what the LiDAR hits.
- **Cost (A40, 4,096 envs):** plaza step 57.3 ms (rigid) against 53.3 ms (capsules) at
  16 humans, and 29.3 ms for both at 4 humans.
- **Checks:**
  - hits equal CPU MuJoCo `mj_ray` on the visual pieces (CPU and A40);
  - the `mjlab_plaza` golden is unchanged (it pins capsules);
  - the unit suite passes.

  Browser interaction with the switch has not been exercised; a fake GUI drives it in
  the tests.
- **Still open:**
  - one body shape: the morphology set (SOMA-X refit or retargeting) is undecided;
  - robot contact geometry (the pieces have `contype` 0);
  - ray budget and scan rate;
  - an mjlab fix for mujoco_warp's off-centre mesh bounds (it affects traced robot
    meshes, not the rigid humans).

### Graph audit and review tools (completed)

- Report: `runs/20261001T055543Z_behavior_report/index.html`; graph mechanics,
  available motion composition, connectivity, stochastic rules and historical video.
  Regenerate with `crowd graph report`.
- Of valid library frames, 43.0% forward, 14.4% sideways, 17.4% backward; only 48.7%
  forward among moving frames. Five of 134,820 edges qualify as slow turning under
  explicit analysis cutoffs. Availability differs from runtime visitation.
  [Conclusions](design/motion-library.md#behavior-audit-and-review-loop-2026-10-01)
  and [audit evidence](../experiments/2026-10-01-graph-behavior-audit.md).
- The saved graph's default score still has zero facing/heading weights. Optional
  orientation terms now exist; legacy plaza goal/spawn RNG remains shared.

### Offline library and graph performance (historical prototype, 2026-09-30)

- **Result:** [motion library](design/motion-library.md#conclusions); numbers in the
  [experiment](../experiments/2026-09-30-ardy-library-graph.md).
  - An 18 h ARDY library takes 2 min of A40 time.
  - The kNN motion graph steps 65,536 humans in 47 ms per 0.4 s: about 5% of an A40
    at m = 16, `S = 0.4` (isolated). Online ARDY of the same responsiveness would need
    about 46 A40s.
  - On a held-out seed at the recommended setting
    (`configs/graph/default.json`), it tracks paths to 0.7–1.4 cm, and its
    median responses are within 0.0–0.4 s of online ARDY. The hunch filter removes
    stooped gait (0.2–0.6% of frames).
  - Costs: more frequent small foot sliding, root-speed seams 1.4–2.0× the interior,
    no local rotations yet, 20 fps output.
- **Waiting on the user:**
  - [robot policy architecture](../decisions/robot-policy-architecture.md): what follows
    the blind-crowd arms now that their endpoint evaluations show a safety/progress
    tradeoff across behaviors;
  - [traversal evaluation](../decisions/traversal-evaluation.md): metric definitions,
    `crowd eval traversal`, the b0 baseline and the fall fix;
  - [library generator](../decisions/library-generator.md): later ARDY/Kimodo expansion;
  - [graph response delay](../decisions/graph-response-delay.md): eventual acceptable
    simulated-time delay; does not block baseline review;
  - [text encoder access](../decisions/text-encoder-access.md): styles;
  - [human body model](../decisions/human-body-model.md): LiDAR geometry settled
    (D15, rigid pieces); still open: morphology source, robot contact geometry, ray
    budget and scan rate.
- **Remaining implementation candidates:**
  - mjlab smoke test: graph-driven humans in the crowd fixture with LiDAR, at m = 4–16;
    also measures simulator `S` at m = 16;
  - local-rotation output with offset decay (needed for mesh skinning);
  - root-velocity inertialization;
  - simulator frame-rate conversion (ARDY 20 fps);
  - a Kimodo/SOMA library through the same graph;
  - styles once prompts work.

## What exists

- **Code:** `src/crowdmotion` (installable; `crowd` CLI), layered as in
  [runtime architecture](design/runtime-architecture.md); module map and commands in
  `docs/architecture.md` and `docs/cli.md`.
  - `behavior/`: `CrowdController` with command sources (random segments, benchmark
    scenarios, plaza intent and steering), planner, graph and online motion sources.
    Library generation, the quality benchmark, the evaluator and mjlab's `GraphCrowd`
    share it.
  - `graph/`: kNN motion graph tables, build, cost terms, runtime, audit.
  - `backends/`: ARDY and Kimodo adapters with typed configs and a registry.
  - `iterate/`: evaluate, diagnose, controlled response probes, compare, sessions (local or Slurm). `bench/`, `library/`,
    `metrics/`, `viz/`, `sim/` (scene, capsules, `mjlab/` plaza, demos, corridor fixture).
  - Configs: `configs/{behavior,graph,library,bench}`; recommended graph setting
    `configs/graph/default.json`.
  - Tests: `tests/unit` (CPU), `tests/golden` (bit-exact, 14 cases), `tests/gpu`.
- **mjlab demos** (2026-09-30; one environment): a 14 × 10 m plaza in which 16
  graph-driven capsule humans walk, seen by an mjlab LiDAR (`crowd demo video|viser`;
  `docs/usage/demos.md`), from the saved graph
  `runs/20261001T013254Z_motion_graph/graph.pt` (441 MB, artifact `graph-default`).
  - **Video:** `runs/20261001T012334Z_mjlab_graph_demo/crowd.mp4`. Closest
    root-to-root gap 0.70 m; about 206 of 5,760 rays hit humans each step.
  - **Viser:** about 1× real time at 50 FPS on a GPU node with every 8th LiDAR hit
    drawn; 1.0× at 30 FPS on a 4-core CPU job; 0.4–0.7× on the busy login node.
  - Some stooped poses survive the mean-head-height filter.
  - This is a visual integration check, not the 4,096-environment smoke test.
- **Library on disk:** `runs/20261001T004918Z_ardy_library/library.pt` (18.2 h,
  2,048 × 32 s streams, Core skeleton; artifact `library-ardy-h40`).
- **Environment:** one uv workspace (crowd-control + mjlab + kimodo + py-soma-x) with
  ARDY from source; third-party pins in `third_party.toml`; see `AGENTS.md`.
- **Weights:** ARDY Core H8 and H40, Kimodo-SOMA-RP-v1.1 and SOMA-X v0.3.2 assets, all
  local and offline (`artifacts.toml` names the HF revisions). ARDY-SOMA is unreleased.
- **mjlab changes (first-party):** LiDAR pattern, minimum range, hit IDs and ray
  targets, with their tests and docs; the corridor fixture moved to `crowdmotion.sim.mjlab.corridor`. See
  [crowd architecture](design/crowd-architecture.md#corridor-fixture-and-our-mjlab-changes).

## Blockers and known issues

- **Gated Llama-3** (403 for the HF account): no prompts. Opt-in geometric posture
  selection now exists, but semantic styles and its geometric calibration remain open.
- **Behavior coverage and selection:** optional orientation costs improve forward
  travel, but slow-turn coverage, long-rollout spacing, wall excursions and slower
  path responses remain limitations. Shared legacy plaza RNG still does not meet
  per-human batch independence; it was preserved as explicitly requested.
- **No end-to-end measurement.** All speeds are isolated, on an A40. The simulator's
  own `S` at m = 16 with LiDAR is unmeasured.
- **`fpo-control/`'s git index stages every file for deletion** (the files are on
  disk). It is unrelated and was left untouched.
- **Old broken env:** `mjlab/.venv.stale-pre-migration` (2026-09-28, gitignored) awaits
  the user's deletion.
- **GPU paths unverified after the refactor** until Phase 10 on the local GPU: bf16
  autocast, ARDY `conditions="library"`, Kimodo post-processing and the throughput OOM
  retry are not covered by the CPU goldens.
