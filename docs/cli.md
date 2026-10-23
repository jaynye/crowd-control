# Command line: `crowd`

One entry point, installed with the package (`.venv/bin/crowd`):

```bash
crowd --help                         # every command
crowd <group> <command> --help       # its options
```

The command table is `COMMANDS` in `src/crowdmotion/cli.py`. Each command is the
`main()` of one package module, imported only when it runs, so `crowd --help` works
without mjlab, ARDY or Kimodo.

Commands that write results create a fresh `runs/<UTC>_<name>/` unless given `--out`,
and refuse to reuse a directory; the one exception is a training lineage, which
`crowd train traversal --run` resumes in its own directory ([train](#train)). Their `run.json` records the status (`running`, then
`completed`, or `failed` with the error), the settings and the environment: host, GPU,
driver, CUDA, torch, the repository commit and the third-party commits, plus
`source.diff` when the tree has uncommitted changes. Replays write `render.json` inside
the evaluation they replay; `crowd graph build` writes `graph.json` beside `graph.pt`.

Source `scripts/env.sh` first. Commands marked GPU need CUDA; commands marked
checkpoints need the ARDY or Kimodo weights ([data](data.md)). `--graph` defaults to
`$CROWD_GRAPH`, else the `graph-default` artifact.

## library

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd library generate` | Offline ARDY motion library under random locomotion commands, or under turning commands with body-heading targets (`"command_source": "turns"`); writes `library.pt` | `--config configs/library/ardy_a40.json` (or `ardy_turns_a40.json`), `--out` | GPU, checkpoints |
| `crowd library turns` | Turn tracking (executed vs commanded in-place and arc turns, measured on the hip line), planted-foot speed, pelvis height and flat-ground admissibility per segment kind of turning libraries; random-command libraries give walk/stand reference rows | `--library <library.pt ...>`, `--labels`, `--overrides` (graph overrides JSON), `--config`, `--lag-s`, `--clips` (in-place clip files for `crowd bench clips`), `--out` | CPU |

## graph

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd graph build` | Build a motion graph (kNN candidates) from one library, or from several (streams concatenated in order, truncated to the shortest), and save its runtime tables | `--library <library.pt ...>`, `--config` (default `configs/graph/default.json`), `--out <dir>/graph.pt`, `--device` | GPU recommended |
| `crowd graph report` | Audit a saved graph's tables and write `audit.json` and a self-contained `index.html` | `--graph`, `--library` (command composition), `--video` (embed a demo), `--out` | CPU |
| `crowd graph edge-review` | Derived posture/height distributions, joint coverage, mask dead ends and sampled contact sheets | `--graph`, `--config` (behavior JSON), `--out` | CPU |
| `crowd graph speed` | Time graph stepping for 4,096–65,536 humans | `--library`, `--out`, `--humans`, `--neighbors`, `--graph-config`, `--reps` | GPU |
| `crowd graph rotations` | Save joint rotations for every graph frame (`rotations.pt`), for rigid mesh humans; checks them against the graph | `--graph`, `--library <library.pt ...>` (the libraries that built it, same order), `--out`, `--device` | Library artifact; GPU recommended |
| `crowd graph turns` | Turning-radius coverage of saved graphs under graph overrides (e.g. the flat-ground mask): in-place, forward-turn radius bins, non-forward turns and 1.2 s turning windows, upright/hunched, per source library of a merged graph and per motion hour | `--graph <graph.pt ...>`, `--labels`, `--overrides`, `--out` | CPU |

## behavior

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd behavior evaluate` | Plaza rollouts on a saved graph for each seed of a config's seed bank: traces (`seed_<n>.npz`) and body-relative metrics | `--config configs/behavior/baseline.json`, `--graph`, `--out`, `--device` (cpu), `--threads` (1) | CPU |
| `crowd behavior diagnose` | Saved-motion seams, vertical heights, posture persistence, ordinary redirects, response and optional native-turn coverage | `--run`, `--coverage`, `--edge-frames` (older traces), `--out` | CPU |
| `crowd behavior probe` | Paired fixed 90/180-degree body and root response through shared steering, one human per empty plaza | `--config` (repeat for paired settings), `--graph`, `--out` | CPU |
| `crowd behavior render` | Replay one seed of an evaluation as a MuJoCo capsule video | `--run`, `--seed`, `--view overview\|follow\|top`, `--human`, `--fps`, `--width`, `--height`, `--out` | `MUJOCO_GL=egl` (or `osmesa`) |
| `crowd behavior compare` | Paired per-seed metric deltas of two evaluations, and a review page with their videos | `--baseline`, `--candidate`, `--out` | CPU |

Guide: [behavior iteration](usage/behavior-iteration.md).

## iterate

Bounded iteration sessions: one run folder per session with a deadline, immutable
request names and a journal that keeps rule changes apart from results.

| Command | Does | Key options |
|---|---|---|
| `crowd iterate agent [name]` | Start a bounded noninteractive Codex process and a local session inside the caller's allocation | `--hours` (2.75), `--prompt`, `--model` (`gpt-6-astra`), `--effort` (`xhigh`), `--check` (validate model, effort and arguments only), `--graph`, `--codex-bin`, `--out` |
| `crowd iterate start <name>` | Open a session folder (`runs/<UTC>_<name>/`) and print its path | `--hours` (deadline, 4), `--executor local\|slurm`, `--note` (plan or authorization), `--graph` (recorded), `--gpu-wait-limit` (Slurm, 300 s), `--sbatch-arg` (Slurm, repeatable), `--out` |
| `crowd iterate run <session> <name> -- <command ...>` | Run (local) or queue (Slurm) a named request; `{session}`, `{name}` and `{out}` in the command are substituted | `--timeout` (900 s, capped by the deadline), `--rule` (the rule change it tests) |
| `crowd iterate status <session>` | Session state, requests and results | |
| `crowd iterate note <session> <title> <text>` | Add a journal entry | `--kind plan\|rule\|observation\|decision` |
| `crowd iterate stop <session>` | Stop the session; later requests are rejected | `--reason` |
| `crowd iterate watch <session>` | Slurm sessions: from the login node, cancel the session's job at the GPU-wait cutoff or the deadline | |
| `crowd iterate worker <session>` | Slurm sessions: drain the queue inside the allocation (started by `scripts/slurm/iterate_worker.sbatch`, not by hand) | |

Guide: [behavior iteration](usage/behavior-iteration.md#iteration-sessions).

## bench

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd bench throughput` | Usable-motion throughput sweep for ARDY and Kimodo | `--config configs/bench/*.json`, `--only`, `--out` | GPU, checkpoints |
| `crowd bench quality` | Closed-loop continuation quality for online backends and motion graphs | `--config`, `--group`, `--only`, `--library` (graph cases), `--out` | GPU; checkpoints for online cases |
| `crowd bench tables` | Markdown tables from throughput and quality runs | `--throughput <runs ...>`, `--quality <runs ...>` | CPU |
| `crowd bench clips <clips_*.npz ...>` | Stick-figure MP4s of quality clips, seams marked | `--humans`, `--max-seconds` | CPU |
| `crowd bench sim-speed` | Simulator-only speed (G1 velocity task; corridor with 4 and 8 humans; opt-in plaza scenes with rigid or capsule humans) | `--num-envs`, `--steps`, `--scenes` (`g1 crowd4 crowd8`, `plaza-{rigid,capsules}-<humans>`), `--out` | GPU |
| `crowd bench body-geometry` | LiDAR cost and range fidelity of capsule, rigid-mesh and skinned humans | `--config configs/bench/body_geometry_*.json`, `--only`, `--no-fidelity`, `--out` | GPU, library artifact |

Guide: [benchmarks](usage/benchmarks.md).

## train

`crowd train traversal --navigation <config.json> --planner-rate 1` enables the
persistent-goal objective. `crowd train navigation-ablations [--submit]` creates a
matched campaign with frozen sources and optionally submits A40 arrays. See
[goal navigation](usage/navigation-training.md) for the reward modes and experiment matrix.

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd train traversal [name]` | Train the crowd traversal policy: a G1 walking to goals through graph-driven rigid-mesh humans, seen by a head LiDAR (`sim.mjlab.traversal`). PPO on LiDAR by default, `--teacher` for PPO on privileged observations (nearest humans, plaza pose), `--distill <teacher run>` to distill that teacher into the LiDAR policy (RSL-RL `Distillation`), `--finetune <student run>` for RL plus imitation (PPO on the LiDAR policy starting from the student as actor and its teacher's critic). Writes one training lineage: `runs/<UTC>_<name>/` with `logs/<experiment>/<time>/` (checkpoints, TensorBoard) per attempt and `run.json` (`attempts`, `target_iterations`, `wandb`). On SIGTERM it checkpoints after the current iteration and exits 75 (`checkpointed`) | `--graph`, `--rotations`, `--behavior` (encounter config, default `encounter_decisive_robot.json`), `--humans` (8; a teacher and its student must match), `--robot-blind` (humans ignore the robot), `--forward-first` (goal command speed scaled by max(cos heading error, 0): turn, then walk), `--no-avoidance` (b0, the non-avoidant baseline: no LiDAR, crowd rewards or collision termination), `--penalty-ramp N` (ramp `collision_end` and `fall_end` in from 0 over N iterations), `--crowd-curriculum RATE` (a training curriculum on the humans: the level d in [0, 1] mixes the behavior config's robot traits toward the robot-blind crowd, oblivious share o + d (1 − o), stubborn share s (1 − d); once per iteration d rises by 0.05 while training collisions stay at or below RATE per robot-minute and falls by 0.05 above 1.5 × RATE; saved in checkpoints, logged as `Curriculum/crowd_difficulty/*`; not with `--robot-blind` or `--no-avoidance`), `--planner-rate 1\|2` (a planning arm: the policy decides at 1 or 2 Hz a residual on the goal command for NVIDIA's released kinematic planner, which frozen SONIC tracks; steps are decisions, see [architecture](architecture.md#planning-arms)), `--teacher`, `--teacher-lidar` / `--no-teacher-lidar` (the teacher also reads the LiDAR, a superset of the student's input; on by default for fresh runs, off for runs recorded before the option; students inherit it), `--distill <run>`, `--finetune <student run>`, `--run <dir>` (the lineage's directory: created when absent, resumed in place from its newest checkpoint to `target_iterations` when present), `--resume <earlier run>` (a branch: a new lineage from the earlier run's newest checkpoint; `--resume` and `--distill` take the earlier run's graph, behavior, humans and robot options and reject contradicting ones; `--resume` also repeats its `--env.*` overrides), `--wandb online\|disabled` (default: the lineage's recorded mode, new lineages `online`), `--out` (a fresh directory); after `--`, mjlab `TrainConfig` options such as `--env.scene.num-envs 4096 --agent.max-iterations 4000`, `--agent.seed 7` or `--env.rewards.collision-end.weight -250` / `--env.rewards.fall-end.weight -250` (the terminal collision and fall penalties, off by default) | GPU |

**Lineages.** `--run runs/<dir>` continues one lineage in place: the same directory, the
same W&B run, and training up to the lineage's `target_iterations` (its first start plus
`--agent.max-iterations`). An attempt repeats the recorded arguments, so give the same
ones (a Slurm requeue does) or only `--run` and `--wandb`; a lineage at its target exits
at once. `--resume <run>`, `--distill` and `--finetune` instead start a new lineage (a
new directory and W&B id) from another run's checkpoint, recording that run as the
parent: for example `crowd train traversal train_traversal_a2p --resume runs/<UTC>_train_traversal_a2
-- --env.rewards.collision-end.weight -1000`. `run.json`'s `attempts` lists each attempt's
Slurm job, host, start and end, checkpoint loaded, start iteration, log directory and
outcome (`done`, `checkpointed`, `failed`, `interrupted`, or `lost` after a kill).

**W&B.** A lineage maps to one W&B run in `$CROWD_WANDB_ENTITY/$CROWD_WANDB_PROJECT`
(default `rl2-group/crowd-control-traversal`) with the directory's name as its id and
the label after `train_traversal_` as its name. `CROWD_WANDB_TAGS` (comma-separated, for
example `test,canary`) adds tags to a new lineage; they are recorded in `run.json` and
kept by every later attempt and evaluation. The live writer sends rows (step metric
`iteration`) only through the last saved checkpoint, and each attempt only rows its own
checkpoints committed; it uploads no checkpoints. `published_through` advances only once
W&B's `finish()` confirms the upload (bounded at 120 s): after a crash, a kill or a finish
timeout the session stays `unconfirmed`, and the next writer (the next attempt, an
evaluation or `crowd wandb sync`) reads the run's history back from W&B and sends only
the rows it lacks. A lineage of at most 20 iterations is a verification run and stays
off W&B on every path. A `WANDB_MODE` other than `online` in the environment is refused
(the attempt trains on TensorBoard only), never silently obeyed. W&B is on by default
for new lineages (`DEFAULT_MODE = "online"` in `src/crowdmotion/tracking/lineage.py`);
`--wandb disabled` keeps a lineage off, and a lineage keeps the mode it recorded (runs
created while the default was off stay off). A W&B failure never stops training: `crowd wandb sync` catches up
([wandb](#wandb)). A lineage created with `--out` resumes with `--run` like any other.

## play

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd play traversal <run>` | Watch a `crowd train traversal` run's policy walk through the crowd, in the env its `run.json` records: live in Viser, the native MuJoCo viewer, or `--video` to `play.mp4` and `summary.json` in a fresh run directory. A teacher plays its privileged actor, a distilled student plays as the LiDAR policy. Episodes end on a collision as in training; collisions are printed, and the video marks each with a red border. Viser's panel switches policy (any run under `--runs` or a run path, and its checkpoints) and crowd (the run's own or a navigation evaluation crowd) without a restart, rebuilding the env in the same process when needed, and shows collisions, falls and goals per robot-minute and episode outcomes since the last switch with `crowd eval traversal`'s definitions. Viser and `--video` play the planning arms; the native viewer does not ([demos](usage/demos.md#trained-traversal-policies)) | `--checkpoint model_<i>.pt` (default newest), `--viewer auto\|viser\|native`, `--crowd trained\|<crowd>` (Viser), `--runs` (Viser menu, default `runs/`), `--conditions <campaign dir>` (Viser: its frozen crowds), `--video`, `--seconds` (30), `--camera follow\|overview`, `--num-envs` (1), `--seed`, `--no-end-on-collision`, `--lidar-stride` (8), `--frame-rate` (60), `--host`, `--port` (8080), `--width`, `--height`, `--device`, `--out` | GPU recommended; `MUJOCO_GL=egl` for `--video` |

## eval

`crowd eval navigation <campaign> --index N` evaluates a navigation arm on its
campaign's fixed behavior/density grid, resuming completed conditions after preemption.

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd eval traversal <run>` | Measure a `crowd train traversal` run's policy, in the env its `run.json` records, with mean actions: command following (cross-track deviation, tracking score, SPL, lag), collisions (rate, time to first collision, by human trait), time to contact, encounters, reactions and falls with their attribution. Writes `metrics.json` and per-leg, encounter, collision, fall and episode CSV tables to a fresh run directory and attaches each result to the run's W&B lineage. Guide: [traversal evaluation](usage/traversal-evaluation.md) | `--checkpoint` (repeatable) or `--last N` (1), `--crowd aware\|blind\|both` (`aware`), `--mode end\|continue\|both` (`both`), `--envs` (4,096), `--seconds` (60), `--episode-s` (none), `--humans`, `--behavior`, `--seed`, `--no-paired`, `--config <scales>.json`, `--no-attach`, `--device`, `--out`, `--planner-zero 1\|2` (P0: the released planner and frozen SONIC following the goal command with a zero residual, in the run's env; no checkpoint, nothing attached) | GPU |

## demo

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd demo video` | Render the mjlab plaza (graph-driven humans, LiDAR) to `crowd.mp4` with `summary.json` | `--graph`, `--config`, `--geometry` (`rigid`, `capsules`), `--humans` (16), `--seconds` (20), `--seed`, `--device`, `--width`, `--height`, `--out` | `MUJOCO_GL=egl`; GPU recommended |
| `crowd demo viser` | Serve the plaza with queued live behavior controls and a rigid-mesh/capsule switch | `--graph`, `--config`, `--geometry` (starting geometry), `--device`, `--humans`, `--num-envs`, `--host`, `--port` (8080), `--lidar-stride`, `--frame-rate`, `--threads`, `--verbose` | CPU or GPU |
| `crowd demo corridor` | Primitive-human corridor fixture with physics and LiDAR; viewer or headless scan export | `--viewer auto\|native\|viser\|none`, `--num-envs`, `--num-humans`, `--seed`, `--moving`, `--device`, `--steps`, `--capture-every`, `--output-dir`, `--render`; `crowd demo corridor plot --scan-file <scans.npz>` | CPU or GPU |

Guides: [demos](usage/demos.md), [corridor](usage/corridor.md).

## wandb

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd wandb status [runs ...]` | Per training lineage (default: every one under `runs/`): W&B mode and status, `published_through`, newest checkpoint, queued evaluations, attempts, who holds the writer lock, and whether it is a check run or has an unconfirmed session. Reads only | | CPU |
| `crowd wandb sync <runs ...>` | Catch lineages up whose writer lock is free: first settle a session that ended unconfirmed against the run's history on W&B, then send the committed TensorBoard rows W&B lacks up to the newest checkpoint, each iteration from the attempt that continued, then the evaluation outbox. A lineage that is training is skipped (its writer publishes); a check run is never published | `--all` (every online lineage), `--enable` (turn W&B on for disabled runs, or adopt runs uploaded by the former `scripts/wandb_sync.py`, whose floor is W&B's last step; not for check runs), `--dry-run` | network |

Evaluations attach to a lineage through `crowdmotion.tracking.attach_eval`: one row per
evaluation, kind and checkpoint, at the checkpoint's iteration, under `eval/<kind>/*`
(`crowd play traversal --video` attaches kind `play`). Without the writer lock the row
waits in `<run>/wandb/outbox/` for the next holder; `run.json`'s `wandb.evals` ledger
keeps it from being logged twice; an entry turns `published` only once W&B confirmed the
session. A lineage with W&B disabled, or a check run, attaches nothing. Many rows of one
evaluation share one W&B session: `queue_eval` each, then `flush_outbox`.

## Scripts outside the CLI

| Script | Does |
|---|---|
| `scripts/bootstrap.sh` | Clone pinned third-party sources, `uv sync`, download SOMA-X ([install](install.md)) |
| `scripts/fetch_artifacts.sh` | Copy run artifacts with sha256 checks; download checkpoints ([data](data.md)) |
| `scripts/env.sh` | Environment for every shell (source it) |
| `scripts/smoke/{ardy_smoke,kimodo_smoke,profile_ardy}.py` | Model load and latency smoke tests, ARDY profiler (GPU, checkpoints) |
| `scripts/smoke/ardy_heading.py` | ARDY body-heading targets: condition parity, angle convention and following from standing (GPU, checkpoints; `scripts/slurm/ardy_heading.sbatch`) |
| `scripts/slurm/*.sbatch` | Cluster jobs wrapping the commands above ([Slurm](slurm.md)) |
| `scripts/check_memory.py`, `scripts/index_runs.py` | Checks for the research memory and its run log |
| `scripts/wandb_sync.py` | Retiring shim for `crowd wandb sync --enable <runs>` (`--entity`, `--project` and `--dry-run` as before). |
