# Behavior iteration

The loop for making crowd behavior look realistic: audit what the saved graph contains,
evaluate behavior on fixed seeds, replay the saved motion as video, compare a candidate
with the baseline on the same seeds, and keep a session record of what changed. All of
it runs on the saved motion graph (artifact `graph-default`); nothing loads a motion
prior or rebuilds the graph. Evaluation and comparison run on CPU; renders need an
offscreen GL context.

Run from the repository root after `source scripts/env.sh`. `--graph` defaults to
`$CROWD_GRAPH`, else the `graph-default` artifact ([data](../data.md)).

## 1. Audit the graph

```bash
crowd graph report [--library <library.pt>] [--video <crowd.mp4>] --out runs/<UTC>_behavior_report
```

Writes `audit.json` (raw data, graph sha256, provenance) and `index.html`, which embeds
the measured composition, connectivity and interactive example edges. It loads no
external scripts or fonts: open it directly, or serve the repository with
`python -m http.server 8090`. `--library` adds the library's command composition;
`--video` embeds a demo video by relative link.

**Definitions** (analysis cutoffs, not human calibration or acceptance thresholds):

- Frame direction compares root travel with the midpoint ARDY body heading of adjacent
  frames, including the edge's predecessor. Below 0.15 m/s is *still/slow*. Moving
  travel within 30° of facing is *forward*, 30–60° *oblique*, 60–120° *sideways*, and
  120° or more *backward*.
- An edge takes the class of its majority of frames (ties go to the earlier class).
  Forward and slow edges are split into *turning* at 10° of net yaw per edge. "Turn in
  place / slow" includes slow translation.
- Candidate transitions leave each edge's end boundary (`edge + 1`). The transition
  matrix shows what is available, weighting valid edges equally, not how often the
  runtime visits them.

## 2. Evaluate

```bash
crowd behavior evaluate --config configs/behavior/baseline.json --out runs/<UTC>_behavior_baseline
```

Each seed of the config's seed bank runs one plaza of `humans` humans for `seconds` at
the motion frame rate (20 fps), through the same `CrowdController` as the mjlab
`GraphCrowd` term: plaza intent and local steering, the path planner, the graph and the
velocity update ([architecture](../architecture.md#the-crowd-loop)). It excludes mjlab
interpolation, sensors and the reset lifecycle, so it is a behavior diagnostic, not a
replay of a simulator run or a throughput benchmark. Defaults: CPU, one thread.

**Config** (`configs/behavior/baseline.json`; unknown keys are rejected):

| Key | Meaning |
|---|---|
| `humans`, `seconds` | Population and duration of each seed (16, 20 s) |
| `seeds` | The seed bank: distinct integers; compare runs only on the same bank |
| `steering` | `PlazaSteeringCfg` fields: plaza half sizes, speed range, wait probability and duration, arrival radius, personal space, avoidance horizon and gain, contact range, wall margin, obstacles `[x, z, radius]` |
| `graph_overrides` | `GraphConfig` fields applied on load: cost weights, `temperature`, correction limits, `inertialize`, `lookahead`. `neighbors`, `vel_weight` and `hunch_ratio` are baked into the graph and cannot change |
| `description` | Free text |

Copy the baseline config for each candidate and change one mechanism at a time.

**Outputs**, in the run directory:

- `seed_<n>.npz`: the executed motion (a `Trace`): joints, planar roots, body headings,
  contacts and foot indices, plus selected edges, commands, goals, wait flags and
  selection diagnostics. Frame 0 is the initial pose.
- `seed_<n>.json`: metrics. Frame composition; shares of moving frames that are
  forward, sideways and backward; facing-to-travel error; body and path turn rates;
  acceleration; pairwise root gaps; planted-foot speeds; natural-successor share; path
  error; rollout timing.
- `summary.json` (all seeds), `skeleton.json` (for replay) and `run.json` (settings,
  graph sha256, environment, status).

The plaza steering draws from one generator shared by all humans in call order, so it
reproduces existing baselines but is not independent per human; see
[architecture](../architecture.md#randomness).

## 3. Replay

```bash
MUJOCO_GL=egl crowd behavior render --run runs/<UTC>_behavior_baseline --seed 0 --view overview
MUJOCO_GL=egl crowd behavior render --run runs/<UTC>_behavior_baseline --seed 0 --view follow --human 3
```

Renders saved joints as MuJoCo capsules (no graph stepping, Warp or LiDAR) into
`render_seed_<seed>_<view>/crowd.mp4` inside the evaluation, with `render.json` (trace
hash, camera, provenance, status). Views: `overview`, `top` and `follow` (the camera
follows `--human`). Arrows on the floor show body facing, which separates curved
forward travel from sidestepping. `--fps`, `--width` and `--height` (20, 960, 540) are
independent of the motion frame rate; `--out` writes another variant elsewhere. Existing
renders are never overwritten.

`MUJOCO_GL=egl` needs a GPU with EGL; on a CPU-only machine use `MUJOCO_GL=osmesa` if
OSMesa is installed. To see a candidate in the simulator with the LiDAR, use
`crowd demo video` ([demos](demos.md)).

## 4. Compare

```bash
crowd behavior compare --baseline runs/<UTC>_behavior_baseline \
    --candidate runs/<UTC>_behavior_candidate --out runs/<UTC>_behavior_compare
```

Both evaluations must share the graph (sha256), frame rate, population, duration and
seed bank. Writes `comparison.json` (candidate minus baseline per seed and metric, both
manifests) and `index.html`, which shows the deltas beside the renders found in either
run. Review spacing, acceleration and planted feet alongside the direction shares. No
automatic pass threshold is applied.

## Iteration order

1. Audit forward and turning coverage, freeze the baseline config and evaluate it.
2. Change one steering or scoring mechanism and evaluate a candidate.
3. Compare matching seeds, then replay the saved motion from several views.
4. Before promoting a candidate to mjlab, inspect coverage failures, sideways and
   backward travel, abrupt yaw, close gaps and foot sliding.

The graph has no explicit facing cost or turn-before-walk state yet; those are behavior
changes to implement and evaluate. Path tracking alone does not enforce them. A new
cost term goes in `graph/costs.py`, an intent state in `behavior/intent.py`
([development](../development.md#extending-the-runtime)).

## Iteration sessions

A session keeps a bounded series of requests in one run folder, with a deadline, request
names that can be used only once, and a journal that records rule changes apart from
results, so a proposed setting is never presented as a tested change.

```bash
S=$(crowd iterate start turn-commit --hours 4 --note "Plan: committed turns biased to 90°")
crowd iterate run $S 000_baseline --timeout 600 -- \
    crowd behavior evaluate --config configs/behavior/baseline.json --out {out}
crowd iterate run $S 001_turns --timeout 900 --rule "Committed turns, 90° mode" -- \
    crowd behavior evaluate --config <candidate.json> --out {out}
crowd iterate note $S "Turns look abrupt" "Seed 0 overview: ..." --kind observation
crowd iterate status $S
crowd iterate stop $S --reason done
```

- `crowd iterate start` prints the session folder, `runs/<UTC>_<name>/`. It holds
  `session.json` (deadline, executor, status), `journal.md`, `queue/` and
  `iterations/<name>/`.
- In the request command, `{session}`, `{name}` and `{out}` (=
  `{session}/iterations/{name}`) are substituted. A leading `crowd` runs this
  environment's CLI.
- Each request leaves `queue/<name>.request.json` → `.working.json` → `.done.json`, plus
  `queue/<name>.log` and `queue/<name>.result.json` (status, exit code, timing).
- **Local executor** (the default): each request runs immediately as a subprocess,
  bounded by its `--timeout`, the session deadline and a `STOP` file in the session
  folder. `crowd iterate stop` writes `STOP`.
- `--kind` of a note is `plan`, `rule`, `observation` or `decision`.

**On the cluster** (`--executor slurm`): `start` submits one worker allocation
(`scripts/slurm/iterate_worker.sbatch`, default `--partition=rl2-lab,overcap
--qos=short`; change with `--sbatch-arg`) and records its job ID. `run` then queues
requests, and the worker drains them (`crowd iterate worker`). From the login node, run
`crowd iterate watch $S`: it journals results as they arrive and cancels only the
recorded job when the GPU-wait limit (`--gpu-wait-limit`, 300 s) or the deadline
passes. Both executors leave identical records. See [Slurm](../slurm.md).
