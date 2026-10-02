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
and refuse to reuse a directory. Their `run.json` records the status (`running`, then
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
| `crowd library generate` | Offline ARDY motion library under random locomotion commands; writes `library.pt` | `--config configs/library/ardy_a40.json`, `--out` | GPU, checkpoints |

## graph

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd graph build` | Build a motion graph (kNN candidates) from a library and save its runtime tables | `--library`, `--config` (default `configs/graph/default.json`), `--out <dir>/graph.pt`, `--device` | GPU recommended |
| `crowd graph report` | Audit a saved graph's tables and write `audit.json` and a self-contained `index.html` | `--graph`, `--library` (command composition), `--video` (embed a demo), `--out` | CPU |
| `crowd graph speed` | Time graph stepping for 4,096–65,536 humans | `--library`, `--out`, `--humans`, `--neighbors`, `--graph-config`, `--reps` | GPU |

## behavior

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd behavior evaluate` | Plaza rollouts on a saved graph for each seed of a config's seed bank: traces (`seed_<n>.npz`) and body-relative metrics | `--config configs/behavior/baseline.json`, `--graph`, `--out`, `--device` (cpu), `--threads` (1) | CPU |
| `crowd behavior render` | Replay one seed of an evaluation as a MuJoCo capsule video | `--run`, `--seed`, `--view overview\|follow\|top`, `--human`, `--fps`, `--width`, `--height`, `--out` | `MUJOCO_GL=egl` (or `osmesa`) |
| `crowd behavior compare` | Paired per-seed metric deltas of two evaluations, and a review page with their videos | `--baseline`, `--candidate`, `--out` | CPU |

Guide: [behavior iteration](usage/behavior-iteration.md).

## iterate

Bounded iteration sessions: one run folder per session with a deadline, immutable
request names and a journal that keeps rule changes apart from results.

| Command | Does | Key options |
|---|---|---|
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
| `crowd bench sim-speed` | Simulator-only speed (G1 velocity task; corridor with 4 and 8 humans) | `--num-envs`, `--steps`, `--out` | GPU |

Guide: [benchmarks](usage/benchmarks.md).

## demo

| Command | Does | Key options | Needs |
|---|---|---|---|
| `crowd demo video` | Render the mjlab plaza (graph-driven humans, LiDAR) to `crowd.mp4` with `summary.json` | `--graph`, `--humans` (16), `--seconds` (20), `--seed`, `--device`, `--width`, `--height`, `--out` | `MUJOCO_GL=egl`; GPU recommended |
| `crowd demo viser` | Serve the plaza in the Viser browser viewer | `--graph`, `--device`, `--humans`, `--num-envs`, `--host`, `--port` (8080), `--lidar-stride`, `--frame-rate`, `--threads`, `--verbose` | CPU or GPU |
| `crowd demo corridor` | Primitive-human corridor fixture with physics and LiDAR; viewer or headless scan export | `--viewer auto\|native\|viser\|none`, `--num-envs`, `--num-humans`, `--seed`, `--moving`, `--device`, `--steps`, `--capture-every`, `--output-dir`, `--render`; `crowd demo corridor plot --scan-file <scans.npz>` | CPU or GPU |

Guides: [demos](usage/demos.md), [corridor](usage/corridor.md).

## Scripts outside the CLI

| Script | Does |
|---|---|
| `scripts/bootstrap.sh` | Clone pinned third-party sources, `uv sync`, download SOMA-X ([install](install.md)) |
| `scripts/fetch_artifacts.sh` | Copy run artifacts with sha256 checks; download checkpoints ([data](data.md)) |
| `scripts/env.sh` | Environment for every shell (source it) |
| `scripts/smoke/{ardy_smoke,kimodo_smoke,profile_ardy}.py` | Model load and latency smoke tests, ARDY profiler (GPU, checkpoints) |
| `scripts/slurm/*.sbatch` | Cluster jobs wrapping the commands above ([Slurm](slurm.md)) |
| `scripts/check_memory.py`, `scripts/index_runs.py` | Checks for the research memory and its run log |
