# crowd-control docs

How to install, run and develop the project code (`src/crowdmotion/`, the `crowd`
command line, `scripts/`, `configs/`). Research notes, results and decisions are kept
separately and are not needed to run anything here.

## Start here

| Doc | Covers |
|---|---|
| [Install](install.md) | Prerequisites, clone, `scripts/bootstrap.sh`, environment check, artifacts, a first run; CPU-only notes |
| [Third-party sources](third-party.md) | kimodo, ardy and SOMA-X pins (`third_party.toml`), how each is wired, upgrading; mjlab as our copy and merging upstream releases |
| [Data and checkpoints](data.md) | Model weights, SOMA-X assets, run artifacts (`artifacts.toml`), the gated text encoder |
| [Command line](cli.md) | Every `crowd` command and its options |
| [Architecture](architecture.md) | Module map and layers, data flow, the controller, motion sources and graph cost terms |
| [Development](development.md) | Tests and markers, golden outputs, the layering rule, lint, extending the runtime |
| [Slurm](slurm.md) | Cluster-only jobs and the lab's scheduling conventions |

## Usage guides

| Guide | Covers |
|---|---|
| [Behavior iteration](usage/behavior-iteration.md) | Graph audit report, plaza evaluation, replay videos, paired comparison, iteration sessions |
| [Library and graph](usage/library-and-graph.md) | Offline ARDY library generation, the motion graph, saved graphs, graph evaluation |
| [Benchmarks](usage/benchmarks.md) | Online ARDY/Kimodo throughput and continuation quality: backends, procedures, metric definitions |
| [Demos](usage/demos.md) | The mjlab plaza with graph-driven humans and LiDAR: video export and the Viser viewer, locally or through an SSH tunnel |
| [Corridor fixture](usage/corridor.md) | Primitive humans with physics and LiDAR in an enclosed corridor |

## Conventions

- Run commands from the repository root after `source scripts/env.sh`. Examples write
  `crowd`; use `.venv/bin/crowd` (or `uv run crowd`) unless `.venv/bin` is on your
  `PATH`.
- Commands that write results create a fresh `runs/<UTC>_<name>/` unless given `--out`,
  and never overwrite an existing directory. `runs/` is local and not tracked in git.
- Motion coordinates follow ARDY: Y-up, meters, heading 0 faces +Z, positive yaw turns
  toward +X. MuJoCo scenes are Z-up: `(x, y, z)_mujoco = (x, -z, y)_motion`.
