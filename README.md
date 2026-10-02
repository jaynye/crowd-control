# crowd-control

Stochastic, reactive human crowds for training a LiDAR-conditioned humanoid traversal
policy in [mjlab](mjlab/). Generative motion models
([ARDY](https://github.com/nv-tlabs/ardy), [Kimodo](https://github.com/nv-tlabs/kimodo))
produce an offline motion library; a stochastic motion graph stitches it online for
thousands of humans; an intent and steering layer decides where each human goes.

## Quickstart (Linux x86_64 with an NVIDIA GPU)

```bash
git clone <this repository> crowd-control && cd crowd-control
scripts/bootstrap.sh                 # pinned kimodo and ardy clones, uv sync, SOMA-X assets
source scripts/env.sh                # ARDY on PYTHONPATH, offline weights, managed Python
scripts/fetch_artifacts.sh --from <user>@<cluster>:/coc/flash7/jni66/crowd-control
.venv/bin/python -m pytest           # unit and golden tests, CPU
.venv/bin/crowd behavior evaluate --config configs/behavior/baseline.json
```

`fetch_artifacts.sh` copies the saved motion graph, which is all behavior work needs.
Model checkpoints (`scripts/fetch_artifacts.sh --checkpoints`) are needed only to
generate new motion. Full instructions: [docs/install.md](docs/install.md).

## Where things are

| Path | Holds |
|---|---|
| [`docs/`](docs/README.md) | How to install, run and develop: CLI, architecture, usage guides |
| `src/crowdmotion/` | The project package and the `crowd` command line |
| `mjlab/` | Our copy of the mjlab simulator, with our sensor changes |
| `configs/` | JSON configs for behavior evaluation, graphs, libraries and benchmarks |
| `scripts/` | Bootstrap, artifact fetching, environment, smoke tests, Slurm jobs |
| `tests/` | Unit, golden (bit-exact) and GPU tests |
| `memory/` | Research memory: status, design rationale, decisions, experiment results, run log |
| [`AGENTS.md`](AGENTS.md) | Binding constraints for agents (and people) working in this repository |
| `third_party.toml`, `artifacts.toml` | Pinned third-party sources; large artifacts outside git |

`third_party/`, `checkpoints/`, `runs/` and `logs/` are local and not tracked.
