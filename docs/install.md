# Install

Supported platform: **Linux x86_64 with an NVIDIA GPU** and a driver recent enough for
CUDA 12.8, the CUDA build of the PyPI torch 2.9.0 wheel. The lock file resolves only for Linux x86_64. Without a GPU
most of the project still runs; see [CPU-only machines](#cpu-only-machines).

## Prerequisites

- `git`, `rsync` and `curl`.
- [uv](https://docs.astral.sh/uv/getting-started/installation/); the cluster uses
  0.12.7. uv installs the managed CPython 3.13 into `.uv-python/`; no system Python is
  needed.
- A C++17 compiler and `make` (on Debian/Ubuntu: `build-essential`). Kimodo builds a
  small C++ extension. CMake is used if
  installed; otherwise `scripts/bootstrap.sh` runs the PyPI `cmake` through `uvx`.
- About 8 GB of disk: the environment (about 7 GB), third-party clones and assets
  (about 0.5 GB) and the saved motion graph (0.44 GB). Checkpoints add 2.6 GB, and the
  motion library another 2 GB, if you fetch them.

## 1. Clone and bootstrap

```bash
git clone <this repository> crowd-control
cd crowd-control
scripts/bootstrap.sh
```

`bootstrap.sh` reads [`third_party.toml`](../third_party.toml) and:

1. clones `kimodo` and `ardy` into `third_party/` at their pinned commits;
2. runs `uv sync` from the repository root. This builds the environment in `.venv/`,
   installs `crowdmotion` (editable) with its `crowd` command, our `mjlab` (editable,
   a workspace member) and Kimodo (built from `third_party/kimodo`);
3. checks that torch is 2.9.0 and warp is 1.14.0;
4. downloads the SOMA-X v0.3.2 assets into `third_party/soma-x/`.

It is safe to rerun. Existing clones are left alone; a clone at a different commit
produces a warning. `scripts/bootstrap.sh --no-sync` only clones. Details of each
dependency: [third-party sources](third-party.md).

## 2. Environment

```bash
source scripts/env.sh
```

Source it in every shell. It:

- puts `third_party/ardy` on `PYTHONPATH` (ARDY runs from source; it is not installed);
- sets `HF_HUB_OFFLINE=1` and the checkpoint directories `CHECKPOINTS_DIR` (ARDY) and
  `CHECKPOINT_DIR` (Kimodo), under `$CROWD_CHECKPOINTS` (default `checkpoints/` in the
  repository);
- sets `UV_PYTHON_INSTALL_DIR=.uv-python`.

Rules that keep the environment intact:

- **Run `uv sync` only from the repository root, never inside `mjlab/`.** Inside
  `mjlab/` it does an exact sync of that member and uninstalls Kimodo and py-soma-x.
  `uv run` inside `mjlab/` is fine.
- After any change to `pyproject.toml` or `uv.lock`, check that torch is still 2.9.0 and
  warp 1.14.0:
  ```bash
  .venv/bin/python -c "import torch, warp; print(torch.__version__, warp.__version__)"
  ```
- After changing the Kimodo clone, rebuild it: `scripts/bootstrap.sh --reinstall-kimodo`
  (applies the CMake fallback; a bare `uv sync --reinstall-package kimodo` needs `cmake`).
- Never install `kimodo[demo]`: its viser fork conflicts with mjlab's viser.

## 3. Artifacts

Behavior work needs only the saved motion graph (0.44 GB). Copy it from the cluster
checkout, where it was built; the script checks its sha256 against
[`artifacts.toml`](../artifacts.toml):

```bash
scripts/fetch_artifacts.sh --from <user>@<cluster login host>:/coc/flash7/jni66/crowd-control
scripts/fetch_artifacts.sh --list       # every artifact, size, present or missing
```

To generate motion (library generation, benchmarks, GPU tests), also download the
ARDY and Kimodo checkpoints from Hugging Face:

```bash
scripts/fetch_artifacts.sh --checkpoints
```

What each artifact is, and the rsync alternative for checkpoints: [data](data.md).

## 4. Check the install

```bash
.venv/bin/python -m pytest                       # unit and golden tests on CPU
source scripts/env.sh && .venv/bin/python -m pytest tests/gpu   # GPU adapter tests (CUDA + checkpoints)
.venv/bin/crowd --help
.venv/bin/crowd behavior evaluate --config configs/behavior/baseline.json
MUJOCO_GL=egl .venv/bin/crowd behavior render --run runs/<UTC>_behavior_eval --seed 0
```

- The golden tests compare outputs bit for bit against `tests/golden/expected/`; those
  that need the saved graph skip without it. The two mjlab golden cases compile Warp
  kernels on their first run (one to two minutes on CPU); later runs use Warp's cache.
- The evaluation (3 seeds × 20 s × 16 humans) takes well under a minute on one CPU
  thread and writes `runs/<UTC>_behavior_eval/`. See [behavior iteration](usage/behavior-iteration.md).
- If a golden test fails on a new machine, see
  [development](development.md#golden-outputs) before changing anything.

## CPU-only machines

Everything except generation and the GPU benchmarks runs on CPU:

| Works on CPU | Needs CUDA |
|---|---|
| Unit and golden tests | `tests/gpu` (adapter parity) |
| `crowd behavior evaluate`, `compare`, `crowd graph report` | `crowd bench throughput`, `crowd bench sim-speed` |
| `crowd behavior render` and `crowd demo video` with `MUJOCO_GL=osmesa`, if OSMesa is installed (slow) | `crowd library generate` (ARDY runs on CPU, but far too slowly for a library) |
| `crowd demo viser --device cpu` (about real time at 30 viewer FPS on 4 cores) | `crowd graph build` (works on CPU, but the kNN build is very slow) |

On shared or busy hosts keep PyTorch to one thread (`torch.set_num_threads(1)`, or
`--threads 1` where a command offers it). With the default thread count, a 20 ms mjlab
plaza step took 500–800 ms on a loaded login node.

## Troubleshooting

- **`uv sync` fails to build kimodo:** install a C++ compiler; check that
  `third_party/kimodo` exists (`scripts/bootstrap.sh --no-sync`).
- **`ModuleNotFoundError: ardy`:** `source scripts/env.sh` in this shell.
- **Checkpoint not found:** the loaders resolve `<dir>/<name>`, for example
  `$CHECKPOINTS_DIR/ARDY-Core-RP-20FPS-Horizon40`. Run
  `scripts/fetch_artifacts.sh --checkpoints`, or set `CROWD_CHECKPOINTS`.
- **Renders fail with a GL error:** set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa`
  (CPU with OSMesa installed).
