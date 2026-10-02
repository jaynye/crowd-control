# crowd-control agent guide

crowd-control builds stochastic, reactive human crowds for training a LiDAR-conditioned
humanoid traversal policy, targeting 4,096 parallel mjlab environments with 4–16 humans
each. The 2026-09-30 benchmark showed the generative motion backends (ARDY, Kimodo)
cannot run online at that scale, so they generate motion **offline**. The library is
stitched online through a stochastic motion graph (or motion matching), or baked per
episode if the policy is trained by imitation instead of RL.

**Start here:** read [memory/core/status.md](memory/core/status.md) for active work,
constraints and next steps. Then check the [open decisions](memory/decisions/README.md)
that touch your task. The user's plan of record is
[memory/core/handoff-2026-09-30.md](memory/core/handoff-2026-09-30.md). How to install
and run everything: [docs/README.md](docs/README.md).

The repository is a git repository. Work runs on a local Linux machine with an NVIDIA GPU
by default; the Slurm cluster is optional (see [Slurm](#slurm)). Commit or push only when
the user asks.

## Where information belongs

| Location | Holds |
|---|---|
| `AGENTS.md` (this file) | Binding code, environment and compute constraints |
| [README.md](README.md), [docs/](docs/README.md) | Human-facing: install, third-party setup, data, architecture, `crowd` commands, usage |
| [memory/README.md](memory/README.md) | Router for research memory, with the categories below |
| [memory/core/](memory/core/README.md) | In-scope knowledge: status, plan, design and implementation rationale, settled decision ledger, conclusions |
| [memory/decisions/](memory/decisions/README.md) | Questions that are still undecided, one file each; archived once resolved |
| [memory/experiments/](memory/experiments/README.md) | Dated experiment results, kept while core or decisions cite them |
| [memory/knowledge/](memory/knowledge/README.md) | External references: upstream repositories (ARDY, Kimodo, SOMA, mjlab), papers, lessons |
| [memory/journal/](memory/journal/README.md) | Dated, ephemeral notes for in-flight non-experiment tasks |
| [memory/log/runs.md](memory/log/runs.md) | One row per run directory: date, kind, command, outcome |
| [memory/archive/](memory/archive/README.md) | Retired notes, dated, historical |
| `src/crowdmotion/` | Project package (layers in its `__init__.py` and `docs/architecture.md`) |
| `configs/` | Run configs: `behavior/`, `graph/`, `library/`, `bench/` (historical A40 runs) |
| `scripts/` | `env.sh`, `bootstrap.sh`, `fetch_artifacts.sh`, memory checks, `smoke/`, `slurm/` |
| `tests/` | `unit/` (CPU), `golden/` (bit-exact outputs), `gpu/` (CUDA and checkpoints) |
| `mjlab/` | **First-party** copy of mujocolab/mjlab (ledger D13), upstream base in `third_party.toml` |
| `third_party.toml`, `third_party/` | Pinned kimodo, ardy, SOMA-X; clones are fetched by `scripts/bootstrap.sh`, not tracked |
| `artifacts.toml` | Large run outputs (graph, library) with sha256; `scripts/fetch_artifacts.sh` copies them |
| `runs/` | Outputs, one UTC-stamped directory per run; never overwritten; not tracked |
| `checkpoints/` | Model weights (`CROWD_CHECKPOINTS`); not tracked |

Binding constraints go here, research rationale and conclusions in `memory/core`,
measured results in `memory/experiments`, transient progress in the journal. Link
canonical notes rather than copying them. Preserve historical evidence and label it as
historical when it is no longer current.

**Docs and code are self-contained.** `docs/`, `src/`, `scripts/`, `tests/` and `configs/`
cite only code, configs, data paths and other docs: never `memory/`, and never a
specific `runs/` directory (`artifacts.toml` maps artifact names to run paths). Memory may
cite docs and code.

## Code rules

- **One crowd loop.** Library generation, the quality benchmark, the behavior evaluator
  and mjlab's `GraphCrowd` all step humans through `behavior.controller.CrowdController`.
  Put a new behavior rule in a command source, planner, motion source or graph cost term,
  never in one consumer, so evaluation and simulation cannot diverge.
- **Layers.** Imports point down: L0 `core`; L1 `graph`, `backends`, `metrics`,
  `sim.scene`, `sim.capsules`; L2 `behavior`; L3 `iterate`, `library`, `bench`,
  `sim.mjlab`, `viz`; L4 `cli`. Only `sim.mjlab` imports mjlab at module level and only
  `backends` imports ARDY or Kimodo. `tests/unit/test_layering.py` enforces this.
- **Goldens.** `tests/golden/expected/` pins today's outputs bit for bit (CPU, one
  thread). A refactor must keep them matching. Never re-capture them to make a change
  pass. A deliberate behavior change needs the user's agreement and a new baseline,
  recorded in the ledger, before expected files change.
- **Randomness.** Per-human streams come from `core.rng`; seed formulas are named schemes
  in `human_seeds`. `PlazaBehavior` still shares one generator (kept for the baselines);
  switching it to per-human streams is a planned behavior change.
- **Commands and outputs.** New tools are `crowd` commands (`src/crowdmotion/cli.py`)
  that write a fresh `runs/<UTC>_<name>/` through `core.rundir.RunDir` (status, config,
  environment, git commit and diff). Configs reject unknown keys: typed dataclasses via
  `core.config.from_dict`, flat library and benchmark cases via `check_keys`
  (`GraphConfig` stays lenient so old saved graphs load; making it strict is Phase 11).
- **mjlab is ours.** Keep simulator changes general (sensors, viewers) and record them
  in `mjlab/docs/source/changelog.rst`; crowd-specific code belongs in
  `crowdmotion.sim.mjlab`. Its upstream history is outside this repository; merging an
  upstream release is described in `docs/third-party.md`.
- **Third-party clones** (`third_party/kimodo`, `third_party/ardy`): do not edit; wrap
  them from `crowdmotion.backends`. Change a pin only in `third_party.toml`.

## Memory workflow

- **Multi-step task:** open `memory/journal/YYYY-MM-DD-topic.md` with the plan and update it
  as you work. Retire it when the task ends, after promoting what lasts.
- **Experiment or benchmark:** open `memory/experiments/YYYY-MM-DD-topic.md` with a
  `Status: running` line, the plan and job IDs, instead of a journal entry. Rewrite it as
  the result when the runs finish.
- **Something needs the user or an experiment to settle:** open a file in
  `memory/decisions/`. When it is settled, record the outcome in the
  [ledger](memory/core/design/decision-ledger.md) and move the file to `memory/archive/`.
- **After an extensive implementation or experiment, update core before finishing:**
  refresh `memory/core/status.md`; revise the affected design note; state each durable
  result as a one- to three-line conclusion in the core note it affects, linking its
  experiment note; update this file for new invariants.
- **Every new run directory** gets a row in [`memory/log/runs.md`](memory/log/runs.md):
  `scripts/index_runs.py` drafts it from `run.json`; write the outcome by hand. Cite runs
  by ID in backticks (`runs/<UTC>_<name>/...`), never as Markdown links: `runs/` is not
  in git.
- **Retire** a journal entry, resolved decision or unreferenced experiment note with
  `git mv` into [`memory/archive/`](memory/archive/README.md), keeping its dated name.
  Never delete memory notes.
- **Before finishing,** run `.venv/bin/python scripts/check_memory.py --check` (it also
  checks run IDs against the run log, and unlogged run directories).
- **`status.md` is a snapshot.** Replace its sections; do not append a log.

## Environment

- One uv workspace at the root: the `crowd-control` package (editable, `src/`) and
  `mjlab` as a member. Run `uv sync` **only from the root**: inside `mjlab/` it does an
  exact member sync and uninstalls kimodo and py-soma-x. `uv run` inside `mjlab/` is
  fine. The managed CPython lives in `.uv-python/` (`UV_PYTHON_INSTALL_DIR`).
- torch is 2.9.0 (PyPI cu128 wheel) and warp 1.14.0. After any lock change, verify both
  are unchanged. The root depends on plain `mjlab`, not `mjlab[cu128]`.
- `kimodo` is a non-editable path dependency on `third_party/kimodo`: after changing that
  clone, run `scripts/bootstrap.sh --reinstall-kimodo` (`uv sync --reinstall-package
  kimodo` plus a CMake fallback through uvx when the system has no CMake). Never install
  `kimodo[demo]` (its viser fork conflicts with mjlab's viser).
- **ARDY runs from source**, not as a dependency. Its `numpy<2` and `transformers==5.8.1`
  pins conflict with this env. `scripts/env.sh` puts `third_party/ardy` on
  `PYTHONPATH`. The root pyproject adds only `vector-quantize-pytorch`. ARDY's
  `motion_correction` extension is the same code Kimodo installs, so it reuses that
  build. Do not add ardy to the lock or build its extension into this venv.
- Weights are local and offline: `source scripts/env.sh` sets `HF_HUB_OFFLINE=1`,
  `CHECKPOINT_DIR` (Kimodo) and `CHECKPOINTS_DIR` (ARDY) under `$CROWD_CHECKPOINTS`
  (default `<root>/checkpoints`). Both loaders resolve `<dir>/<display name>`
  (`Kimodo-SOMA-RP-v1.1`, `ARDY-Core-RP-20FPS-Horizon{8,40}`).
  `scripts/fetch_artifacts.sh --checkpoints` downloads them under those names.
- Package code does not need `env.sh`; ARDY and the checkpoints do.
- Do not load the ARDY `soma` nickname: no SOMA checkpoint has been released.
- **CPU runs:** set PyTorch to one thread (or a few) on shared hosts. The evaluator,
  Viser demo and goldens do this by default.
- Never call `soma.setup_warp_for_ddp()` in an mjlab process: it disables PyTorch's CUDA
  caching allocator for the whole process.
- **Text encoder:** both models use LLM2Vec on the gated
  `meta-llama/Meta-Llama-3-8B-Instruct`, which currently returns 403 for the HF account.
  Do not route around the gate with an ungated mirror without the user's explicit
  approval ([decision](memory/decisions/text-encoder-access.md)).

## Project constraints (user direction, 2026-09-30 handoff)

- **Scale and backend (user, 2026-09-30):** 4,096 environments, 4–16 humans each. ARDY
  and Kimodo run offline only; do not put generator calls in the training loop. Small
  online runs for evaluation are fine.
- Order of work: benchmark ARDY and Kimodo (done), choose a backend (done: offline),
  build the offline library and motion graph, then implement the simulator behavior model. Do not begin by training a
  new motion prior or building a custom human rig. Keep MotionBricks outside the initial
  comparison.
- Separate intent, local steering and body-motion generation. Stochastic behavior
  decisions stay independent of the motion backend.
- **Motion graph and behavior (user, 2026-10-01):** the motion graph of offline
  prior-generated clips stitched through keypoint state is the chosen runtime.
  Ordinary travel should be mostly forward, with turning motion for direction changes;
  lateral/backward motion is deliberate, including abrupt close avoidance. Measure
  travel relative to body facing rather than treating curved forward paths as sidesteps.
- Benchmarks measure **usable** motion: count only newly generated frames (ARDY returns
  history + horizon), apply the consumed fraction `u`, exclude model loading, GUI and
  export from steady-state timing, and report startup separately. Label isolated
  generation capacity as such; it is necessary, not sufficient, for end-to-end speed.
- Do not invent pass thresholds. If slowdown and response-delay limits are unspecified,
  report achieved performance and the break-even simulator speed.
- Record GPU model, driver, CUDA/PyTorch versions, repository commits, checkpoint
  revisions, precision and acceleration settings with every measurement.
- Per-human random streams must be independent of batch order and composition.

## Slurm

The cluster (GT skynet) is optional: the local machine runs everything. Cluster-only job
files are in `scripts/slurm/`; see `docs/slurm.md`. The pre-migration checkout stays at
`/coc/flash7/jni66/crowd-control` with the run outputs `artifacts.toml` points at, and
mjlab's upstream git directory at `/coc/flash7/jni66/mjlab-upstream.git`.

Lab conventions (shared with `~/flash/hrc`, user direction 2026-09-28):

- Prefer L40S, A40 or A5000. Never Titan X or 2080 Ti. Exclude `bishop`.
- Submit with `--partition=rl2-lab,overcap --qos=short`. `rl2-lab` is not preemptible
  and caps jobs at 4 h; `overcap` is preemptible.
- Compare speed only within one GPU type, and record the type. The 2026-09-28 Kimodo
  smoke ran on an L40S; the 2026-09-30 benchmark runs on A40.
- Memory is not a scheduled resource (`--mem` is not enforced). Check `FreeMem` before
  stacking jobs on one node.
