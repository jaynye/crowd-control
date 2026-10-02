# Development

How to test, check and extend the code. The structure itself is described in
[architecture](architecture.md).

## Tests

```bash
.venv/bin/python -m pytest                    # unit tests and fast goldens (CPU; tests/gpu only when named)
.venv/bin/python -m pytest tests/unit         # unit tests only (seconds; no data needed)
.venv/bin/python -m pytest -m golden          # golden outputs only
.venv/bin/python -m pytest -m slow            # real ARDY/Kimodo goldens on CPU (minutes; env.sh + checkpoints)
.venv/bin/python -m pytest tests/gpu          # adapter parity against upstream (CUDA + checkpoints)
```

| Directory | What | Needs |
|---|---|---|
| `tests/unit/` | Core utilities, backend registry, CLI table, graph runtime and cost terms, crowd controller, behavior metrics and evaluation, iteration sessions (mocked scheduler), corridor fixture, layering | Nothing (synthetic library in `tests/synthetic.py`) |
| `tests/golden/` | Bit-exact outputs against `tests/golden/expected/` | The saved graph for some cases (skipped without it) |
| `tests/gpu/` | ARDY and Kimodo adapters against upstream's own calls on the same GPU | CUDA, `source scripts/env.sh`, checkpoints |

Markers (`pyproject.toml`): `gpu` (skipped without CUDA or checkpoints), `slow` (deselected by
default), `golden`. Tests set PyTorch to one thread (`CROWD_TEST_THREADS` overrides).
The mjlab cases compile Warp kernels on first use; later runs use the cache.

## Golden outputs

`tests/golden/harness.py` recomputes a fixed set of outputs with the current code and
compares them **bit for bit** with manifests captured from the code before the
2026-10 reorganization. It is what proves a refactor changed no behavior.

| Case | Pins |
|---|---|
| `evaluate`, `compare`, `replay` | Behavior evaluator traces and metrics on the saved graph; the paired comparison; replay scene and capsule poses |
| `runtime_step`, `audit` | `GraphRuntime.step` on the saved graph in four configurations (saved, lookahead, no correction or inertialization, greedy); the graph audit |
| `graph_build` | kNN build, tables and save/load on a synthetic library |
| `quality_graph`, `quality_online` | Quality-benchmark rollouts and metrics for a synthetic graph and for deterministic fake ARDY/Kimodo backends |
| `library_fake`, `throughput_fake` | Library streams, random commands and throughput populations with fake backends |
| `mjlab_plaza`, `corridor` | mjlab plaza and corridor on CPU: compiled-model hash, mocap poses or joint positions, LiDAR, command-term state |
| `ardy_cpu`, `kimodo_cpu` (`slow`) | Real ARDY and Kimodo on CPU: library streams, a quality rollout, throughput calls |

Each manifest records, per array, dtype, shape, sha256 and summary statistics (sum,
absolute sum, min, max), and stores JSON outputs in full. Every case has a fast variant
(the pytest default) and a full one:

```bash
.venv/bin/python tests/golden/harness.py check --out /tmp/goldens            # full set
.venv/bin/python tests/golden/harness.py check --fast --cases evaluate --out /tmp/goldens
.venv/bin/python tests/golden/harness.py check --slow --cases ardy_cpu --out /tmp/goldens
```

`--out` receives the recomputed arrays (`<case>.npz`) and a report, for diagnosing a
mismatch. The harness runs on CPU with one thread; only its "API adapters" section may
change when code moves.

Rules:

- **Never re-capture `tests/golden/expected/` to make a test pass.** A mismatch means
  behavior changed: find the cause first. Re-capturing (`harness.py capture`) is for a
  deliberate, recorded behavior change only, in its own change, with the old and new
  numbers compared.
- **Another machine.** Results are bit-exact on one CPU type. If a different instruction
  set changes only float rounding, the summary statistics in the manifest show it (equal
  shapes, statistics equal to many digits). Record the tolerance and give that machine
  its own captured set rather than overwriting the reference.
- **GPU results** are bit-exact only on the same GPU model; GPU checks compare two code
  versions on one card.

## Layering and lint

- `tests/unit/test_layering.py` fails if a module imports from a higher layer, if a
  module outside `sim.mjlab` imports mjlab at module level, or if a module outside
  `backends` imports ARDY or Kimodo ([architecture](architecture.md#layers)). A new
  subpackage needs an entry in its `LAYERS` table.
- Lint with `.venv/bin/ruff check src tests` (configured in `pyproject.toml`: two-space
  indents, 120 columns).
- Docs and code never cite `memory/` or a specific run directory; reference code,
  configs, artifacts by name, and other docs. `scripts/check_memory.py --check` enforces
  this.

## Extending the runtime

### Adding a graph cost term

`graph/costs.py` (for example a facing or turn-commitment cost):

1. Add the weight to `GraphConfig` (`graph/config.py`) with a default of `0.0`.
2. Write a class with `name`, `weight_key` (that field), `skip_when_zero = True` and
   `__call__(ctx) -> [H, k]`, reading what it needs from `CandidateContext` (candidate
   edge ids, their placed root paths, the request, the graph tables).
3. Append an instance to `TERMS`. Appending keeps earlier sums, and so every existing
   result, bit-identical; a zero weight skips the term entirely.
4. Unit-test it in `tests/unit/test_graph_costs.py`, then evaluate a candidate config
   with `graph_overrides` setting the weight
   ([behavior iteration](usage/behavior-iteration.md)).

The term's raw value is returned in `CostTotal.values` under its name for diagnostics.

### Adding an intent state or steering rule

Intent (where a human wants to go, waits, turn commitments) lives in
`behavior/intent.py` (`GoalIntent`); collision avoidance and walls in
`behavior/steering.py` (`LocalSteering`). `PlazaBehavior` combines them as the command
source of both the evaluator and mjlab, so a change reaches both.

`GoalIntent` and `LocalSteering` share one generator and draw in a fixed order to
reproduce existing baselines. A new random draw changes every later draw: treat it as a
behavior change, evaluate it against a fresh baseline, and do not expect the goldens to
match. New randomness should come from per-human streams (`core.rng`).

### Other extension points

- **A command source:** implement `command(t, kinematics, groups) -> (speed, heading)`
  (`behavior/commands.py`).
- **A motion source:** implement `reset`, `step`, `commit` and `path_frames`
  (`behavior/sources.py`); the controller needs nothing else.
- **An online generator:** a config dataclass and an adapter with `generate(request)`,
  plus a `BACKENDS` row in `backends/registry.py`.
- **A command:** a module with `main(argv)`, plus a row in `COMMANDS` in `cli.py`. Use
  `core.rundir.RunDir` for its output directory and provenance.

## Running on CPU

- Use one PyTorch thread on shared machines (`torch.set_num_threads(1)`; commands offer
  `--threads`). More threads oversubscribe busy hosts and slow mjlab steps by an order
  of magnitude.
- The evaluator, comparison, graph report, tests and goldens are CPU-first.
- mjlab runs on CPU through MuJoCo-Warp's CPU backend; the first run of a scene compiles
  kernels.
- Never call `soma.setup_warp_for_ddp()` in an mjlab process: it disables PyTorch's CUDA
  caching allocator for the whole process.
