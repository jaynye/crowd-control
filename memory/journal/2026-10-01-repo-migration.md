# Repository migration to git and local iteration

Opened 2026-10-01. State: **Phases 0–9 done (2026-10-01); waiting for the user's git init,
initial commit and local bring-up (Phase 10).** Retire this entry after Phase 10.

## Request (user, 2026-10-01)

Move the repo to a local machine for motion-graph behavior iteration instead of Slurm,
because GPU waits stall the loop. Before pushing to git:

- organize the code at the functional level (consolidate code, move functions, add
  classes), not only move files;
- `docs/` holds human-facing instructions: install, cloning third-party repos, usage;
- `memory/` holds agent- and human-facing memory: knowledge base, implementation and
  architecture details, dated logs and journal entries of topics and runs;
- exclude `runs/` and third-party git repos.

First deliverable: an artifact with the current organization and the post-migration
plan: https://claude.ai/artifact/XSpHps2aJht4tSiXrPxfrP (version 3; version 4 adds the outcome, §0).

## User answers (2026-10-01)

1. Local machine: **Linux + NVIDIA GPU**.
2. mjlab: "We can have our own clone of mjlab (i.e. mjlab should not be considered
   thirdparty)." Recorded as ledger D13. The plan reads this as "track mjlab's files in
   our repo and move `mjlab/.git` aside", which drops upstream history. A fork used as
   a submodule, or `git subtree`, would keep it. The user has been asked which one they
   want. **Until Phase 1 executes, the AGENTS.md rule "do not reset or clean" mjlab still
   applies.**
3. Git remote: "I can create the repo and create the initial commit myself." No
   `git init`, commit or push by the agent.
4. Large artifacts: **rsync from the cluster**, with sha256 values in a tracked manifest.

Decisions 5 (`src/` layout with a `crowd` CLI) and 6 (keep and port the online benchmark
harness) proceed with the recommendation unless the user objects.

## Plan summary (details in the artifact)

- Phase 0: snapshot, export mjlab diff, CPU golden outputs (evaluate on `graph.pt`,
  `audit_graph`, `GraphRuntime.step`, synthetic quality metrics, mjlab plaza mocap).
- Phases 1–8: behavior-preserving refactor, each gated on bit-exact goldens.
  - Third-party bootstrap for kimodo, ardy and SOMA-X; mjlab first-party.
  - Packaging and `core/`.
  - Backend registry.
  - `CrowdController` with motion sources replacing six rollout copies.
  - Graph split with cost terms.
  - Shared plaza scene and viz.
  - Local executor for iteration sessions.
- Phase 9: docs and memory restructure.
  - `docs/install.md`, `third-party.md`, `data.md`, `architecture.md`.
  - `memory/log/runs.md` and `memory/archive/`.
  - Retired notes rescued from `runs/`; `check_memory.py` made git-aware.
- Phase 10: the user commits and pushes. Local bring-up, then GPU bit-exact checks on
  the local card.
- Phase 11: behavior changes against a new baseline.
  - Per-human steering streams, strict `GraphConfig`.
  - Then committed turns and a facing cost.

## Open questions for the user

- mjlab history: plain tracked files (history dropped), fork as submodule, or subtree.
- Initial-commit timing: right after Phase 1, so that later phases become reviewable
  commits, or at the end, with snapshot tarballs for rollback until then.

## Execution (user approval, 2026-10-01)

User: "Execute and implement the migration in full (steps 0-9)", with up to two
implementation subagents (Opus 5.5) and two adversarial reviewers of the other provider
(Codex, gpt-6.1-sol, read-only).

The two open questions were not answered; the plan's defaults apply and are reversible:

- mjlab history: plain tracked files. `mjlab/.git` is moved aside, not deleted, so a
  fork-as-submodule or `git subtree` can still be adopted.
- No commits by the agent: per-phase source snapshots under `runs/` for rollback, and
  `diff -ruN` against them for review. No shadow git repository.

Rules for every agent during the refactor: never run `uv sync` except the orchestrator
in Phase 1/2; never edit expected golden values; on a golden mismatch, stop and report.

### Phase log

- Phase 0 (done 2026-10-01): snapshot `runs/20261001T225208Z_pre_migration_snapshot/`
  (`source.tar` without runs/.venv/third-party clones, extracted `tree/`, `mjlab.diff`
  against c2e1e0640, `mjlab_untracked.tar`, `pins.json`). Goldens: `tests/golden/harness.py`
  with expected manifests in `tests/golden/expected/` (14 cases: evaluate, audit,
  runtime_step, graph_build, quality_graph, quality_online (fake ARDY/Kimodo),
  library_fake, throughput_fake, mjlab_plaza, corridor, replay, compare, and slow
  ardy_cpu/kimodo_cpu on the real models). Full and fast sets reproduce bit for bit in a
  second process (`goldens/check1_*.log`). Kimodo on CPU: 45–65 s per 2-human call.
- Phase 1 (done): `third_party.toml`, `scripts/bootstrap.sh`, `artifacts.toml`,
  `scripts/fetch_artifacts.sh`, `.gitignore`. kimodo, ardy, soma-x moved to `third_party/`
  (root `.gitattributes` and `.cache/` moved into `third_party/soma-x/`). `mjlab/.git` moved
  to `/coc/flash7/jni66/mjlab-upstream.git` (usable with `--git-dir`). `checkpoints` is a
  symlink to `~/flash/checkpoints`; env.sh defaults to `<root>/checkpoints`. The login node
  has no cmake: bootstrap falls back to PyPI cmake through uvx; Kimodo's stale CMake build
  cache moved to the snapshot dir (`kimodo-build-pre-move`). torch 2.9.0, warp 1.14.0
  unchanged; fast and slow goldens match. Dry run: 708 files, 63.6 MB, none over 10 MB,
  no embedded repo, no secrets.
- Phase 2 (done): package at `src/crowdmotion` (uv_build, editable, `crowd` entry point);
  tests in `tests/{unit,gpu,golden}` with `tests/synthetic.py`; layering test report-only.
  Unit and fast goldens pass without env.sh.
- Phase 3 (done): `core/{rng,geometry,timing,paths,provenance,rundir,config,skeleton,trace}`,
  `behavior/planner.py`, `graph/` package (motion_graph.py, build.py, audit.py), metrics
  modules; `runtime.py`, `bench/{behavior,paths}.py`, `library/commands.py` removed.
  Snapshot `phase3_source.tar`. Fast goldens 12/12.
- Phase 4 (done): `backends/registry.py` (ArdyConfig, KimodoConfig, `config_from_case`,
  `make_backend`), `backends/text.py`, shared `tf32_matmul`/`denoise_autocast`; both
  adapters expose `generate(req)`, `path_frames`, `frames_per_call`, `skeleton_info()`,
  `configure(cfg)` (throughput sweeps reuse loaded weights). Harness fakes follow the protocol.
- Phase 5 (done): `behavior/{commands,intent,steering,plaza,kinematics,sources,controller}.py`;
  quality (online + graph in one `run_rollout`), library generation, the evaluator
  (`iterate/evaluate.py`) and GraphCrowd all use `CrowdController`. One aliasing bug found
  by the goldens (kinematics were views of the start tensors) and fixed by cloning in reset.
  Full + slow (real ARDY) goldens match.
- Phase 6 (done, subagent B): `graph/{config,tables,build,costs,runtime,audit}.py`, cost
  terms with flat weight keys, `format` field and `read_blob` compat loader;
  `configs/graph/default.json`. Goldens match (fast and full).
- Phase 7 (done except corridor, which B is moving): `sim/{scene,capsules}.py`,
  `sim/mjlab/{command,plaza,demo}.py`, `viz/{replay,report,stick,html}.py`, templates,
  `bench/{tables,sim_speed}.py`; migrated scripts deleted; CLI table complete. Fast goldens
  12/12 (mjlab model hash and replay model hash unchanged). Snapshot `phase7_source.tar`.
- Corridor and Phase 8 (done, subagent B): `sim/mjlab/corridor/` (mjlab keeps only the
  sensor edits: 5 files +125/-6 vs c2e1e0640, plus `tests/test_lidar_pattern.py`);
  `iterate/{session,executors}.py` (LocalExecutor default, SlurmExecutor);
  `scripts/slurm/*.sbatch` (15) and `scripts/smoke/`. Layering test now strict.
- Codex reviewer 1 (gpt-6.1-sol; the read-only bwrap sandbox fails on this host, so it ran
  with `-s workspace-write -c sandbox_workspace_write.network_access=true` in a scratch
  cwd, reading the repo by absolute path). Seven findings (`review1_output.md`), all
  addressed:
  1. `history: 0` used to pass the whole executed history (`[:, -0:]`); no config used it.
     Quality and library generation now reject history < 1 instead of silently differing.
  2. Root/provenance lookup outside a checkout: `CROWD_ROOT`, cwd search, clear errors.
  3. Quality/library honored a `device` key the old factories ignored: per-caller key
     allowlists (`config_from_case(case, keys, ...)`) restore the old key sets
     (also `to_soma77` for quality).
  4. Grouped stepping with `WindowVelocity` crashed: the controller refuses it up front.
  5. Resuming after a partial graph step desynced: the controller raises instead.
  6. Same-second default run names collided: `RunDir` appends `_2`, `_3`, ...
  7. Cost terms needed a separate `TERMS` edit: `@cost_term` registers them.
  Also fixed: `tests/gpu` adapters still used the old constructors.
- Phase 9 (done, subagent M): `docs/` (install, third-party, data, architecture, cli,
  development, slurm, usage/*), root `README.md`, `memory/core/design/runtime-architecture.md`,
  code-path second pass; orchestrator rewrote AGENTS.md and `status.md`.
- Codex reviewer 2 (end state; `review2` prompt and findings in the scratchpad, summary
  here). It cloned kimodo/ardy at the pins with `bootstrap.sh --no-sync`, built a copy from
  the tracked-file list, ran 90 tests and the baseline evaluation. Nine findings, all fixed:
  pytest default paths no longer include `tests/gpu` (and GPU tests skip without
  checkpoints); install docs require `make`; `fetch_artifacts.sh` refetches files that fail
  their hash (resume tested) and always reruns `hf download`; bootstrap's SOMA-X download
  ignores `HF_HUB_OFFLINE`; `scripts/bootstrap.sh --reinstall-kimodo` carries the CMake
  fallback; `mkdir -p` before checkpoint rsync; library and benchmark cases reject unknown
  keys (`check_keys`; every shipped config passes, `tests/unit/test_configs.py`);
  `check_memory.py` flags any concrete run ID in non-test code.
- Final gates: full and fast goldens 12/12 and slow ARDY/Kimodo 2/2 after both reviews;
  a copy built only from the tracked-file list passes unit + fast golden tests; final
  `pytest` 106 passed (unit + fast goldens). Every file in `tests/golden/expected/` dates
  from Phase 0 (18:56–19:11 local); none was re-captured. No shipped config uses
  `history: 0`. Final tracked-file list: `tracked_files.txt` in the snapshot dir
  (733 files, 63.7 MB, none over 10 MB). `mjlab/AGENTS.md` and `CLAUDE.md` gained a pointer
  to the root rules (upstream's workflows assume mjlab's own repository).
- Memory restructure (Phase 9a) delegated to a subagent that owns `memory/**`,
  `check_memory.py`, `index_runs.py`, the handoff move and `logs/`.

Golden commands (cluster, CPU, one thread):

    source scripts/env.sh
    CROWD_GRAPH=$PWD/runs/20261001T013254Z_motion_graph/graph.pt \
      .venv/bin/python tests/golden/harness.py check [--fast] [--slow] --out <dir>
