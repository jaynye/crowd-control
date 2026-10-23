# Archive

Retired notes, kept for history. Nothing here is current: each note holds as of its date,
and any lasting conclusion or decision it contained was promoted to
[core](../core/README.md), the [decision ledger](../core/design/decision-ledger.md) or
[knowledge](../knowledge/README.md) before it was retired. An archived note is never the
only home of a live decision or result.

## Rules

- **Retire** a journal entry, a resolved decision file or an unreferenced experiment
  note by moving it here (`git mv`), keeping or adding a `YYYY-MM-DD-` prefix. Fix the
  links that the move breaks (`scripts/check_memory.py --check` lists them).
- **Do not edit** archived notes, except to fix links. Their body text is verbatim.
- **Link the promoted note** from core and decisions; cite an archived note only as a
  historical source. Docs and code never cite memory.

Notes rescued on 2026-10-01 from places that git does not track carry a one-line
"Archived from" header naming their original path; their relative links were adjusted.

## Index

Oldest first.

| Note | What it was | Superseded by |
|---|---|---|
| [mjlab Slurm Viser and agentic crowds](2026-09-26-mjlab-slurm-viser-and-agentic-crowds.md) | Implementation note in the mjlab clone (`mjlab/memory/`): Viser over Slurm with an SSH tunnel, uv vs containers, an MCP control plane proposal for GPT-6 Astra | [knowledge/mjlab.md](../knowledge/mjlab.md) |
| [mjlab G1 crowd simulation plan](2026-09-28-mjlab-g1-crowd-simulation-plan.md) | Astra's G1 crowd plan (`mjlab/docs/plans/`): intent layer, LiDAR, contact, scene, training; primitive corridor fixture | Human embodiment: the Kimodo note below, then the motion graph ([crowd architecture](../core/design/crowd-architecture.md)) |
| [mjlab Kimodo stochastic humans](2026-09-28-mjlab-kimodo-stochastic-humans.md) | "Decisions online, bodies offline": Kimodo-baked library plus motion matching (`mjlab/docs/plans/`) | Ledger D11: the motion graph ([motion library](../core/design/motion-library.md)) |
| [Memory system and benchmark journal](2026-09-30-memory-and-benchmark.md) | Journal of the memory setup and the online-generation benchmark | [Motion backends](../core/design/motion-backends.md) |
| [Demand budget](2026-09-30-decision-demand-budget.md) | Decision file: throughput a backend must sustain | Ledger D09 |
| [Backend choice](2026-09-30-decision-backend-choice.md) | Decision file: which backend drives the humans | Ledger D10 |
| [Offline runtime](2026-10-01-decision-offline-runtime.md) | Decision file: how to stitch an offline library online (ARDY-tabulated graph proposal) | Ledger D11; open parts in [library generator](../decisions/library-generator.md) and [graph response delay](../decisions/graph-response-delay.md) |
| [Graph behavior report](2026-10-01-graph-behavior-report.md) | Journal of the graph audit report and the evaluation tooling | [Graph behavior audit](../experiments/2026-10-01-graph-behavior-audit.md) |
| [Repo migration journal](2026-10-01-repo-migration.md) | Journal of the move to git and local iteration: Phase 0 goldens, the behavior-preserving refactor (Phases 1–8), two Codex reviews, docs and memory restructure (Phase 9); Phase 10's commit and push (done by the user; its local GPU checks remain open in status) | [Runtime architecture](../core/design/runtime-architecture.md), ledger D13, [status](../core/status.md) (outstanding GPU checks, Phase 11), `docs/development.md` |
| [Behavior session journal](2026-10-01-behavior-session-journal.md) | Session journal of the bounded behavior iteration (`runs/20261001T061749Z_behavior_iteration/journal.md`) | [Bounded behavior iteration](../experiments/2026-10-01-graph-behavior-iteration.md) |
| [Codex batch iteration launcher](2026-10-04-codex-batch-iteration.md) | Implementation journal: automatic Codex startup inside a bounded Slurm allocation, CPU lifecycle checks and test-only submission validation | [Motion library](../core/design/motion-library.md), [batch setup](../../docs/slurm.md#start-codex-automatically-in-a-batch-job) |
| [Codex launcher fix](2026-10-04-codex-launcher-fix.md) | Journal: why batch job 4013182 did no work (`--sandbox` conflicts with `--approve-for-me`), the real-CLI preflight, `--model`/`--effort` defaults and the bubblewrap observation on sky1 | [Batch setup](../../docs/slurm.md#start-codex-automatically-in-a-batch-job), [status](../core/status.md), run log |
| [Rigid mesh humans](2026-10-06-rigid-body-geometry.md) | Implementation journal: rotation sidecar, Warp ray target and fused pose pass, mjlab ray-target hook, Viser geometry switch, tests and A40 timing | [Ledger D15](../core/design/decision-ledger.md), [body geometry benchmark](../experiments/2026-10-05-body-geometry-benchmark.md), [status](../core/status.md) |
| [Encounter design journal](2026-10-06-encounter-design.md) | Journal of the encounter behavior design and the turning-library delegation (2026-10-06) | [Encounter design](../experiments/2026-10-06-encounter-design.md), [turning library](../experiments/2026-10-06-turning-library.md), ledger D16 |
| [Encounter turning-graph comparison journal](2026-10-07-encounter-turns-comparison.md) | Journal of the comparison video, Viser twist controls and checks after Astra's round-3 run (2026-10-07) | [Encounter turning-graph iteration](../experiments/2026-10-07-encounter-turns-iteration.md) |
| [Learning paradigm decision](2026-10-12-decision-learning-paradigm.md) | Open decision (2026-09-30): RL in mjlab vs TANGO-style offline expert data plus imitation, with the baked-crowd cost estimate | Ledger D18 (RL, teacher then distillation); [robot policy architecture](../decisions/robot-policy-architecture.md) |
| [Recentering toward robot training journal](2026-10-12-recentering-training.md) | Journal: Astra round 4 in main while the traversal env, `crowd train traversal`, the teacher and distillation were built in a snapshot checkout (git tag `snapshot/2026-10-12-train`, directory removed 2026-10-23), then merged | [Robot-aware teachers](../experiments/2026-10-12-robot-aware-teachers.md), [encounter decisiveness](../experiments/2026-10-12-encounter-decisive.md), ledger D18 and D19, [robot policy architecture](../decisions/robot-policy-architecture.md) |
| [Traversal evaluation harness journal](2026-10-18-traversal-eval-harness.md) | Implementation journal: env options for the next arms, paired goals, the metric suite, `crowd eval traversal`, the smoke fixes (TTC, zero-length records), Astra review 1, GPU checks 4133387 and 4133388 | [Traversal evaluation](../decisions/traversal-evaluation.md) decision, `docs/usage/traversal-evaluation.md` |
| [W&B run continuity journal](2026-10-18-wandb-continuity.md) | Implementation journal: tracking package, live writer, in-place lineages and requeue, evaluation attachment, fake-SDK tests, GPU check 4133159 and the cross-node lock probe | Ledger D20, [runtime architecture](../core/design/runtime-architecture.md#training-lineages-and-wb) |
| [W&B run continuity decision](2026-10-18-decision-wandb-run-continuity.md) | Decision file (2026-10-18): one W&B run per training lineage, publish through the last checkpoint, requeue on SIGTERM, evaluations under `eval/*`; compared with hrc; the five open questions | Ledger D20; `AGENTS.md` (training lineages and W&B); [CLI](../../docs/cli.md#train), [Slurm](../../docs/slurm.md) |
| [Paper formalism preparation](2026-10-14-paper-formalism.md) | Documentation task: inspect graph and learning code, formalize the implemented algorithms, collect measured results and figure specifications | [Paper formalism and figure handoff](../core/paper-formalism.md) |
| [W&B fixes and canary](2026-10-18-wandb-canary.md) | Journal: the seven open D20 findings fixed, a second Astra review (four more, fixed), the real W&B canary in `crowd-control-traversal-debug` (job 4138449) and its read-back | [Ledger D20](../core/design/decision-ledger.md), [runtime architecture](../core/design/runtime-architecture.md#training-lineages-and-wb), run log |
| [SONIC planning arms journal](2026-10-18-sonic-planning-arms.md) | Implementation journal: planner and tracker ports, the gate, `PlanningVecEnv`, the planner MatMul fold, launches, requeues and evaluations of P0, P1/P2, the teachers and the students | [Planning arms experiment](../experiments/2026-10-19-sonic-planning-arms.md), [robot policy architecture](../decisions/robot-policy-architecture.md) |
| [Curriculum avoidance analysis](2026-10-21-curriculum-analysis.md) | Analysis journal: completed curriculum/fixed-blind/two-stage evaluations, joint goal and collision comparisons across behavior conditions, robot-prefixed parameter semantics | [Blind-crowd experiment](../experiments/2026-10-21-blind-crowd-arms.md), [crowd architecture](../core/design/crowd-architecture.md), evaluation requirement in AGENTS.md |
| [Sparse-goal and crowd proposal](2026-10-21-sparse-goal-crowd-proposal.md) | Assessment journal: sparse successful-goal/time rewards, non-interacting roaming humans, density and coherent flow | [Planner objective and crowd decision](../decisions/planner-objective-and-crowd.md) |
| [Live viewer journal](2026-10-22-live-viewer.md) | Build log of the live Viser traversal viewer: GPU check jobs, viser workarounds, golden comparison | [Status](../core/status.md) (Tools), [mjlab knowledge](../knowledge/mjlab.md) (Traps), `docs/usage/demos.md` |
