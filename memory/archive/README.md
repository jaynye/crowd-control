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
| [Behavior session journal](2026-10-01-behavior-session-journal.md) | Session journal of the bounded behavior iteration (`runs/20261001T061749Z_behavior_iteration/journal.md`) | [Bounded behavior iteration](../experiments/2026-10-01-graph-behavior-iteration.md) |
