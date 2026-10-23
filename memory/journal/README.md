# Journal

Dated, ephemeral work notes, one file per task: `YYYY-MM-DD-topic.md`. Use an entry to
survive an interrupted session. Record the plan, progress, commands to resume, and what
remains. Keep it short. Experiments do not use the journal: their dated note in
[`../experiments/`](../experiments/README.md) holds the plan while running and the
result afterwards.

## Lifecycle

1. **Open** an entry when a task spans several steps or sessions. State the request,
   the plan and any constraints the user gave.
2. **Update** it as work lands: tick steps, and note job IDs and run directories.
3. **Promote** before retiring:
   - measured results → an experiment note, with its conclusion in the core note it
     affects;
   - design facts → the relevant `../core/` note;
   - new open questions → `../decisions/`;
   - binding constraints → `AGENTS.md`;
   - current state → `../core/status.md`.
4. **Retire** the entry once nothing in it is needed for handoff. Move it (`git mv`) to
   [`../archive/`](../archive/README.md), keeping its dated file name. A journal entry
   must never be the only home of an important result or decision.

Do not link journal entries from code, docs or core notes; link the promoted note
instead.

## Active entries

| Entry | Task |
|---|---|
| [2026-10-01-repo-migration](2026-10-01-repo-migration.md) | Move the repo to git and a local Linux GPU machine; functional reorganization, docs and memory restructure. Plan published, awaiting approval |

Retired: [memory system and online-generation benchmark](../archive/2026-09-30-memory-and-benchmark.md)
(2026-09-30); [graph behavior report](../archive/2026-10-01-graph-behavior-report.md)
(2026-10-01); [rigid mesh humans](../archive/2026-10-06-rigid-body-geometry.md) (2026-10-06); [encounter design](../archive/2026-10-06-encounter-design.md) (2026-10-06); [encounter turning-graph comparison](../archive/2026-10-07-encounter-turns-comparison.md) (2026-10-07). The bounded iteration session's own journal is archived as
[behavior session journal](../archive/2026-10-01-behavior-session-journal.md).
