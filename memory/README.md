# Project memory

Research memory for agents and people, modeled on `~/flash/hrc/memory`. Code and API
usage live in [`docs/`](../docs/README.md). Binding code, environment and compute
constraints live in [`AGENTS.md`](../AGENTS.md). Memory may cite docs and code, but docs
and code never cite memory, so a note can move or be retired without breaking them
(`scripts/check_memory.py` checks both).

| Category | Holds | Lifetime |
|---|---|---|
| [`core/`](core/README.md) | In-scope knowledge: [current status](core/status.md), the [plan](core/plan.md), design, settled decisions, and the conclusions drawn from experiments | Long-lived; revised in place |
| [`decisions/`](decisions/README.md) | Questions that are **still undecided**: options, evidence, what would decide them | Deleted once resolved; the outcome moves to core |
| [`experiments/`](experiments/README.md) | Dated results: what ran, what it measured, links to the runs | Kept while core or decisions cite it; then archived |
| [`knowledge/`](knowledge/README.md) | References outside the project's scope: upstream repositories and models (ARDY, Kimodo, SOMA-X), their APIs and pitfalls | Long-lived; rarely changes |
| [`journal/`](journal/README.md) | Dated, ephemeral work notes for non-experiment tasks: plans, progress, handoffs | Retired when the task ends |
| [`log/`](log/README.md) | Dated logs: the [run log](log/runs.md), one row per run directory (command, job, GPU, outcome, artifacts) | Append-only; every new run gets a row |
| [`archive/`](archive/README.md) | Retired notes: finished journal entries, resolved decision files, unreferenced experiment notes, and earlier sessions' plans | Historical; never edited except to fix links |

## Where to look

- **Starting work:** read [core/status.md](core/status.md), then the
  [decisions index](decisions/README.md) for anything touching your task.
- **Why a design choice was made:** [core design](core/README.md#design) and the
  [decision ledger](core/design/decision-ledger.md).
- **Can a backend run online?** The conclusions live in
  [motion backends](core/design/motion-backends.md), which links the dated
  [experiment notes](experiments/README.md). Each run directory under `runs/` is that
  run's immutable record; `runs/` is not in git, so the [run log](log/runs.md) is how
  memory resolves a run ID on a machine that lacks the directory.
- **How ARDY or Kimodo works, and its traps:** [knowledge](knowledge/README.md).
- **An in-flight task:** its entry in the [journal](journal/README.md).
- **Why something was retired, or an earlier session's plan:** the [archive](archive/README.md).

## Where to write

- **Starting a multi-step task:** open a journal entry with the plan, and update it as
  you go.
- **Starting an experiment or benchmark:** open its dated note in `experiments/` with
  `Status: running`, and rewrite that note as the result when it finishes.
- **Any new run directory:** add its row to the [run log](log/runs.md)
  (`scripts/index_runs.py` drafts it; write the outcome by hand).
- **A question that needs the user or an experiment:** open a file in `decisions/`.
- **Finishing an extensive implementation or experiment:** update core. Refresh
  `core/status.md`. Revise the affected design note, stating each durable result as a
  short conclusion that links its experiment note. Record resolved decisions in the
  ledger and move their `decisions/` file to `archive/`. Then retire the journal entry, and archive
  any experiment notes that are no longer cited (`scripts/check_memory.py`).
- **A binding code, environment or compute constraint:** [`AGENTS.md`](../AGENTS.md),
  not memory.
- **Upstream material:** `knowledge/`.

Memory is tracked in git; `runs/` is not. Retire a note by moving it (`git mv`) into
[`archive/`](archive/README.md) with a dated name, never by deleting it, and fix the
links the move breaks. Cite runs by ID (`runs/<UTC>_<name>/...`); `scripts/check_memory.py`
checks every cited run ID against the [run log](log/runs.md), not against the disk.

## Earlier sessions' notes

Plans and notes that earlier sessions wrote inside the mjlab clone, and notes retired
before memory was tracked in git, are now in the [archive](archive/README.md). The
lasting facts from the mjlab work are in [knowledge/mjlab.md](knowledge/mjlab.md).
