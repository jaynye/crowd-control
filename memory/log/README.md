# Logs

Dated, append-only logs of what ran. They index evidence; they do not hold conclusions
(those are in [core](../core/README.md)) or measured tables (those are in
[experiments](../experiments/README.md)).

| Log | Holds |
|---|---|
| [Run log](runs.md) | One row per run directory under `runs/`: date, kind, command and config as run, Slurm job, GPU, outcome, artifacts |

## Rules

- **Every new run directory gets a row,** in the same change that cites it.
  `scripts/index_runs.py` drafts rows from each run's `run.json`; write the outcome by
  hand from the experiment note or journal, and never guess it ("unknown" is allowed).
  `scripts/index_runs.py --check` fails while a run on disk has no row.
- **Rows are historical.** The command, config and paths are as run at the time; code
  and configs may have moved since. Correct a row only for factual errors.
- **Cite runs by ID** (`runs/<UTC>_<name>/...`) anywhere in memory.
  `scripts/check_memory.py` resolves every cited run ID against the run log, so a
  citation stays valid on a machine without that run directory. Large artifacts that
  other machines need are listed in `artifacts.toml` and fetched by
  `scripts/fetch_artifacts.sh`.
