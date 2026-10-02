# Experiments

Dated records of what was run and what it measured, one file per experiment or group of
related runs: `YYYY-MM-DD-topic.md`. Each note links its run directories under `runs/`,
which stay the immutable record; every run has a row in the
[run log](../log/runs.md). Core holds the conclusions; this folder holds the
evidence behind them.

## Lifecycle

1. **Open** the note when the experiment starts, with a `Status: running` line under the
   title. Record the question, plan, configs, job IDs and run directories. Experiment
   work uses this note instead of a journal entry.
2. **Finish** by rewriting the same note as the result. Remove the status line. Lead
   with the outcome, then the numbers, scope and caveats.
3. **Conclude in core.** State the conclusion in one to three lines in the core note it
   affects (usually [motion backends](../core/design/motion-backends.md)) and link this
   note. A link from `status.md` alone is not enough: status is a snapshot, and its next
   refresh would orphan the note. If the result settles an open question, record the
   outcome in the ledger and retire the `decisions/` file.
4. **Retire** a note once nothing live cites it (reachable by links from `core/` or
   `decisions/`, directly or through other live notes). Move it (`git mv`) to
   [`../archive/`](../archive/README.md), keeping its dated file name, and fix the links
   the move breaks.

## Rules

- **Only core, decisions, other experiment notes, journal entries, the run log and the
  archive link here.** Docs, code, `knowledge/` and root files link the core conclusion
  instead. Index files
  (`README.md`) do not count as citations.
- **Notes are historical.** Numbers hold as of their date and hardware. A new question
  gets a new dated note that links the one it supersedes. Fix factual errors in place
  with a dated correction.
- **Cite runs as code paths, not links:** `runs/<UTC>_<name>/file`. `runs/` is not in
  git, so a Markdown link into it breaks on a fresh clone.
- **Record hardware on every number.** GPU type, driver, precision and settings; speed
  is compared only within one GPU type.
- **Check before finishing:** `.venv/bin/python scripts/check_memory.py --check`.

## Index

Oldest first. One line per note; remove the line when retiring a note.

| Note | Result |
|---|---|
| [Kimodo smoke (L40S)](2026-09-28-kimodo-smoke.md) | Historical: 7–28 clip-s/wall-s, compute-bound past batch 16, TF32 2×; dense root path 1.5 cm mean error; SOMA xlo skinning 24 ms / 2,400 poses |
| [Motion-backend benchmark (A40)](2026-09-30-motion-backend-benchmark.md) | Kimodo ≤ 60 new-motion-s/wall-s online. ARDY H40 (TF32, no guidance, 5 steps) 2,800, with clean continuation: `S*` 0.69 / 0.17 / 0.086 for m = 1 / 4 / 8 at u = 1. Response = replan wait + ~0.5 s. Simulator alone S 0.41–0.68 |
| [ARDY library and motion graph (A40)](2026-09-30-ardy-library-graph.md) | 18 h ARDY library in 2 min. The graph steps 65,536 humans in 47 ms per 0.4 s edge (k = 128). With an end-velocity cost and a hunch filter (held-out seed): path 0.7–1.4 cm, median responses within 0.0–0.4 s of online ARDY, hunched frames 0.2–0.6%. Costs: more foot sliding, root-speed seams 1.4–2.0×, no rotations yet |
| [Graph behavior audit (CPU)](2026-10-01-graph-behavior-audit.md) | Valid library frames: 43.0% forward, 14.4% sideways, 17.4% backward; 48.7% forward among moving frames. Slow-turn coverage scarce. Local HTML and evaluation/replay/compare loop prepared; tuning not executed at user's request |
| [Bounded behavior iteration](2026-10-01-graph-behavior-iteration.md) | Stopped at five-minute GPU-wait limit (QOSGrpGRES); session journal, baseline, turn proposal and worker preserved. No real evaluations, renders or runtime-rule changes |
