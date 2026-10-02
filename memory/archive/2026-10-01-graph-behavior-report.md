# Motion graph behavior report and iteration tooling

> Archived from `runs/20261001T055543Z_behavior_report/archive/2026-10-01-graph-behavior-report.md` on 2026-10-01; historical. Links were adjusted to the archive location.

Opened 2026-10-01. User requested a local HTML explanation of the current graph,
measured motion composition and connectivity, and stochastic behavior, before an
efficient behavior iteration/render loop is executed. Scope question pending:
stop after report/tooling or continue tuning after the report is ready. Resolved:
the user requested stopping after setup for review.

## Plan

1. Inspect graph tables, library commands, steering, demo provenance and video.
2. Measure body-relative motion classes and candidate connectivity; generate a
   local HTML report with definitions, source links and embedded baseline video.
3. Prepare deterministic config-driven evaluation/render tools, saved traces and
   comparisons so candidates can be assessed without regenerating the library.
4. Validate tools, promote durable findings and the user's graph choice into core,
   retire this entry, and run the memory checker.

## Progress

- Required status, open runtime decision and plan read. Only root AGENTS.md applies.
- Sandbox initialization fails (`bwrap` network namespace); local tool execution
  uses reviewed escalation. No upstream clones or environment dependencies changed.
- Current graph samples by trajectory + pose + end velocity, with no explicit
  locomotion-facing penalty. Steering goal/avoidance vectors can jump direction.
- Historical video predates the saved graph and current mjlab command-term demo;
  treat it as historical evidence, not an exact replay of current source.
- Final report: `runs/20261001T055543Z_behavior_report/index.html`. Graph composition,
  connectivity, examples, original video and provenance are measured/explained.
- Config-driven evaluation, saved-joint replay, paired HTML and Slurm wrapper ready.
  14 synthetic CPU tests pass; actual replay scene builds 400 mocap bodies. No real
  behavior loop or new render was executed. Findings promoted to the audit experiment,
  motion-library/crowd core notes and status; graph choice recorded in D11–D12.

Completed: final HTML controls/resources, static checks and replay-arrow API verified.
The memory-router stale runtime link was updated; journal retired after promotion.
