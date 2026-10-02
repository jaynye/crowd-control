# Motion graph behavior iteration

The session stopped at the user's GPU-wait limit. **No behavior improvement was
measured:** the queued baseline never executed, no new rollout video was rendered,
and stochastic runtime rules remain unchanged.

User authorized continual evaluation/rendering to improve naturalistic forward travel
and committed sampled turns, with 90° weighted above 45°/60°. One run folder:
`runs/20261001T061749Z_behavior_iteration/`; its `journal.md` is the ephemeral,
continually appended rule-change record. Deadline 2026-10-01 10:15:24 UTC (four hours
from task start), or stop if GPU waiting exceeds 300 seconds.

Slurm scheduler exposes the lab allocation as partition/account `rl2-lab`, not
`rl2-group`; use `qos=short`, one A40, exclude bishop. Job 3994253 requested on trublu
(38,206 MB free host memory), with a deadline-aware request worker. The request
remained PENDING on `QOSGrpGRES` and was canceled at 06:23:23.125 UTC, 300.125 s
after submission. No GPU allocation or measurement was obtained. CPU validation
ran on sky1; no timing comparison was collected.

Plan: freeze baseline → paired config-driven evaluation → saved-motion render →
committed stochastic turn/forward rule changes → assess behavior and spacing/foot
tradeoffs → validate held-out seeds and mjlab integration. No prior in the runtime.
Preceding report and coverage evidence:
[audit](2026-10-01-graph-behavior-audit.md).

## Preserved work

The session's local index (`runs/20261001T061749Z_behavior_iteration/index.html`)
and [journal](../archive/2026-10-01-behavior-session-journal.md) distinguish
proposal from executed changes. The baseline retains old rules and expands to
60 s × seeds 0/1/2. Proposed turn centers 45/60/90/135/180° carry weights
15/20/50/10/5%, sampled once on TURN entry with a retained body-heading target.
These are uncalibrated settings to implement/evaluate in the next allocated session.

The allocation worker processes immutable queued requests with per-request logs,
deadline cancellation and independent replay/evaluation commands. The session monitor
cancels only its recorded job on GPU waiting/deadline/STOP, and appends the journal.
Sources and request records are preserved inside the single run folder. The tooling
is now `crowd iterate` with local and Slurm executors; usage is in
[behavior iteration](../../docs/usage/behavior-iteration.md#iteration-sessions).
