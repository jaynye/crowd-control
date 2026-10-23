# Core project knowledge

In-scope, long-lived knowledge about this project: what we are building, what exists,
and what we have concluded. Start with [status](status.md). Open questions are in
[decisions](../decisions/README.md), dated results in
[experiments](../experiments/README.md), and upstream repositories in
[knowledge](../knowledge/README.md).

**Update core after any extensive implementation or experiment:**
- refresh [status](status.md);
- revise the affected design note;
- state each durable result as a one- to three-line conclusion in the note it affects,
  linking its dated [experiment note](../experiments/README.md). Those links keep the
  experiment note live; remove one only when its conclusion is superseded.

Keep numbers in the experiment notes and link to them; run directories under `runs/`
stay the immutable record of each run, indexed in the [run log](../log/runs.md).

## Status and plan

| Note | Scope |
|---|---|
| [Status](status.md) | Active work, what exists, known issues, next steps |
| [Plan](plan.md) | The user's staged plan (2026-09-30 handoff) with progress and promotion evidence per stage |
| [Handoff](handoff-2026-09-30.md) | The user's plan of record, 2026-09-30 (moved here from the repository root); not edited |
| [Paper formalism](paper-formalism.md) | Three-page manuscript core, implemented motion-graph and teacher/student mathematics, figure specifications, and results provenance (2026-10-14) |

## Design

| Note | Scope |
|---|---|
| [Crowd architecture](design/crowd-architecture.md) | Intent → local steering → body motion; per-human state; stochastic switching; scheduling; kinematic humans; geometry |
| [Motion backends](design/motion-backends.md) | Backend adapter contract, benchmark metrics (`R_usable`, `u`, break-even `S*`), and the measured conclusions per backend |
| [Motion library](design/motion-library.md) | Offline ARDY library + stochastic motion graph: design, runtime cost, quality against online ARDY, curation |
| [Runtime architecture](design/runtime-architecture.md) | Why one crowd controller, motion sources and ordered cost terms; what was kept bit-exact; deferred behavior changes; invariants |
| [Decision ledger](design/decision-ledger.md) | Settled decisions with rationale, date and who decided |
