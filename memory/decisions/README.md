# Open decisions

One file per question that is **still undecided**. Settled choices live in
[core](../core/README.md): rationale in the
[decision ledger](../core/design/decision-ledger.md), binding constraints in
[`AGENTS.md`](../../AGENTS.md).

## Lifecycle

- **Open** a file when a question needs a user decision or an experiment to settle it,
  and you cannot resolve it from the code, the data or an existing constraint.
- **Update** the evidence section when results arrive. Link experiment notes and run
  directories instead of copying tables.
- **Resolve** by moving the outcome, with its date and who decided, into core: the
  ledger, the relevant design note, `AGENTS.md` for a binding constraint, and
  [status](../core/status.md) if it changes active work. Then move the file to
  [`../archive/`](../archive/README.md) with a dated name (`git mv`) and remove its row
  below.

Each file states: the question; whose call it is; the options; the evidence so far;
what would decide it; and links.

## Index

| Decision | Whose call | Blocking? |
|---|---|---|
| [Planner objective and crowd](planner-objective-and-crowd.md): persistent-goal reward, non-reactive graph humans and reward/credit/density ablations; implementation and dense experiment authorized, best objective still empirical | user, informed by experiment | which objective/crowd to promote after results |
| [Robot policy architecture](robot-policy-architecture.md): (a) from scratch, (b) SONIC residual, (c) LiDAR planner replacing SONIC's kinematic planner, (d) velocity-conditioned SONIC; recommended order (a) now, SONIC-in-mjlab gate, (c), (b) ablation, (d). User strongly favors (c) with SONIC frozen (2026-10-18); the FP32 hand port batches at 239 plans/s on A40: 3.4–5.7× training time at 1–2 Hz replans, 25× at 10 Hz | user | what to build after the (a) baseline; the SONIC gate |
| [Traversal evaluation](traversal-evaluation.md): command-following and collision metrics (trajectory deviation, time to first collision, rates per robot-minute, TTC, encounters), a paired `crowd eval traversal` (built), the non-avoidant baseline b0 and the fall fix (training since 2026-10-18) | user | which evaluations to run on the arms |
| [Library generator](library-generator.md): keep ARDY Core, add Kimodo/SOMA, or mix priors behind the chosen graph | empirical, then user | no; current graph uses ARDY |
| [Graph response delay](graph-response-delay.md): acceptable simulated-time reaction delay after turning/steering changes | user, informed by measurements | no baseline review |
| [Encounter progress](encounter-progress.md): failed ordinary turns beside walls/obstacles after close-pair recovery | empirical, then user | candidate readiness; opt-in development continues |
| [Posture and height calibration](posture-height-calibration.md): geometric style classes, targets and acceptable flat-ground heights | empirical review, then user | calibrated promotion; no opt-in development |
| [Text encoder access](text-encoder-access.md): gated Llama-3 blocks prompts; guidance with prompts would cost ARDY 1.9–2.8× | user | style prompts; the no-guidance setting |
| [Human body model](human-body-model.md): LiDAR geometry settled (D15: full-resolution rigid Core-skin pieces); morphology set, contact geometry, ray budget and scan rate open | user | morphology variety, robot contact |

Retired 2026-09-30: demand budget (ledger D09) and backend choice (ledger D10), archived
as [demand budget](../archive/2026-09-30-decision-demand-budget.md) and
[backend choice](../archive/2026-09-30-decision-backend-choice.md).

Retired 2026-10-01: offline runtime (ledger D11: motion graph selected). Proposal
archived as [offline runtime](../archive/2026-10-01-decision-offline-runtime.md);
remaining generator and reaction-delay questions are separate above.

Retired 2026-10-18: W&B run continuity (ledger D20: in-place lineages, one writer per W&B run, writer off until a canary), archived as
[W&B run continuity](../archive/2026-10-18-decision-wandb-run-continuity.md).
