# Paper formalism and figure handoff

Status: complete (2026-10-14).

Request: prepare a Markdown document with LaTeX for a short three-page paper on the
stochastic motion graph, human behavior, robot POMDP, and teacher/student training.
The user's separate Astra session will create visuals; this task supplies their
mathematical content and figure specifications.

Plan:

1. Check current code, design decisions, and experiment evidence; separate historical
   measurements, current results, and untested claims.
2. Formalize the offline prior, graph topology and stochastic traversal, encounter
   state, robot POMDP, PPO teacher, and implemented student objective.
3. Write a compact paper core with supporting notation, diagram specifications,
   evidence/provenance, and a three-page allocation.
4. Check equations against code and reported numbers against experiment notes; run
   the memory checker and archive this journal when complete.

Constraints: no training or benchmark runs, no baseline or implementation changes,
no commits or pushes. Preserve the extensive pre-existing working-tree changes.

Progress:

- Inspected exact graph tables, descriptor, sampling, execution, encounter modes,
  observation/reward implementation, PPO configuration, and installed distillation.
- Verified primary literature references and historical/current experiment evidence.
- Wrote [paper formalism](../core/paper-formalism.md): a roughly 1,400-word manuscript
  core plus supporting equations, figure specifications, provenance, and code map.
- Recorded a code/notes discrepancy: `--finetune` is PPO initialized from student and
  teacher critic, not a simultaneous PPO-plus-imitation loss. The paper describes
  the implemented algorithm; no training code changed.
- Baseline memory check passed. Reviewed the equations against code, checked all
  local document links and 18 display-math blocks, and recomputed the reported
  percentages. The final memory check is the remaining closeout command; no new
  implementation, training, experiment, or run directory was created.

Deliverable and durable outcome: [paper formalism](../core/paper-formalism.md),
indexed in core. Its three-page core has eight numbered equations; optional detail
is separated from the manuscript so Astra can shorten it for a venue template.
