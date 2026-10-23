# Curriculum avoidance analysis

Status: complete (2026-10-21). Retired after promotion to the
[blind-crowd experiment](../experiments/2026-10-21-blind-crowd-arms.md),
[crowd architecture](../core/design/crowd-architecture.md), current status and the
architecture/evaluation decisions. User's condition-wise safety/progress evaluation
requirement is recorded in AGENTS.md. Five training outcomes and ten existing evaluation
directories are indexed; no new simulation jobs or behavior changes.

User request: assess whether the existing kinematic-planner curriculum teaches actual
human avoidance, to inform additional Stage 2 interaction-parameter curricula.

Plan:
- Read the implemented difficulty controller, rewards, action space and evaluation definitions.
- Compare completed curriculum, fixed-blind, original/+30 and two-stage penalty runs using
  paired aware/blind evaluations; inspect training difficulty and close-approach behavior.
- Separate improvements in collision risk from stopping, reduced exposure and human yielding.
- Promote results to the blind-crowd experiment, core status and architecture decision;
  index completed runs, retire this entry and check memory.

Constraints: preserve existing workspace edits; no commits/pushes; no simulation or heavy
analysis on sky1. Use existing saved metrics and small log reads; submit compute work if needed.

Findings: curriculum improves blind collisions/goals but loses aware collision safety;
fixed blind trades more aware safety for blind safety. Difficulty mostly plateaus, with
restart-related final increments. Assertiveness and independent robot stubbornness are
candidate next axes; retain easy-condition exposure and evaluate both metrics per condition.
