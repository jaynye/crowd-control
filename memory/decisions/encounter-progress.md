# Encounter progress after close-pair recovery

Opened 2026-10-12. Status: open. Whose call: empirical investigation, then user review
before treating the development candidate as ready for robot training.

## Question

How should ordinary turns recover when the saved graph remains nearly stationary
beside a wall or obstacle, while keeping the contact gains and minimal detour goal?
This is distinct from two stopped humans resolving an encounter.

## Evidence

The [decisiveness experiment](../experiments/2026-10-12-encounter-decisive.md) selected
`configs/behavior/encounter_decisive.json` as a development candidate. Close-pair
stalls at root gap <=1.1 m fell from 43 to 19 over three 60 s development seeds,
and all-frame joint-sphere contacts fell to zero. However individual pending-goal
stops reach 48 s (development seed1/human2, censored) and 47.2 s (held-out seed101/human3).
Both are non-stubborn, near the west wall and scanner obstacle, with ordinary turns
and movement commands but almost no executed progress. Frequently no nearby human
threat exists, so the mutual-pair recovery clock stays zero.

Stepwise evidence and inspected follow sheets are in
`runs/20261012T152601Z_encounter_decisive/session/iterations/109_individual_stop_tails`,
`runs/20261012T152601Z_encounter_decisive/session/iterations/110_selected_wall_stall_stills`
and `runs/20261012T152601Z_encounter_decisive/session/iterations/111_heldout_wall_stall_stills`.
Longer root-gap cutoffs also expose residual pair stalls: at 1.5 m the maximum is 6.4 s.
Goal arrivals fall slightly, and foot-speed/acceleration diagnostics worsen. No
post-held-out runtime tuning was performed.

## Options to investigate

1. Isolate the saved wall/obstacle states and compare ordinary-turn target retention,
   confinement, wall-mirror clearance costs and admissible graph edges. Add a
   one-human wall/obstacle probe plus a free-space control before choosing a fix.
2. An opt-in executed-progress watchdog could end an unrealized ordinary turn and
   request a feasible in-place turn or waypoint around the obstacle. It must retain
   the stop rule and avoid merely moving a jam beyond the close-pair cutoff.
3. Existing backward edges might provide an escape, but the saved-table audit found
   poor commanded execution from rest. Establish reachable entry transitions and
   cost selection before relying on them or proposing an offline supplement.

## What would decide it

Trace selected versus available graph costs at the frozen development failure;
test one mechanism at a time, then compare individual stop tails, pair stalls across
gap cutoffs, goal-directed progress, contacts, feet/seams and redirects. No new pass
thresholds are implied. Preserve the current candidate and exact all-off baseline.
Seeds 101/307/911 were used once after freezing this candidate; they are now evidence,
not an unseen selection bank. A later promotion needs a separately reserved bank.
