# Sparse-goal reward and non-interacting crowd proposal

Status: complete (2026-10-21), retired after preserving the proposal in the
[objective/crowd decision](../decisions/planner-objective-and-crowd.md), status and crowd
architecture. No implementation or new runs. The proposal distinguishes reward alignment
from sparse exploration, requires persistent observable goals and a suitable credit horizon,
and treats flow coherence/relative angle separately from nominal density.

User asks whether to replace planner rewards with successful-goal/time rewards and
collision termination, and train against randomly roaming humans that ignore each
other and the robot, with density and common-direction coherence controlling difficulty.
This is an assessment request, not an instruction to implement or launch runs.

Plan: inspect current reward/goal/crowd mechanics; check sparse-reward and shaping
references; propose a matched experiment and evaluation grid; preserve the proposal in
an open decision, then retire this journal and run memory checks.

Initial findings: velocity/facing rewards oppose detours; planner discount is 0.75 per
second, current goals can be replaced after 10–15 s, and body geoms already have physical
contacts disabled. Logical human stopping/separation/graph clearance remain active even
when the robot is invisible or stochastic switching is disabled.
