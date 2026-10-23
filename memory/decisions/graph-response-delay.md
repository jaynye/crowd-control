# Acceptable graph reaction delay

Opened 2026-10-01 from the remaining limit in the retired offline-runtime question.
**Whose call:** user, informed by measurements. **Blocking:** no baseline behavior
review; it can inform later branch frequency and turn selection.

The graph uses 0.4 s edges. The user has not set an acceptable simulated-time
reaction delay. The prior [library evaluation](../experiments/2026-09-30-ardy-library-graph.md)
measured median responses 0.0–0.4 s behind online ARDY and p95 turn-180 delay 2.1 s.
Those results measured root-path response, not turn-before-walk body-facing realism.

Report actual response after steering/facing changes. Do not invent a limit.
The user can set the eventual requirement once the behavior and simulator context
are concrete. Historical context:
[retired runtime proposal](../archive/2026-10-01-decision-offline-runtime.md).


## Evidence added 2026-10-05

The [evening behavior iteration](../experiments/2026-10-05-evening-graph-behavior-iteration.md)
measured held-out controlled 90°/180° requests through shared steering. The selected
opt-in candidate completed 48/48 body responses in each case (median 2.77/6.10 s),
while baseline reached 28/48 and 0/48. Root-path medians slowed from 0.90/1.06 s to
2.33/5.83 s; candidate p95 was 4.08/9.60 s. These include replan waiting and use
explicit diagnostic response definitions. This adds evidence; it does not settle the
user's acceptable-delay limit or promote the candidate, which has long-run spacing
regressions.

The later [transition/posture iteration](../experiments/2026-10-05-transition-posture-iteration.md)
adds development-only evidence: the peak-mask/forward-floor candidate completes
47/48 controlled body responses for each 90°/180° case, versus incumbent 48/48.
Conditional per-seed body-delay medians are 2.25–2.47 / 4.82–5.29 s versus
2.72–2.80 / 6.06–6.80 s. Faster successful responses coexist with one censored
human per case. Held-out evaluation and new integration were not run after an
extraction failure, so this does not replace the earlier selected incumbent.
