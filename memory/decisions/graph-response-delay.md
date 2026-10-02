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
