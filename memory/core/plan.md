# Plan

The plan of record is the user's handoff,
[handoff-2026-09-30.md](handoff-2026-09-30.md)
(2026-09-30). This note tracks progress against it and the evidence each stage needs
before the next begins. Do not restate the handoff here; link its sections.

**Goal.** Stochastic, reactive human crowds for training a LiDAR-conditioned humanoid
traversal policy at 4,096 parallel environments with m = 4–16 humans each (user,
2026-09-30; ledger D09).

**Route to body motion: offline** (ledger D10). The benchmark ruled out online
generation at this scale. ARDY and Kimodo generate diverse motion offline (walking
styles, turns, starts/stops, idles), stitched online through a stochastic motion graph
(user, 2026-10-01; ledger D11). On a TANGO-style
branch, humans can instead be baked per episode
([learning paradigm](../decisions/learning-paradigm.md)).

## Stages

| # | Stage | State | Evidence needed to advance |
|---|---|---|---|
| 1 | Working backends and thin adapters (handoff §1) | Done 2026-09-30: `src/crowdmotion/backends/`, parity-tested against upstream | Both load offline, generate batched motion, continue from executed history; conventions in [knowledge](../knowledge/README.md) |
| 2 | Usable-motion benchmark (§2) | Isolated-capacity part done: [benchmark](../experiments/2026-09-30-motion-backend-benchmark.md), conclusions in [motion backends](design/motion-backends.md#conclusions). End-to-end vs cached replay not started | Throughput, quality, VRAM, seeding, provenance (done); end-to-end run in mjlab (open) |
| 3 | Choose a backend (§3) | Done 2026-09-30 (user): offline only (ledger D09–D10). ARDY is the first library generator; Kimodo remains the SOMA option | — |
| 4 | Offline library and graph (§4) | Prototype done 2026-09-30: ARDY library + kNN motion graph, near online-ARDY quality at about 5% of an A40 for 65k humans (isolated) ([motion library](design/motion-library.md)). Graph selected by the user 2026-10-01 (D11); behavior audit and review tools prepared | Local rotations; root-velocity seams; an mjlab smoke test with the graph |
| 5 | Behavior model and batched scheduling (§5) | Plaza demo exists; facing/turning audit and iteration tooling prepared 2026-10-01. User requested report/setup review before tuning | Seeded replay; intent → steering → body separation; refill/switch bucketing |
| 6 | Geometry and simulator smoke test (§6) | One-environment capsule/LiDAR demo exists; high-scale test not started | Mesh or capsule proxies in mjlab LiDAR and contact; end-to-end run vs cached replay |

## Deliverables (handoff §6)

- Reproducible benchmark script and configuration; raw CSV/JSON measurements.
- Short backend recommendation with rollout videos; the selected adapter.
- Optional graph, only if justified; then the batched behavior controller and a
  simulator smoke test.
