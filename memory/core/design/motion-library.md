# Offline motion library and stochastic motion graph

How body motion reaches 16k–65k simulated humans now that the generators run offline
only (ledger D09–D10). Code: `src/crowdmotion/library/` (generation) and
`src/crowdmotion/graph/` (tables, build, cost terms, runtime, audit); usage:
[library and graph](../../../docs/usage/library-and-graph.md). How the graph is driven:
[runtime architecture](runtime-architecture.md). The user selected the
motion graph on 2026-10-01 (ledger D11). ARDY supplies the current library; later
[generator expansion](../../decisions/library-generator.md) remains open.

## Behavior audit and review loop (2026-10-01)

The CPU audit of 134,820 valid edges found only 48.7% of moving frames forward,
14.4% of all frames sideways and 17.4% backward. Only 5 edges qualify as slow
turning under the explicit analysis cutoffs; these are availability, not visitation
figures ([audit](../../experiments/2026-10-01-graph-behavior-audit.md)).

The score audited on 2026-10-01 had no facing/turn-before-walk cost, and demo steering could request
abrupt direction changes. Root tracking alone does not establish realistic forward
walking. Deliberate turn coverage and graph selection both need review; the prior
claim of no sideways/backward library motion was incorrect.

[Behavior iteration](../../../docs/usage/behavior-iteration.md) describes the local
report and config-driven evaluate → saved-motion replay → paired comparison tools
(`crowd graph report`, `crowd behavior evaluate|render|compare`, `crowd iterate`). The user
accepted the report and authorized iteration, but the first allocation remained
pending on `QOSGrpGRES` and stopped after five minutes; no real tuning or new render
ran ([session](../../experiments/2026-10-01-graph-behavior-iteration.md)). The bounded
session worker/monitor and proposal are preserved for the next allocation.
Synthetic metrics/persistence tests and replay scene setup pass. The evaluator
preserves current demo RNG for baseline behavior; training still needs independent
steering streams.

Batch agent execution (2026-10-04): `crowd iterate agent` starts Codex and a nested
local session after the GPU allocation begins, so scheduler waiting does not consume
the iteration budget. The outer RunDir saves provenance, prompt, agent logs and final
message; STOP, deadline and signals close the session and terminate processes.
The batch wrapper and [setup](../../../docs/slurm.md#start-codex-automatically-in-a-batch-job)
are prepared; fake-agent CPU validation does not establish rendering or realism gains.
A fake CLI also cannot establish that the real one accepts the launch arguments: the
first batch run failed on a flag conflict, so the launcher now preflights the exact
argv and the model/effort catalog with the real Codex before opening a run.

## Tested behavior controls (2026-10-05)

The saved graph now supports optional facing/heading costs, a heading-error deadband,
and shared forward steering with per-human avoidance priority. Defaults remain
unchanged. The [evening iteration](../../experiments/2026-10-05-evening-graph-behavior-iteration.md)
validated the pre-existing opt-in implementation and added the deadband without
rebuilding the graph or loading a motion prior.

- **Direction improved on held-out seeds:** 20 s moving-forward shares rose from
  20–24% to 64–70%; 60 s ordinary-command travel was 81–83% forward. Speed decreased,
  and foot sliding was mixed, so forward percentage alone is not an acceptance test.
- **Spacing is unresolved:** the 60 s candidate regressed on two seeds, reaching a
  0.501 m root gap despite both humans already having evasive priority. A short
  paired rollout would have missed this; keep the candidate opt-in.
- **Body turning has a path-response cost:** all 48 held-out 90°/180° probes turned
  their bodies, but median root responses were 2.33/5.83 s. Committed sampled turns
  alone completed 8/49 attempts and accumulated at walls; native forward-turn
  candidates were available in only 25–39% of their large-turn states.
- **Compatibility/integration:** the frozen baseline reproduces exactly; committed
  HEAD and working defaults match 158 arrays and six JSON outputs on the same host.
  Both retain historical golden numeric mismatches. A 2-env × 4-human A40 mjlab
  check passed; neither this nor capsule stills establish training-scale readiness.

## Load-time posture and flat-ground evidence (2026-10-05)

The [transition iteration](../../experiments/2026-10-05-transition-posture-iteration.md)
adds development evidence and opt-in code, not a promoted configuration. It stopped
on frame-extraction failure before fresh held-out and new GPU integration checks.

- Saved body-local Y is absolute ground height. Signed Hips–Neck pitch separates
  torso lean more directly than head ratio; 62.6% of its variance lies between streams.
  Reviewed 15° classes remain experimental, with [calibration open](../../decisions/posture-height-calibration.md).
- A predecessor-inclusive height mask plus peak lower-foot criterion removes 3.02%
  of valid edges, leaving no dead ends after retained edges. Executed blending still
  needs measurement; empty candidate rows explicitly hold and report a fallback.
- Soft sticky posture and matched starts reduce development class crossings from
  18–19 to 4–6 per human-minute. Independent latent changes and actual pose-class
  crossings are different metrics; a soft preference does not guarantee a class.
- Only two upright and three hunched slow-turn edges survive. Slow explicit turn
  phases stalled; walking-turn phases help controlled response but retain crowd
  tradeoffs. Forward percentage alone cannot establish turn-then-walk success.
- The candidate's 60 s ordinary forward share is 93.7–94.4% and peak lower-foot height
  .112 m, but seed-0 gap worsens to .433 m and controlled completion drops to 47/48.
  Keep it opt-in and preserve the earlier incumbent pending unfinished review.

## Heading-conditioned turning supplement (2026-10-06)

ARDY follows body-heading targets: its in-place turns reach 0.98 of the commanded
hip-line angle, against 0.00 without targets. The opt-in `graph-turns` combines the base
library with a 9.1 h turning library. Under the round-2 mask it holds 4,763 in-place
edges without a planted twist (5,873 gross), against 8 in graph-default.

Twists are not masked. In a twist the pelvis turns while the feet stay planted, and 19%
of in-place edges and 32% of in-place windows contain one. The supplement also adds
2,354 standing or straight edges that hold a twist, against 67 in the base, so humans
can stand twisted without a turn request
([turning library](../../experiments/2026-10-06-turning-library.md)).

## Backward execution audit (2026-10-12)

**Conclusion:** the saved turning graph has admissible short backward/back-diagonal
edges, including 402 with entry speed <=0.15 m/s and 3,218 <=0.5 m/s. Coverage alone
does not establish an executable escape: from 32 standing starts a fixed-facing
0.4 m/s backward command produced median 0.004 m backward progress over 2.4 s.
[Audit and inspected edge sheets](../../experiments/2026-10-12-encounter-decisive.md).

The audit used existing saved tables and inspected twelve examples; no generator,
library or graph rebuild ran. No backward-step strategy was added. First test graph
entry connectivity, cost selection and posture/mask constraints from actual blocked
states. An offline supplement is justified only if that isolates missing reachable
motion, and would need short standing-to-backward/back-diagonal starts and stops,
paired turns, checked body-facing/foot behavior and rebuilt rotations.

## Design

- **Library.** ARDY H40 at its online setting (TF32, no guidance, 5 steps) streams
  closed-loop motion under a random semi-Markov command process: stand, walk at
  0.3–2.0 m/s, heading changes up to ±180°, some curving.
  - Every 8-frame (0.4 s) edge is one ARDY call from the executed 4-frame history, so
    each edge is a genuine ARDY continuation under a known command.
  - Per-human command and noise streams are batch-independent.
- **Graph.** Nodes are boundaries (the frame before each edge). A boundary's candidate
  edges are the successors of its k nearest boundaries, by root-relative joint position
  and velocity. Its own successor is always a candidate at distance 0.
  - The prototype uses k-nearest neighbors rather than the k-means clusters first
    proposed, which avoids cluster-boundary effects.
- **Runtime step**, per human and per edge:
  1. Score candidates by root-path error against the steering layer's requested path,
     descriptor distance, and end-of-edge velocity error. Optional terms also score
     body-relative forward travel and terminal body heading.
  2. Sample with per-human Gumbel noise.
  3. Bend the root path by at most ±5° yaw and ±10% speed.
  4. Decay the pose offset over the edge.

  Pure torch: no generator in the loop.
- **Curation** originally used a baked library-side head-height filter: edges with
  mean head height below 85% of neutral are dropped. Reviewed load-time posture and
  vertical annotations now provide separate opt-in runtime costs/masks without
  changing that saved filter or graph format.
- **Shared behavior layer.** Intent and steering produce a world root path as for
  online backends (ledger D05); optional graph-orientation priorities flow through
  `CrowdController`. Baseline intent/steering and RNG defaults remain unchanged.

## Conclusions

As of 2026-09-30, one A40, 18 h library from 2,048 streams. Quality numbers are from a
held-out seed with 128 humans
([experiment](../../experiments/2026-09-30-ardy-library-graph.md)):

- **Cost is solved.**
  - Generating the 18 h library takes 2 min. ARDY makes about 540 kept motion-s per
    wall-s while replanning every edge.
  - At the recommended setting, the graph steps 65,536 humans in 47 ms per 0.4 s of
    motion (k = 128): about 5% of an A40 at m = 16 and `S = 0.4`. That is isolated
    stepping. Feeding the poses to mjlab
    and ray-casting LiDAR against them is not measured.
  - At m = 16 and `S = 0.4`, online ARDY at the quality reference used here would need
    about 46 A40s: it replans every 0.4 s, so u = 0.2. The floor is 9.3 A40s, for
    buffer refill only, which responds in 1.3–1.6 s.
- **Quality is close to online ARDY** at the recommended setting
  (`configs/graph/default.json`, formerly `configs/library/graph_default.json`:
  k = 128, correction, end-velocity weight 0.25 s, hunch filter), against ARDY replanned
  every 0.4 s:
  - path error 0.7–1.4 cm vs 0.6–0.75;
  - speed error 0.04–0.09 vs 0.04–0.05 m/s;
  - median command responses 0.0–0.4 s slower (turn-180: 1.1 vs 0.7 s; p95 2.1 vs
    0.95 s);
  - 1.6% of humans never complete a turn-180.
- **Historical head-height result, not torso-style calibration.** The filter and the metric both use
  head height, so this is true by construction: hunched output frames fall to
  0.2–0.6% (60 s: 0.4%). For comparison, online ARDY without text, replanned every
  0.4 s, stooped in 3–17% of frames over 12 s and 24% over 60 s (128 humans, quality
  harness). The benchmark's 3–6% / 24% came from refill rollouts, measured ad hoc on
  6 clips.
- **Scope of the evaluation.** It covered walking at 0.5–1.6 m/s with stops, starts,
  90°/180° turns and speed steps. Never tested:
  - starting from standing toward a new direction (the wait, yield and reroute case);
  - turning in place, sidestepping or walking backward as deliberate behaviors.
    Correction 2026-10-01: the commands do not label these, but root-only constraints
    allow considerable body-relative lateral/backward output, as audited above.
- **Cost terms matter more than library size.** Without an end-velocity cost, 22% of
  humans never stopped, although a fitting edge was available. Position-only path cost
  over one edge cannot tell "start decelerating" from "keep walking". k = 32 is
  coverage-limited for 180° turns; k = 128 is not.
- **Remaining costs of stitching:**
  - planted-foot sliding happens more often (fraction above 0.1 m/s: 0.39–0.52 vs
    0.22–0.35). Root correction is a main source: without it the fraction is
    0.30–0.40, but tracking and speed response degrade;
  - root-speed seams are 1.4–2.0× the interior root-step change (online 0.9–1.4×);
  - rare path excursions up to 18–29 cm.
- **Not yet supported:**
  - local rotations, so no mesh skinning ([human body model](../../decisions/human-body-model.md));
  - semantic styles or prompts ([text encoder access](../../decisions/text-encoder-access.md));
    opt-in geometric upright/hunched selection was added in the later iteration above;
  - high-scale mjlab integration. The one-environment capsule demo now interpolates
    20 fps motion to the simulator rate; this is a visual integration check only.
