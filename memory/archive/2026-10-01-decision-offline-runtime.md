# Offline runtime: how to stitch an offline motion library online

> Archived from `runs/20261001T055543Z_behavior_report/archive/offline-runtime.md` on 2026-10-01; historical. Links were adjusted to the archive location.

Opened 2026-09-30. **Whose call:** the user, informed by a prototype. **Active:** the
user set 4,096 environments with m = 4–16 and moved both generators offline (ledger
D09–D10). The runtime is required for RL in mjlab and for closed-loop evaluation; a
TANGO-style branch can bake humans per episode instead
([learning paradigm](../decisions/learning-paradigm.md)).

## Question

If body motion comes from an offline-generated library, how is it stitched online, and
which model generates the library?

## Options

1. **Stochastic motion graph** (the user's and the handoff's framing). Clips and
   bridges are generated offline. Transitions match short boundary windows on pose,
   velocity, heading and support-foot state. Transition opportunities must be frequent,
   so clip length does not force long reaction delays (handoff §4).
2. **Motion matching** over a baked database. This is the 2026-09-28 proposal,
   `mjlab/docs/plans/kimodo_stochastic_humans.md`, still unreviewed.
   - A nearest-neighbour query runs every 0.1–0.2 s and blends with inertialization.
   - The animation owns the root, with bounded yaw and speed correction.
   - It argued that graphs quantize decisions to branch points 0.5–1 s apart, and that
     continuous steering needs combinatorially many clips.
3. **ARDY-tabulated graph** (agent proposal, 2026-09-30). This combines 1 and 2 around
   ARDY's own conditional structure.
   - **Library generation.** Offline, roll out ARDY H40 closed-loop under a command
     generator covering the intent layer's range: speeds 0–1.8 m/s, turns up to 180°,
     stops and starts, idles and, once prompts work, styles.
   - **Cost.** Measured throughput is 2,800 motion-seconds per wall-second, so 10 hours
     of motion takes about 13 s on an A40. Diversity is cheap.
   - **Edges.** Cut the streams at 4-frame token boundaries into 0.5 s edges. Each edge
     carries its start state (the preceding token: pose, velocities, contacts), the
     command it realized (speed and turn-rate bins) and its end state.
   - **Nodes.** Cluster start and end states (for example k-means over root-relative
     features, K ≈ 1–2k).
   - **Online runtime.** At each 0.5 s boundary, each human samples an edge whose start
     node matches its current node and whose command bin is nearest the intent layer's
     request. Then:
     - sampling is softmax over cost, with per-human streams;
     - the edge is rotated and translated onto the current root;
     - the small pose offset is inertialized over a few frames.
   - **Why it fits.**
     - Every transition is a genuine ARDY continuation from a nearby state, which ARDY
       continues cleanly: its seams measured no rougher than its interior.
     - Response delay is bounded by the edge length plus about 0.5 s of motion, which
       matches online ARDY at P = 0.5.
     - The runtime is pure-torch gather and blend, so 32k humans cost almost nothing.
   - **Risks.**
     - Coverage gaps: a node with no edge for the requested command. Mitigate with the
       nearest command bin, plus a coverage report.
     - Blend artifacts when clusters are loose.
     - The Core skin is not SOMA ([human body model](../decisions/human-body-model.md)).
     - Behavior diversity is limited to what the command generator covered.
4. **Kimodo-generated library** with any runtime above. Kimodo gives SOMA natively;
   10 hours of motion takes 10–23 min on an A40 (10–25 steps, bf16). Its continuation
   seams do not matter offline, where upstream's cross-fade is valid.

All options expose the same backend contract as the online generators, so the behavior
layer does not change (ledger D04, D05).

## Evidence

- ARDY continues cleanly from 4-frame histories: seams no rougher than the interior, and
  0.5 s response after a replan
  ([benchmark](../experiments/2026-09-30-motion-backend-benchmark.md)).
- **Option 3 prototype, 2026-09-30**
  ([experiment](../experiments/2026-09-30-ardy-library-graph.md); conclusions in
  [motion library](../core/design/motion-library.md)):
  - 65,536 humans step in 47 ms per 0.4 s edge on an A40.
  - On a held-out seed: path error 0.7–1.4 cm; median responses within 0.0–0.4 s of
    online ARDY replanned every 0.4 s; clean pose seams.
  - The hunch filter removes stooped gait: 0.2–0.6% of frames, by construction.
  - Costs: more frequent small foot sliding, root-speed seams 1.4–2.0× the interior,
    no local rotations yet.
  - The proposal's k-means nodes were replaced by k-nearest-neighbor candidates. An
    end-velocity cost was needed for stops and sharp turns.
- Options 1 (classic graph) and 2 (motion matching) are not prototyped. Option 3 is a
  graph whose dense, 0.4 s branch points already behave much like matching restricted
  to edge boundaries.

## Open limits

- **Response delay** (moved here from the retired demand-budget decision): the user has
  not set an acceptable simulated-time reaction delay. It now sets edge length and
  branch frequency. Report achieved delay; do not invent a threshold.

## What the prototype must measure

The benchmark's clean-seam result does not transfer automatically. In a graph, an edge's
history comes from a **different** rollout, so seams are clean only as far as clusters
are tight.

- **Seam jump before inertialization:** pose and velocity discontinuity against K,
  compared with ARDY's own interior frame-to-frame change.
- **Root drift:** command binning will raise path error far above online ARDY's
  0.5–0.9 cm. Report path error with and without a bounded yaw and speed root
  correction.
- **Coverage:** the fraction of (node, command-bin) requests served by an exact-bin edge.
- **Response delay and foot sliding:** in the same harness, against online ARDY at
  P = 0.5.

## What would decide it

- The user's preference between graph and matching semantics. Their 2026-09-30 framing
  was a stochastic motion graph, and option 3 now meets the measured bar above.
- Whether the remaining costs matter downstream: foot sliding and root-speed seams for
  LiDAR returns or policy learning, and local rotations for mesh LiDAR. An mjlab smoke
  test with the graph would show this.
- The [learning paradigm](../decisions/learning-paradigm.md): on a TANGO-style branch the runtime
  serves evaluation only.
