# 2026-10-06: encounter behavior design

Status: complete. The design is implemented, opt-in, and measured on development seeds.
Tuning is left to the iteration agent.

On 2026-10-06 the user asked for an **alternate** behavior design beside the original
random-goal plaza design, so that the iteration agent can work on either (ledger D16):

- waypoint or velocity conditioning, with random waypoints for comparison;
- a configurable collision bubble;
- a per-step switch probability that depends on the human's assertiveness and on the
  distance of the closest point in the bubble;
- committed interventions: sidestep, turns of various radii (in place and gradual), and
  yield;
- humans stop before colliding, with a starting tolerance of 1 cm.

Two humans may decide at once; there is no tie-break.

Usage and every rule: `docs/usage/behavior-iteration.md#behavior-designs`. Generating
in-place turns is a separate note: [turning library](2026-10-06-turning-library.md).

## Implementation

- **New modules:**
  - `behavior/encounter.py`: `EncounterCfg` and `EncounterBehavior`.
  - `behavior/designs.py`: builds either design and gives the scene section and
    context radii.
  - `planner.plan_arc`: arcs of a chosen radius, and in-place heading sweeps.
  - `metrics/encounters.py`: decisions, stops, cylinder clearance, and joint-sphere
    clearance at capsule radii.
- **Graph runtime:** `ClearanceCost`, enabled by `clearance_weight` (default 0, so it
  is skipped and earlier results stay bit-identical). A command source passes neighbors
  through `motion_options`.
- **Seed scheme:** `encounter`.
- **Consumers:**
  - The evaluator, diagnose, replay, `GraphCrowd`, the plaza env and the demo
    `--config` select a design from the config (`"design"`; default `plaza`, and the
    serialized config is unchanged).
  - Viser shows encounter controls in place of the plaza steering controls.
  - Diagnose adds joint-sphere clearance for both designs.
- **Configs:**
  - `configs/behavior/encounter.json`: random waypoints, the plaza graph settings plus
    `turn_weight` 1, `turn_preserve_yaw` and `clearance_weight` 20.
  - `encounter_head_on.json` and `encounter_crossing.json`: two-human probes, 8 seeds.
  - `plaza_forward_posture.json`: the round-2 plaza development candidate.
- **Tests:** `tests/unit/test_encounter.py` (16).

## Design iterations on the default graph

All runs: CPU, one thread, zhurong, development seeds. Each defect below was found by
tracing saved motion, and each fix is a rule, not a weight search. Per-run outcomes are
in the run log.

1. **Stop rule predicted others at their intended velocity** (job 4024324). A human
   already stopped by the rule still looked like it was walking forward, so standing
   never looked safer. Head-on contacts occurred, and crowd contact frames reached
   23–90%. Fix: predict others at their executed velocity and check the human's own
   still-executing motion as well; a stopped turner pivots in place (job 4024359).
2. **Bodies stuck touching.** Pairs stopped while touching near nearby waypoints, for up
   to 15 s. Fix: bodies within the clearance step apart, and a human blocked within
   1.5 m of its waypoint counts as arrived (job 4024380).
3. **Commands are only requests.** One pivot was realized by a walking-turn clip that
   walked *into* its neighbor while commanded to stand, because the turn cost
   outweighed path tracking and the graph has no in-place turns. Separation commands of
   0.3 m/s executed at 0.01 m/s. Fix: the graph `ClearanceCost`, weight 20, applies the
   stop rule when clips are selected (job 4024393).
4. **Standoffs.** Both humans stood face to face for up to 18 s:
   - the side was taken from a degenerate closest-approach point;
   - yield was drawn by humans already blocked;
   - the all-or-nothing stop cancelled sidesteps.

   Fix: fall back to the current offset for the side, exclude yield when blocked, and
   remove only the closing velocity component so humans can slide past (job 4024466).
5. **Walls** (job 4024528). Veers walked into walls, the wall push commanded 2–3 m/s,
   and stalled in-place turns stood at a wall. Fix:
   - interventions end early when the way ahead is blocked;
   - arcs near walls become in-place turns, and stalled in-place turns become tight arcs;
   - the push and command speed are capped;
   - stationary wall mirrors join the clearance cost.

   Job 4024580 then made the clearance cost per neighbor, so a wall mirror could no
   longer mask a person.
6. **Fast tangential passes cut inside** because execution lags commands. Fix: the speed
   toward each neighbor is limited to (gap − clearance) / `approach_time_s` (0.8 s)
   (job 4024609).
7. **The 0.25 m cylinder was the wrong body.** Joint-sphere clearance at capsule radii
   showed contact in 13–19% of crowd frames at root gaps of 0.52–0.70 m, mostly arm
   against head or arm. The per-human maximum horizontal joint reach from the root has a
   median of 0.39 m and p95 of 0.50 m. Fix: default `body_radius_m` 0.35 (final job
   4024647).

## Final measurements (job 4024647; plaza from job 4024411)

Plaza is `plaza_forward_posture.json`; encounter is `encounter.json`. Same graph,
16 humans, seeds 0/1/2, at 20 s and 60 s. Joint-sphere clearance is the minimum over
all pairs and frames (stride 2) at capsule radii, so it misses mid-bone surfaces.

| 60 s, seeds 0/1/2 | Plaza | Encounter |
|---|---|---|
| Forward / sideways / backward share of moving frames | 70–77 / 9–10 / 5–10% | 94 / 1 / 0.1% |
| Planted-foot speed p95 | 0.43–0.47 m/s | 0.30–0.32 m/s |
| Acceleration p95 | 4.6–5.0 m/s² | 2.6–2.8 m/s² |
| Mean speed | 0.58–0.64 m/s | 0.43–0.50 m/s |
| Minimum root gap | 0.43–0.60 m | 0.68–0.70 m |
| Joint-sphere clearance (contact frames) | −0.088 / −0.075 / −0.054 m (0.5–1.2%) | +0.008 / +0.056 / −0.011 m (0–0.3%) |
| Ordinary redirects: turn then forward / lateral / unresolved | 24–31 / 28–37 / 167–189 | 12–16 / 0–1 / 72–105 |
| Wall excursion frames | 0 | 0 |

The encounter crowd at 60 s also shows:

- humans stopped in 7% of steps;
- 9.0–9.4 switches per human-minute;
- strategies over the three seeds: 123 sidestep, 73 veer, 75 pivot, 174 yield.

At 20 s the encounter crowd has no joint-sphere contact, minimum +0.008 m. The
probes have no contact on any measure: head-on joint clearance is at least 0.068 m and
crossing at least 0.263 m. In seed 0 at 60 s (development settings before the final
radius change), the per-step switch percentage given a threat rises with urgency from 3%
to 35% and falls with assertiveness from 31% to 9%, as designed.

## What executes on the default graph (60 s, seeds 0/1/2)

| Commanded | Executed |
|---|---|
| Ordinary arc turns | Median yaw 18°/s (p90 48°/s) at 0.81 m/s |
| Ordinary in-place turns | Median yaw 2.8°/s and about 0.01 m/s, effectively standing; 88 of 180 hit the stall fallback and became arcs |
| Pivot interventions (in place) | Same as in-place turns |
| Sidesteps | Of moving frames, 9% sideways, 36% oblique, 55% forward |
| Veers, yields, cruise | 94–98% forward |

This matches the [turning audit](2026-10-06-turning-library.md): the graph has 8
admissible in-place turning edges, none faster than 31°/s. Gradual turns are realized;
in-place turns and true sidesteps need new motion.

## Limits

- The 1 cm stop is a target. Executed motion lags commands, so the worst crowd overlap
  is about 1 cm by joint spheres at 60 s. The cylinder metric at 0.35 m is stricter:
  seed 2 reaches −0.023 m in 10.9% of frames.
- Ordinary redirects remain mostly unresolved, but the lateral and backward resolutions
  that dominated the plaza design are gone.
- Development seeds only. Held-out seeds 101/307/911 were not used.
- These are heuristics; no human calibration.

- **A multi-human jam.** In seed 0 at 60 s, three or four humans near (2.3, 3.5) freeze
  for more than 10 s from about 45 s. They are in pivots or yields; even commanded
  walk-away speeds of up to 1.16 m/s execute at 0 m/s. With the body unturned, walking
  along the pivot target is sideways relative to the body. With neighbors on several
  sides, every moving clip pays the forward-preference and clearance costs, so standing
  wins. In-place turns would let them face away. Until the graph has them, pivots and
  in-place ordinary turns can deadlock small groups.

## First look on the merged turning graph (job 4024666)

This uses the same encounter configs with `--graph` set to the merged default-plus-turning
graph (`runs/20261007T021516Z_motion_graph_turns/graph.pt`). That graph's own
validation, including heading twists with planted feet, is in the
[turning library note](2026-10-06-turning-library.md).

| 60 s, seeds 0/1/2 | Default graph | Turning graph |
|---|---|---|
| Ordinary in-place turns, median (p90) yaw rate | 2.8 (17) deg/s | 83 (187) deg/s, root 0.24 m/s |
| Pivot phase 0, median yaw rate | 2.8 deg/s | 58 deg/s |
| In-place turns that stalled into arcs | 87 of 179 | 0 of 147 |
| Redirects: turn then forward / lateral / unresolved | 12–16 / 0–1 / 72–105 | 45–47 / 2–5 / 63–64 |
| Forward share; mean speed | 94%; 0.43–0.50 m/s | 94–96%; 0.52–0.61 m/s |
| Joint-sphere clearance (contact frames) | +0.008 / +0.056 / −0.011 m (0–0.3%) | −0.070 / −0.022 / −0.047 m (0.3–5.0%) |
| Planted-foot p95; acceleration p95 | 0.30–0.32 m/s; 2.6–2.8 m/s² | 0.38–0.40 m/s; 3.3–3.4 m/s² |

The probes stay contact-free: head-on joint clearance is at least 0.064 m and crossing
at least 0.039 m. The 20 s crowd seeds all touch, by up to 0.07 m.

In-place turning is now realized, and turn-then-walk resolutions roughly triple, which
is the original redirection objective. Contact, sliding and acceleration regress.
In-place clips still translate (median 0.24 m/s), and arms swing during turns. This is
the first evidence for the agent to iterate on, not a tuned result. The graph is still
unmasked: 1,110 of its in-place edges and 2,354 standing or straight edges hold planted
twists. No MuJoCo render of this run was made.

## Visual review and integration

- **Stills inspected** (capsule replay; still-only tools, so continuous playback was not
  reviewed):
  - round-4 head-on seed 0 overview, 5.4–9.4 s: a mutual-stop standoff of about 3 s
    before the two pass;
  - round-4 crowd seed 0 overview, 1–16 s;
  - final 60 s crowd seed 0 overview at 5/15/25/35/45/55 s, which shows the late jam
    near the wall;
  - final follow view of human 8, 5.6–8.8 s around its sidestep (6.0 s) and pivot
    (7.6 s): close passes and no visible interpenetration at this scale; many bodies
    lean forward.
- **mjlab** (`runs/20261007T021950Z_mjlab_graph_demo/`, job 4024652, A40): the first
  `GraphCrowd` run of the encounter design.
  - Run: 16 rigid humans for 10 s, the encounter config with live-tuning controls built.
  - Spacing: minimum root gap 0.72 m.
  - LiDAR: 5,760 rays and about 123 human hits per frame.
  - Cost: env step 3.9 ms; render 34 ms.

  Stills at 1–9.8 s look like the evaluator's crowd. The Viser panel itself was not
  opened.
- **Validation** (job 4024648):
  - 226 CPU unit tests pass, 3 skipped;
  - all 12 fast golden cases match the expected files on zhurong;
  - Ruff passes after an import-order fix;
  - the plaza 60 s seed 0 trace is bit-exact with the previous day's evaluation.
