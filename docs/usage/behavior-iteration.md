# Behavior iteration

The loop for making crowd behavior look realistic: audit what the saved graph contains,
evaluate behavior on fixed seeds, replay the saved motion as video, compare a candidate
with the baseline on the same seeds, and keep a session record of what changed. All of
it runs on the saved motion graph (artifact `graph-default`); nothing loads a motion
prior or rebuilds the graph. Evaluation and comparison run on CPU; renders need an
offscreen GL context.

Run from the repository root after `source scripts/env.sh`. `--graph` defaults to
`$CROWD_GRAPH`, else the `graph-default` artifact ([data](../data.md)).

## 1. Audit the graph

```bash
crowd graph report [--library <library.pt>] [--video <crowd.mp4>] --out runs/<UTC>_behavior_report
```

Writes `audit.json` (raw data, graph sha256, provenance) and `index.html`, which embeds
the measured composition, connectivity and interactive example edges. It loads no
external scripts or fonts: open it directly, or serve the repository with
`python -m http.server 8090`. `--library` adds the library's command composition;
`--video` embeds a demo video by relative link.

**Definitions** (analysis cutoffs, not human calibration or acceptance thresholds):

- Frame direction compares root travel with the midpoint ARDY body heading of adjacent
  frames, including the edge's predecessor. Below 0.15 m/s is *still/slow*. Moving
  travel within 30° of facing is *forward*, 30–60° *oblique*, 60–120° *sideways*, and
  120° or more *backward*.
- An edge takes the class of its majority of frames (ties go to the earlier class).
  Forward and slow edges are split into *turning* at 10° of net yaw per edge. "Turn in
  place / slow" includes slow translation.
- Candidate transitions leave each edge's end boundary (`edge + 1`). The transition
  matrix shows what is available, weighting valid edges equally, not how often the
  runtime visits them.

## 2. Evaluate

```bash
crowd behavior evaluate --config configs/behavior/baseline.json --out runs/<UTC>_behavior_baseline
```

Each seed of the config's seed bank runs one plaza of `humans` humans for `seconds` at
the motion frame rate (20 fps), through the same `CrowdController` as the mjlab
`GraphCrowd` term: plaza intent and local steering, the path planner, the graph and the
velocity update ([architecture](../architecture.md#the-crowd-loop)). It excludes mjlab
interpolation, sensors and the reset lifecycle, so it is a behavior diagnostic, not a
replay of a simulator run or a throughput benchmark. Defaults: CPU, one thread.

**Config** (`configs/behavior/baseline.json`; unknown keys are rejected):

| Key | Meaning |
|---|---|
| `humans`, `seconds` | Population and duration of each seed (16, 20 s) |
| `seeds` | The seed bank: distinct integers; compare runs only on the same bank |
| `design` | Behavior design: `plaza` (default) or `encounter`; see [behavior designs](#behavior-designs) |
| `steering` | Plaza design only. `PlazaSteeringCfg` fields: plaza half sizes, speed range, wait probability and duration, arrival radius, personal space, avoidance horizon and gain, contact range, wall margin, obstacles `[x, z, radius]` |
| `encounter` | Encounter design only. `EncounterCfg` fields ([below](#encounter-design)) |
| `graph_overrides` | `GraphConfig` fields applied on load: cost weights, `temperature`, correction limits, `inertialize`, `lookahead`. `neighbors`, `vel_weight` and `hunch_ratio` are baked into the graph and cannot change |
| `description` | Free text |

Copy the baseline config for each candidate and change one mechanism at a time.

**Outputs**, in the run directory:

- `seed_<n>.npz`: the executed motion (a `Trace`): joints, planar roots, body headings,
  contacts and foot indices, plus selected edges, commands, goals, wait flags and
  selection diagnostics. Frame 0 is the initial pose.
- `seed_<n>.json`: metrics. Frame composition; shares of moving frames that are
  forward, sideways and backward; facing-to-travel error; body and path turn rates;
  acceleration; pairwise root gaps; planted-foot speeds; natural-successor share; path
  error; rollout timing.
- `summary.json` (all seeds), `skeleton.json` (for replay) and `run.json` (settings,
  graph sha256, environment, status).

The plaza steering draws from one generator shared by all humans in call order, so it
reproduces existing baselines but is not independent per human; see
[architecture](../architecture.md#randomness).

### Behavior designs

Two designs drive the same `CrowdController`, graph and metrics, so either can be
evaluated, rendered, compared, diagnosed and run in mjlab (`behavior/designs.py`):

- **`plaza`** (the original): random goals and waits, with continuous
  time-to-closest-approach avoidance that re-plans every edge
  (`behavior/plaza.py`). Its options are listed under [iteration order](#iteration-order).
  `configs/behavior/baseline.json` is the original baseline, and
  `configs/behavior/plaza_forward_posture.json` adds forward steering, the flat-ground
  mask and sticky posture.
- **`encounter`**: waypoint or velocity conditioning, a collision bubble, and
  stochastic decisions that a human keeps once made (`behavior/encounter.py`).
  `configs/behavior/encounter.json` uses random waypoints and the graph settings of
  `plaza_forward_posture.json` plus `turn_weight`, `turn_preserve_yaw` and
  `clearance_weight`, so the two designs compare on the same seeds and graph.
  `encounter_head_on.json` and `encounter_crossing.json` are two-human probes with
  fixed routes. `encounter_turns.json` is `encounter.json` plus the planted-twist mask,
  for the opt-in turning graph (artifact `graph-turns`, [data](../data.md)); pass that
  graph with `--graph`.

Compare only evaluations with the same population, duration, seed bank and graph.
Comparing the two designs measures the whole behavior, not one mechanism.

#### Encounter design

Each command step (one 0.4 s graph edge), every human:

1. **Follows its condition.**
   - `conditioning: "random_waypoints"`: random waypoints, preferring legs of at least
     `waypoint_min_leg_m`, with optional waits on arrival.
   - `"waypoints"`: fixed per-human routes (`waypoints`, `loop_waypoints`).
   - `"velocity"`: a commanded `(speed, heading_deg)` per human (`velocities`).

   Code can replace routes or velocities at run time with `set_waypoints` and
   `set_velocities`. A direction change of at least `turn_start_deg` becomes a turn.
   Turns of at least `inplace_min_deg` are made in place with probability
   `inplace_probability`. Otherwise the turn is an arc whose radius is drawn from
   `turn_radius_m` and limited to half the distance to the waypoint.
2. **Detects threats.** A neighbor is a threat when it is inside the bubble
   (`bubble_radius_m`) and approaching. It must also come within `pass_margin_m` of
   this human's body within `horizon_s` if both keep their intended velocities.
   `urgency` is 0 when the closest threat is at the bubble's edge and 1 at body contact
   (`2 * body_radius_m`).
3. **May switch to an intervention.** A cruising human with a threat switches with
   probability `1 - exp(-rate * dt)` per step, where
   `rate = switch_rate_hz * (1 - assertiveness) * urgency ** urgency_exponent`.
   Assertiveness is drawn per human at reset from `assertiveness_range`. While a human
   is blocked, its assertiveness decays with time constant `patience_s`.
4. **Picks a strategy and keeps it until it finishes.** The strategy weights are
   interpolated by urgency between `strategy_far` and `strategy_near`, in the order
   sidestep, veer, pivot, yield.

   | Strategy | What the human does |
   |---|---|
   | sidestep | Keeps its facing and moves sideways `sidestep_m` at `sidestep_speed_mps` while walking on slowly |
   | veer | Turns away by `veer_angle_deg` along an arc whose radius is interpolated from `veer_radius_m` (far, near) by urgency, walks on for `hold_s`, then turns back toward its condition |
   | pivot | Turns away in place by `pivot_angle_deg`, walks on for `hold_s`, then turns back |
   | yield | Slows to a fraction `yield_speed_scale` of its speed (possibly zero) until clear, at most `yield_max_s` |

   The side is away from the threat's closest-approach point. If that point is
   straight ahead, the side is away from the threat's current position. Head-on, the
   human passes on its right with probability `prefer_right`. With probability
   `side_noise` it takes the other side. If walls or obstacles block a side, it takes
   the other one; if both are blocked, it yields. A human the stop rule has already
   stopped does not draw yield. A sidestep, veer or pivot that would walk into a wall
   or obstacle ends early, and an ordinary arc that would leave the open area becomes
   an in-place turn.
5. **Obeys the stop rule.** Others are predicted at their executed velocities. Motion
   conflicts when the intended or still-executing velocity would bring the two bodies
   within `stop_clearance_m` (default 0.01 m) within `stop_horizon_s`, and standing
   still would keep them further apart. The intended velocity loses its component
   toward each conflicting neighbor, so a human may slide past. It stops if a conflict
   remains or less than 0.15 m/s is left. Before that check, a human's speed toward
   each neighbor is limited to (gap − clearance) / `approach_time_s` (default 0.8 s),
   so approaches slow down as the gap closes while passing stays free. A stopped human may still turn in place
   toward where it wants to face. Bodies already within the clearance step apart at
   `separation_speed_mps`. Executed motion lags commands, so the clearance is a
   target, not a guarantee; the diagnostics measure it. Two humans may make the same
   decision at once; only the stop rule resolves it. A human the rule stops within
   `arrive_blocked_m` of its waypoint counts as arrived.

Sidesteps, in-place turns and separation hold a body target through the graph's
retained-turn cost, so set `graph_overrides.turn_weight` (and usually
`turn_preserve_yaw`) when using this design. Commands are requests, and the graph
may pick a clip that walks on or turns forward into a neighbor. Set
`graph_overrides.clearance_weight` so that the stop rule also holds when the graph
selects clips. The design passes each human's four nearest neighbors and stationary
mirror images behind the nearest x and z walls, so the same clearance keeps bodies off
the walls. An in-place turn that the graph cannot realize within its expected time
(angle over rate, plus 1.5 s) continues as a tight arc. Walls and obstacles remove travel into
them. Randomness comes from per-human streams (`core.rng` scheme `encounter`), one
fixed block of uniforms per human per step, so results never depend on batch order or
composition. All settings are heuristics, not calibrated pedestrian behavior.

### Optional decisive encounter controls

These controls belong in `encounter`; each mechanism defaults off. The development
candidate `configs/behavior/encounter_decisive.json` combines them with the saved
`graph-turns` artifact. Use its paired `graph-turns-rotations` artifact for rigid
mjlab humans. Changing a command does not guarantee that the graph executes it.

The candidate uses a 5% stubborn reset probability, lower non-stubborn assertiveness,
0.8 s mutual-stall recovery, geometric turns, and bounded bypass completion. Its
head-on convention is always each person's own right; either human may decide at
the same time. Sidesteps and yields have small strategy weights. The body radius is
0.4 m, passing margin 0.2 m and bubble radius 2.5 m; the stop-clearance target remains
0.01 m. Existing graph costs balance end-velocity tracking, forward travel and clearance.
`encounter_blocked_head_on.json` adds an eight-seed slow, close-start probe.

This remains a development candidate. Close-pair stalls and joint-sphere contact
improved on its development bank, but long individual stops beside walls/obstacles,
foot sliding and unresolved redirects remain. A geometric minimum-angle command is
not proof of a shortest executed path. The browser controls have headless binding
tests; their interactive behavior has not been visually verified.

| Setting | Default | Effect when enabled |
|---|---|---|
| `retain_avoidance_heading` | `false` | Keeps the veer's body target when approach limiting removes its arc yaw |
| `stall_recovery_s` | `0` | After this much mutual near-stationarity, finishes an unsuccessful cruise, sidestep, yield or straight bypass attempt and requests a pivot; an ongoing turn gets time to finish |
| `minimal_avoidance` | `false` | Searches headings at one-degree increments for the least goal-direction deviation that clears all nearby bodies at their executed velocities; derives sidestep distance from the closest-point lateral deficit |
| `avoidance_margin_m` | `[0.02, 0.08]` | Random extra margin used by geometric avoidance, beyond `pass_margin_m` |
| `pass_commitment_s` | `0` | Replaces fixed post-turn `hold_s`: resumes the goal path when it clears nearby bodies, or after this maximum bypass duration |
| `pass_commitment_jitter` | `0` | For nearby parallel motion with little goal progress, shortens each person's deadline independently to a duration between `(1-jitter) * pass_commitment_s` and the maximum, using its existing hold draw |
| `stubborn_fraction` | `0` | Draws a persistent stubborn trait at reset; active stubborn humans initiate no intervention, while still obeying the stop rule |
| `nonstubborn_assertiveness_range` | `[0, 0.35]` | Reset assertiveness range used when the stubborn-population option is enabled; with fraction zero, the original `assertiveness_range` still applies |
| `stubborn_patience_s` | `[8, 14]` | Independently drawn finite stall patience; after it expires, retires stubbornness for the episode and forces a pivot attempt |
| `blocked_turn_first` | `false` | Experimental alternative: converts a constrained, nearly stationary veer into a retained in-place turn |
| `project_safe_velocity` | `false` | Experimental alternative: projects the command onto all radial approach/stop constraints instead of summing neighbor corrections |
| `anticipate_braking` | `false` | Experimental alternative: graph clearance also checks a slowing/stopping neighbor forecast while retaining its executed-velocity forecast |

Recovery uses executed speed below 0.15 m/s, a pending goal, and a nearby nearly
stationary neighbor. It gives a new straight phase at least the configured recovery
delay before ending it. Stubborn patience uses executed near-stationarity rather than
the instantaneous stop-rule flag. Two stubborn humans can both retire their traits
and act; there is no deterministic pair precedence. Finite patience guarantees a
resolution attempt, not physical progress through an impossible graph or geometry.
The optional parallel-motion trigger requires both speeds above 0.15 m/s, velocity
directions within 20 degrees, a root gap below body contact plus passing margin plus
0.1 m, and own goal-directed speed below 25% of executed speed. Useful forward bypasses
keep the maximum deadline. These cutoffs identify failed co-motion; they are not
calibrated human timing rules.

Geometric avoidance preserves the independently drawn head-on passing side; otherwise
it prefers the least detour toward the goal. Inside the passing margin it asks for no
further closing. If no heading clears every nearby body, it chooses the best minimum
clearance and retains the stop rule. These are constant-velocity predictions, not a
globally shortest collision-free path. Fixed angle ranges remain the default.

The Viser encounter panel queues these controls for the simulation thread. Population
fraction, assertiveness ranges and stubborn patience apply at the next reset; current
humans retain their drawn traits. The extra reset draw uses a separate counter in the
same per-human `encounter` stream, without shifting legacy random slots.

### External agents (the robot)

An encounter crowd can perceive bodies it does not control, such as the robot in the
traversal env. `EncounterBehavior.set_external` supplies their positions and
velocities each edge; in mjlab, `GraphCrowdCfg.perceive_entity` reads a scene
entity's root. Each external agent is one more neighbor in the threat test, stop rule,
separation, geometric avoidance, bypass completion and the graph's clearance cost; its
current velocity also stands in for its intended one. Without external agents nothing
changes. A human's attitude toward external agents is drawn at reset, apart from its
stubbornness toward humans, on its own counter of the `encounter` stream:

| Setting | Default | Effect |
|---|---|---|
| `robot_radius_m` | `0.3` | Body radius of an external agent for every rule above (humans use `body_radius_m`) |
| `robot_oblivious_fraction` | `0` | Share that does not perceive external agents at all, and walks into them |
| `robot_stubborn_fraction` | `0` | Share that obeys the stop rule and separation toward them but never intervenes for them; the trait does not retire |
| `robot_assertiveness_range` | `[0, 0.35]` | Assertiveness of the cooperative rest, used when an external agent is the closest threat |

Human stubbornness and patience do not apply to external agents, and the robot traits
do not apply to humans. The Viser encounter panel exposes the shares and range (next
reset) and the radius. `configs/behavior/encounter_decisive_robot.json` is the
decisive candidate with provisional shares of 5% oblivious and 10% stubborn.

### Standoff diagnostics

Pass `--standoffs` to `crowd behavior evaluate` to include the shared diagnostics in
each seed's metrics. `crowd behavior diagnose` always computes them from saved traces;
`--joint-stride 1` checks joint-sphere contact at every saved frame (default stride 2).
Paired comparisons include standoff deltas when both evaluations contain them.

The output states its definitions: a pair standoff lasts at least 1.2 s with both
mean edge speeds below 0.15 m/s and root gap at most 1.1 m, at least one pending goal,
and neither human in a locomotor intervention or active turn. Yield counts as idle.
A second measure includes stalled maneuvers so repeated decisions cannot hide a jam.
Both report counts per human-minute and duration p50/p95/max. Pending-goal stop
episodes, stopped seconds per human-minute, mean speed and goal arrivals accompany
them. Resolution is passing, or a gap increasing at least 0.25 m above its minimum;
unresolved encounters are reported as censored, not successful delays. Threat
encounters also report whether either human intervenes or accumulates at least 0.2 m
of lateral/backward displacement relative to facing. Cutoffs are diagnostic definitions,
not pass thresholds or calibrated human statistics. Resolution precision is one edge.

**Extra outputs.** Encounter traces also save, per human and command step:
`encounter_mode` (0 cruise, 1 sidestep, 2 veer, 3 pivot, 4 yield),
`encounter_phase`, `_threat`, `_eligible`, `_urgency`, `_switch`, `_stopped`,
`_separating`, `_assertiveness`, `_side`, `_ordinary_turn` and
`_ordinary_turn_radius`. They also save `turn_active`, `turn_target` and
`steering_evasive`, which is set while intervening, stopped or separating.
Decision traces additionally retain pair threats, effective assertiveness, hazard,
preferred/intended/commanded/executed velocities, stop-rule conflicts and limiting,
stubborn status/releases and execution-stall recoveries for stepwise investigation.

`seed_<n>.json` and `diagnose` add an `encounter` block:

- mode shares, switches per human-minute and strategy counts;
- the switch percentage given a threat, binned by urgency and by assertiveness;
- stop episodes, and in-place against arc ordinary turns with their radii;
- body clearance between `body_radius_m` cylinders: minimum, contact frames, frames
  within the tolerance, and contact events with their duration in pair-seconds.

Diagnostics use the bubble and the passing distance as their clear/close radii. The
ordinary redirect metric counts encounter interventions as non-ordinary.

## 3. Replay

```bash
MUJOCO_GL=egl crowd behavior render --run runs/<UTC>_behavior_baseline --seed 0 --view overview
MUJOCO_GL=egl crowd behavior render --run runs/<UTC>_behavior_baseline --seed 0 --view follow --human 3
```

Renders saved joints as MuJoCo capsules (no graph stepping, Warp or LiDAR) into
`render_seed_<seed>_<view>/crowd.mp4` inside the evaluation, with `render.json` (trace
hash, camera, provenance, status). Views: `overview`, `top` and `follow` (the camera
follows `--human`). Arrows on the floor show body facing, which separates curved
forward travel from sidestepping. `--fps`, `--width` and `--height` (20, 960, 540) are
independent of the motion frame rate; `--out` writes another variant elsewhere. Existing
renders are never overwritten.
Use `--azimuth` and `--elevation` (degrees, negative elevation looks down) to
inspect a follow view whose feet are hidden by a wall or another human.

`MUJOCO_GL=egl` needs a GPU with EGL; on a CPU-only machine use `MUJOCO_GL=osmesa` if
OSMesa is installed. To see a candidate in the simulator with the LiDAR, use
`crowd demo video` ([demos](demos.md)).

## 4. Compare

```bash
crowd behavior compare --baseline runs/<UTC>_behavior_baseline \
    --candidate runs/<UTC>_behavior_candidate --out runs/<UTC>_behavior_compare
```

Both evaluations must share the graph (sha256), frame rate, population, duration and
seed bank. Writes `comparison.json` (candidate minus baseline per seed and metric, both
manifests) and `index.html`, which shows the deltas beside the renders found in either
run. Review spacing, acceleration and planted feet alongside the direction shares. No
automatic pass threshold is applied.

## Iteration order

1. Audit forward and turning coverage, freeze the baseline config and evaluate it.
2. Change one steering or scoring mechanism and evaluate a candidate.
3. Compare matching seeds, then replay the saved motion from several views.
4. Before promoting a candidate to mjlab, inspect coverage failures, sideways and
   backward travel, abrupt yaw, close gaps and foot sliding.

The baseline keeps all optional orientation mechanisms disabled. Candidate configs can
enable them separately; they are heuristics, not calibrated pedestrian behavior:

| Setting | Effect |
|---|---|
| `graph_overrides.facing_weight` | Penalize speed minus forward velocity in the moving body frame. Midpoint body headings keep curved forward motion distinct from sidesteps. Default `0`. |
| `graph_overrides.heading_weight` | Prefer native end-of-edge body heading aligned with the requested path's terminal velocity. Stationary requests have no heading preference. Default `0`. |
| `graph_overrides.heading_deadband_deg` | Ignore this many degrees of heading error before applying the heading cost, letting path and pose scores choose among aligned clips. Default `0`. |
| `steering.forward_steering` | Limit ordinary travel's lead over executed body facing (`forward_lead_deg`, default 30), slowing large redirects (`forward_speed_floor`, default 0.5). Default `false`. |
| `steering.relax_orientation_for_avoidance` | Release facing/heading costs for humans with imminent conflicts or close humans, walls or obstacles. Evasive steering retains lateral/backward escape commands. Default `false`. |
| `steering.committed_turns` | Replace destination intent with WALK/WAIT/TURN states. Sample a retained heading target, biased toward 90 degrees, and finish on executed body facing. Default `false`. |

`contact_m` controls the hard neighbor push and also the close/imminent-conflict
classification used by the optional steering rules. Changes can affect spacing,
speed and direction shares together. Committed turns can remain unfinished or
accumulate at walls when steering and available graph turns cannot realize the target.
Do not treat forward shares alone as evidence of successful turning.

These mechanisms run through `CrowdController` in every consumer. Costs live in
`graph/costs.py`, intent in `behavior/intent.py` and steering in
`behavior/steering.py` ([development](../development.md#extending-the-runtime)).

### Response and continuity diagnostics

```bash
crowd behavior diagnose --run <evaluation> --coverage --out runs/<UTC>_diagnostics
crowd behavior probe --config <baseline.json> --config <candidate.json> --out runs/<UTC>_response
```

`diagnose` reads saved motion to measure pose/root seams, progress despite moving
commands, wall excursions, direction by ordinary/evasive context and command response.
`--coverage` also loads the same graph to audit native turning candidates at visited
boundaries. Conditional response windows end when commands change; report censored
events with successful delays. Older traces require `--edge-frames` when their
manifest lacks it.

`probe` uses the shared controller and local steering for fixed 90/180-degree requests,
with one human per empty plaza. It separates root-path and body-heading response,
including replan waiting and nonresponses. It does not test stochastic intent or
crowd spacing. Its angular/speed/duration cutoffs are diagnostic definitions, not
acceptance thresholds.

### Posture and flat-ground transitions

```bash
crowd graph edge-review --config <candidate.json> --out runs/<UTC>_edge_review
```

This derives annotations from the saved tables without changing the graph file:
head-height ratio, signed Hips-to-Neck pitch, trunk inclination, forward head offset,
pelvis height range and foot heights. Body-local Y is absolute ground height; only
planar translation and yaw were removed. The report includes distributions,
correlations, stream variance, joint posture/motion coverage, mask dead ends and
deterministically sampled contact sheets. Inspect the sheets before choosing a
class split. These geometric labels are experimental, not calibrated human styles.
Use `--edges <id> <id> ...` to add a contact sheet of specific edges found in a
saved rollout, plus a top view of their feet and hip line; the sampled population
sheets remain in the same report. The report also counts what the planted-twist mask
removes per motion class, and the dead ends it adds.

All new behavior switches default off. Graph settings belong in `graph_overrides`:

| Setting | Meaning |
|---|---|
| `vertical_mask` | Enable a flat-ground mask over saved valid edges. Default `false`. |
| `vertical_pelvis_range_m` | Maximum within-edge pelvis height range, including the predecessor; default `0.15` m. |
| `vertical_foot_min_m` | The lower foot must reach this absolute height at least once, including the predecessor; default `0.12` m. |
| `vertical_foot_max_m` | Reject an edge if any contact-foot joint exceeds this height; default `0.60` m. |
| `vertical_lowest_foot_max_m` | Optional maximum height of the lower foot over an edge, including the predecessor; default `0` disables this extra criterion. It catches brief airborne segments missed by the minimum-height test. |
| `posture_enabled` | Sample and retain a per-human upright/hunched style, and select a matching reset pose. Default `false`. |
| `posture_weight` | Soft squared pitch error cost, default `0`. A 20° error costs one weight unit; no hard posture filter. |
| `posture_cutoff_deg`, `posture_target_margin_deg` | Signed-pitch class split and target offset; defaults `15` and `10` degrees give targets of 5°/25°. These are reviewable heuristics. |
| `posture_initial_hunched` | Hunched probability at reset only, default `0.10`. |
| `posture_upright_to_hunched_rate`, `posture_hunched_to_upright_rate` | Continuous-time switching rates per simulated second, defaults `0.005`/`0.020`; read live. |
| `turn_weight` | Soft retained body-target error cost during an ordinary turn phase, default `0`. Unlike the heading cost it does not require translation. |
| `turn_preserve_yaw` | Disable path-straightening yaw correction during that phase, keeping native clip turns; default `false`. |
| `start_vel_weight` | Soft candidate entry-velocity error against the last executed planar velocity, in seconds; default `0`. It can reduce transition acceleration but delay avoidance or turns. |
| `clearance_weight` | Cost per meter by which a candidate clip closes on a neighbor inside the clearance distance, beyond what standing still would. Neighbors move on at their executed velocities. Only command sources that pass neighbors (the encounter design) use it; default `0`. |
| `twist_mask` | Mask planted twists: edges whose foot-contact labels are all on for the whole edge while the mean foot direction is yawed from body forward by more than `twist_max_foot_yaw_deg` (edge mean). It removes twisted standing and straight edges as well as in-place turns on planted feet; stepping turns, whose contact labels change, stay. Like the flat-ground mask, it leaves the saved tables intact and uses the same hold fallback at an all-masked boundary. Default `false`. |
| `twist_max_foot_yaw_deg` | Foot yaw limit of the twist mask; default `21.4`, the p99 over graph-default's admissible edges (a heuristic, not a calibrated limit). |

Style uses the independent `core.rng` scheme `graph_posture`; it does not consume
the legacy plaza RNG or motion-selection stream. Exact two-state transition
probabilities apply at edge boundaries, with the first edge retaining the reset
draw. Diagnostics count observable boundary switches, not hypothetical intervening
continuous-time switches. Class-matched starts require matching predecessor and
edge-average pitch; slow starts are preferred. A missing requested start class
raises an explicit error.

The vertical mask leaves saved tables intact. If a current boundary has no finite
admissible candidate, the runtime holds the current pose/root/heading, counts the
fallback and retries at the next edge. It never silently chooses a masked edge.
Reset fails clearly when no eligible start exists. Executed vertical diagnostics
include inertialization, which can differ from a candidate's table heights.

`steering.turn_before_walk` (default `false`) retains an ordinary direction target
when it exceeds `redirect_start_deg` (45°), requests forward motion at up to
`redirect_speed_mps` (0.20 m/s), then resumes travel within `redirect_finish_deg`
(15°). Waiting or evasion cancels the phase. This request is soft: a graph with
scarce slow turns may need walking-turn speeds and may still fail to realize it.
Tune it together with `turn_weight`; raising costs alone does not add coverage.

`diagnose` adds executed pelvis/lowest-foot heights, actual posture shares and
class crossings, latent shares/switches and configured expectations. It also
classifies ordinary goal/body mismatches of at least 45°: body alignment within
20° followed by forward travel toward the target for 0.25 s is turn-then-forward;
a preceding sustained 0.25 s of sideways/backward travel takes precedence. Attempts
last at most 6 s and are censored by avoidance, waiting or changed goals. Unresolved
attempts and initial alignments remain in the denominator. All angles/durations
are stated diagnostic definitions, not acceptance thresholds. Curved forward paths
remain forward because direction is measured against moving body facing.
Counts are also split into initial alignment, a newly sampled goal, and resumed
alignment after interruption; do not describe every attempt as a new goal change.

Use the [Viser behavior panel](demos.md#live-behavior-controls) to adjust these
runtime preferences without restarting; initial posture changes apply at reset.

Review held-out seeds and longer rollouts before promotion. Extract and inspect
timestamped stills or dense frame sequences from saved replay, recording the seed,
view and times: a completed MP4 alone is not a visual review.

## Iteration sessions

A session keeps a bounded series of requests in one run folder, with a deadline, request
names that can be used only once, and a journal that records rule changes apart from
results, so a proposed setting is never presented as a tested change.

```bash
S=$(crowd iterate start turn-commit --hours 4 --note "Plan: committed turns biased to 90°")
crowd iterate run $S 000_baseline --timeout 600 -- \
    crowd behavior evaluate --config configs/behavior/baseline.json --out {out}
crowd iterate run $S 001_turns --timeout 900 --rule "Committed turns, 90° mode" -- \
    crowd behavior evaluate --config <candidate.json> --out {out}
crowd iterate note $S "Turns look abrupt" "Seed 0 overview: ..." --kind observation
crowd iterate status $S
crowd iterate stop $S --reason done
```

- `crowd iterate start` prints the session folder, `runs/<UTC>_<name>/`. It holds
  `session.json` (deadline, executor, status), `journal.md`, `queue/` and
  `iterations/<name>/`.
- In the request command, `{session}`, `{name}` and `{out}` (=
  `{session}/iterations/{name}`) are substituted. A leading `crowd` runs this
  environment's CLI.
- Each request leaves `queue/<name>.request.json` → `.working.json` → `.done.json`, plus
  `queue/<name>.log` and `queue/<name>.result.json` (status, exit code, timing).
- **Local executor** (the default): each request runs immediately as a subprocess,
  bounded by its `--timeout`, the session deadline and a `STOP` file in the session
  folder. `crowd iterate stop` writes `STOP`.
- `--kind` of a note is `plan`, `rule`, `observation` or `decision`.

**On the cluster** (`--executor slurm`): `start` submits one worker allocation
(`scripts/slurm/iterate_worker.sbatch`, default `--partition=rl2-lab,overcap
--qos=short`; change with `--sbatch-arg`) and records its job ID. `run` then queues
requests, and the worker drains them (`crowd iterate worker`). From the login node, run
`crowd iterate watch $S`: it journals results as they arrive and cancels only the
recorded job when the GPU-wait limit (`--gpu-wait-limit`, 300 s) or the deadline
passes. Both executors leave identical records. See [Slurm](../slurm.md).

**Agent inside a batch allocation:** submit `scripts/slurm/iterate_codex.sbatch` to
start `crowd iterate agent` and Codex's noninteractive process on the compute node.
The agent can iterate either [behavior design](#behavior-designs): its prompt names
the design and the frozen reference configs, and every request is an ordinary
evaluation, render or comparison.
It creates one outer run with a local `session/`, records the prompt and agent logs,
and enforces both the agent and request deadlines. The default prompt freezes and
reviews a baseline, refines shared behavior/cost mechanisms, and compares paired seeds
and saved-motion renders. Read [batch Codex setup](../slurm.md#start-codex-automatically-in-a-batch-job)
for submission, authentication, monitoring and stop commands.
