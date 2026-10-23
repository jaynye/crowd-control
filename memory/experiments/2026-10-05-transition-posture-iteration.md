# Saved-graph transitions, posture and live tuning

Development improvements were measured, but **no new candidate was promoted**.
190 local requests completed; final-candidate frame extraction failed because its
command resolved virtualenv Python to the managed base interpreter, which cannot
import imageio. Experiments stopped at 00:12 UTC on 2026-10-06 under the user's
explicit stop-on-failure instruction. Deadline was 02:42 UTC. Fresh held-out seeds
101/307/911 remain unseen. No failed request was overwritten or retried.

Outer run: `runs/20261005T225742Z_behavior_iteration/`, existing local session and
launcher allocation 4018869, A40 on chitti, driver 595.91.07, torch 2.9.0+cu128,
CUDA 12.8, commit 5c1ef9c plus preserved initial dirty work and new changes.
CPU evaluation uses float32, one thread; capsule replay uses EGL. Each request
records provenance, source diff, graph hash, rule and timeout. The outer run was
logged before experiments. No priors, rebuilds, dependency/third-party edits,
Slurm submissions, commits or golden expected-file writes.

Full report, per-seed tables, rejected candidates and image inventory:
`runs/20261005T225742Z_behavior_iteration/report.md`. Chronological evidence:
`runs/20261005T225742Z_behavior_iteration/session/journal.md`.

## Plan and implementation

Freeze original and incumbent, evaluate dev seeds 0/1/2 before changes (000/001),
audit saved tables and inspect classes, add one opt-in mechanism at a time,
reserve fresh held-out seeds and GPU integration. Initial work is preserved in
`initial/`; reference hashes are in `reference_freeze.json` under the outer run.

Implemented load-time edge annotations/contact sheets; vertical masks with
explicit hold-and-report fallback; matched starts and soft sticky posture with
independent `graph_posture` RNG; optional retained-target/native-yaw/entry-velocity
costs and turn-before-walk steering; executed height/posture/redirect metrics;
queued Viser controls and demo config loading. All mechanisms default off, all
consumers remain on `CrowdController`, and legacy plaza RNG is unchanged.

## Audit and reviewed calibration

002 verifies 134,820 valid edges, absolute body-local Y, neutral head 1.684192 m.
Head-ratio median .93521, 20.68% below .90. Signed Hips–Neck pitch median 12.56°,
quartiles 5.89/18.81; inclination median 12.97°, quartiles 6.78/19.09. Signed pitch
correlates −.628 with head ratio, .969 with head offset, .059 with speed;
62.6% of pitch variance is between streams.

Nine sheets (72 edges, three poses each) inspected: straighter torsos under 10°,
forward lean at 18–25° and over 25°. Low head also reflects bent knees/suspension.
Signed pitch with split 15° and targets 5°/25° is an **experimental heuristic**,
not calibrated human posture. Natural successors retain class often, but not always.

Initial mask 003: pelvis range ≤.15 m, lower foot reaches ≤.12 m, all feet ≤.60 m,
including predecessors; removes 3,069 edges. Long development exposed a brief
.211 m lower-foot excursion amplified by pose blending. Reviewed 091 brief-airborne
sheet and native edges 136330/331/332 motivated peak lower foot ≤.12 m (population
p95 .053 m, p99 .232 m). Final audit 094 removes 4,072 edges (3.02%), affects 14,306
boundaries; 437 dead overall, 413 after originally valid edges, **zero after retained
edges**. Upright forward-straight/forward-turn/slow-turn counts 31,639/3,957/**2**;
hunched 19,164/1,255/**3**. Same-class forward-turn availability 57.4%/46.6%; natural
successor class persistence 93.5%/86.8%. This supports soft posture preference.

## Development evidence

Candidate for continued review, **not selected/frozen**:
`runs/20261005T225742Z_behavior_iteration/configs/forward_floor080_peak012.json`.
Original destination intent, facing weight 1.5, forward speed floor .8, posture
weight .3 and final mask; explicit turn phase disabled. Matched 60 s, 16 humans,
seeds 0/1/2, incumbent 087 versus candidate 096:

| Metric | Incumbent | Candidate |
|---|---|---|
| All moving forward % | 67.9 / 68.8 / 61.1 | 76.5 / 76.1 / 70.5 |
| Ordinary moving forward % | 81.9 / 83.9 / 78.2 | 94.4 / 94.0 / 93.7 |
| Minimum root gap m | .635 / .573 / .470 | .433 / .605 / .550 |
| Planted-foot speed p95 m/s | .470 / .442 / .449 | .429 / .431 / .468 |
| Seam acceleration p95 m/s² | 11.63 / 10.91 / 11.48 | 11.30 / 11.48 / 12.62 |
| Maximum lower-foot height m | 1.336 / .332 / .871 | .062 / .093 / .112 |
| Posture crossings / human-minute | 18.00 / 19.44 / 18.31 | 5.75 / 4.31 / 5.44 |

No fallback/wall excursion or lower-foot frames above .12 m in candidate runs.
Small negative foot heights remain. Executed pelvis edge-range maximum .152 m
slightly exceeds the table criterion because blending changes poses. Seed-0 spacing
regresses, while sliding, acceleration and seams are mixed. No invented pass limits.

Initial hunched sample 7/48 (14.58%) versus configured 10%; latent share 21.46%
versus 14.79% finite-horizon expectation; observed pose share 22.40%, agreement
96.2–97.1%. Thirty latent changes over 48 human-minutes; realized U→H .00801/s,
H→U .01958/s versus configured .005/.020/s. Small-bank variability was not tuned
away. A 50,000-human unit test separately checks CTMC probabilities.

Ordinary ≥45° reorientation counts (body-then-forward / lateral-backward /
unresolved): incumbent 62/176/473, candidate 80/97/536. New-goal subset:
12/2/65 versus 15/0/78. Most attempts are resumed alignments; most remain unresolved.
Forward share does not establish mostly completed turns. Controlled probes 099:
candidate 47/48 body responses in each 90°/180° case versus incumbent 48/48.
Successful per-seed body-delay medians 2.25–2.47 / 4.82–5.29 s versus incumbent
2.72–2.80 / 6.06–6.80 s; one seed-2 human censored in each case.

Rejected/limited alternatives: .2 m/s explicit turn phase (029) stalls, only 16/48
controlled 180° responses. Weight 3 (036) restores motion but loses forward travel.
Native-yaw preservation removes counteracting correction (057); walking turns .8 m/s
(066) complete controlled responses but do not dominate crowded metrics. Stronger
posture/pose/turn costs (073/075/076) trade spacing and sliding. Entry velocity .2
(081) improves seams but worsens spacing/response. Contact 1.4 m (097/098) improves
gaps but loses forward share and briefly crosses a wall; 1.3 (100/101) also loses
forward share. Full one-mechanism iteration sequence is preserved in the journal.

## Inspected images, validation and unfinished work

Actually inspected: all 002 edge sheets and 091 brief-height/requested edges;
seed-0 overview 009/010/035/042; seed-0 upright human 0/hunched human 2 follow sheets
022–025; seed-0 human 0 turn alternatives 063–065 and 070–072; seed-2 human 7 at
39.8–41 s in 092/093. Reversed camera 093 exposes feet hidden in 092. Exact times
are in the report. Capsule occlusion/artifacts and still-only inspection limit
conclusions. Continuous playback and browser interaction were not verified.
Final candidate 102 overview MP4 is **not** a visual review: extraction failed.

Final CPU suite: **165 passed**, one Torch deprecation warning; Ruff passes.
Earlier parity request 080: all 12 fast golden cases match preserved initial code
(283 arrays/27 JSON), and both reference trace banks reproduce bit for bit. Both
trees retain historical expected-file mismatches on this host; expectations unchanged.
This check predates final peak-mask/diagnostic additions and was not repeated after
stopping. Final inventory confirms graph/references/dependencies/expected unchanged.

Viser callbacks, queued mutation, private configs, live rate/reset semantics and
parameters have headless CPU tests. The prepared `mjlab_live_check.py` was **not run**;
prior-run GPU validation does not validate these new controls.

Unfinished: final candidate images, freeze/held-out evaluation, new GPU integration,
browser review, dependable mostly-turn-then-walk behavior, calibration, spacing/seams,
and training-scale performance. Next: use literal `.venv/bin/python`, finish review,
freeze and test unseen 101/307/911 at 20/60 s, then small mjlab check. Later behavior
work should test avoidance over the executed .4 s horizon and reachable upright
walking-turn sequences; cost tuning cannot supply missing slow turns.

## Follow-up: comparison video on seen seeds (2026-10-06)

For the user's visual review, the unchanged candidate was evaluated on seeds 0/11/23/47
at 60 s (`runs/20261006T172820Z_behavior_eval/`). Seed 0 reproduces request 096 bit for bit
in every motion array. The video `runs/20261006T175357Z_behavior_video/comparison.mp4`
sets the candidate beside the original baseline and the round-1 selected candidate on the
round-1 held-out seeds 11/23/47, which are **seen**, so this is not held-out evidence.
Compared with round 1:
- ordinary forward travel rises from 81–83% to 93%;
- hunched frames fall from 36–42% to 13–21%;
- posture switches fall from 17–19 to 5–6 per human-minute.

Ordinary redirects stay mostly unresolved. Minimum root gap is mixed (0.49/0.47/0.63 m
against 0.50/0.58/0.57 m). Seed 47 executes a 0.138 m lowest-foot peak (0.016% of
frames), above the 0.12 m table mask, which did not happen on the dev seeds; round 1
reached 0.27 m there. Seed 11 has 0.109% outside-wall human frames, at parity with
round 1's 0.104% on that seed. The baseline's ordinary and redirect figures use
steering context reconstructed by `diagnose`, since its traces did not save it.

Stills were spot-checked for layout only. This is not the unfinished image review.

Why redirects stay unresolved (same traces and `diagnose_round2`, seeds 11/23/47):
- 138–168 of the 162–200 unresolved attempts per seed end because steering turns evasive.
  Only 2 end in a wait. Median attempt length is 0.8–1.2 s.
- The evasive flag is on for 33–40% of command steps. It starts about 19–21 times per
  human-minute, with median episodes of 2 edges and median ordinary gaps of 3 edges.
- At 76–84% of evasive steps a neighbour is within `contact_m` (1.2 m in this config).
  At 7–10% only a predicted conflict is present, with no close neighbour and no wall.
  These two buckets overlap with the wall bucket. The median nearest neighbour
  at evasive steps is 1.1 m.
- The flag therefore reacts to proximity rather than to a likely collision, and it
  interrupts turns before they finish. This matches the 47/48 body responses in the
  empty-plaza probes.
