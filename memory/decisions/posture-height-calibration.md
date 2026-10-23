# Posture classes and flat-ground height calibration

Opened 2026-10-05. **Whose call:** empirical review, then user.
**Blocking:** promotion of geometric classes/limits as calibrated human behavior;
does not block opt-in development or live tuning.

Question: what signed trunk-pitch classes/targets and vertical limits match the
intended upright/hunched styles and acceptable flat-ground motion?

Current development uses pitch split 15° and targets 5°/25°, soft weight .3,
pelvis range .15 m and peak lower-foot .12 m. Alternatives include a continuous
posture score or reviewed stream-level styles, and different height criteria.
Head ratio alone confounds torso lean, knee flexion and suspension. A hard posture
filter risks already scarce turning coverage.

The [transition iteration](../experiments/2026-10-05-transition-posture-iteration.md)
verified geometric distributions, inspected 72 initial sampled edges plus brief
airborne examples, and measured persistence/height gains on development seeds.
These choices are heuristics; executed blending can exceed table bounds, and
only two upright/three hunched slow-turn edges survive the chosen mask.

To decide: complete final candidate image and held-out review, have the user label
representative posture examples (including borderline cases), compare continuous
and binary scores on those labels, and inspect executed transitions and contacts.
Do not infer acceptance solely from percentile cutoffs or forward-travel shares.

Links: [motion library](../core/design/motion-library.md),
[graph response delay](graph-response-delay.md).
