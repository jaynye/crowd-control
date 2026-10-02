# Stochastic crowd behavior iteration journal

> Archived from `runs/20261001T061749Z_behavior_iteration/journal.md` on 2026-10-01; historical. Links were adjusted to the archive location.

Status: running

User authorization: continual iteration and rendering, one run folder; qos=short,
rl2-group allocation. Stop by 2026-10-01T10:15:24+00:00 or after
more than five minutes waiting for a GPU. No motion prior calls in the runtime.

## Plan

Freeze current behavior baseline, evaluate/render paired seeds, implement forward
locomotion and committed sampled turns (45/60/90 degrees, strongest mass at 90),
review direction/turn completion/spacing/foot quality, and record every rule change.
All configurations, traces, comparisons, logs and videos live under this folder.

## 2026-10-01T06:17:49.635574+00:00 — Session setup

Prior report accepted. Starting from the saved 134,820-edge graph. Turning coverage
is scarce, so yaw/facing costs and turn-state motion availability will be measured.

## 2026-10-01T06:18:57.066187+00:00 — GPU request / baseline

Job 3994253: A40 on trublu, partition/account rl2-lab, qos short. Node FreeMem
38,206 MB before allocation. Scheduler has no rl2-group partition; lab account
is rl2-lab. Baseline retains current rules, expanded to 60 s × seeds 0/1/2.
Five-minute queue timer starts at submission; four-hour deadline unchanged.

## 2026-10-01T06:21:32.641094+00:00 — Proposed committed-turn rule

No stochastic runtime rule has changed yet. Proposal: sample turn magnitude once
on entering TURN from 45/60/90/135/180 degrees with weights 15/20/50/10/5%,
plus small jitter, and commit the heading target until executed body motion reaches
it. Bias 90 degrees most strongly, retain forward WALK dwell 4–12 s, use slowdown
for ordinary avoidance and reserve lateral steps for imminent conflict. These are
uncalibrated initial settings. Native-motion facing/turn costs need measured trials.

GPU request remains pending on QOSGrpGRES. Exact queue cutoff: 06:23:23 UTC.

## 2026-10-01T06:23:23.129695+00:00 — Session stopped

Reason: gpu_wait_limit. Canceled session-owned Slurm job 3994253; no more requests will run.

## 2026-10-01T06:26:32.372720+00:00 — Final record

The GPU limit stopped the session at 06:23:23.125 UTC, 300.125 s after
submission. Job 3994253 never left PENDING (QOSGrpGRES) and was canceled.
The queued baseline was marked not_executed; no baseline metrics, candidate
evaluations, videos or runtime stochastic-rule changes were produced. The
committed-turn distribution remains a proposal. Session/worker sources are saved
under source/; safeguard checks use mocked scheduler calls, not an allocation.
