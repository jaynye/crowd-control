# 2026-10-07: encounter-on-turning-graph comparison and Viser support

**Request (user, 2026-10-07):** after Astra's round-3 run
(`runs/20261007T110618Z_encounter_turns`), prepare a comparison and make it work in the
Viser viewer so the parameters can be tuned live.

**Found:** Astra stopped itself after 13 of 225 minutes. Its own analysis script raised
a KeyError, it applied the prompt's "stop on execution failure" rule, and its
`crowd iterate stop` killed the agent before it wrote a report. It finished the frozen
references and one candidate: an opt-in planted-twist mask (`twist_mask`,
`twist_max_foot_yaw_deg`), evaluated as request 08.

## Plan

1. [x] Narrow the failure rule in both Codex prompts; forbid `crowd iterate stop`.
2. [x] Finalize Astra's run-log row and experiment note.
3. [x] Executed planted-twist share in the reference runs; probe deltas.
4. [x] Viser: twist controls in `sim/mjlab/tuning.py`, mask checks for any mask field, test.
5. [x] Validate on zhurong: unit tests, fast goldens with `CROWD_GRAPH` unset, Ruff.
6. [x] GPU: rigid `crowd demo video` on graph-turns with the candidate; headless Viser start.
7. [x] Comparison video: plaza/default, encounter/default, encounter/turns,
       encounter/turns + twist mask; 60 s, seeds 0/1/2.
8. [x] Promote `configs/behavior/encounter_turns.json`; docs rows; status; run log;
       `check_memory`.

## Outcome (retired 2026-10-07)

- Verification `runs/20261007T165724Z_golden_check` (job 4027189, zhurong): 233 unit
  tests, Ruff, 12 fast goldens match.
- Job 4027190 (A40 deebot): rigid demo `runs/20261007T165748Z_mjlab_graph_demo`, Viser
  check `runs/20261007T165839Z_viser_check` (passed), video
  `runs/20261007T165926Z_encounter_turns_video`.
- Results promoted to the [experiment note](../experiments/2026-10-07-encounter-turns-iteration.md)
  and `core/status.md`; the round-3 prompt now starts from `encounter_turns.json`.
