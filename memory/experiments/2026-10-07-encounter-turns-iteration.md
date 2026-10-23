# 2026-10-07: encounter iteration on the turning graph (Astra, round 3)

Status: complete, stopped early. The agent ran 13 of its 225 minutes. It finished the
frozen references and one candidate, an opt-in planted-twist mask; priorities 2–5 of
its prompt were not started.

- **Run:** `runs/20261007T110618Z_encounter_turns` (job 4026225, A40 chappie).
- **Prompt:** `configs/behavior/iterate_codex_encounter.txt`.
- **Graph:** `graph-turns`, with rotations `graph-turns-rotations`; the agent verified
  both paths and hashes.
- **Evaluations:** CPU, one thread, development seeds 0/1/2. Prior evidence:
  [encounter design](2026-10-06-encounter-design.md) and
  [turning library](2026-10-06-turning-library.md).

## Why it stopped

Request 09 ran the agent's own script, `trace_analysis.py`, which read a
`joint_names` key that `skeleton.json` does not have and raised a KeyError. The prompt
said "if access, rendering or execution fails, record the failure and exit". The agent
applied that to its own bug and ran `crowd iterate stop`. STOP makes the launcher
terminate the agent, so it never wrote `report.md`, `final.md` or this note's result.

Both Codex prompts now say:

- a failing script the agent wrote is a bug to fix;
- only graph, rendering, tool or Codex access failures end a run early;
- the agent must never run `crowd iterate stop`.

## What it did

1. **References** (requests 00–05). Encounter and plaza at 20 and 60 s, plus both
   probes, on graph-turns. They reproduce the earlier first look: seed 0 at 60 s has a
   joint-sphere clearance of −0.070 m and 45/2/63 redirects.
2. **Twist distribution and sheets** (06–07). On graph-turns at 21.4° (graph-default's
   p99):
   - The mask removes 3,533 of 206,782 admissible edges: 1,110 in-place, 2,421 straight
     or gentle, and 2 non-forward moving turns. It adds no dead ends.
   - On graph-default it would remove 67.
   - In the top-view sheets the masked edges show static feet with a rotating or
     held-twisted pelvis. The all-contact condition keeps stepping turns, but it also
     keeps some severe foot/body lag where a single contact label releases (edges
     200447 and 222132).
3. **Mechanism: `twist_mask`, `twist_max_foot_yaw_deg`** (default off, 21.4).
   - It masks every edge whose contact labels are all on and whose edge-mean foot yaw
     exceeds the limit. Labels come from saved tables at load (`annotate_twist`, shared
     with `crowd graph turns`).
   - It masks candidates and starts and reuses the hold fallback; validity and the file
     are unchanged.
   - Tests: `tests/unit/test_twist_mask.py`.
4. **Candidate 08** = `encounter.json` + mask, promoted as
   `configs/behavior/encounter_turns.json`.

## Results (60 s, seeds 0/1/2; graph-turns unless stated)

Comparison video and executed-twist shares: `runs/20261007T165926Z_encounter_turns_video`.

| | Encounter, graph-default | Encounter, graph-turns | + twist mask |
|---|---|---|---|
| Executed planted-twist edges | 0 / 0 / 0% | 2.6 / 1.8 / 3.6% | 0 (by construction) |
| Joint-sphere clearance (contact frames) | +0.008 / +0.056 / −0.011 m (0–0.3%) | −0.070 / −0.022 / −0.047 m (5.0 / 0.3 / 1.7%) | −0.027 / −0.010 / +0.043 m (3.8 / 0.3 / 0%) |
| Redirects: turn then forward / lateral / unresolved | 12–16 / 0–1 / 72–105 | 45–47 / 2–5 / 63–64 | 47–49 / 4–6 / 59–72 |
| Planted-foot p95; acceleration p95 | 0.30–0.32 m/s; 2.6–2.8 | 0.38–0.40 m/s; 3.3–3.4 | 0.38 m/s; 3.3–3.4 |
| Mean speed; stopped steps | 0.43–0.50 m/s; 7% | 0.52–0.61 m/s; 8.5–12.9% | 0.59–0.63 m/s; 8.4–9.7% |

At 20 s the mask's clearance is −0.027 / +0.024 / +0.043 m against −0.070 / −0.022 /
−0.022 m, but seed 0 still has 11.5% contact frames. The probes stay contact-free with
and without the mask: head-on at least 0.064 → 0.041 m, crossing at least
0.039 → 0.071 m.

**Reading.**

- On graph-turns, 2–4% of executed edges were planted twists, almost all slow or
  standing ones. The mask removes them and keeps the in-place turning gain.
- Minimum joint-sphere clearance improves in all 6 crowd runs (20 and 60 s), and
  contact frames in 5 (one tie). But each configuration is a single unpaired run per
  seed, and trajectories diverge after the first changed selection. The mask is not
  shown to cause the contact change.
- Sliding, acceleration and unresolved redirects did not improve.
- This is a development candidate: not frozen, and held-out seeds 101/307/911 are
  unused.

## Checked afterwards (2026-10-07)

- **Code:** unit tests (233 passed, 3 skipped), Ruff, and all 12 fast golden cases on
  zhurong with `CROWD_GRAPH` unset (`runs/20261007T165724Z_golden_check`). The agent's
  own golden attempts on chappie mismatched at about 1e-8 before its edits as well, so
  chappie's CPU differs; they are not evidence of a regression.
- **Rigid mjlab on graph-turns:** `runs/20261007T165748Z_mjlab_graph_demo`, the first
  such run.
  - Setup: 16 humans for 20 s with the mask.
  - Spacing: minimum root gap 0.69 m.
  - Cost: env step 3.4 ms.
  - Stills show intact rigid poses.
- **Viser:** `runs/20261007T165839Z_viser_check`. The twist mask and its limit are now
  live controls. Queued edits apply at motion edges, and an empty mask is rejected. The
  real `crowd demo viser` serves HTTP. No browser client was connected.

## Not done

From the prompt:

- contact tracing and tuning: stop look-ahead, approach time, body radius;
- the remaining unresolved redirects and the multi-human jam;
- the decision-realism checks;
- visual review of replay by the agent;
- held-out seeds;
- whether the heading or turn cost favors twisted clips.
