"""Classify observed ordinary goal redirects by executed body-relative motion."""

from __future__ import annotations

import numpy as np

from crowdmotion.core.geometry import wrap_np
from crowdmotion.metrics.locomotion import classify, motion_features


def _first_run(mask, hold):
  if len(mask) < hold:
    return None
  found = np.flatnonzero(np.convolve(mask.astype(int), np.ones(hold, dtype=int), mode="valid") == hold)
  return int(found[0]) if len(found) else None


def ordinary_redirects(trace, fps: float, edge_frames: int, ordinary, arrive_m=.6,
                       angle_deg=45., window_s=6.) -> dict:
  """Non-overlapping ordinary reorientation attempts; retain unresolved attempts.

  An attempt begins when the goal bearing is >=angle_deg from body facing, outside
  arrival distance, while not waiting/evasive. Track that goal for up to window_s;
  censor when avoidance, waiting, goal replacement or >30deg target drift occurs.
  Rearm on alignment <30deg, a new goal, or a new ordinary interval. Starts are
  included and identified, so this does not pretend all attempts are new goal draws.
  """
  root, body, goals = trace["root"], trace["heading"], trace["goal"]
  speed, body_angle, _ = motion_features(root, body, fps)
  labels = classify(speed, body_angle)
  travel = np.arctan2(np.diff(root, axis=1)[..., 0], np.diff(root, axis=1)[..., 1])
  steps = goals.shape[1]
  hold, max_steps = max(1, round(.25 * fps)), max(1, round(window_s * fps / edge_frames))
  records = []
  for h in range(len(root)):
    k, armed = 0, True
    last_goal = None
    while k < steps:
      f = k * edge_frames
      delta = goals[h, k] - root[h, f]
      target = float(np.arctan2(delta[0], delta[1]))
      error = abs(float(wrap_np(target - body[h, f])))
      new_goal = last_goal is None or np.linalg.norm(goals[h, k] - last_goal) > .1
      if new_goal or not ordinary[h, k] or error < np.pi / 6:
        armed = True
      last_goal = goals[h, k]
      if not (armed and ordinary[h, k] and np.linalg.norm(delta) > arrive_m and error >= np.radians(angle_deg)):
        k += 1
        continue
      end_k = min(steps, k + max_steps)
      reason = "end" if end_k == steps else "window"
      for q in range(k + 1, end_k):
        new_delta = goals[h, q] - root[h, q * edge_frames]
        bearing = np.arctan2(new_delta[0], new_delta[1])
        if not ordinary[h, q]:
          end_k, reason = q, "avoidance_or_wait"
          break
        if np.linalg.norm(goals[h, q] - goals[h, k]) > .1 or abs(wrap_np(bearing - target)) > np.pi / 6:
          end_k, reason = q, "goal_changed_or_drifted"
          break
      end = min(end_k * edge_frames, root.shape[1] - 1)
      # Displacements f..end-1 lead to body frames f+1..end.
      aligned = np.abs(wrap_np(body[h, f + 1:end + 1] - target)) <= np.radians(20)
      body_at = _first_run(aligned, hold)
      forward = ((labels[h, f:end] == 1)
                 & (np.abs(wrap_np(travel[h, f:end] - target)) <= np.pi / 6))
      forward_at = None if body_at is None else _first_run(forward[body_at:], hold)
      if forward_at is not None:
        forward_at += body_at
      until = len(aligned) if body_at is None else body_at
      lateral = (labels[h, f:f + until] >= 3)
      lateral_at = _first_run(lateral, hold)
      category = ("lateral_or_backward" if lateral_at is not None else
                  "body_turn_then_forward" if forward_at is not None else "unresolved")
      records.append({
        "human": h, "start_s": f / fps, "end_s": end / fps, "end_reason": reason,
        "initial_alignment": k == 0, "requested_angle_deg": float(np.degrees(error)),
        "trigger": "initial_alignment" if k == 0 else "new_goal" if new_goal else "resumed_alignment",
        "category": category, "body_alignment_delay_s": None if body_at is None else (body_at + 1) / fps,
        "forward_response_delay_s": None if forward_at is None else (forward_at + 1) / fps,
        "lateral_backward_seconds_before_alignment": float(lateral.sum()) / fps,
      })
      armed = False
      k = max(k + 1, end_k)
  counts = {key: sum(r["category"] == key for r in records)
            for key in ("body_turn_then_forward", "lateral_or_backward", "unresolved")}
  return {"attempts": len(records), "counts": counts,
          "percent": {k: 100 * v / max(1, len(records)) for k, v in counts.items()},
          "by_trigger": {trigger: {category: sum(r["trigger"] == trigger and r["category"] == category
                                                for r in records) for category in counts}
                         for trigger in ("initial_alignment", "new_goal", "resumed_alignment")},
          "initial_alignment_attempts": sum(r["initial_alignment"] for r in records), "records": records,
          "definition": f"Ordinary goal/body mismatch >= {angle_deg:g}deg, away from arrival, no wait/evasion. "
                        f"Nonoverlapping attempts, at most {window_s:g}s, censored on avoidance, wait, goal replacement or >30deg drift. "
                        "Body within20deg then forward travel toward target within30deg for.25s: body_turn_then_forward. "
                        "A prior sustained.25s of sideways/backward (>60deg from midpoint body facing at >=.15m/s) "
                        "takes lateral_or_backward precedence. Curved forward arcs remain forward. Incomplete windows are unresolved. "
                        "Initial alignments counted explicitly; cutoffs are diagnostics, not human calibration."}
