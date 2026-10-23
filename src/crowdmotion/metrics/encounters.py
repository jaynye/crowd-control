"""Encounter-design diagnostics: decisions, interventions, stops and body clearance.

Bodies are the design's vertical cylinders (`body_radius_m`) unless joint radii are
given. Cutoffs are diagnostics, not calibrated acceptance limits.
"""

from __future__ import annotations

import numpy as np

from crowdmotion.metrics.spacing import root_gaps

MODES = ("cruise", "sidestep", "veer", "pivot", "yield")


def _percentiles(values):
  values = np.asarray(values, dtype=float)
  return np.percentile(values, [50, 95]).tolist() if values.size else None


def _episodes(mask):
  """Half-open runs, including episodes censored at either recording boundary."""
  changes = np.diff(np.r_[False, mask, False].astype(int))
  return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def standoff_metrics(trace: dict, fps: float, edge_frames: int, cfg: dict, *, pair_gap_m: float = 1.1) -> dict:
  """Common executed-motion diagnostics for both designs, at command-step resolution.

  Pair records retain times and human IDs so long tails can be traced and replayed.
  Threat delays include a separate censored count; unresolved encounters never enter
  the successful-delay percentiles. These cutoffs define diagnostics, not success.
  """
  root = np.asarray(trace["root"])
  if not np.isfinite(pair_gap_m) or pair_gap_m <= 0:
    raise ValueError("pair_gap_m must be positive and finite")
  humans, frames = root.shape[:2]
  steps = trace["command"].shape[1]
  starts = np.arange(steps) * edge_frames
  ends = np.minimum(starts + edge_frames, frames - 1)
  durations = (ends - starts) / fps
  dt = edge_frames / fps
  minutes = humans * (frames - 1) / fps / 60
  pos = root[:, starts]
  displacement = root[:, ends] - pos
  velocity = displacement / durations[None, :, None]
  speed = np.linalg.norm(velocity, axis=-1)
  near_still = speed < .15
  waiting = trace["waiting"]
  scene = cfg.get("encounter", cfg.get("steering", {}))
  goals = trace["goal"]
  pending = ~waiting & (np.linalg.norm(goals - pos, axis=-1) > scene.get("arrive_m", .6))
  mode = trace.get("encounter_mode", np.zeros((humans, steps), dtype=int))
  maneuver = ((mode > 0) & (mode != 4)) | trace.get("turn_active", np.zeros_like(near_still))
  maneuver |= trace.get("encounter_ordinary_turn", np.zeros_like(near_still))
  if "encounter_mode" not in trace:
    maneuver |= trace.get("steering_evasive", np.zeros_like(near_still))
  stops = []
  for human in range(humans):
    for first, end in _episodes(near_still[human] & pending[human]):
      stops.append({"human": human, "start_s": float(starts[first] / fps),
                    "duration_s": float(durations[first:end].sum()), "censored_at_end": bool(end == steps)})
  # Goal changes count random-route arrivals; a non-looping fixed route also has a
  # final arrival with no goal change, detected by first entry into its arrival disk.
  changes = np.linalg.norm(np.diff(goals, axis=1), axis=-1) > 1e-5
  if scene.get("conditioning") == "velocity":
    arrivals = 0
  else:
    arrivals = int(changes.sum())
    if scene.get("conditioning") == "waypoints" and not scene.get("loop_waypoints", True):
      close_goal = np.linalg.norm(goals - pos, axis=-1) <= scene.get("arrive_m", .6)
      final = np.asarray([route[-1] for route in scene["waypoints"]])
      final_active = np.linalg.norm(goals - final[:, None], axis=-1) < 1e-5
      arrivals += int((final_active & (close_goal | waiting)).any(1).sum())
  # Intended pair threats are saved by the encounter design. For plaza, use each
  # person's goal direction at its requested speed as a counterfactual intent.
  pair_threat = trace.get("encounter_pair_threat")
  goal_vector = goals - pos
  desired = goal_vector / np.linalg.norm(goal_vector, axis=-1, keepdims=True).clip(1e-9)
  desired *= np.maximum(trace["command"][..., :1], .3)
  desired *= (~waiting)[..., None]
  heading = trace["heading"][:, starts]
  forward = np.stack((np.sin(heading), np.cos(heading)), -1)
  right = np.stack((np.cos(heading), -np.sin(heading)), -1)
  # Integrate only steps actually travelling sideways/backward relative to facing.
  longitudinal = (displacement * forward).sum(-1)
  lateral = np.abs((displacement * right).sum(-1))
  lateral = np.where(lateral > np.abs(longitudinal), lateral, 0)
  backward = np.maximum(-longitudinal, 0)
  idle_records, all_records, encounters = [], [], []
  for i in range(humans):
    for j in range(i + 1, humans):
      relative = pos[j] - pos[i]
      gap = np.linalg.norm(relative, axis=-1)
      stalled = near_still[i] & near_still[j] & (gap <= pair_gap_m) & (pending[i] | pending[j])
      for mask, records in ((stalled, all_records), (stalled & ~maneuver[i] & ~maneuver[j], idle_records)):
        for first, end in _episodes(mask):
          duration = float(durations[first:end].sum())
          if duration >= 1.2 - 1e-6:
            records.append({"humans": [i, j], "start_s": float(starts[first] / fps),
                            "duration_s": duration, "min_root_gap_m": float(gap[first:end].min()),
                            "censored_at_end": bool(end == steps)})
      if pair_threat is not None:
        threats = pair_threat[i, :, j] | pair_threat[j, :, i]
      else:
        rv = desired[j] - desired[i]
        dot = (relative * rv).sum(-1)
        tau = np.clip(-dot / (rv * rv).sum(-1).clip(1e-9), 0, scene.get("horizon_s", 3.))
        miss = np.linalg.norm(relative + rv * tau[:, None], axis=-1)
        threats = (gap < scene.get("bubble_radius_m", 2.5)) & (dot < 0) & (miss < 1.0)
      first = None
      low = np.inf
      for k in range(steps):
        if first is None:
          if threats[k] and (k == 0 or not threats[k - 1]):
            first, low = k, gap[k]
          else:
            continue
        low = min(low, gap[k])
        passed = (relative[k] * relative[first]).sum() < 0
        separating = gap[k] >= low + .25 and k > first and gap[k] > gap[k - 1]
        resolved = (passed or separating) and k > first
        if resolved or k == steps - 1:
          sl = slice(first, k + 1)
          intervention = bool(((mode[[i, j], sl] > 0)).any())
          locomotor = bool(((mode[[i, j], sl] > 0) & (mode[[i, j], sl] != 4)).any())
          measured = bool((lateral[[i, j], sl].sum(1) >= .2).any()
                          or (backward[[i, j], sl].sum(1) >= .2).any())
          encounters.append({"humans": [i, j], "start_s": float(starts[first] / fps),
                             "observed_s": float((starts[k] - starts[first]) / fps),
                             "resolved": bool(resolved), "intervention": intervention,
                             "locomotor_intervention": locomotor, "measured_deviation": measured,
                             "either_deviates": intervention or measured})
          first = None

  def summary(records):
    values = [r["duration_s"] for r in records]
    return {"count": len(records), "per_human_minute": len(records) / max(minutes, 1e-9),
            "duration_s_p50_p95": _percentiles(values), "duration_s_max": max(values, default=None),
            "censored_at_end": sum(r["censored_at_end"] for r in records), "records": records}

  resolved = [r["observed_s"] for r in encounters if r["resolved"]]
  deviations = sum(r["either_deviates"] for r in encounters)
  return {
    "human_minutes": minutes, "pair_standoffs": summary(idle_records),
    "pair_stalls_including_maneuvers": summary(all_records), "pending_goal_stops": summary(stops),
    "stopped_pending_goal_s_per_human_minute": float((near_still * pending * durations).sum()) / max(minutes, 1e-9),
    "mean_speed_mps": float(np.linalg.norm(np.diff(root, axis=1) * fps, axis=-1).mean()),
    "goal_arrivals": arrivals, "goal_arrivals_per_human_minute": arrivals / max(minutes, 1e-9),
    "threat_encounters": {"count": len(encounters), "resolved": len(resolved),
                          "censored_at_end": len(encounters) - len(resolved),
                          "resolution_s_p50_p95": _percentiles(resolved),
                          "either_deviates_percent": 100 * deviations / len(encounters) if encounters else None,
                          "neither_deviates_percent": 100 * (1 - deviations / len(encounters)) if encounters else None,
                          "locomotor_intervention_percent": 100 * sum(r["locomotor_intervention"] for r in encounters)
                          / len(encounters) if encounters else None, "records": encounters},
    "definition": {
      "sampling_s": dt, "near_stop_mps": .15, "pair_root_gap_m": pair_gap_m, "minimum_standoff_s": 1.2,
      "pair_standoff": f"Both mean edge speeds <.15 m/s, root gap <={pair_gap_m:g} m, at least one pending goal, neither "
                        "in a locomotor intervention or active turn; yields count as idle. Also report stalls including maneuvers.",
      "pending_goal": "Not waiting and current waypoint further than arrive_m. Stops include recording-end censoring.",
      "threat": "Encounter: saved intended-velocity bubble test. Plaza: goal direction at requested speed (floor .3), "
                "zero while waiting; approaching, within 2.5 m, miss <1 m within 3 s. Start on threat rising edge.",
      "resolution": "First step after onset with pair longitudinal order reversed, or root gap .25 m above its running "
                    "minimum and increasing. Unresolved at recording end are censored, excluded from delay percentiles.",
      "deviation": "Either any intervention (including yield), or .2 m accumulated sideways/backward displacement "
                   "relative to edge-start facing. Locomotor interventions excluding yield also reported.",
      "arrivals": "Goal changes, plus entry into the final arrival disk for non-looping fixed routes; velocity mode has none.",
      "scope": "Diagnostic cutoffs, no pass thresholds or human calibration. Duration precision is one command step.",
    },
  }


def cylinder_clearance(root: np.ndarray, body_radius_m: float, tolerance_m: float, fps: float) -> dict:
  """Root-gap clearance between `body_radius_m` cylinders, over every frame and pair."""
  gaps = root_gaps(root)[1:]  # exclude the reset frame
  clear = gaps - 2 * body_radius_m
  frames = clear.min((1, 2))
  pairs = np.triu(np.ones(clear.shape[1:], dtype=bool), 1)
  touching = (clear < 0)[:, pairs]  # [T, pairs]
  entries = int((touching[1:] & ~touching[:-1]).sum() + touching[:1].sum()) if len(touching) else 0
  return {
    "min_clearance_m": float(frames.min()) if frames.size else None,
    "contact_frame_percent": 100 * float((frames < 0).mean()) if frames.size else None,
    "within_tolerance_frame_percent": 100 * float((frames < tolerance_m).mean()) if frames.size else None,
    "contact_pair_events": entries,
    "contact_pair_seconds": float(touching.sum()) / fps,
  }


def joint_clearance(joints: np.ndarray, radii: np.ndarray, stride: int = 1) -> dict:
  """Minimum distance between any two humans' joint spheres (`radii [J]`, meters).

  Joint spheres approximate the capsule bodies at their joints only (not along bones).
  """
  frames = joints[:, 1::stride]  # [H, T, J, 3]
  H = len(frames)
  worst = np.full(frames.shape[1], np.inf)
  for i in range(H):
    for j in range(i + 1, H):
      d = np.linalg.norm(frames[i][:, :, None] - frames[j][:, None], axis=-1)  # [T, J, J]
      d = d - radii[None, :, None] - radii[None, None, :]
      worst = np.minimum(worst, d.min((1, 2)))
  return {"min_joint_sphere_clearance_m": float(worst.min()) if worst.size else None,
          "joint_sphere_contact_frame_percent": 100 * float((worst < 0).mean()) if worst.size else None,
          "frame_stride": stride}


def encounter_metrics(trace: dict, fps: float, edge_frames: int, cfg: dict) -> dict:
  """Decision statistics from an encounter-design trace (`trace` holds `encounter_*`)."""
  mode, switch = trace["encounter_mode"], trace["encounter_switch"]
  threat, urgency = trace["encounter_threat"], trace["encounter_urgency"]
  stopped, assertive = trace["encounter_stopped"], trace["encounter_assertiveness"][:, 0]
  steps = mode.shape[1]
  step_s = edge_frames / fps
  minutes = len(mode) * steps * step_s / 60
  chosen = mode[switch]
  # Empirical per-step switch probability given a threat, binned by urgency and assertiveness.
  forced = (trace.get("encounter_stubborn_release", np.zeros_like(switch))
            | trace.get("encounter_stall_recovery", np.zeros_like(switch)))
  at_risk = trace["encounter_eligible"] & ~forced
  urgency_bins = [0, .25, .5, .75, 1.0001]
  by_urgency = {}
  for lo, hi in zip(urgency_bins[:-1], urgency_bins[1:]):
    sel = at_risk & (urgency >= lo) & (urgency < hi)
    by_urgency[f"{lo:g}-{min(hi, 1):g}"] = {"steps": int(sel.sum()),
                                           "switch_percent": 100 * float(switch[sel].mean()) if sel.any() else None,
                                           "expected_percent": 100 * float(trace["encounter_switch_probability"][sel].mean())
                                           if sel.any() and "encounter_switch_probability" in trace else None}
  by_assertiveness = {}
  a = np.broadcast_to(assertive[:, None], mode.shape)
  for lo, hi in ((0, .3), (.3, .6), (.6, 1.0001)):
    sel = at_risk & (a >= lo) & (a < hi)
    by_assertiveness[f"{lo:g}-{min(hi, 1):g}"] = {"steps": int(sel.sum()),
                                                 "switch_percent": 100 * float(switch[sel].mean()) if sel.any() else None}
  # Stops: consecutive stopped steps per human.
  runs = []
  for row in stopped:
    edges = np.diff(np.r_[0, row.astype(int), 0])
    runs += list((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) * step_s)
  # Executed ordinary turns: in place versus arcs, from the turn radius at onset.
  turning = trace["encounter_ordinary_turn"]
  onset = turning & ~np.concatenate((np.zeros((len(turning), 1), dtype=bool), turning[:, :-1]), 1)
  radius = trace["encounter_ordinary_turn_radius"][onset]
  stubborn = trace.get("encounter_stubborn", np.zeros_like(stopped))
  active_stubborn = trace.get("encounter_stubborn_active", np.zeros_like(stopped))
  effective = trace.get("encounter_effective_assertiveness", a)
  by_effective = {}
  for lo, hi in ((0, .3), (.3, .6), (.6, 1.0001)):
    sel = at_risk & (effective >= lo) & (effective < hi)
    by_effective[f"{lo:g}-{min(hi, 1):g}"] = {
      "steps": int(sel.sum()), "switch_percent": 100 * float(switch[sel].mean()) if sel.any() else None,
      "expected_percent": 100 * float(trace["encounter_switch_probability"][sel].mean())
      if sel.any() and "encounter_switch_probability" in trace else None}
  return {
    "human_minutes": minutes,
    "mode_step_percent": {name: 100 * float((mode == k).mean()) for k, name in enumerate(MODES)},
    "threat_step_percent": 100 * float(threat.mean()),
    "switches": int(switch.sum()),
    "switches_per_human_minute": float(switch.sum()) / max(minutes, 1e-9),
    "strategy_counts": {name: int((chosen == k).sum()) for k, name in enumerate(MODES) if k},
    "switch_percent_given_threat_by_urgency": by_urgency,
    "switch_percent_given_threat_by_assertiveness": by_assertiveness,
    "switch_percent_given_threat_by_effective_assertiveness": by_effective,
    "stubborn_initial_percent": 100 * float(stubborn[:, 0].mean()),
    "stubborn_active_step_percent": 100 * float(active_stubborn.mean()),
    "stubborn_patience_releases": int(trace.get("encounter_stubborn_release", np.zeros_like(stopped)).sum()),
    "stall_recoveries": int(trace.get("encounter_stall_recovery", np.zeros_like(stopped)).sum()),
    "switch_percent_given_threat_by_stubbornness": {
      name: {"steps": int(sel.sum()), "switch_percent": 100 * float(switch[sel].mean()) if sel.any() else None}
      for name, sel in (("stubborn", at_risk & active_stubborn), ("nonstubborn", at_risk & ~active_stubborn))},
    "stopped_step_percent": 100 * float(stopped.mean()),
    "separating_step_percent": 100 * float(trace["encounter_separating"].mean()),
    "stop_episodes_per_human_minute": len(runs) / max(minutes, 1e-9),
    "stop_episode_s_p50_p95": _percentiles(runs),
    "ordinary_turns": int(onset.sum()),
    "ordinary_turns_in_place": int((radius == 0).sum()),
    "ordinary_turn_radius_m_p50_p95": _percentiles(radius[radius > 0]),
    "body_clearance": cylinder_clearance(trace["root"], cfg["body_radius_m"], cfg["stop_clearance_m"], fps),
    "definition": "Switch percentages are per command step while eligible (cruising, not waiting, with a "
                  "threat in the bubble), excluding separately counted forced recovery/patience-release steps. "
                  "Clearance uses body_radius_m cylinders around roots; contact means "
                  "negative clearance. Diagnostics, not limits.",
  }
