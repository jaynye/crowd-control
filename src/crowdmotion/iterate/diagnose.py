"""Additional diagnostics from saved plaza motion; no graph stepping or rendering."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from crowdmotion.behavior.designs import context_radii, design_of, scene_section
from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.behavior.steering import LocalSteering
from crowdmotion.core.geometry import wrap_np
from crowdmotion.core.provenance import file_sha256
from crowdmotion.core.rng import human_seeds
from crowdmotion.core.rundir import RunDir
from crowdmotion.graph import MotionGraph
from crowdmotion.metrics.encounters import encounter_metrics, joint_clearance
from crowdmotion.metrics.locomotion import classify, motion_features
from crowdmotion.metrics.posture import posture_metrics, vertical_metrics
from crowdmotion.metrics.redirects import ordinary_redirects
from crowdmotion.metrics.response import smoothed_velocity
from crowdmotion.metrics.seams import seam_stats
from crowdmotion.sim.capsules import BONE_RADIUS, HEAD_RADIUS


def add_evasive_context(trace, cfg, fps, edge_frames):
  """Older baselines did not save the context. Reconstruct it with shared steering.

  BufferVelocity uses the last six root frames, with reset padding. The context
  depends on kinematics/geometry only, not preferred velocity or random intent.
  No motion is generated and no random numbers are drawn.
  """
  if "steering_evasive" in trace:
    return "saved steering context"
  steering_cfg = dataclasses.replace(PlazaSteeringCfg(**cfg["steering"]),
                                    forward_steering=False, relax_orientation_for_avoidance=True, turn_before_walk=False)
  steering = LocalSteering(steering_cfg, len(trace["root"]), "cpu")
  roots = torch.from_numpy(trace["root"])
  rows = []
  for k in range(trace["command"].shape[1]):
    f = k * edge_frames
    xz = roots[:, f][None]
    vel = (roots[:, f] - roots[:, max(0, f - 5)])[None] * fps / 5
    steering.command(xz, vel, torch.from_numpy(trace["heading"][:, f])[None],
                     torch.zeros_like(xz), torch.ones(xz.shape[:2]), torch.zeros(xz.shape[:2], dtype=torch.bool))
    rows.append(steering.evasive[0].numpy().copy())
  trace["steering_evasive"] = np.stack(rows, 1)
  return "reconstructed with shared LocalSteering and saved BufferVelocity roots (no motion stepping)"


def joint_radii(names) -> np.ndarray:
  """Capsule radius at each joint (`sim.capsules`): the bone ending there; Hips takes the
  Spine radius and unlisted joints such as thumbs 3 cm."""
  return np.array([HEAD_RADIUS if n == "Head" else BONE_RADIUS.get("Spine" if n == "Hips" else n, 0.03)
                   for n in names])


def _percentiles(values):
  return np.percentile(values, [50, 95]).tolist() if len(values) else None


def command_response(trace: dict, fps: float, edge_frames: int) -> dict:
  """Large command changes, censored when the command changes again or ends.

  Analysis definitions: changes >=45 degrees at requested speed >=0.3 m/s;
  response within 30 degrees for 0.25 s (root speed >=0.15 m/s). Windows end
  when the request differs by >30 degrees or stops. These are diagnostics,
  not acceptance limits; report the censoring alongside successful delays.
  """
  commands, heading = trace["command"], trace["heading"]
  velocity = smoothed_velocity(torch.from_numpy(trace["root"]), fps).numpy()
  speed = np.linalg.norm(velocity, axis=-1)
  travel = np.arctan2(velocity[..., 0], velocity[..., 1])
  hold = max(1, round(0.25 * fps))
  events, short = 0, 0
  delays = {"root": [], "body": []}
  eligible = 0
  for human in range(len(commands)):
    cmd = commands[human]
    changed = (np.abs(wrap_np(np.diff(cmd[:, 1]))) >= np.pi / 4) & (cmd[1:, 0] >= 0.3)
    for k in np.flatnonzero(changed) + 1:
      events += 1
      target = cmd[k, 1]
      different = (np.abs(wrap_np(cmd[k + 1:, 1] - target)) > np.pi / 6) | (cmd[k + 1:, 0] < 0.15)
      end_k = k + 1 + np.flatnonzero(different)[0] if different.any() else len(cmd)
      start, end = k * edge_frames + 1, min(end_k * edge_frames + 1, heading.shape[1])
      if end - start < hold:
        short += 1
        continue
      eligible += 1
      for name, angle in (("root", travel), ("body", heading)):
        ok = np.abs(wrap_np(angle[human, start:end] - target)) <= np.pi / 6
        if name == "root":
          ok &= speed[human, start:end] >= 0.15
        reached = np.flatnonzero(np.convolve(ok.astype(int), np.ones(hold, dtype=int), mode="valid") == hold)
        if len(reached):
          delays[name].append((1 + int(reached[0])) / fps)
  return {
    "events": events, "eligible": eligible, "short_windows": short,
    **{name: {"reached": len(values), "not_reached_before_command_changed_or_end": eligible - len(values),
              "delay_s_p50_p95": _percentiles(values)} for name, values in delays.items()},
    "definition": "Command changes >=45 deg at >=.3 m/s; within 30 deg for .25 s; root >=.15 m/s. "
                  "Censor at next command >30 deg away or stop. Conditional delays are not a controlled reaction benchmark.",
  }


def diagnose_trace(trace: dict, cfg: dict, fps: float, edge_frames: int) -> dict:
  root = trace["root"]
  speed, angle, _ = motion_features(root, trace["heading"], fps)
  labels = classify(speed, angle)
  cmd = np.repeat(trace["command"], edge_frames, axis=1)[:, :speed.shape[1]]
  active = cmd[..., 0] >= 0.3
  travel = np.diff(root, axis=1) * fps
  direction = np.stack((np.sin(cmd[..., 1]), np.cos(cmd[..., 1])), -1)
  progress = (travel * direction).sum(-1)
  delta = root[:, None] - root[None, :]
  gaps = np.linalg.norm(delta, axis=-1)
  gaps[np.arange(len(root)), np.arange(len(root))] = np.inf
  nearest = gaps.min(1)[:, 1:]
  moving = speed >= 0.15
  context = {}
  close_m, clear_m = context_radii(cfg)
  for name, mask in (("close", nearest < close_m), ("clear", nearest >= clear_m)):
    selected = mask & moving
    count = int(selected.sum())
    context[name] = {"moving_frames": count, **{
      key: 100 * int(((labels == label) & selected).sum()) / max(1, count)
      for key, label in (("forward_percent", 1), ("sideways_percent", 3), ("backward_percent", 4))}}
  if "steering_evasive" in trace:
    evasive = np.repeat(trace["steering_evasive"], edge_frames, axis=1)[:, :speed.shape[1]]
    for name, mask in (("ordinary_command", ~evasive), ("evasive_command", evasive)):
      selected = mask & moving
      count = int(selected.sum())
      context[name] = {"all_frame_percent": 100 * float(mask.mean()), "moving_frames": count, **{
        key: 100 * int(((labels == label) & selected).sum()) / max(1, count)
        for key, label in (("forward_percent", 1), ("sideways_percent", 3), ("backward_percent", 4))}}
  rt = torch.from_numpy(root)
  velocity = torch.diff(rt, dim=1) * fps
  velocity = torch.cat((velocity[:, :1], velocity), 1)
  seams = seam_stats(torch.from_numpy(trace["joints"]), velocity, rt,
                     list(range(1 + edge_frames, root.shape[1], edge_frames)), fps)
  scene = scene_section(cfg)
  half = np.array([scene["half_x"], scene["half_z"]])
  return {
    "seams": seams,
    "response": command_response(trace, fps, edge_frames),
    "commanded_moving_frames": int(active.sum()),
    "slow_despite_move_command_percent": 100 * int((active & ~moving).sum()) / max(1, int(active.sum())),
    "commanded_progress_mps_p50_p95": _percentiles(progress[active]),
    "speed_mps_mean": float(speed.mean()),
    "outside_wall_human_frame_percent": 100 * float((np.abs(root[:, 1:]) > half).any(-1).mean()),
    "direction_by_neighbor_context": context,
    "note": "Root gaps are not capsule collision distances; close/clear are the design's context radii "
            "(plaza: contact/personal; encounter: body contact plus passing margin, and the bubble). "
            "Seams exclude the initial reset. Cutoffs are diagnostics, not calibrated acceptance limits.",
  }


def committed_turns(trace: dict, fps: float, edge_frames: int) -> dict | None:
  if "intent_mode" not in trace:
    return None
  records = []
  mode, count = trace["intent_mode"], trace["intent_turn_count"]
  for h in range(len(mode)):
    for turn in range(1, int(count[h, -1]) + 1):
      rows = np.flatnonzero(count[h] == turn)
      turning = rows[mode[h, rows] == 2]
      if not len(turning):
        continue  # Still waiting: no executed turn attempt yet.
      start = int(turning[0])
      complete = rows[(rows > start) & (mode[h, rows] == 0)]
      end = int(complete[0]) if len(complete) else mode.shape[1]
      target = trace["intent_target"][h, start]
      last = min(end * edge_frames, trace["heading"].shape[1] - 1)
      records.append({"human": h, "turn": turn, "start_s": start * edge_frames / fps,
                      "sampled_angle_deg": float(np.degrees(trace["intent_sampled_angle"][h, start])),
                      "completed": bool(len(complete)), "observed_s": (end - start) * edge_frames / fps,
                      "final_error_deg": float(abs(np.degrees(wrap_np(target - trace["heading"][h, last]))))})
  completed = [r for r in records if r["completed"]]
  return {"attempted": len(records), "completed": len(completed), "censored_at_end": len(records) - len(completed),
          "completion_s_p50_p95": _percentiles([r["observed_s"] for r in completed]), "records": records}


def reachable_turns(graph, trace: dict, seed: int) -> dict:
  """Availability at the actual visited boundaries, not a claim of realizable routes."""
  edges = torch.from_numpy(trace["edge"])
  if "boundary" in trace:
    boundaries = torch.from_numpy(trace["boundary"])
  else:
    initial = graph.start_boundaries(human_seeds(seed, torch.arange(len(edges)), "evaluator"))
    boundaries = torch.cat((initial[:, None], edges[:, :-1] + 1), 1)
  cand = graph.cand[boundaries]
  yaw = graph.edge_yaw[cand].numpy()
  local = graph.edge_root[cand].numpy()
  speed, angle, _ = motion_features(local.reshape(-1, graph.L, 2), yaw.reshape(-1, graph.L), graph.fps,
                                    np.zeros((yaw.size // graph.L, 2)), np.zeros(yaw.size // graph.L))
  forward = (classify(speed, angle) == 1).mean(-1).reshape(cand.shape) >= 0.5
  target = trace.get("intent_target", trace["command"][..., 1])
  if "turn_target" in trace:
    target = np.where(trace["turn_active"], trace["turn_target"], target)
  body = trace["heading"][:, ::graph.L][:, :edges.shape[1]]
  error = wrap_np(target - body)
  requested = np.abs(error) >= np.pi / 4
  if "intent_mode" in trace:
    requested &= trace["intent_mode"] == 2
  if "turn_active" in trace:
    requested &= trace["turn_active"]
  directed = np.degrees(yaw[..., -1]) * np.sign(error)[..., None]
  admissible = graph.admissible_edges()[cand].numpy()
  directed = np.where(admissible, directed, -np.inf)
  eligible = directed[requested]
  fwd = forward[requested]
  valid = graph.valid
  all_yaw = graph.edge_yaw[valid, -1].abs().numpy() * (180 / np.pi)
  result = {
    "large_turn_states": int(requested.sum()),
    "has_native_10deg_turn_candidate_percent": 100 * float((eligible.max(-1) >= 10).mean()) if len(eligible) else None,
    "has_forward_native_10deg_turn_candidate_percent": 100 * float(((eligible >= 10) & fwd).any(-1).mean()) if len(eligible) else None,
    "best_directed_native_yaw_deg_p50_p95": _percentiles(eligible.max(-1)) if len(eligible) else None,
    "valid_edge_abs_yaw_deg_p50_p95": _percentiles(all_yaw),
    "note": "Native .4-s candidates at visited boundaries, before root correction; requested target >=45 deg away. "
            "Forward candidate means >=half its frames forward relative to midpoint facing; 10 deg is an audit cutoff, not a pass limit. "
            "Configured vertical mask applies. Availability alone does not prove continuity or successful turning.",
  }
  if "posture_hunched" in trace:
    pitch = graph.ensure_annotations().trunk_pitch_deg[cand].numpy()
    matches = (pitch >= graph.cfg.posture_cutoff_deg) == trace["posture_hunched"][..., None]
    result["has_posture_matched_forward_native10deg_percent"] = (
      100 * float(((directed >= 10) & forward & matches)[requested].any(-1).mean()) if requested.any() else None)
    result["visited_large_turn_states_without_admissible_candidate"] = int((~admissible[requested].any(-1)).sum())
  if "turn_active" in trace:
    # Difference between played net yaw and native table yaw is the applied
    # root correction (apart from trace float roundoff and a final partial edge).
    played = trace["edge"]
    end = np.minimum((np.arange(played.shape[1]) + 1) * graph.L, trace["heading"].shape[1] - 1)
    actual_yaw = wrap_np(trace["heading"][:, end] - body)
    native_frame = end - np.arange(played.shape[1]) * graph.L - 1
    native = graph.edge_yaw[torch.from_numpy(played.clip(min=0)), torch.from_numpy(native_frame)].numpy()
    correction = wrap_np(actual_yaw - native)
    active = trace["turn_active"] & (played >= 0)
    result["executed_turn_steps"] = int(active.sum())
    result["turn_yaw_correction_deg_p50_p95"] = _percentiles(np.abs(np.degrees(correction[active])))
    result["turn_steps_correction_opposes_native_percent"] = (
      100 * float(((correction * native < -1e-5) & active).sum()) / max(1, int(active.sum())))
  return result


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--run", type=Path, required=True)
  p.add_argument("--out", type=Path)
  p.add_argument("--edge-frames", type=int, help="required for older evaluations without edge_frames metadata")
  p.add_argument("--coverage", action="store_true", help="audit candidate turn availability using the evaluation's saved graph")
  args = p.parse_args(argv)
  torch.set_num_threads(1)
  manifest = json.loads((args.run / "run.json").read_text())
  if manifest["status"] != "completed":
    raise ValueError("Diagnostics require a completed evaluation")
  cfg = manifest["config"]
  skeleton = json.loads((args.run / "skeleton.json").read_text())["meta"]
  edge_frames = manifest.get("edge_frames", args.edge_frames)
  if edge_frames is None or edge_frames < 1:
    raise ValueError("Supply --edge-frames for an older evaluation without that metadata")
  if args.edge_frames is not None and edge_frames != args.edge_frames:
    raise ValueError("--edge-frames disagrees with the recorded evaluation")
  graph = None
  if args.coverage:
    graph_path = Path(manifest["graph_path"])
    if file_sha256(graph_path) != manifest["graph_sha256"]:
      raise ValueError("Graph differs from the evaluation")
    graph = MotionGraph.load(graph_path, device="cpu", cfg_overrides=cfg["graph_overrides"])
  with RunDir.create("behavior_diagnose", out=args.out,
                     record={"evaluation": str(args.run.resolve()), "manifest": manifest}) as run:
    results = []
    for seed in cfg["seeds"]:
      path = args.run / f"seed_{seed}.npz"
      with np.load(path) as saved:
        trace = {k: saved[k] for k in saved.files}
      frames, steps = trace["root"].shape[1] - 1, trace["command"].shape[1]
      if (frames + edge_frames - 1) // edge_frames != steps:
        raise ValueError("Missing or inconsistent edge_frames metadata")
      context_source = (add_evasive_context(trace, cfg, manifest["fps"], edge_frames) if design_of(cfg) == "plaza"
                        else "saved encounter context (intervening or stopped)")
      row = {"seed": seed, "trace_sha256": file_sha256(path),
             **diagnose_trace(trace, cfg, manifest["fps"], edge_frames),
             "vertical": vertical_metrics(trace, manifest["fps"], edge_frames),
             "ordinary_context_source": context_source,
             "ordinary_redirects": ordinary_redirects(trace, manifest["fps"], edge_frames,
                ~trace["steering_evasive"] & ~trace["waiting"], scene_section(cfg)["arrive_m"])}
      if design_of(cfg) == "encounter":
        row["encounter"] = encounter_metrics(trace, manifest["fps"], edge_frames, cfg["encounter"])
      row["joint_clearance"] = joint_clearance(trace["joints"], joint_radii(skeleton["joint_names"]), stride=2)
      if "Hips" in skeleton["joint_names"] and "Neck" in skeleton["joint_names"]:
        row["posture"] = posture_metrics(trace, manifest["fps"], edge_frames, skeleton["joint_names"],
          cfg["graph_overrides"].get("posture_cutoff_deg", 15.))
        if cfg["graph_overrides"].get("posture_enabled"):
          from math import exp

          from crowdmotion.graph.config import GraphConfig
          pc = GraphConfig.from_dict(cfg["graph_overrides"])
          a, b = pc.posture_upright_to_hunched_rate, pc.posture_hunched_to_upright_rate
          times = np.arange(steps) * edge_frames / manifest["fps"]
          p0 = pc.posture_initial_hunched
          expected = (a / (a + b) + (p0 - a / (a + b)) * np.exp(-(a + b) * times)
                      if a + b else np.full_like(times, p0))
          row["posture"]["configured"] = {
            "initial_hunched_probability": p0, "upright_to_hunched_rate_per_s": a,
            "hunched_to_upright_rate_per_s": b, "expected_hunched_edge_percent": 100 * float(expected.mean()),
            "boundary_switch_probabilities": [r * (1 - exp(-(a+b) * edge_frames / manifest["fps"])) / (a+b)
                                               if a+b else 0. for r in (a, b)],
          }
      if turns := committed_turns(trace, manifest["fps"], edge_frames):
        row["committed_turns"] = turns
      if graph is not None:
        row["turn_coverage"] = reachable_turns(graph, trace, seed)
      run.write_json(f"seed_{seed}.json", row)
      results.append(row)
    run.write_json("summary.json", {"seeds": results})
  print(run.path)
