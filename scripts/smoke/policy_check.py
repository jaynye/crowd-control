"""Diagnostics for a trained traversal policy (not a crowd command).

(a) Teacher-observation sanity: `crowd_nearest`'s first entry against geometry recomputed
    here (min-clearance human's hip in the robot yaw frame; its velocity against the hip
    finite difference over one env step).
(b) Collision attribution: at each `crowd_collision`, the pre-step robot speed, standing
    flag, the human's and the robot's velocity components toward each other, the
    bearing of the human in the robot frame, and the human's trait toward the robot
    (oblivious, stubborn or cooperative) and stop flag. Episode outcomes split by
    standing envs.

(f) Approach: the robot's speed and its velocity toward the nearest human, in bins of
    their clearance (moving envs, human within 60° of the robot's heading).

(c) Backward walking: the share of moving-env steps with commanded and executed body
    vx below -0.1 m/s. (d) Turn-rate tracking and the heading error. (e) A sign check of
    d(heading_w)/dt against the body yaw rate.

Play the run under the env options it was trained with (`--behavior`, `--robot-blind`,
`--forward-first`, `--no-teacher-lidar`).

  python scripts/smoke/policy_check.py <run dir> [--policy teacher|lidar] [--checkpoint model_<i>.pt]
  (a distilled student's checkpoint plays with `--policy lidar`)
      [--envs 512] [--steps 1500]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("run", type=Path)
  p.add_argument("--policy", default="teacher")
  p.add_argument("--envs", type=int, default=512)
  p.add_argument("--steps", type=int, default=1500)
  p.add_argument("--behavior", default="configs/behavior/encounter_turns.json")
  p.add_argument("--robot-blind", action="store_true")
  p.add_argument("--forward-first", action="store_true")
  p.add_argument("--teacher-lidar", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--checkpoint", type=Path, help="a model_<i>.pt to play (default: the run's newest)")
  args = p.parse_args()

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from crowdmotion.core.paths import default_graph, default_graph_rotations
  from crowdmotion.sim.mjlab.train import actor_checkpoint, newest_checkpoint
  from crowdmotion.sim.mjlab.traversal import _clearance, crowd_nearest, human_joints, register

  task = register(str(default_graph()), args.behavior, str(default_graph_rotations()), 8, policy=args.policy,
                  perceive_robot=not args.robot_blind, forward_first=args.forward_first,
                  teacher_lidar=args.teacher_lidar)
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = args.envs
  agent = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
  ckpt = actor_checkpoint(args.checkpoint or newest_checkpoint(args.run))  # a student plays with `--policy lidar`
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  print("checkpoint", ckpt)

  crowd = env.command_manager.get_term("crowd")
  twist = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  dt = env.step_dt
  obs = wrapped.get_observations()

  pos_err, cos_v, ratio_v, n_checks = [], [], [], 0
  rec = {k: [] for k in ("robot_speed", "robot_toward", "human_toward", "human_speed", "bearing_deg",
                         "standing", "cmd_speed", "cmd_toward", "cmd_vx_b", "exec_vx_b", "heading_err_deg",
                         "trait", "human_stopped")}
  behavior = crowd.behavior
  traits = ("oblivious", "stubborn", "cooperative")
  back_cmd, back_exec, moving_steps = 0, 0, 0
  yaw = {"cmd": [], "act": [], "err": [], "dh": [], "wz": []}
  outcomes = {"standing": {}, "moving": {}}
  bins = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
  approach = {"clear": [], "toward": [], "speed": []}
  for step in range(args.steps):
    with torch.no_grad():
      actions = policy(obs)
    # Pre-step state.
    hips0 = human_joints(crowd)[:, :, 0, :2].clone()
    vel_xz = crowd.vel
    vel_h = torch.stack((vel_xz[..., 0], -vel_xz[..., 1]), -1).clone()
    rpos = robot.data.root_link_pos_w[:, :2].clone()
    rvel = robot.data.root_link_lin_vel_w[:, :2].clone()
    heading = robot.data.heading_w.clone()
    _, per_human = _clearance(env)(env)
    nearest = per_human.argmin(-1)
    rows = torch.arange(args.envs, device="cuda:0")
    rel = hips0[rows, nearest] - rpos  # human relative to robot, world
    unit = rel / rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    robot_toward = (rvel * unit).sum(-1)
    human_toward = -(vel_h[rows, nearest] * unit).sum(-1)
    cmd_w = twist.vel_command_w[:, :2] if hasattr(twist, "vel_command_w") else torch.zeros_like(rvel)
    c, s = torch.cos(heading), torch.sin(heading)
    rel_b = torch.stack((c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]), -1)
    bearing = torch.rad2deg(torch.atan2(rel_b[:, 1], rel_b[:, 0]))
    standing = twist.is_standing_env.clone()
    cmd_vx_b = twist.vel_command_b[:, 0].clone()
    exec_vx_b = c * rvel[:, 0] + s * rvel[:, 1]
    heading_err = torch.rad2deg(twist.heading_error.abs()).clone()
    trait = torch.where(~behavior.robot_perceives, 0, torch.where(behavior.robot_decides, 2, 1))[rows, nearest]
    human_stopped = behavior.stopped[rows, nearest].float()
    moving = ~standing
    moving_steps += int(moving.sum())
    ahead = moving & (bearing.abs() < 60)
    approach["clear"].append(per_human[rows, nearest][ahead].cpu())
    approach["toward"].append(robot_toward[ahead].cpu())
    approach["speed"].append(rvel.norm(dim=-1)[ahead].cpu())
    back_cmd += int((moving & (cmd_vx_b < -0.1)).sum())
    back_exec += int((moving & (exec_vx_b < -0.1)).sum())
    if step % 10 == 0:
      yaw["cmd"].append(twist.vel_command_b[moving, 2].cpu())
      yaw["act"].append(robot.data.root_link_ang_vel_b[moving, 2].cpu())
      yaw["err"].append(heading_err[moving].cpu())
    if step % 100 == 50:
      feats = crowd_nearest(env, k=6).view(args.envs, 6, 5)
      pos_err.append((feats[:, 0, :2] - rel_b).norm(dim=-1).median().item())
    wz_before = robot.data.root_link_ang_vel_b[:, 2].clone()
    obs, _, dones, _ = wrapped.step(actions)
    if step % 10 == 0:
      keep = ~dones.bool()
      dh = torch.atan2(torch.sin(robot.data.heading_w - heading), torch.cos(robot.data.heading_w - heading)) / dt
      yaw["dh"].append(dh[keep].cpu())
      yaw["wz"].append(((wz_before + robot.data.root_link_ang_vel_b[:, 2]) / 2)[keep].cpu())
    if step % 100 == 50:
      hips1 = human_joints(crowd)[:, :, 0, :2]
      fd = (hips1 - hips0) / dt
      moving = (vel_h.norm(dim=-1) > 0.3) & (fd.norm(dim=-1) > 0.3) & ~dones[:, None].bool()
      if moving.any():
        a, b = vel_h[moving], fd[moving]
        cos_v.append(torch.nn.functional.cosine_similarity(a, b, dim=-1).median().item())
        ratio_v.append((a.norm(dim=-1) / b.norm(dim=-1)).median().item())
        n_checks += int(moving.sum())
    terms = {n: env.termination_manager.get_term(n).clone() for n in env.termination_manager.active_terms}
    hit = terms["crowd_collision"]
    for name, mask in terms.items():
      for group, sel in (("standing", standing), ("moving", ~standing)):
        outcomes[group][name] = outcomes[group].get(name, 0) + int((mask & sel).sum())
    if hit.any():
      for k, v in (("robot_speed", rvel.norm(dim=-1)), ("robot_toward", robot_toward),
                   ("human_toward", human_toward), ("human_speed", vel_h[rows, nearest].norm(dim=-1)),
                   ("bearing_deg", bearing), ("standing", standing.float()),
                   ("cmd_speed", cmd_w.norm(dim=-1)), ("cmd_toward", (cmd_w * unit).sum(-1)),
                   ("cmd_vx_b", cmd_vx_b), ("exec_vx_b", exec_vx_b), ("heading_err_deg", heading_err),
                   ("trait", trait), ("human_stopped", human_stopped)):
        rec[k].append(v[hit].cpu())

  print("(a) crowd_nearest[0] position error vs recomputed, median per check (m):",
        [round(x, 4) for x in pos_err])
  print("(a) crowd.vel vs hip finite difference: median cosine", [round(x, 3) for x in cos_v],
        "median |vel|/|fd|", [round(x, 3) for x in ratio_v], "moving humans checked", n_checks)
  print("(b) episode ends by group:", outcomes)
  yc, ya, ye = (torch.cat(yaw[k]) for k in ("cmd", "act", "err"))
  agree = (torch.sign(yc) == torch.sign(ya))[yc.abs() > 0.3].float().mean().item()
  print(f"(d) yaw: |cmd| p50 {yc.abs().median().item():.2f}, |actual| p50 {ya.abs().median().item():.2f}, "
        f"sign agreement when |cmd|>0.3 {agree:.2f}, corr {torch.corrcoef(torch.stack((yc, ya)))[0, 1].item():.2f}, "
        f"|heading error| p10/p50/p90 {[round(float(ye.quantile(q)), 1) for q in (.1, .5, .9)]}")
  dh, wz = torch.cat(yaw["dh"]), torch.cat(yaw["wz"])
  print(f"(e) d(heading_w)/dt vs body yaw rate: corr {torch.corrcoef(torch.stack((dh, wz)))[0, 1].item():.3f}, "
        f"median ratio {(dh / wz)[wz.abs() > 0.2].median().item():.3f}")
  print(f"(c) moving-env steps with commanded body vx < -0.1: {back_cmd / max(moving_steps, 1):.3f}; "
        f"executed body vx < -0.1: {back_exec / max(moving_steps, 1):.3f}")
  a = {k: torch.cat(v) for k, v in approach.items()}
  print(f"(f) nearest human ahead, {len(a['clear'])} env-steps; share of moving env-steps with it within 0.25 m: "
        f"{(a['clear'] < 0.25).sum().item() / max(moving_steps, 1):.4f}")
  for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (a["clear"] >= lo) & (a["clear"] < hi)
    if sel.any():
      print(f"    clearance {lo:.2f}-{hi:.2f} m: {int(sel.sum()):7d} steps; toward p50/p90 "
            f"{a['toward'][sel].median().item():.2f}/{a['toward'][sel].quantile(.9).item():.2f}; "
            f"speed p50 {a['speed'][sel].median().item():.2f}")
  if rec["robot_speed"]:
    r = {k: torch.cat(v) for k, v in rec.items()}
    n = len(r["robot_speed"])
    q = lambda x: [round(float(x.quantile(t)), 3) for t in (0.1, 0.5, 0.9)]  # noqa: E731
    print(f"(b) collisions: {n}")
    for k in ("robot_speed", "robot_toward", "human_toward", "human_speed", "cmd_speed", "cmd_toward", "bearing_deg",
              "cmd_vx_b", "exec_vx_b", "heading_err_deg"):
      print(f"    {k:14s} p10/p50/p90 {q(r[k])}")
    human_led = (r["human_toward"] > r["robot_toward"]).float().mean().item()
    robot_still = (r["robot_speed"] < 0.15).float().mean().item()
    front = (r["bearing_deg"].abs() < 60).float().mean().item()
    behind = (r["bearing_deg"].abs() > 120).float().mean().item()
    print(f"    human closing faster than robot: {human_led:.2f}; robot speed < 0.15: {robot_still:.2f}; "
          f"standing env: {r['standing'].mean().item():.2f}; human in front (|bearing|<60): {front:.2f}; "
          f"behind (>120): {behind:.2f}")
    cb, eb = r["cmd_vx_b"] < -0.1, r["exec_vx_b"] < -0.1
    print(f"    at collision: executed backward {eb.float().mean().item():.2f}; of those, commanded backward "
          f"{(cb & eb).float().sum().item() / max(eb.sum().item(), 1):.2f}")
    ends = sum(sum(g.values()) for g in outcomes.values())
    print(f"    collisions per episode end: {n / max(ends, 1):.3f} ({ends} ends)")
    for i, name in enumerate(traits):
      sel = r["trait"] == i
      if sel.any():
        print(f"    {name:11s} {int(sel.sum()):5d} ({sel.float().mean().item():.2f}); human stopped "
              f"{r['human_stopped'][sel].mean().item():.2f}; robot_toward p50 "
              f"{r['robot_toward'][sel].median().item():.2f}; human_toward p50 {r['human_toward'][sel].median().item():.2f}; "
              f"human closing faster {(r['human_toward'] > r['robot_toward'])[sel].float().mean().item():.2f}")
  env.close()


if __name__ == "__main__":
  main()
