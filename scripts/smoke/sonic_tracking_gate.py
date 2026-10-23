"""SONIC-in-mjlab gate: the frozen tracker on mjlab's G1, following the released kinematic
planner. Run on a GPU compute node; writes a fresh `runs/<UTC>_sonic_tracking_gate/`.

    .venv/bin/python scripts/smoke/sonic_tracking_gate.py open --envs 1024 --seconds 20
    .venv/bin/python scripts/smoke/sonic_tracking_gate.py goal --envs 1024 --seconds 60
    .venv/bin/python scripts/smoke/sonic_tracking_gate.py time --envs 4096

`open`: every env walks a random command sequence (a new direction and speed every 2 s,
some idle), replanning on each change and on the release's 1 s walking timer. `goal`
(P0 without a crowd): commands point at random goals 3-8 m away, replanned every second.
`time`: the tracker's milliseconds per control step at --envs, on a fixed plan.

Reported, without pass thresholds: falls per robot-minute and the share of envs that fell;
joint error to the playing reference; root orientation error to it (deg); planar speed of the
robot against the reference's; goals per robot-minute (`goal`); env-step, tracker and
planner times. Planner calls go through `backends.sonic.SonicPlanner` (the batched ONNX port, FP32).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crowdmotion.core.rundir import RunDir  # noqa: E402

TASK = "Mjlab-Velocity-Flat-Unitree-G1"


def planner_port(device: str):
  from crowdmotion.backends.sonic import SonicPlanner

  planner = SonicPlanner(device=device)
  return planner, planner.stats


def make_env(envs: int, seconds: float, device: str):
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = envs
  cfg.episode_length_s = seconds + 5.0
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def ctl_leg(i: int) -> bool:
  """IsaacLab index i is a leg joint (hip, knee or ankle)."""
  from crowdmotion.backends.sonic import TO_ISAACLAB
  return TO_ISAACLAB[i] < 12


def angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  dot = (a * b).sum(-1).abs().clamp(max=1.0)
  return torch.rad2deg(2 * torch.acos(dot))


def run(args, rd: RunDir) -> dict:
  from crowdmotion.sim.mjlab.sonic_control import IDLE, WALK, SonicController

  device = "cuda:0"
  torch.manual_seed(args.seed)
  plan, pstats = planner_port(device)
  env = make_env(args.envs, args.seconds, device)
  env.reset()
  ctl = SonicController(env, plan)
  n = env.num_envs
  all_ids = torch.arange(n, device=device)
  ctl.reset(all_ids)
  robot = ctl.robot
  dt = env.step_dt
  steps = int(round(args.seconds / dt))
  per_s = int(round(1.0 / dt))

  g = torch.Generator(device=device).manual_seed(args.seed)
  heading = torch.rand(n, generator=g, device=device) * 2 * math.pi  # world command direction
  speed = 0.4 + 0.6 * torch.rand(n, generator=g, device=device)
  idle = torch.zeros(n, dtype=torch.bool, device=device)
  goal = robot.data.root_link_pos_w[:, :2].clone()
  if args.mode == "goal":
    r = 3 + 5 * torch.rand(n, generator=g, device=device)
    a = torch.rand(n, generator=g, device=device) * 2 * math.pi
    goal += torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)

  falls = 0
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  goals = 0
  sums = {"joint_err": 0.0, "leg_err": 0.0, "arm_err": 0.0, "ori_err": 0.0, "speed": 0.0, "ref_speed": 0.0}
  legs = torch.tensor([i for i in range(29) if ctl_leg(i)], device=device)
  arms = torch.tensor([i for i in range(29) if not ctl_leg(i)], device=device)
  count = 0
  step_s = tracker_s = 0.0
  prev_ref = ctl.reference_world()[0].clone()

  def command(ids):
    if args.mode == "goal":
      d = goal[ids] - robot.data.root_link_pos_w[ids, :2]
      d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      mode, spd = WALK, torch.full((ids.numel(),), args.speed, device=device)
    else:
      d = torch.stack([torch.cos(heading[ids]), torch.sin(heading[ids])], -1)
      mode = torch.where(idle[ids], IDLE, WALK)
      spd = torch.where(idle[ids], -1.0, speed[ids])
    move = ctl.to_planner(ids, d)
    move = torch.where(torch.as_tensor(mode, device=device).reshape(-1, 1) == IDLE, 0.0, move)
    return mode, spd, move, ctl.to_planner(ids, d)

  for t in range(steps):
    if t > 0 and t % per_s == 0:  # the 1 s walking timer (and, in `open`, a new command every 2 s)
      if args.mode == "open" and t % (2 * per_s) == 0:
        heading += (torch.rand(n, generator=g, device=device) - 0.5) * math.pi
        speed = 0.4 + 0.6 * torch.rand(n, generator=g, device=device)
        idle = torch.rand(n, generator=g, device=device) < 0.1
      ctl.replan(all_ids, *command(all_ids))
    torch.cuda.synchronize()
    a = time.perf_counter()
    action = ctl.act()
    torch.cuda.synchronize()
    b = time.perf_counter()
    _, _, terminated, truncated, _ = env.step(action)
    torch.cuda.synchronize()
    c = time.perf_counter()
    if t >= per_s:  # timing after warm-up
      tracker_s += b - a
      step_s += c - b
    done = (terminated | truncated).nonzero().flatten()
    fell_now = terminated & ~truncated
    falls += int(fell_now.sum())
    fell |= fell_now
    if done.numel():
      ctl.reset(done)
      if args.mode == "goal":
        goal[done] = robot.data.root_link_pos_w[done, :2]
    ref_xy, ref_q = ctl.reference_world()
    live = torch.ones(n, dtype=torch.bool, device=device)
    live[done] = False
    if live.any():
      i = ctl.plans.frame
      err = (robot.data.joint_pos[:, ctl._robot_to_isaaclab] - ctl.plans.joint_pos[all_ids, i]).abs()
      sums["joint_err"] += float(err.mean(-1)[live].sum())
      sums["leg_err"] += float(err[:, legs].mean(-1)[live].sum())
      sums["arm_err"] += float(err[:, arms].mean(-1)[live].sum())
      sums["ori_err"] += float(angle_deg(robot.data.root_link_quat_w, ref_q)[live].sum())
      sums["speed"] += float(robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)[live].sum())
      sums["ref_speed"] += float(((ref_xy - prev_ref).norm(dim=-1) / dt)[live].clamp(max=5.0).sum())
      count += int(live.sum())
    prev_ref = ref_xy.clone()
    if args.mode == "goal":
      reached = (goal - robot.data.root_link_pos_w[:, :2]).norm(dim=-1) < 0.5
      if reached.any():
        goals += int(reached.sum())
        ids = reached.nonzero().flatten()
        r = 3 + 5 * torch.rand(ids.numel(), generator=g, device=device)
        a = torch.rand(ids.numel(), generator=g, device=device) * 2 * math.pi
        goal[ids] = robot.data.root_link_pos_w[ids, :2] + torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)
        ctl.replan(ids, *command(ids))
    if t % per_s == 0:
      print(f"t={t * dt:5.1f}s falls={falls} fell_share={fell.float().mean():.3f} goals={goals}", flush=True)

  robot_min = n * args.seconds / 60.0
  timed = max(steps - per_s, 1)
  return {
    "task": TASK, "mode": args.mode, "envs": n, "seconds": args.seconds, "seed": args.seed,
    "falls_per_robot_min": falls / robot_min, "falls": falls, "fell_share": float(fell.float().mean()),
    "joint_err_mean_rad": sums["joint_err"] / max(count, 1),
    "leg_joint_err_mean_rad": sums["leg_err"] / max(count, 1), "upper_joint_err_mean_rad": sums["arm_err"] / max(count, 1),
    "orientation_err_mean_deg": sums["ori_err"] / max(count, 1),
    "robot_speed_mean": sums["speed"] / max(count, 1), "reference_speed_mean": sums["ref_speed"] / max(count, 1),
    "goals_per_robot_min": goals / robot_min if args.mode == "goal" else None,
    "tracker_ms_per_step": 1000 * tracker_s / timed, "env_step_ms": 1000 * step_s / timed,
    "planner": {**pstats, "ms_per_plan": 1000 * pstats["seconds"] / max(pstats["plans"], 1)},
    "gpu": torch.cuda.get_device_name(0),
  }


def time_tracker(args) -> dict:
  from crowdmotion.backends.sonic import FRAMES, HISTORY, SonicTracker

  device = "cuda:0"
  tr = SonicTracker().to(device)
  n = args.envs
  x = [torch.randn(n, FRAMES, d, device=device) for d in (29, 29, 6)]
  h = [torch.randn(n, HISTORY, d, device=device) for d in (3, 29, 29, 29, 3)]
  out = {}
  with torch.inference_mode():
    for _ in range(10):
      tr.decode(tr.encode(*x), *h)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(200):
      tr.decode(tr.encode(*x), *h)
    torch.cuda.synchronize()
    out["tracker_ms_per_step_fp32"] = 1000 * (time.perf_counter() - t) / 200
  out.update(envs=n, gpu=torch.cuda.get_device_name(0))
  return out


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("mode", choices=("open", "goal", "time"))
  p.add_argument("--envs", type=int, default=1024)
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--speed", type=float, default=0.8, help="goal mode: commanded speed, m/s")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  with RunDir.create("sonic_tracking_gate", record={"kind": "sonic_tracking_gate", **vars(args)}) as rd:
    result = time_tracker(args) if args.mode == "time" else run(args, rd)
    rd.write_json("result.json", result)
    print(result, flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
