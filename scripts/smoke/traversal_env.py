"""Smoke test of the crowd traversal env: build, step with zero and random actions,
report observation sizes, LiDAR coverage of humans, clearance, terminations and the
environment step time.

  python scripts/smoke/traversal_env.py [--envs 64] [--steps 200] [--timing-envs 4096]
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=200)
  p.add_argument("--timing-envs", type=int, default=4096)
  p.add_argument("--timing-steps", type=int, default=50)
  p.add_argument("--humans", type=int, default=8)
  p.add_argument("--behavior", default="configs/behavior/encounter_turns.json")
  args = p.parse_args()

  from mjlab.envs import ManagerBasedRlEnv

  from crowdmotion.core.paths import default_graph, default_graph_rotations
  from crowdmotion.sim.mjlab.traversal import _clearance, human_joints, make_traversal_env_cfg

  def build(n):
    cfg = make_traversal_env_cfg(str(default_graph()), args.behavior, str(default_graph_rotations()), args.humans)
    cfg.scene.num_envs = n
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")

  env = build(args.envs)
  obs, _ = env.reset()
  print({k: tuple(v.shape) for k, v in obs.items()})
  print("actor terms", env.observation_manager.active_terms["actor"])
  A = env.action_manager.total_action_dim
  done_by = {}
  clear_min, lidar_near = [], []
  for i in range(args.steps):
    act = torch.zeros(args.envs, A, device="cuda:0") if i < args.steps // 2 else 0.3 * torch.randn(args.envs, A, device="cuda:0")
    obs, rew, term, trunc, extras = env.step(act)
    closest, _ = _clearance(env)(env)
    clear_min.append(closest.min().item())
    lidar = env.observation_manager.compute_group("actor")[:, -72 * 4:-72 * 3]
    lidar_near.append((lidar < 0.5).float().mean().item())
    for name in env.termination_manager.active_terms:
      done_by[name] = done_by.get(name, 0) + int(env.termination_manager.get_term(name).sum())
  # LiDAR returns on humans: hit points within 0.3 m of any human joint.
  data = env.scene["lidar"].data
  joints = human_joints(env.command_manager.get_term("crowd")).flatten(1, 2)
  near_human = (torch.cdist(data.hit_pos_w, joints).amin(-1) < 0.3) & (data.distances > 0)
  print("LiDAR rays hitting humans per env: mean", near_human.sum(1).float().mean().item(),
        "envs with any", (near_human.sum(1) > 0).float().mean().item())
  robot = env.scene["robot"].data.root_link_pos_w
  hips = human_joints(env.command_manager.get_term("crowd"))[:, :, 0]
  print("robot z mean", robot[:, 2].mean().item(), "xy range", robot[:, :2].abs().amax(0).tolist())
  print("human hip z mean", hips[..., 2].mean().item(), "xy range", hips[..., :2].abs().amax((0, 1)).tolist())
  print("min clearance over steps", min(clear_min), "final per-env min p50", closest.median().item())
  print("share of lidar bins < 3 m", sum(lidar_near) / len(lidar_near))
  print("terminations", done_by)
  print("reward last", rew.mean().item(), "command", env.command_manager.get_command("twist")[:3].tolist())
  env.close()
  del env
  torch.cuda.empty_cache()

  if args.timing_envs:
    env = build(args.timing_envs)
    env.reset()
    A = env.action_manager.total_action_dim
    act = torch.zeros(args.timing_envs, A, device="cuda:0")
    for _ in range(10):
      env.step(act)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(args.timing_steps):
      env.step(act)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / args.timing_steps
    print(f"{args.timing_envs} envs x {args.humans} humans: {dt * 1e3:.1f} ms per env step, "
          f"{args.timing_envs / dt:.0f} env steps/s, peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
  main()
