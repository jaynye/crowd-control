"""Simulator-speed reference points for the motion budget (zero actions, no policy).

Measures batched env steps per second, and so `S` (simulated seconds per wall
second per environment), for:
  * `g1`: Mjlab-Velocity-Flat-Unitree-G1, robot only (an upper bound on crowd
    training speed: humans, LiDAR and the policy only add cost);
  * `crowd4`, `crowd8`: the corridor fixture (primitive humans + 180x16 LiDAR, no robot).

  crowd bench sim-speed [--num-envs 1024 4096] [--out runs/<UTC>_sim_speed]   # needs CUDA
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from crowdmotion.core.rundir import RunDir


def measure(env, steps: int, warmup: int) -> dict:
  action = torch.zeros(env.action_space.shape, device=env.device)
  for _ in range(warmup):
    env.step(action)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(steps):
    env.step(action)
  torch.cuda.synchronize()
  wall = time.perf_counter() - t0
  step_dt = env.step_dt
  return {
    "num_envs": env.num_envs,
    "step_dt": step_dt,
    "batched_steps_per_s": steps / wall,
    "env_steps_per_s": steps * env.num_envs / wall,
    "S": steps * step_dt / wall,
    "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
  }


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_sim_speed)")
  parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 4096])
  parser.add_argument("--steps", type=int, default=200)
  args = parser.parse_args(argv)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from crowdmotion.sim.mjlab.corridor.scene import make_crowd_env_cfg

  with RunDir.create("sim_speed", out=args.out, record={"args": {k: str(v) for k, v in vars(args).items()}}) as run:
    rows = []
    for n in args.num_envs:
      for kind in ("g1", "crowd4", "crowd8"):
        if kind == "g1":
          cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
          cfg.scene.num_envs = n
        else:
          cfg = make_crowd_env_cfg(n, num_humans=int(kind[-1]), seed=0, moving=True)
        torch.cuda.reset_peak_memory_stats()
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
          row = {"scene": kind, **measure(env, args.steps, warmup=20)}
        finally:
          env.close()
        rows.append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()
    run.write_json("rows.json", rows)


if __name__ == "__main__":
  main()
