"""Simulator-speed reference points for the motion budget (zero actions, no policy).

Measures batched env steps per second, and so `S` (simulated seconds per wall
second per environment), for:
  * `g1`: Mjlab-Velocity-Flat-Unitree-G1, robot only (an upper bound on crowd
    training speed: humans, LiDAR and the policy only add cost);
  * `crowd4`, `crowd8`: the corridor fixture (primitive humans + 180x16 LiDAR, no robot);
  * `plaza-<geometry>-<humans>` (opt-in via `--scenes`, e.g. `plaza-rigid-16`): the
    graph-driven plaza (`crowdmotion.sim.mjlab.plaza`, 360x16 LiDAR, no robot) with
    `rigid` mesh or `capsules` humans; needs the graph and, for rigid, its rotations.
    These rows also time `command_manager.compute` (graph stepping and pose writes)
    and `sim.sense` (BVH refit and ray casts) with CUDA events.

  crowd bench sim-speed [--num-envs 1024 4096] [--scenes g1 crowd4 crowd8] [--out runs/<UTC>_sim_speed]   # needs CUDA
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from crowdmotion.core.rundir import RunDir


def _timed(obj, name: str, events: list) -> None:
  """Wrap `obj.name` so every call records a pair of CUDA events into `events`."""
  fn = getattr(obj, name)

  def wrapper(*args, **kwargs):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    events.append((start, end))
    return out

  setattr(obj, name, wrapper)


def measure(env, steps: int, warmup: int, stages: bool = False) -> dict:
  action = torch.zeros(env.action_space.shape, device=env.device)
  for _ in range(warmup):
    env.step(action)
  timed = {"command_ms": [], "sense_ms": []}
  if stages:
    _timed(env.command_manager, "compute", timed["command_ms"])
    _timed(env.sim, "sense", timed["sense_ms"])
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(steps):
    env.step(action)
  torch.cuda.synchronize()
  wall = time.perf_counter() - t0
  step_dt = env.step_dt
  extra = {k: float(torch.tensor([s.elapsed_time(e) for s, e in v]).median()) for k, v in timed.items() if v}
  return {
    **extra,
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
  parser.add_argument("--scenes", nargs="+", default=["g1", "crowd4", "crowd8"],
                      help="g1, crowd4, crowd8, plaza-{rigid,capsules}-<humans>")
  args = parser.parse_args(argv)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from crowdmotion.sim.mjlab.corridor.scene import make_crowd_env_cfg

  with RunDir.create("sim_speed", out=args.out, record={"args": {k: str(v) for k, v in vars(args).items()}}) as run:
    rows = []
    for n in args.num_envs:
      for kind in args.scenes:
        plaza = kind.startswith("plaza-")
        if kind == "g1":
          cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
          cfg.scene.num_envs = n
        elif plaza:
          from crowdmotion.core.paths import default_graph
          from crowdmotion.sim.mjlab.plaza import make_plaza_env_cfg

          _, geometry, humans = kind.split("-")
          cfg = make_plaza_env_cfg(str(default_graph()), humans=int(humans), num_envs=n, seed=0,
                                   human_geometry=geometry)
        else:
          cfg = make_crowd_env_cfg(n, num_humans=int(kind[-1]), seed=0, moving=True)
        torch.cuda.reset_peak_memory_stats()
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
          row = {"scene": kind, **measure(env, args.steps, warmup=20, stages=plaza)}
        finally:
          env.close()
        rows.append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()
    run.write_json("rows.json", rows)


if __name__ == "__main__":
  main()
