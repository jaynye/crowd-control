"""Check folding ONNX MatMuls with a 2-D right operand into one mm (`OnnxPort`): equal
outputs to the unfolded port on varied requests, and the plans/s of each at --batch.

    .venv/bin/python scripts/smoke/sonic_planner_fold_check.py --batch 4096
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from crowdmotion.backends import onnx_port
from crowdmotion.backends.sonic import SonicPlanner, example_planner_inputs
from crowdmotion.core.rundir import RunDir

UNFOLDED = onnx_port.OnnxPort.op


def folded(op, x, a):
  if op == "MatMul" and x[1].ndim == 2 and x[0].ndim >= 2:
    k, n = x[1].shape
    return (x[0].reshape(-1, k) @ x[1]).reshape(*x[0].shape[:-1], n)
  return UNFOLDED(op, x, a)


def requests(b: int, device: str) -> dict:
  g = np.random.default_rng(0)
  ex = example_planner_inputs()
  out = {k: np.repeat(v[None], b, 0) for k, v in ex.items()}
  ang = g.uniform(-np.pi, np.pi, b)
  out["movement_direction"][:, 0, 0], out["movement_direction"][:, 0, 1] = np.cos(ang), np.sin(ang)
  face = ang + g.uniform(-1, 1, b)
  out["facing_direction"][:, 0, 0], out["facing_direction"][:, 0, 1] = np.cos(face), np.sin(face)
  out["target_vel"][:, 0] = g.uniform(0.3, 1.2, b)
  out["mode"][:, 0] = g.choice([0, 1, 2], b)
  out["random_seed"][:, 0] = g.integers(0, 10000, b)
  out["context_mujoco_qpos"][:, :, :, 7:] += g.normal(0, 0.05, (b, 1, 4, 29)).astype(np.float32)
  return {k: torch.from_numpy(v).to(device) for k, v in out.items()}


def timed(planner, inputs, repeats=3):
  planner(inputs)
  best = 1e9
  for _ in range(repeats):
    t = time.perf_counter()
    q, f = planner(inputs)
    best = min(best, time.perf_counter() - t)
  return best, q, f


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--batch", type=int, default=4096)
  args = p.parse_args()
  with RunDir.create("sonic_planner_fold_check", record={"kind": "sonic_planner_fold_check", **vars(args)}) as rd:
    planner = SonicPlanner(device="cuda")
    small = requests(64, "cuda")
    ref_q, ref_f = planner(small)
    onnx_port.OnnxPort.op = staticmethod(folded)
    new_q, new_f = planner(small)
    res = {"max_abs_qpos": float((new_q - ref_q).abs().max()), "frames_equal": bool((new_f == ref_f).all())}
    big = requests(args.batch, "cuda")
    res["folded_s"], _, _ = timed(planner, big)
    onnx_port.OnnxPort.op = staticmethod(UNFOLDED)
    res["unfolded_s"], _, _ = timed(planner, big, repeats=1)
    res.update(batch=args.batch, folded_plans_per_s=args.batch / res["folded_s"],
               unfolded_plans_per_s=args.batch / res["unfolded_s"], gpu=torch.cuda.get_device_name(0))
    rd.write_json("result.json", res)
    print(res, flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
