"""Profile ARDY generation and try torch.compile on the denoiser.

  sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/profile_ardy.sbatch runs/<UTC>_ardy_profile

Writes `profile_<model>.txt` (top CUDA and CPU ops for one steady-state call)
and `compile.jsonl` (wall time per call for eager vs compiled variants).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from crowdmotion.backends.base import MotionRequest
from crowdmotion.backends.registry import ArdyConfig, make_backend
from crowdmotion.behavior.planner import random_arc_paths as steering_paths
from crowdmotion.core.provenance import environment
from crowdmotion.core.rundir import write_json


def request(backend, batch, history):
  dev = backend.device
  xz = torch.zeros(batch, 2, device=dev)
  heading = torch.linspace(-math.pi, math.pi, batch, device=dev)
  seeds = torch.arange(batch)
  counters = torch.zeros(batch, dtype=torch.long)
  path = steering_paths(xz, heading, backend.horizon + backend.future_frames, backend.fps, seeds, counters)
  return MotionRequest(
    text_feat=backend.empty_text(batch), seeds=seeds, counters=counters, history=history,
    root_path=path, start_xz=xz if history is None else None,
    start_heading=heading if history is None else None,
  )


def bench(backend, batch, reps=5):
  fresh = backend.generate(request(backend, batch, None))
  req = request(backend, batch, fresh.features[:, -4:])
  for _ in range(2):
    backend.generate(req)
  torch.cuda.synchronize()
  walls = []
  for _ in range(reps):
    t0 = time.perf_counter()
    backend.generate(req)
    torch.cuda.synchronize()
    walls.append(time.perf_counter() - t0)
  walls.sort()
  return walls[len(walls) // 2], req


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 4096])
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  write_json(args.out / "run.json", {"environment": environment()})
  log = (args.out / "compile.jsonl").open("a")
  for name in ("ARDY-Core-RP-20FPS-Horizon8", "ARDY-Core-RP-20FPS-Horizon40"):
    backend = make_backend(ArdyConfig(model=name, precision="bf16", timing=False))
    # Profile one eager steady-state call at batch 1024.
    _, req = bench(backend, 1024, reps=1)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      backend.generate(req)
      torch.cuda.synchronize()
    table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    (args.out / f"profile_{name}.txt").write_text(table_cuda + "\n\n" + table_cpu)
    print(name, "\n", table_cuda[:6000], flush=True)

    for variant in ("eager", "inductor-default", "inductor-max-autotune-no-cudagraphs"):
      if variant != "eager":
        torch._dynamo.reset()
        backend = make_backend(ArdyConfig(model=name, precision="bf16", timing=False))
        mode = variant.split("-", 1)[1]
        t0 = time.perf_counter()
        try:
          backend.model.compile_denoiser(backend="inductor", mode=mode, dynamic=False)
        except Exception as exc:  # noqa: BLE001
          row = {"model": name, "variant": variant, "error": repr(exc)[:500]}
          log.write(json.dumps(row) + "\n")
          log.flush()
          print(json.dumps(row), flush=True)
          continue
        compile_s = time.perf_counter() - t0
      else:
        compile_s = 0.0
      for batch in args.batches:
        t0 = time.perf_counter()
        try:
          wall, _ = bench(backend, batch)
          row = {"model": name, "variant": variant, "batch": batch, "wall_p50": wall,
                 "R_new": batch * backend.horizon / backend.fps / wall,
                 "first_call_incl_compile_s": time.perf_counter() - t0 + compile_s}
        except Exception as exc:  # noqa: BLE001
          row = {"model": name, "variant": variant, "batch": batch, "error": repr(exc)[:500]}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
    del backend
    torch.cuda.empty_cache()
  log.close()


if __name__ == "__main__":
  main()
