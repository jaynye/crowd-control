"""GPU smoke test and latency benchmark for the unified crowd-control env.

Checks that Kimodo-SOMA-RP-v1.1 loads from local weights, that SOMALayer runs
on local assets with Warp, and measures generation latency. Text features are
zeros (text-free, constraint-driven sampling), so the gated Llama-3 encoder is
not needed. Denoiser latency does not depend on the prompt.

  source scripts/env.sh && .venv/bin/python scripts/smoke/kimodo_smoke.py
  sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/kimodo_smoke.sbatch

Writes `runs/<UTC>_kimodo_smoke/smoke.json` unless given `--out`. SOMA-X assets
come from third_party/soma-x (scripts/bootstrap.sh).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from crowdmotion.backends.text import ZeroTextEncoder
from crowdmotion.core.paths import ROOT, RUNS
from crowdmotion.core.rundir import utc_stamp


def quiet(it):
  return it


def timed(fn, reps: int = 1) -> tuple[float, object]:
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  out = None
  for _ in range(reps):
    out = fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - t0) / reps, out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, default=RUNS / f"{utc_stamp()}_kimodo_smoke" / "smoke.json")
  parser.add_argument("--seconds", type=float, default=5.0)
  parser.add_argument("--batches", type=int, nargs="*", default=[1, 16, 64, 256])
  parser.add_argument("--steps", type=int, nargs="+", default=[100, 25])
  parser.add_argument("--tf32", action="store_true", help="Allow TF32 matmuls.")
  args = parser.parse_args()
  torch.backends.cuda.matmul.allow_tf32 = args.tf32
  torch.backends.cudnn.allow_tf32 = args.tf32

  from kimodo.constraints import Root2DConstraintSet
  from kimodo.model.load_model import load_model
  from soma import SOMALayer

  results: dict = {"gpu": torch.cuda.get_device_name(0), "tf32": args.tf32}
  print("GPU:", results["gpu"], flush=True)

  t0 = time.perf_counter()
  model = load_model(
    "Kimodo-SOMA-RP-v1.1", device="cuda", text_encoder=ZeroTextEncoder()
  )
  results["load_s"] = time.perf_counter() - t0
  fps = model.fps
  frames = int(args.seconds * fps)
  results.update(fps=fps, frames=frames)
  print(f"loaded in {results['load_s']:.1f}s; fps={fps}", flush=True)

  def gen(batch: int, steps: int, **kw):
    return model(
      [""] * batch,
      frames,
      num_denoising_steps=steps,
      progress_bar=quiet,
      **kw,
    )

  gen(1, 5)  # Warm-up.
  latency = []
  for steps in args.steps:
    for batch in args.batches:
      torch.cuda.reset_peak_memory_stats()
      try:
        dt, _ = timed(lambda b=batch, s=steps: gen(b, s))
      except torch.OutOfMemoryError:
        print(f"steps={steps} batch={batch}: OOM", flush=True)
        torch.cuda.empty_cache()
        continue
      row = {
        "steps": steps,
        "batch": batch,
        "seconds": dt,
        "clip_seconds_per_wall_second": batch * args.seconds / dt,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
      }
      latency.append(row)
      print(json.dumps(row), flush=True)
  results["latency"] = latency

  # Dense root path: a 90-degree left arc at 1.2 m/s. Y-up; +Z is forward.
  batch, speed, radius = 8, 1.2, 3.0
  t = torch.arange(frames) / fps
  theta = speed * t / radius
  path = torch.stack((radius * (1 - torch.cos(theta)), radius * torch.sin(theta)), -1)
  constraint = Root2DConstraintSet(model.skeleton, torch.arange(frames), path)
  constraint.to("cuda")
  constraints = [[constraint] for _ in range(batch)]
  for post in (False, True):
    dt, out = timed(
      lambda p=post: gen(batch, 100, constraint_lst=constraints, post_processing=p)
    )
    root = out["smooth_root_pos"][..., [0, 2]].cpu()
    err = (root - path).norm(dim=-1)
    feet = out["foot_contacts"].float().mean().item()
    row = {
      "post_processing": post,
      "batch": batch,
      "seconds": dt,
      "root_err_mean_m": err.mean().item(),
      "root_err_max_m": err.max().item(),
      "foot_contact_fraction": feet,
    }
    results[f"root_path_post{int(post)}"] = row
    print(json.dumps(row), flush=True)

  # Multi-prompt continuation: segment 2 is conditioned on segment 1's tail.
  dt, out = timed(
    lambda: model(
      ["", ""],
      [frames, frames],
      num_denoising_steps=100,
      multi_prompt=True,
      num_samples=8,
      progress_bar=quiet,
    )
  )
  results["multiprompt_2x"] = {"batch": 8, "seconds": dt}
  print(json.dumps(results["multiprompt_2x"]), flush=True)

  # SOMA xlo skinning on local assets with the Warp LBS path.
  layer = SOMALayer(
    data_root=ROOT / "third_party" / "soma-x", identity_model_type="soma", lod="xlo", device="cuda"
  )
  rots = out["local_rot_mats"].reshape(-1, 77, 3, 3).float()
  identity = torch.zeros(1, layer.num_shape_components, device="cuda")
  layer.prepare_identity(identity)
  dt, posed = timed(lambda: layer.pose(rots, pose2rot=False), reps=5)
  results["soma_xlo"] = {
    "poses": rots.shape[0],
    "vertices": tuple(posed.vertices.shape),
    "seconds": dt,
    "finite": bool(torch.isfinite(posed.vertices).all()),
  }
  print(json.dumps(results["soma_xlo"]), flush=True)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(results, indent=2))
  print("wrote", args.out, flush=True)
  assert math.isfinite(results["load_s"])


if __name__ == "__main__":
  main()
