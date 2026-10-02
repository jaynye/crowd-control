"""GPU smoke test for ARDY Core checkpoints in the unified crowd-control env.

Checks that ARDY-Core-RP-20FPS-Horizon{8,40} load from local weights without a
text encoder, that `autoregressive_step` returns history + horizon frames, and
that the Core skin mesh poses. Text features are zeros (the gated Llama-3
encoder is unavailable); the denoiser cost does not depend on the prompt.

  sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/ardy_smoke.sbatch

Writes `runs/<UTC>_ardy_smoke/smoke.json` unless given `--out`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from crowdmotion.core.paths import RUNS
from crowdmotion.core.rundir import utc_stamp


def timed(fn, reps: int = 3):
  fn()
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  out = None
  for _ in range(reps):
    out = fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - t0) / reps, out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, default=RUNS / f"{utc_stamp()}_ardy_smoke" / "smoke.json")
  args = parser.parse_args()

  from ardy.model.load_model import load_model
  from ardy.viz.core_skin import CoreSkin

  results: dict = {"gpu": torch.cuda.get_device_name(0), "models": {}}
  for name in ("ARDY-Core-RP-20FPS-Horizon8", "ARDY-Core-RP-20FPS-Horizon40"):
    t0 = time.perf_counter()
    model = load_model(name, device="cuda", text_encoder=False)
    load_s = time.perf_counter() - t0
    rep = model.motion_rep
    H = model.gen_horizon_len
    row: dict = {
      "load_s": load_s,
      "horizon": H,
      "fps": rep.fps,
      "frames_per_token": model.num_frames_per_token,
      "num_base_steps": model.diffusion.num_base_steps,
      "motion_rep_dim": rep.motion_rep_dim,
      "joints": rep.skeleton.nbjoints,
      "joint_names": list(rep.skeleton.bone_order_names),
      "root_idx": rep.skeleton.root_idx,
    }

    def step(batch, history=None, frames=None, steps=None, cfg=(2.0, 2.0)):
      feat = torch.zeros(batch, 1, 4096, device="cuda")
      mask = torch.ones(batch, 1, dtype=torch.bool, device="cuda")
      hist_len = 0 if history is None else history.shape[1]
      return model.autoregressive_step(
        num_frames=frames or hist_len + H,
        num_denoising_steps=steps or model.diffusion.num_base_steps,
        motion_mask=None,
        observed_motion=None,
        cfg_weight=cfg,
        text_feat=feat,
        text_pad_mask=mask,
        init_history_sequence=history,
      )

    fresh = step(4)
    row["fresh_out_shape"] = list(fresh.shape)
    hist = fresh[:, -4:]
    cont = step(4, history=hist)
    row["cont_out_shape"] = list(cont.shape)
    # Seam: re-encoded history vs the frames it was built from.
    dec_hist = rep.inverse(hist, is_normalized=True)
    dec_cont = rep.inverse(cont, is_normalized=True)
    row["history_reencode_joint_err_m"] = (
      (dec_cont["posed_joints"][:, :4] - dec_hist["posed_joints"]).norm(dim=-1).mean().item()
    )
    row["decoded_keys"] = {k: list(v.shape) for k, v in dec_cont.items() if torch.is_tensor(v)}

    lat = []
    for batch in (1, 64, 1024):
      torch.cuda.reset_peak_memory_stats()
      dt, _ = timed(lambda b=batch: step(b, history=torch.cat([hist] * (b // 4 + 1))[:b]))
      lat.append(
        {
          "batch": batch,
          "seconds": dt,
          "new_motion_s_per_wall_s": batch * H / rep.fps / dt,
          "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
      )
      print(name, json.dumps(lat[-1]), flush=True)
    row["latency_hist4_default"] = lat
    results["models"][name] = row

  skin = CoreSkin(model.skeleton)
  results["core_skin"] = {
    "vertices": int(skin.bind_vertices.shape[0]),
    "faces": int(skin.faces.shape[0]),
    "lbs_width": int(skin.lbs_weights.shape[1]),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(results, indent=2))
  print(json.dumps({k: v for k, v in results.items() if k != "models"}), flush=True)
  for name, row in results["models"].items():
    print(name, {k: v for k, v in row.items() if k not in ("joint_names", "latency_hist4_default")})
  print("wrote", args.out, flush=True)


if __name__ == "__main__":
  main()
