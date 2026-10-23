"""ARDY body-heading targets: angle convention, following, and condition parity (CUDA).

  source scripts/env.sh
  .venv/bin/python scripts/smoke/ardy_heading.py [--out runs/<UTC>_ardy_heading]

Uses the library setting (H40, TF32, no guidance, 5 steps, 4 history frames, 8 frames
kept per call). Writes `heading_check.json` and `heading_check.npz` to a fresh run
directory.

1. Condition parity: the "library" (upstream constraint objects) and "vectorized"
   condition paths give the same mask and observation for random paths and headings,
   with NaN (unconstrained) frames, and are unchanged without headings.
2. Convention and following: humans start fresh at spread headings `h0`, are asked to
   stand still (path = start position) and, for `calls` closed-loop calls, to hold a
   constant heading `h0 + delta` for delta in {0, +90, -90, 180} degrees, with and
   without the heading target. Reported per delta: the returned heading, the hip-line,
   shoulder-line and toe headings measured from the posed joints (independent of the
   heading feature), root drift, and label-free planted-foot speed and stepping.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from crowdmotion.behavior.sources import OnlineMotionSource
from crowdmotion.core.rundir import RunDir
from crowdmotion.library.generate import library_backend
from crowdmotion.metrics.feet import label_free_skate

DELTAS_DEG = (0.0, 90.0, -90.0, 180.0)


def line_heading(right: np.ndarray, left: np.ndarray) -> np.ndarray:
  """Facing from a right-minus-left body line (0 faces +Z, positive toward +X)."""
  d = right - left
  return np.arctan2(d[..., 2], -d[..., 0])


def toe_heading(joints: np.ndarray, names: list[str]) -> np.ndarray:
  """Mean of both foot-to-toe directions (0 faces +Z, positive toward +X)."""
  out = []
  for side in ("Left", "Right"):
    d = joints[..., names.index(f"{side}ToeBase"), :] - joints[..., names.index(f"{side}Foot"), :]
    out.append(np.stack((d[..., 0], d[..., 2]), -1))
  v = out[0] / np.linalg.norm(out[0], axis=-1, keepdims=True) + out[1] / np.linalg.norm(out[1], axis=-1, keepdims=True)
  return np.arctan2(v[..., 0], v[..., 1])


def wrap_deg(a: np.ndarray) -> np.ndarray:
  return (np.asarray(a) + 180.0) % 360.0 - 180.0


def parity(backend, batch: int = 8, hist: int = 4) -> dict:
  g = torch.Generator().manual_seed(0)
  n = backend.path_frames
  path = (torch.randn(batch, n, 2, generator=g).cumsum(1) * 0.05).to(backend.device)
  heading = (torch.rand(batch, n, generator=g) * 2 * math.pi - math.pi).to(backend.device)
  heading[0, 5:9] = float("nan")
  heading[1] = float("nan")
  out = {}
  num_frames = hist + backend.horizon + backend.future_frames
  for name, rh in (("path_only", None), ("with_heading", heading), ("all_heading", torch.nan_to_num(heading, 0.3))):
    res = {}
    for mode in ("library", "vectorized"):
      backend.conditions_mode = mode
      res[mode] = backend.conditions(path, hist, num_frames, batch, rh)
    backend.conditions_mode = "vectorized"
    (ml, ol), (mv, ov) = res["library"], res["vectorized"]
    out[name] = {"mask_equal": bool(torch.equal(ml.float(), mv.float())),
                 "observed_max_abs_diff": float((ol - ov).abs().max()),
                 "observed_finite": bool(torch.isfinite(ov).all() and torch.isfinite(ol).all())}
  return out


@torch.no_grad()
def hold_heading(backend, names: list[str], calls: int, take: int, use_targets: bool) -> dict:
  per = 16
  h0 = torch.linspace(-math.pi, math.pi, per + 1)[:-1] + 0.1
  deltas = torch.tensor([math.radians(d) for d in DELTAS_DEG])
  start = h0.repeat(len(deltas))
  target = start + deltas.repeat_interleave(per)
  b = len(start)
  dev = backend.device
  seeds = torch.arange(b) + 7_000_000
  src = OnlineMotionSource(backend, history_frames=4, cast_history=True)
  state = src.reset(seeds, torch.zeros(b, 2, device=dev), start.to(dev))
  path = torch.zeros(b, src.path_frames, 2, device=dev)
  rh = target.to(dev)[:, None].expand(b, src.path_frames).contiguous()
  heads, joints, roots, contacts = [], [], [], []
  for _ in range(calls):
    step = src.step(state, path, root_heading=rh if use_targets else None)
    src.commit(state, step, take)
    heads.append(step.heading[:, :take].cpu())
    joints.append(step.joints[:, :take].cpu())
    roots.append(step.root[:, :take].cpu())
    contacts.append(step.contacts[:, :take].cpu())
  heading = torch.cat(heads, 1).numpy()
  j = torch.cat(joints, 1).numpy()
  root = torch.cat(roots, 1).numpy()
  hip = line_heading(j[..., names.index("RightUpLeg"), :], j[..., names.index("LeftUpLeg"), :])
  shoulder = line_heading(j[..., names.index("RightArm"), :], j[..., names.index("LeftArm"), :])
  toe = toe_heading(j, names)
  fps = backend.fps
  rows = {}
  for k, d in enumerate(DELTAS_DEG):
    sl = slice(k * per, (k + 1) * per)
    ref = start[sl].numpy()[:, None]
    end = slice(-int(fps), None)  # the last second
    rel = {name: wrap_deg(np.degrees(np.unwrap(a[sl], axis=1) - ref)) for name, a in
           (("returned", heading), ("hip", hip), ("shoulder", shoulder), ("toe", toe))}
    xz = root[sl][..., [0, 2]]
    feet = torch.from_numpy(j[sl][:, :, backend_feet(names)])
    rows[f"{d:+.0f}"] = {
      "final_heading_rel_deg_median": {k2: float(np.median(v[:, end])) for k2, v in rel.items()},
      "final_heading_error_deg_median_abs": {k2: float(np.median(np.abs(wrap_deg(v[:, end] - d))))
                                             for k2, v in rel.items()},
      "time_to_within_15deg_s_median": float(np.median([first_within(v, d, 15.0) for v in rel["returned"]]) / fps),
      "root_drift_m_median": float(np.median(np.linalg.norm(xz - xz[:, :1], axis=-1).max(1))),
      "root_drift_m_max": float(np.linalg.norm(xz - xz[:, :1], axis=-1).max()),
      "skate": label_free_skate(feet, fps),
      "contact_switches_per_s": float(np.abs(np.diff(torch.cat(contacts, 1)[sl].numpy().astype(int), axis=1)).sum()
                                      / (per * 4 * (len(heading[0]) / fps))),
      "pelvis_height_m": [float(np.percentile(j[sl][..., names.index("Hips"), 1], q)) for q in (5, 50, 95)],
    }
  return {"rows": rows, "arrays": {"heading": heading, "hip": hip, "shoulder": shoulder, "toe": toe,
                                   "joints": j.astype(np.float32), "start": start.numpy(), "target": target.numpy(),
                                   "contacts": torch.cat(contacts, 1).numpy()}}


def backend_feet(names):
  return [names.index(n) for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]


def first_within(rel_deg: np.ndarray, target: float, tol: float) -> float:
  ok = np.abs(wrap_deg(rel_deg - target)) <= tol
  idx = np.flatnonzero(ok)
  return float(idx[0]) if len(idx) else float("nan")


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--out", type=Path)
  parser.add_argument("--calls", type=int, default=15)
  args = parser.parse_args(argv)
  cfg = {"model": "ARDY-Core-RP-20FPS-Horizon40", "steps": 5, "precision": "tf32", "cfg_type": "nocfg"}
  with RunDir.create("ardy_heading", out=args.out, record={"config": cfg, "calls": args.calls}) as run:
    backend = library_backend(cfg)
    names = backend.joint_names
    result = {"gpu": torch.cuda.get_device_name(0), "parity": parity(backend), "take": 8}
    arrays = {}
    for flag in (True, False):
      res = hold_heading(backend, names, args.calls, 8, flag)
      result["targets_on" if flag else "targets_off"] = res["rows"]
      arrays.update({f"{'on' if flag else 'off'}_{k}": v for k, v in res["arrays"].items()})
    arrays["parents"] = np.array(backend.skeleton_info()["parents"])
    run.write_json("heading_check.json", result)
    np.savez_compressed(run.path / "heading_check.npz", **arrays)
    run.record["summary"] = {"parity": result["parity"]}
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
