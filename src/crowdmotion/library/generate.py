"""Generate an offline ARDY motion library under random locomotion commands.

  source scripts/env.sh
  crowd library generate --config configs/library/ardy_a40.json --out runs/<UTC>_ardy_library

Every human streams closed-loop ARDY motion. At each edge boundary (every
`edge_frames`, a multiple of ARDY's 4-frame token) the human's command is read,
a root path is planned from its executed state (as in the quality harness),
ARDY generates from the last `history` executed frames, and only the first
`edge_frames` new frames are kept. Each edge is therefore a genuine ARDY
continuation of the frames before it under a known command.

`command_source` picks the commands: "random" (default, `RandomCommands`: root path
only) or "turns" (`TurnCommands`: stand / walk / in-place turn / arc segments whose
plans also carry per-frame body-heading targets). `commands` holds the selected
source's ranges (`CommandRanges` or `TurnRanges`).

Writes `library.pt` (a dict of tensors, see `save_library`) and `run.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from crowdmotion.backends.registry import ArdyConfig, config_from_case, make_backend
from crowdmotion.behavior.commands import CommandRanges, RandomCommands, TurnCommands, TurnRanges
from crowdmotion.behavior.controller import CrowdController
from crowdmotion.behavior.kinematics import WindowVelocity
from crowdmotion.behavior.sources import OnlineMotionSource
from crowdmotion.core.config import check_keys
from crowdmotion.core.rng import human_seeds
from crowdmotion.core.rundir import RunDir
from crowdmotion.core.skeleton import SkeletonInfo
from crowdmotion.core.timing import synchronize

# Library generation defaults: the recommended online setting (TF32, no guidance, 5 steps).
LIBRARY_DEFAULTS = {"model": ArdyConfig.model, "steps": 5, "precision": "tf32", "cfg_type": "nocfg", "future_frames": 0}


LIBRARY_KEYS = ("model", "steps", "precision", "cfg_type", "future")
# Every key a library config may hold (unknown keys, e.g. typos, are errors).
CONFIG_KEYS = (*LIBRARY_KEYS, "name", "description", "history", "edge_frames", "seed", "humans", "batch",
               "duration", "commands", "command_source")
COMMAND_SOURCES = ("random", "turns")


def library_backend(cfg: dict):
  return make_backend(config_from_case({"backend": "ardy", **cfg}, LIBRARY_KEYS, **LIBRARY_DEFAULTS))


@torch.no_grad()
def generate_streams(backend, cfg: dict, seed_offset: int, humans: int) -> dict:
  """Closed-loop ARDY streams for one batch of humans. Tensors stay on the device."""
  dev = backend.device
  fps = backend.fps
  edge = cfg["edge_frames"]
  hist = cfg.get("history", 4)
  nfpt = backend.frames_per_token
  if edge % nfpt or hist % nfpt or hist < 1:
    raise ValueError(f"edge_frames and history must be positive multiples of {nfpt}")
  edges = int(round(cfg["duration"] * fps)) // edge
  seeds = human_seeds(cfg.get("seed", 0), torch.arange(humans), "stream", offset=seed_offset)
  gen = torch.Generator().manual_seed(cfg.get("seed", 0) * 7919 + seed_offset)
  h0 = (torch.rand(humans, generator=gen) * 2 * math.pi - math.pi).to(dev)
  kind = cfg.get("command_source", "random")
  source = OnlineMotionSource(backend, history_frames=hist, cast_history=True)
  if kind == "random":
    commands = RandomCommands(seeds, h0, CommandRanges.from_dict(cfg.get("commands")))
    ctrl = CrowdController(commands, source, WindowVelocity(fps))
  elif kind == "turns":
    commands = TurnCommands(seeds, h0, TurnRanges.from_dict(cfg.get("commands")))
    ctrl = CrowdController(commands, source, WindowVelocity(fps), planner=commands.plan)
  else:
    raise ValueError(f"command_source must be one of {COMMAND_SOURCES}, not {kind!r}")
  ctrl.reset(seeds, torch.zeros(humans, 2, device=dev), h0)
  feats, joints, roots, heads, contacts, rots = [], [], [], [], [], []
  cmd_speed, cmd_heading, cmd_mode = [], [], []
  turn_rows = {"heading_target": [], "cmd_rate": [], "cmd_angle": [], "cmd_radius": [], "cmd_segment": []}
  for k in range(edges):
    step = ctrl.advance(k * edge / fps, take=edge)
    m = step.motion
    feats.append(m.features[:, :edge])
    joints.append(m.joints[:, :edge])
    roots.append(m.root[:, :edge])
    heads.append(m.heading[:, :edge])
    contacts.append(m.contacts[:, :edge].bool())
    rots.append(m.local_rot[:, :edge].half())
    cmd_speed.append(step.speed)
    cmd_heading.append(step.heading)
    cmd_mode.append(torch.as_tensor(commands.state()["mode"], device=dev))
    if kind == "turns":
      st = commands.state()
      turn_rows["heading_target"].append(commands.reference[:, :edge])
      for key, name in (("cmd_rate", "rate"), ("cmd_angle", "angle"), ("cmd_radius", "radius"),
                        ("cmd_segment", "segment")):
        turn_rows[key].append(torch.as_tensor(st[name], device=dev))
  out = {
    "features": torch.cat(feats, 1),
    "posed_joints": torch.cat(joints, 1),
    "root_pos": torch.cat(roots, 1),
    "heading": torch.cat(heads, 1),
    "contacts": torch.cat(contacts, 1),
    "local_rot": torch.cat(rots, 1),
    "cmd_speed": torch.stack(cmd_speed, 1),
    "cmd_heading": torch.stack(cmd_heading, 1),
    "cmd_mode": torch.stack(cmd_mode, 1),
    "seeds": seeds,
  }
  if kind == "turns":
    out["heading_target"] = torch.cat(turn_rows.pop("heading_target"), 1)
    out.update({key: torch.stack(rows, 1) for key, rows in turn_rows.items()})
  return out


def save_library(path: Path, parts: list[dict], backend, cfg: dict) -> dict:
  """Concatenate batches and save.

  Per-stream tensors are `[N, T, ...]` with T = edges * edge_frames: features
  (fp16), posed_joints, root_pos, heading, contacts, local_rot (fp16). Per-edge
  commands are `[N, edges]`: cmd_speed, cmd_heading (world), cmd_mode (0 walk,
  1 stand; with turning commands also 2 in-place turn, 3 arc). Turning commands add
  `heading_target [N, T]` (the executed frames' dense planned heading reference, also
  where it was not constrained or not passed to the generator) and per-edge `cmd_rate` (signed
  rad/s), `cmd_angle` (signed segment turn, rad), `cmd_radius` (arcs, m; NaN
  otherwise) and `cmd_segment` (segment counter).
  """
  lib = {key: torch.cat([p[key].cpu() for p in parts]) for key in parts[0]}
  lib["features"] = lib["features"].half()
  skel = SkeletonInfo.from_upstream(backend.model.skeleton, backend.joint_names)
  if skel.head_idx is None:
    raise ValueError("library skeletons need a 'Head' joint")
  lib["meta"] = {
    "fps": backend.fps,
    "edge_frames": cfg["edge_frames"],
    "history": cfg.get("history", 4),
    "joint_names": backend.joint_names,
    "parents": skel.parents,
    "foot_idx": skel.foot_idx,
    "head_idx": skel.head_idx,
    "neutral_head_height": skel.neutral_head_height,
    "model": backend.model_name,
    "steps": backend.steps,
    "cfg_type": backend.cfg_type,
    "precision": backend.precision,
  }
  if cfg.get("command_source", "random") != "random":
    lib["meta"]["command_source"] = cfg["command_source"]
  torch.save(lib, path)
  return lib


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_ardy_library)")
  args = parser.parse_args(argv)
  cfg = json.loads(args.config.read_text())
  check_keys(cfg, CONFIG_KEYS, "library config")
  with RunDir.create("ardy_library", out=args.out, record={"config": cfg}) as run:
    t0 = time.perf_counter()
    backend = library_backend(cfg)
    load_s = time.perf_counter() - t0
    parts, batch, total = [], cfg.get("batch", 1024), cfg["humans"]
    synchronize(backend.device)
    t0 = time.perf_counter()
    for start in range(0, total, batch):
      parts.append(generate_streams(backend, cfg, start, min(batch, total - start)))
      synchronize(backend.device)
      print(f"humans {start + len(parts[-1]['seeds'])}/{total} at {time.perf_counter() - t0:.1f} s", flush=True)
    gen_s = time.perf_counter() - t0
    lib = save_library(run.path / "library.pt", parts, backend, cfg)
    frames = lib["posed_joints"].shape[0] * lib["posed_joints"].shape[1]
    stats = {
      "load_s": load_s,
      "generate_s": gen_s,
      "streams": lib["posed_joints"].shape[0],
      "frames": frames,
      "motion_hours": frames / backend.fps / 3600,
      "kept_motion_s_per_wall_s": frames / backend.fps / gen_s,
      "peak_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
      "nonfinite_streams": int((~torch.isfinite(lib["posed_joints"]).all(dim=(1, 2, 3))).sum()),
    }
    print(json.dumps(stats), flush=True)
    run.record["stats"] = stats


if __name__ == "__main__":
  main()
