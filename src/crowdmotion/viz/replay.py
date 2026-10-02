"""Replay saved evaluator traces as MuJoCo capsule videos (no graph, prior or LiDAR).

  crowd behavior render --run runs/<eval> [--seed 0] [--view overview|follow|top]

Writes `<run>/render_seed_<seed>_<view>/crowd.mp4` and `render.json`. Arrows on
the floor show body facing, which separates curved forward travel from
sidestepping. Needs an offscreen GL context (`MUJOCO_GL=egl` on a GPU machine).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from crowdmotion.core.geometry import motion_to_mujoco
from crowdmotion.core.provenance import environment, file_sha256
from crowdmotion.core.rundir import write_json
from crowdmotion.sim.capsules import CapsuleSkeleton
from crowdmotion.sim.scene import replay_spec, set_camera


def replay_scene(skeleton, humans, steering):
  """Compiled MuJoCo replay scene for a resolved steering config (see `sim.scene`)."""
  return replay_spec(steering["half_x"], steering["half_z"], steering["obstacles"], skeleton, humans).compile()


def render(args):
  import imageio.v2 as imageio
  import mujoco
  from PIL import Image, ImageDraw

  torch.set_num_threads(1)
  manifest = json.loads((args.run / "run.json").read_text())
  if manifest["status"] != "completed":
    raise ValueError("Render requires a completed evaluation")
  if args.seed not in manifest["config"]["seeds"]:
    raise ValueError("Seed is not present in this run")
  if not 0 <= args.human < manifest["config"]["humans"] or args.fps <= 0:
    raise ValueError("Choose a valid human and positive render FPS")
  out = args.out or args.run / f"render_seed_{args.seed}_{args.view}"
  out.mkdir(parents=True, exist_ok=False)
  trace = np.load(args.run / f"seed_{args.seed}.npz")
  saved = json.loads((args.run / "skeleton.json").read_text())
  skeleton = CapsuleSkeleton.from_rest(
    saved["meta"], torch.tensor(saved["rest_joints"])
  )
  model = replay_scene(
    skeleton, manifest["config"]["humans"], manifest["config"]["steering"]
  )
  model.vis.global_.offwidth, model.vis.global_.offheight = args.width, args.height
  data = mujoco.MjData(model)
  camera = mujoco.MjvCamera()
  set_camera(camera, args.view)
  render_info = {
    "status": "running",
    "evaluation": str(args.run.resolve()),
    "seed": args.seed,
    "view": args.view,
    "human": args.human,
    "fps": args.fps,
    "width": args.width,
    "height": args.height,
    "environment": environment(),
    "trace_sha256": file_sha256(args.run / f"seed_{args.seed}.npz"),
    "scope": "MuJoCo capsule replay of saved motion; no graph, prior or LiDAR.",
  }
  write_json(out / "render.json", render_info)
  motion = torch.from_numpy(trace["joints"])
  fps = manifest["fps"]
  seconds = (motion.shape[1] - 1) / fps
  start = time.perf_counter()
  try:
    with (
      mujoco.Renderer(model, height=args.height, width=args.width) as renderer,
      imageio.get_writer(
        out / "crowd.mp4",
        fps=args.fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        ffmpeg_params=["-threads", "1"],
      ) as writer,
    ):
      for i in range(round(seconds * args.fps)):
        t = i / args.fps
        f = min(t * fps, motion.shape[1] - 1)
        lo, hi = int(f), min(int(f) + 1, motion.shape[1] - 1)
        pose = motion[:, lo] * (1 - (f - lo)) + motion[:, hi] * (f - lo)
        pos, quat = skeleton.poses(motion_to_mujoco(pose))
        data.mocap_pos[:] = pos.reshape(-1, 3).numpy()
        data.mocap_quat[:] = quat.reshape(-1, 4).numpy()
        mujoco.mj_forward(model, data)
        if args.view == "follow":
          p = pose[args.human, 0]
          camera.lookat[:] = (float(p[0]), -float(p[2]), 0.9)
        renderer.update_scene(data, camera=camera)
        # Facing arrows help distinguish curved forward travel from sidestepping.
        for h in range(len(pose)):
          p = pose[h, 0].numpy()
          yaw = trace["heading"][h, lo]
          if renderer.scene.ngeom >= renderer.scene.maxgeom:
            break
          geom = renderer.scene.geoms[renderer.scene.ngeom]
          mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([0.95, 0.95, 0.9, 1]),
          )
          mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.015,
            [p[0], -p[2], 0.05],
            [p[0] + 0.45 * np.sin(yaw), -p[2] - 0.45 * np.cos(yaw), 0.05],
          )
          renderer.scene.ngeom += 1
        image = Image.fromarray(renderer.render())
        ImageDraw.Draw(image).text(
          (12, 12),
          f"Seed {args.seed} | {args.view} | t={t:.2f}s | arrows: body facing",
          fill="white",
          stroke_width=2,
          stroke_fill="black",
        )
        writer.append_data(np.asarray(image))
    render_info.update(
      status="completed", seconds=seconds, render_wall_s=time.perf_counter() - start
    )
  except Exception as exc:
    render_info.update(status="failed", error=str(exc))
    raise
  finally:
    trace.close()
    write_json(out / "render.json", render_info)
  print(out / "crowd.mp4")


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--view", choices=("overview", "follow", "top"), default="overview")
  parser.add_argument("--human", type=int, default=0)
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=540)
  render(parser.parse_args(argv))


if __name__ == "__main__":
  main()
