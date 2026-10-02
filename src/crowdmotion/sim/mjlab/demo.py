"""mjlab plaza demos: graph-driven capsule humans seen by a LiDAR rig.

  crowd demo video [--graph <graph.pt>] [--out runs/<UTC>_mjlab_graph_demo] [--seconds 20]
  crowd demo viser [--graph <graph.pt>] [--device cpu] [--port 8080]

`video` renders offscreen to `crowd.mp4` at the environment rate (50 fps; needs
`MUJOCO_GL=egl` on a GPU machine or `osmesa` on CPU) and writes `summary.json`.
`viser` serves the stock `ViserPlayViewer`: play/pause, step, speed, reset (a new
random crowd) and debug overlays (LiDAR hits, goals, commanded velocities).
Open the printed port in a browser, through an SSH tunnel on a remote machine.
Both use the `GraphCrowd` command term (`crowdmotion.sim.mjlab.command`).
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np
import torch

from crowdmotion.core.paths import default_graph
from crowdmotion.core.rundir import RunDir


def _common(parser: argparse.ArgumentParser, default_device: str) -> None:
  parser.add_argument("--graph", type=Path, help="saved graph (default: $CROWD_GRAPH or artifact graph-default)")
  parser.add_argument("--device", default=default_device)
  parser.add_argument("--humans", type=int, default=16)
  parser.add_argument("--seed", type=int, default=0)


def video(argv=None) -> None:
  parser = argparse.ArgumentParser(prog="crowd demo video", description="Render the mjlab plaza to MP4.")
  _common(parser, "cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_mjlab_graph_demo)")
  parser.add_argument("--seconds", type=float, default=20.0)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  args = parser.parse_args(argv)
  graph = args.graph or default_graph()
  arg_info = {k: str(v) for k, v in vars(args).items()}

  import imageio.v2 as imageio
  import mujoco
  from mjlab.envs import ManagerBasedRlEnv

  from crowdmotion.sim.mjlab.plaza import make_plaza_env_cfg, set_hit_sphere_radius

  with RunDir.create("mjlab_graph_demo", out=args.out, record={"args": arg_info}) as run:
    cfg = make_plaza_env_cfg(str(graph), humans=args.humans, seed=args.seed, width=args.width, height=args.height)
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array")
    set_hit_sphere_radius(env)
    env.reset()
    term = env.command_manager.get_term("crowd")
    model = env.sim.mj_model
    crowd_geoms = torch.tensor(
      [g for g in range(model.ngeom)
       if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("crowd/")],
      device=args.device,
    )
    sensor = env.scene["lidar"]
    action = torch.zeros(env.action_space.shape, device=args.device)
    steps = round(args.seconds / env.step_dt)
    writer = imageio.get_writer(run.path / "crowd.mp4", fps=round(1 / env.step_dt), codec="libx264", quality=8,
                                macro_block_size=1)
    timing = {"env_step": [], "render": []}
    hits, gaps, natural = [], [], []
    try:
      for step in range(steps):
        t0 = time.perf_counter()
        env.step(action)
        if args.device.startswith("cuda"):
          torch.cuda.synchronize()
        timing["env_step"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        writer.append_data(env.render())
        timing["render"].append(time.perf_counter() - t0)
        hits.append(int(torch.isin(sensor.data.geom_ids[0], crowd_geoms).sum()))
        r = term.state.root_xz.view(env.num_envs, term.M, 2)[0]
        d = (r[:, None] - r[None]).norm(dim=-1) + torch.eye(term.M, device=r.device) * 1e3
        gaps.append(float(d.min()))
        natural.append(float(term.metrics["natural_successor"][0]))
        if step % 50 == 0:
          print(f"t={(step + 1) * env.step_dt:5.1f}s  min gap {gaps[-1]:.2f} m  human LiDAR hits {hits[-1]}",
                flush=True)
    finally:
      writer.close()
      env.close()
    summary = {
      "humans": args.humans,
      "seconds": steps * env.step_dt,
      "device": args.device,
      "video": "crowd.mp4",
      "video_fps": round(1 / env.step_dt),
      "lidar_rays": int(sensor.num_rays),
      "human_lidar_hits_mean": float(np.mean(hits)),
      "min_pair_root_gap_m": float(np.min(gaps)),
      "min_pair_root_gap_p05_m": float(np.percentile(gaps, 5)),
      "natural_successor_frac": float(np.mean(natural)),
      "env_step_ms_mean": 1000 * float(np.mean(timing["env_step"][5:])),
      "render_ms_mean": 1000 * float(np.mean(timing["render"][5:])),
    }
    run.write_json("summary.json", summary)
  print(summary, flush=True)


def viser(argv=None) -> None:
  parser = argparse.ArgumentParser(prog="crowd demo viser", description="Serve the mjlab plaza in Viser.")
  _common(parser, "cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8080)
  parser.add_argument("--lidar-azimuth", type=int, default=360)
  parser.add_argument(
    "--lidar-stride", type=int, default=None,
    help="draw every n-th LiDAR hit (default 8 on CUDA, 16 on CPU; drawing is per-frame Python)",
  )
  parser.add_argument("--frame-rate", type=float, default=None, help="viewer FPS (default 60 on CUDA, 30 on CPU)")
  parser.add_argument(
    "--threads", type=int, default=None,
    help="torch CPU threads (default 1 on CPU: more threads oversubscribe busy shared hosts)",
  )
  parser.add_argument("--verbose", action="store_true", help="print viewer FPS and real-time factor")
  args = parser.parse_args(argv)
  cpu = args.device == "cpu"
  if args.threads or cpu:
    torch.set_num_threads(args.threads or 1)
  stride = args.lidar_stride or (16 if cpu else 8)
  frame_rate = args.frame_rate or (30.0 if cpu else 60.0)

  import viser as viser_lib
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer import ViserPlayViewer
  from mjlab.viewer.base import VerbosityLevel

  from crowdmotion.sim.mjlab.plaza import make_plaza_env_cfg, set_hit_sphere_radius

  cfg = make_plaza_env_cfg(
    str(args.graph or default_graph()), humans=args.humans, num_envs=args.num_envs, seed=args.seed,
    lidar_azimuth=args.lidar_azimuth, lidar_stride=stride,
  )
  env = ManagerBasedRlEnv(cfg, device=args.device)
  set_hit_sphere_radius(env)
  wrapped = RslRlVecEnvWrapper(env)
  action = torch.zeros(env.action_space.shape, device=args.device)

  server = viser_lib.ViserServer(host=args.host, port=args.port, label="crowd-control motion graph")
  port = server.get_port()
  print(f"VISER host={socket.getfqdn()} port={port} device={args.device}", flush=True)
  verbosity = VerbosityLevel.DEBUG if args.verbose else VerbosityLevel.SILENT
  viewer = ViserPlayViewer(
    wrapped, lambda obs: action, frame_rate=frame_rate, viser_server=server, verbosity=verbosity
  )
  viewer.run()


def main(argv=None) -> None:
  argv = list(argv or [])
  if not argv or argv[0] not in ("video", "viser"):
    raise SystemExit("usage: crowd demo {video,viser} [options]")
  (video if argv[0] == "video" else viser)(argv[1:])


if __name__ == "__main__":
  import sys

  main(sys.argv[1:])
