"""Inspect primitive pedestrians and LiDAR in the corridor fixture without a robot.

  crowd demo corridor --viewer viser
  crowd demo corridor --viewer none --device cpu --steps 250 --capture-every 25 \
      --output-dir runs/<UTC>_corridor
  MUJOCO_GL=egl crowd demo corridor --viewer none --output-dir runs/<UTC>_corridor --render True
  crowd demo corridor plot --scan-file runs/<UTC>_corridor/scans.npz

Headless runs write `scans.npz` (ranges, hit positions, hit geom ids and names,
sensor positions, human x/y/yaw, contact-force histories at each capture),
`summary.json` and the compiled `scene/`. `plot` draws XY/XZ views of one
capture (`corridor.plot`) without a simulator or OpenGL context.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

import mediapy as media
import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from crowdmotion.sim.mjlab.corridor.scene import make_crowd_env_cfg


def run(
  viewer: Literal["auto", "native", "viser", "none"] = "auto",
  num_envs: int = 1,
  num_humans: int = 4,
  seed: int = 0,
  moving: bool = True,
  device: str | None = None,
  steps: int = 250,
  capture_every: int = 25,
  output_dir: Path | None = None,
  render: bool = False,
) -> None:
  """Headless exports contain environment 0, including the initial/final scan."""
  if steps < 0 or capture_every < 1:
    raise ValueError("Require steps >= 0 and capture_every >= 1")
  if output_dir is not None and viewer != "none":
    raise ValueError("Use --viewer none for bounded capture runs")
  if render and output_dir is None:
    raise ValueError("--render requires --output-dir")
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = make_crowd_env_cfg(num_envs, num_humans, seed, moving)
  env = ManagerBasedRlEnv(
    cfg, device=device, render_mode="rgb_array" if render else None
  )
  wrapped = RslRlVecEnvWrapper(env)
  action = torch.zeros(env.action_space.shape, device=device)

  def policy(obs) -> torch.Tensor:
    del obs
    return action

  try:
    if viewer == "auto":
      viewer = "native" if os.environ.get("DISPLAY") else "viser"
    if viewer == "native":
      NativeMujocoViewer(wrapped, policy).run()
      return
    if viewer == "viser":
      ViserPlayViewer(wrapped, policy).run()
      return

    sensor: RayCastSensor = env.scene["lidar"]
    contacts: list[ContactSensor] = [
      env.scene[f"human_{i}_contacts"] for i in range(num_humans)
    ]
    peak_force = torch.zeros(num_humans, device=device)
    scans: list[dict[str, np.ndarray]] = []
    if output_dir is not None:
      output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(steps + 1):
      if step:
        wrapped.step(action)
      for i, contact in enumerate(contacts):
        history = contact.data.force_history
        assert history is not None
        peak_force[i] = torch.maximum(peak_force[i], history[0].norm(dim=-1).max())
      if step % capture_every and step != steps:
        continue
      data = sensor.data
      assert data.geom_ids is not None
      positions = (
        np.stack(
          [
            env.scene[f"human_{i}"].data.joint_pos[0].cpu().numpy().copy()
            for i in range(num_humans)
          ]
        )
        if num_humans
        else np.empty((0, 3), dtype=np.float32)
      )
      scans.append(
        {
          "time": np.asarray(step * env.step_dt),
          "distances": data.distances[0].cpu().numpy().copy(),
          "geom_ids": data.geom_ids[0].cpu().numpy().copy(),
          "hit_pos_w": data.hit_pos_w[0].cpu().numpy().copy(),
          "sensor_pos_w": data.pos_w[0].cpu().numpy().copy(),
          "human_xyyaw": positions,
          "contact_force_history": np.stack(
            [
              contact.data.force_history[0].cpu().numpy().copy()
              for contact in contacts
              if contact.data.force_history is not None
            ]
          )
          if contacts
          else np.empty((0, 1, cfg.decimation, 3), dtype=np.float32),
        }
      )
      if render:
        assert output_dir is not None
        frame = env.render()
        assert frame is not None
        media.write_image(output_dir / f"frame_{step:06d}.png", frame)

    names = [
      mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
      for i in range(env.sim.mj_model.ngeom)
    ]
    summary = {
      "seed": seed,
      "device": device,
      "num_envs": num_envs,
      "num_humans": num_humans,
      "moving": moving,
      "steps": steps,
      "step_dt": env.step_dt,
      "ray_count": sensor.num_rays,
      "sensor_model": "instantaneous angular raster, MID-360 FOV approximation",
      "sensor_mount": "fixed test rig at (-5, 0, 1.15); not calibrated to G1",
      "valid_returns_final": int((scans[-1]["distances"] >= 0).sum()),
      "peak_contact_force_n": peak_force.cpu().tolist(),
      "contact_scope": "per-human maximum contact with any other body, not a sum",
      "hit_geoms_final": sorted(
        {names[int(i)] for i in scans[-1]["geom_ids"] if i >= 0}
      ),
    }
    if output_dir is not None:
      np.savez_compressed(
        output_dir / "scans.npz",
        allow_pickle=False,
        geom_names=np.asarray(names),
        **{key: np.stack([s[key] for s in scans]) for key in scans[0]},
      )
      (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
      env.scene.write(output_dir / "scene")
    print(json.dumps(summary, indent=2))
  finally:
    env.close()


def main(argv: list[str] | None = None) -> None:
  argv = list(sys.argv[1:] if argv is None else argv)
  if argv[:1] == ["plot"]:
    from crowdmotion.sim.mjlab.corridor.plot import plot

    tyro.cli(plot, args=argv[1:])
    return
  tyro.cli(run, args=argv, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
