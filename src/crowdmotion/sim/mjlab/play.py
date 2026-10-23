"""`crowd play traversal`: watch a trained traversal policy walk through the crowd.

  crowd play traversal <run dir> [--checkpoint model_<i>.pt] [--viewer auto|viser|native]
      [--num-envs 1] [--seed 0] [--no-end-on-collision] [--lidar-stride 8] [--port 8080]
      [--crowd trained|<navigation evaluation crowd>] [--runs runs/] [--conditions <campaign>]
  crowd play traversal <run dir> --video [--seconds 30] [--camera follow|overview]
      [--width 1280] [--height 720] [--out runs/<UTC>_play_traversal_<run>]

The run's `run.json` fixes the env (graph, rotations, behavior, humans, the humans'
perception of the robot, the teacher's LiDAR), as `crowd train traversal --resume` would.
The run's `--env.*` overrides are not repeated: they were reward weights, which do not
change the policy's actions. A teacher plays its privileged actor; a distilled student
plays as the LiDAR policy. The checkpoint defaults to the run's newest.

Episodes end on a collision, as in training (`--no-end-on-collision` keeps walking, in
states the policy never trained on). The auto-reset happens inside the step, so the
collision itself is never drawn: each one is printed, and the video holds a red border
and a label for half a second.

The viewer writes nothing. `--viewer auto` picks the native MuJoCo viewer with a display
and Viser without one. Viser is live (`live_viewer`): its panel switches policy (any run
under `--runs`, or a run path, and its checkpoints) and crowd (the run's own, or the
navigation campaign's evaluation crowds) while it plays, and shows collisions, falls and
goals per robot-minute since the last switch with the evaluator's definitions. It plays
the planning arms too (the native viewer does not). It listens on all interfaces without
authentication: on a remote machine, forward the printed port over SSH
(docs/usage/demos.md). `--video` renders offscreen
(`MUJOCO_GL=egl` on a GPU machine) to `play.mp4` with `summary.json` in a fresh run
directory, then attaches the video and its collision counts to the training lineage's
W&B run (`eval/play/*` at the checkpoint's iteration; `crowdmotion.tracking.attach_eval`)
when that lineage is online.
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from crowdmotion.core.paths import ROOT
from crowdmotion.core.rundir import RunDir
from crowdmotion.sim.mjlab.train import actor_checkpoint, env_options, newest_checkpoint
from crowdmotion.tracking import attach_eval
from crowdmotion.tracking.lineage import checkpoint_iteration

# The policy each recorded run plays as: a distilled student is the LiDAR policy.
PLAY_AS = {"lidar": "lidar", "teacher": "teacher", "student": "lidar"}
FLASH_S = 0.5  # how long the video marks a collision
FOLLOW = {"distance": 6.0, "elevation": -35.0, "azimuth": 90.0}  # tracks the robot's torso
PLAY_NCONMAX = 128


def run_policy(run: Path) -> tuple[str, str, dict]:
  """`(trained, plays_as, env options)` of a `crowd train traversal` run."""
  record = json.loads((Path(run) / "run.json").read_text())
  trained = record.get("policy", "lidar")
  if trained not in PLAY_AS:
    raise ValueError(f"{Path(run).name}: unknown policy {trained!r}")
  given = Namespace(graph=None, rotations=None, behavior=None, humans=None, robot_blind=None,
                    forward_first=None, teacher_lidar=None, no_avoidance=None, penalty_ramp=None,
                    planner_rate=None, crowd_curriculum=None)
  return trained, PLAY_AS[trained], env_options(given, Path(run))


def label(run: Path) -> str:
  """`teacher_t7` for `runs/<UTC>_train_traversal_teacher_t7`."""
  name = Path(run).name
  return name.split("_train_traversal_", 1)[-1] if "_train_traversal_" in name else name


class Collisions:
  """Wraps the policy and reports the collisions of the step just taken (`crowd_collision`
  as in training). Without the termination, an overlap counts once, when it begins. Times
  count this player's own steps: loading a checkpoint restores the env's training step
  counter (`common_step_counter`, for curricula)."""

  def __init__(self, env, policy, echo: bool = True):
    from crowdmotion.sim.mjlab.traversal import crowd_collision

    self.env, self.policy, self.echo = env, policy, echo
    self.ends = "crowd_collision" in env.termination_manager.active_terms
    self.check = crowd_collision
    self.inside = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.times: list[tuple[float, int]] = []
    self.steps = 0  # env steps taken since the player started
    self.started = False

  def after_step(self) -> torch.Tensor:
    """Envs that collided in the last step; records and prints them."""
    if self.ends:
      new = self.env.termination_manager.get_term("crowd_collision").clone()
    else:
      hit = self.check(self.env)
      new, self.inside = hit & ~self.inside, hit
    self.steps += 1
    t = self.steps * self.env.step_dt
    for e in new.nonzero().flatten().tolist():
      self.times.append((t, e))
      if self.echo:
        print(f"t={t:7.2f} s  env {e}: collision #{len(self.times)}", flush=True)
    return new

  def __call__(self, obs):
    if self.started:  # the viewer steps after each call
      self.after_step()
    self.started = True
    return self.policy(obs)


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="crowd play traversal", description=__doc__.split("\n\n")[0])
  p.add_argument("run", type=Path, help="a `crowd train traversal` run directory")
  p.add_argument("--checkpoint", type=Path, help="model_<i>.pt: a path, or a name in the run's log directory "
                                                 "(default: the run's newest)")
  p.add_argument("--viewer", choices=("auto", "viser", "native"), default="auto")
  p.add_argument("--video", action="store_true", help="render to MP4 instead of opening a viewer")
  p.add_argument("--num-envs", type=int, default=1,
                 help="worlds to simulate (they share the origin; viewers and videos show env 0)")
  p.add_argument("--seed", type=int, default=0, help="crowd and env seed")
  p.add_argument("--end-on-collision", action=argparse.BooleanOptionalAction, default=True,
                 help="end the episode on a collision, as in training (default: on)")
  p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default 1 on CPU)")
  p.add_argument("--lidar-stride", type=int, default=8,
                 help="draw every n-th LiDAR hit (drawing is per-frame Python; the policy sees every ray)")
  p.add_argument("--host", default="0.0.0.0")
  p.add_argument("--port", type=int, default=8080)
  p.add_argument("--seconds", type=float, default=30.0, help="video length")
  p.add_argument("--camera", choices=("follow", "overview"), default="follow",
                 help="video camera: tracking the robot, or the whole plaza")
  p.add_argument("--width", type=int, default=1280)
  p.add_argument("--height", type=int, default=720)
  p.add_argument("--out", type=Path, help="video run directory (default: runs/<UTC>_play_traversal_<run>)")
  p.add_argument("--crowd", default="trained",
                 help="Viser: the crowd to start in, `trained` (the run's own) or a navigation evaluation crowd "
                      "(independent_n4|8|16, coherent90_n8|16, reactive_aware_n8, reactive_blind_n8)")
  p.add_argument("--runs", type=Path, default=ROOT / "runs",
                 help="Viser: the folder whose *_train_traversal_* runs the policy menu offers (default: runs/)")
  p.add_argument("--conditions", type=Path, metavar="CAMPAIGN",
                 help="Viser: offer a navigation ablation campaign directory's frozen evaluation crowds "
                      "(default: the same crowds, built from the campaign code)")
  p.add_argument("--frame-rate", type=float, default=60.0, help="Viser: viewer frames per second")
  args = p.parse_args(argv)
  if args.device == "cpu" or args.threads:
    torch.set_num_threads(args.threads or 1)
  trained, plays_as, env = run_policy(args.run)
  ckpt = args.checkpoint or newest_checkpoint(args.run)
  if not ckpt.exists() and len(ckpt.parts) == 1:
    ckpt = newest_checkpoint(args.run).parent / ckpt
  if not ckpt.exists():
    p.error(f"no checkpoint {ckpt}")
  viewer = args.viewer
  if viewer == "auto":
    viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
  if viewer == "viser" and not args.video:
    from crowdmotion.sim.mjlab.live_viewer import serve

    return serve(args, args.run, ckpt)
  if args.crowd != "trained":
    p.error("--crowd applies to the Viser viewer; --video and the native viewer play the run's own crowd")

  import mjlab.tasks  # noqa: F401  (populates the registry)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.viewer import ViewerConfig

  from crowdmotion.core.paths import default_graph, default_graph_rotations
  from crowdmotion.sim.mjlab.plaza import set_hit_sphere_radius
  from crowdmotion.sim.mjlab.traversal import collision_termination, register
  from crowdmotion.sim.scene import CAMERAS

  task = register(str(env["graph"] or default_graph()), env["behavior"],
                  str(env["rotations"] or default_graph_rotations()), env["humans"], policy=plays_as,
                  perceive_robot=env["perceive_robot"], forward_first=env["forward_first"],
                  teacher_lidar=env["teacher_lidar"], no_avoidance=env["no_avoidance"],
                  planner_rate=env["planner_rate"], navigation=env.get("navigation"))
  if env["planner_rate"] and not args.video:
    p.error("a planning arm plays in Viser (--viewer viser) or with --video; the native viewer steps whole "
            "decisions")
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.commands["crowd"].seed = args.seed
  if args.end_on_collision:
    cfg.terminations["crowd_collision"] = collision_termination()
  cfg.viewer.max_extra_envs = 0  # every world sits at the origin
  # Contacts are pooled across worlds: training's 35 per world overflows with a few worlds.
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, PLAY_NCONMAX)
  cfg.viewer.width, cfg.viewer.height = args.width, args.height
  if args.camera == "overview":
    view = CAMERAS["overview"]
    cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    cfg.viewer.lookat = view["lookat"]
    cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = (view[k] for k in ("distance", "azimuth",
                                                                                     "elevation"))
  else:
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = (FOLLOW[k] for k in ("distance", "elevation",
                                                                                       "azimuth"))

  agent = load_rl_cfg(task)
  raw = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array" if args.video else None)
  set_hit_sphere_radius(raw)
  raw.scene["lidar"].cfg.viz.stride = args.lidar_stride
  if env["planner_rate"]:
    from crowdmotion.sim.mjlab.planning import PlanningVecEnv

    wrapped = PlanningVecEnv(raw, env["planner_rate"])
  else:
    wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(wrapped, asdict(agent), device=args.device)

  def load(path: Path):
    runner.load(str(actor_checkpoint(path)), load_cfg={"actor": True}, strict=True, map_location=args.device)
    return runner.get_inference_policy(device=args.device)

  print(f"{args.run.name}: {trained} policy playing as {plays_as}, {ckpt.name}", flush=True)
  if args.video:
    return video(args, raw, wrapped, load(ckpt), ckpt, trained, plays_as, env)

  from mjlab.viewer import NativeMujocoViewer

  NativeMujocoViewer(wrapped, Collisions(raw, load(ckpt))).run()
  return 0


def video(args, raw, wrapped, policy, ckpt: Path, trained: str, plays_as: str, env: dict) -> int:
  import imageio.v2 as imageio
  from PIL import Image, ImageDraw, ImageFont

  name = f"play_traversal_{label(args.run)}"
  record = {"run": str(args.run), "checkpoint": str(ckpt), "trained": trained, "plays_as": plays_as,
            "env": env, "args": {k: str(v) for k, v in vars(args).items()}}
  steps = round(args.seconds / raw.step_dt)
  flash = round(FLASH_S / raw.step_dt)
  with RunDir.create(name, out=args.out, record=record) as run:
    collisions = Collisions(raw, policy, echo=True)
    fps = round(1 / raw.step_dt)
    writer = imageio.get_writer(run.path / "play.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1)
    obs = wrapped.get_observations()
    since = flash  # steps since env 0's last collision
    terms = raw.termination_manager
    ends = dict.fromkeys(terms.active_terms, 0)
    font = ImageFont.load_default(size=max(14, args.height // 36))
    bar = font.size + 12
    taken = 0  # env steps rendered

    def frame_after_step(done0: bool) -> None:
      """One env step: tally env 0's episode ends and collisions, render and write a frame."""
      nonlocal since, taken
      taken += 1
      hit = collisions.after_step()
      if done0:
        for name in ends:
          ends[name] += int(terms.get_term(name)[0])
      since = 0 if hit[0] else since + 1
      frame = Image.fromarray(raw.render())
      draw = ImageDraw.Draw(frame)
      n = sum(e == 0 for _, e in collisions.times)
      text = f"{label(args.run)}  {ckpt.stem}  t={taken * raw.step_dt:5.1f} s  collisions {n}"
      draw.rectangle((0, 0, frame.width, bar), fill=(0, 0, 0))
      draw.text((10, 6), text, fill=(255, 255, 255), font=font)
      if since < flash:
        w = 10
        for box in ((0, 0, frame.width, w), (0, frame.height - w, frame.width, frame.height),
                    (0, 0, w, frame.height), (frame.width - w, 0, frame.width, frame.height)):
          draw.rectangle(box, fill=(220, 30, 30))
        draw.text((14, bar + 8), "COLLISION", fill=(220, 30, 30), font=font)
      writer.append_data(np.asarray(frame))

    substeps = getattr(wrapped, "substeps", 1)  # a planning arm's step is one decision
    try:
      with torch.no_grad():
        if substeps > 1:
          wrapped.on_substep = lambda: frame_after_step(bool(terms.dones[0]))
          for _ in range(steps // substeps):
            obs, _, _, _ = wrapped.step(policy(obs))
        else:
          for _ in range(steps):
            obs, _, dones, _ = wrapped.step(policy(obs))
            frame_after_step(bool(dones[0]))
    finally:
      writer.close()
      raw.close()
    seconds = taken * raw.step_dt
    env0 = [t for t, e in collisions.times if e == 0]
    summary = {"video": "play.mp4", "fps": fps, "seconds": seconds, "checkpoint": ckpt.name,
               "end_on_collision": args.end_on_collision, "camera": args.camera,
               "collisions_env0": len(env0), "collision_times_env0_s": [round(t, 2) for t in env0],
               "episode_ends_env0": ends, "collisions_all_envs": len(collisions.times), "num_envs": args.num_envs}
    run.write_json("summary.json", summary)
    iteration = checkpoint_iteration(ckpt)
    if iteration is None:
      iteration = int(torch.load(ckpt, weights_only=False, map_location="cpu").get("iter") or 0)
    scalars = {"collisions_env0": len(env0), "collisions_all_envs": len(collisions.times), "seconds": seconds,
               **{f"episode_ends_env0/{k}": v for k, v in ends.items()}}
    attached = attach_eval(args.run, run.path, iteration=iteration, kind="play", scalars=scalars,
                           video=run.path / "play.mp4")
    run.update(wandb=attached)
  print(json.dumps(summary), flush=True)
  print(f"W&B: {attached}", flush=True)
  return 0
