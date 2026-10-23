"""`crowd eval traversal`: measure trained traversal policies in the crowd.

  crowd eval traversal <run dir> [--checkpoint model_<i>.pt ...] [--last N]
      [--crowd aware|blind|both] [--mode end|continue|both] [--envs 4096] [--seconds 60]
      [--episode-s S] [--humans N] [--behavior <encounter config>] [--seed 0] [--no-paired]
      [--config <eval config>.json] [--no-attach] [--out runs/<UTC>_eval_traversal_<run>]

Every checkpoint plays its mean actions for `--seconds` in `--envs` worlds, once per
crowd and mode, and `crowdmotion.metrics.traversal` measures command following,
collisions, time to contact, encounters, reactions and falls (definitions there):

- crowds: `aware` humans perceive the robot (the training default); `blind` humans are
  all oblivious to it, so only the robot can avoid them;
- modes: `end` ends an episode on a collision, as in training (time to first collision);
  `continue` lets episodes run through collisions (collisions per full episode; states
  after a contact are unfamiliar to the policy, so read those numbers as pessimistic).
  Falls and time-outs (`--episode-s`, default none within the rollout) end episodes in
  both; episodes still running at the end are censored there.

The run's `run.json` fixes the env (graph, rotations, behavior, humans, observation
options), as `crowd play traversal` does; `--humans` and `--behavior` override it. Goals,
spawns, standing flags and resample times are paired by default: every checkpoint, and
every run evaluated with the same seed, gets the same goal chain in the same env and
episode (`CrowdGoalCommandCfg.paired_seed`). In the blind crowd the humans are then
paired too; the aware crowd reacts to each robot. Scales (`TraversalEvalCfg`) come from
`--config`.

Writes a fresh run directory: `metrics.json` (one entry per checkpoint, crowd and mode),
`tables/<iteration>_<crowd>_<mode>_<table>.csv` (legs, encounters, collisions, falls,
fall controls, episodes) and `run.json`. Each entry attaches to the training run's W&B
lineage as `eval/traversal/<crowd>/<mode>/*` at the checkpoint's iteration (`--no-attach`
skips it).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from crowdmotion.core.config import load_json
from crowdmotion.core.rundir import RunDir
from crowdmotion.metrics.traversal import CommandState, StepState, TraversalMetrics, cfg_from_dict
from crowdmotion.sim.mjlab.play import label, run_policy
from crowdmotion.sim.mjlab.train import LINKED, actor_checkpoint
from crowdmotion.tracking import flush_outbox, queue_eval
from crowdmotion.tracking.evals import ledger_key

CROWDS, MODES = ("aware", "blind"), ("end", "continue")
SMALL_NCONMAX = 128  # contacts are pooled across worlds; training's per-world budget overflows with few


def run_checkpoints(run: Path) -> list[Path]:
  """The run's own `model_<i>.pt` by iteration (linked runs skipped; a later log
  directory wins an iteration both hold)."""
  found: dict[int, Path] = {}
  for p in sorted(Path(run).glob("logs/*/*/model_*.pt")):
    m = re.fullmatch(r"model_(\d+)\.pt", p.name)
    if m and not p.parent.name.startswith(LINKED):
      found[int(m.group(1))] = p
  if not found:
    raise FileNotFoundError(f"no logs/<experiment>/*/model_<i>.pt under {run}")
  return [found[i] for i in sorted(found)]


def iteration_of(ckpt: Path) -> int:
  return int(re.fullmatch(r"model_(\d+)\.pt", ckpt.name).group(1))


# The env side -----------------------------------------------------------------------------
def capture(env, env_ids) -> None:
  """Step event: every env's state at the end of the step's physics, before the
  auto-reset, stored as `env._eval_pre` (`metrics.traversal.StepState`)."""
  del env_ids
  from crowdmotion.sim.mjlab.traversal import _clearance, human_joints

  robot, twist = env.scene["robot"], env.command_manager.get_term("twist")
  crowd = env.command_manager.get_term("crowd")
  d, origins = robot.data, env.scene.env_origins[:, :2]
  terms = env.termination_manager
  _, per_human = _clearance(env)(env)
  vel = crowd.vel
  behavior = crowd.behavior
  if crowd.cfg.perceive_entity is None:
    trait = torch.zeros_like(per_human, dtype=torch.long)  # nobody sees the robot
  else:
    trait = torch.where(~behavior.robot_perceives, 0, torch.where(behavior.robot_decides, 2, 1))
  none = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  env._eval_pre = StepState(
    pos=(d.root_link_pos_w[:, :2] - origins).clone(), vel=d.root_link_lin_vel_w[:, :2].clone(),
    heading=d.heading_w.clone(),
    vel_b=torch.cat((d.root_link_lin_vel_b[:, :2], d.root_link_ang_vel_b[:, 2:3]), -1).clone(),
    cmd_w=twist.vel_command_w[:, :2].clone(), cmd_b=twist.vel_command_b.clone(),
    standing=twist.is_standing_env.clone(), clearance=per_human.clone(),
    human_pos=human_joints(crowd)[:, :, 0, :2] - origins[:, None],
    human_vel=torch.stack((vel[..., 0], -vel[..., 1]), -1).clone(), trait=trait.clone(),
    fell=terms.get_term("fell_over").clone(),
    timed_out=terms.get_term("time_out").clone() if "time_out" in terms.active_terms else none,
    collided=terms.get_term("crowd_collision").clone() if "crowd_collision" in terms.active_terms else none,
    done=env.reset_buf.clone(),
    succeeded=terms.get_term("goal_success").clone() if "goal_success" in terms.active_terms else None,
    deadline=terms.get_term("goal_deadline").clone() if "goal_deadline" in terms.active_terms else None,
  )


def command_state(env) -> CommandState:
  twist, robot = env.command_manager.get_term("twist"), env.scene["robot"]
  origins = env.scene.env_origins[:, :2]
  return CommandState(pos=robot.data.root_link_pos_w[:, :2] - origins, goal=twist.goal_w - origins,
                      speed=twist.speed.clone(), standing=twist.is_standing_env.clone(),
                      arrivals=twist.metrics["goals_reached"].clone())


def make_env(env_opts: dict, crowd: str, mode: str, args, clip_actions, device: str):
  """An eval env: the play config of the run's env, with `crowd` and `mode` applied."""
  return build_env(eval_env_cfg(env_opts, crowd, mode, args), env_opts, clip_actions, device)


def eval_env_cfg(env_opts: dict, crowd: str, mode: str, args):
  """The config `make_env` builds (`args`: behavior, humans, seed, envs, seconds,
  episode_s, paired and cfg, as `crowd eval traversal` parses them). `crowd play
  traversal`'s live viewer builds the same config and changes only what it displays."""
  from mjlab.managers.event_manager import EventTermCfg

  from crowdmotion.core.paths import default_graph, default_graph_rotations
  from crowdmotion.sim.mjlab.traversal import collision_termination, make_traversal_env_cfg

  cfg = make_traversal_env_cfg(
    str(env_opts["graph"] or default_graph()), args.behavior or env_opts["behavior"],
    str(env_opts["rotations"] or default_graph_rotations()), args.humans or env_opts["humans"], seed=args.seed,
    play=True, perceive_robot=crowd == "aware", forward_first=env_opts["forward_first"],
    teacher_lidar=env_opts["teacher_lidar"], no_avoidance=env_opts["no_avoidance"],
    planner_rate=env_opts.get("planner_rate") or 0.0,
    navigation=env_opts.get("navigation"),
  )
  cfg.scene.num_envs = args.envs
  if env_opts.get("navigation") is None:
    cfg.episode_length_s = args.episode_s or args.seconds + 1.0
  elif args.episode_s is not None:
    if args.episode_s != env_opts["navigation"]["deadline_s"]:
      raise ValueError("navigation evaluation keeps the recorded task deadline")
  if args.envs < 64:
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, SMALL_NCONMAX)
  if mode == "end":
    cfg.terminations["crowd_collision"] = collision_termination(args.cfg.penetration_m)
  else:
    cfg.terminations.pop("crowd_collision", None)
  if args.paired:
    cfg.commands["twist"].paired_seed = args.seed
  cfg.commands["twist"].debug_vis = False
  cfg.events["eval_capture"] = EventTermCfg(func=capture, mode="step")
  return cfg


def build_env(cfg, env_opts: dict, clip_actions, device: str, planner=None):
  """`(raw env, the runner's env)`: the planning wrapper for a planning arm (`planner`: a
  loaded `SonicPlanner` to reuse, else a fresh one), RSL-RL's wrapper otherwise."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper

  raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  if env_opts.get("planner_rate"):  # a planning arm decides at 1 or 2 Hz over frozen SONIC
    from crowdmotion.sim.mjlab.planning import PlanningVecEnv

    return raw, PlanningVecEnv(raw, env_opts["planner_rate"], planner=planner)
  return raw, RslRlVecEnvWrapper(raw, clip_actions=clip_actions)


def rewind(env, seed: int) -> None:
  """Put the reset counters that key the crowd's and the command's streams back where a
  fresh env has them, so every checkpoint starts the same episodes."""
  crowd, twist = env.command_manager.get_term("crowd"), env.command_manager.get_term("twist")
  crowd.resets.zero_()
  if hasattr(crowd.behavior, "episodes"):
    crowd.behavior.episodes.fill_(-1)
  twist.episodes.zero_()
  torch.manual_seed(seed)


def rollout(env, wrapped, policy, steps: int, cfg, seed: int) -> tuple[dict, dict]:
  twist = env.command_manager.get_term("twist")
  rewind(env, seed)
  obs, _ = wrapped.reset()
  m = TraversalMetrics(env.num_envs, env.command_manager.get_term("crowd").M, env.step_dt, steps,
                       twist.cfg.slow_m, twist.cfg.arrive_m, cfg, env.device, seed=seed)
  m.start(command_state(env))
  flow_order = torch.zeros((), device=env.device)
  human_speed = torch.zeros((), device=env.device)
  measured_steps = 0

  def record() -> None:
    nonlocal flow_order, human_speed, measured_steps
    m.step(env._eval_pre, command_state(env))
    v = env._eval_pre.human_vel
    speeds = v.norm(dim=-1)
    flow_order += (v.sum(1).norm(dim=-1) / speeds.sum(1).clamp(min=1e-6)).mean()
    human_speed += speeds.mean()
    measured_steps += 1

  substeps = getattr(wrapped, "substeps", 1)  # a planning arm's step is one decision
  if substeps > 1:
    wrapped.on_substep = record
  with torch.no_grad():
    for _ in range(steps // substeps):
      obs, _, _, _ = wrapped.step(policy(obs))
      if substeps == 1:
        record()
  if substeps > 1:
    wrapped.on_substep = None
  metrics, tables = m.finish()
  metrics["humans/flow_order"] = float(flow_order) / max(measured_steps, 1)
  metrics["humans/speed_mps"] = float(human_speed) / max(measured_steps, 1)
  nav = getattr(twist.cfg, "navigation", None)
  if nav is not None:
    from crowdmotion.metrics.traversal import EPISODE_ENDS
    ep = tables["episodes"]
    complete = ep["end"] != EPISODE_ENDS.index("eval_end")
    success = ep["end"] == EPISODE_ENDS.index("success")
    penalized = np.where(success, ep["duration"], nav.deadline_s)
    metrics["goal/deadline_penalized_time_s"] = float(penalized[complete].mean()) if complete.any() else None
  return metrics, tables


# Output -------------------------------------------------------------------------------------
def _clean(value):
  """JSON-safe: numpy scalars to Python, NaN and infinities to null."""
  if isinstance(value, dict):
    return {k: _clean(v) for k, v in value.items()}
  if isinstance(value, (np.floating, float)):
    return float(value) if math.isfinite(value) else None
  if isinstance(value, np.integer):
    return int(value)
  return value


def write_tables(folder: Path, stem: str, tables: dict[str, dict[str, np.ndarray]]) -> None:
  folder.mkdir(parents=True, exist_ok=True)
  for name, cols in tables.items():
    names = list(cols)
    data = np.stack([cols[n] for n in names], 1) if names and len(cols[names[0]]) else np.zeros((0, len(names)))
    np.savetxt(folder / f"{stem}_{name}.csv", data, delimiter=",", header=",".join(names), comments="", fmt="%.6g")


HEADLINE = ("command/success", "command/spl", "command/cross_track_mean_m", "command/tracking_score",
            "command/goals_per_moving_min", "collision/per_robot_min", "first_collision/median_s",
            "first_collision/rmst_s", "fall/per_robot_min", "encounter/per_robot_min", "collision/per_encounter",
            "ttc/moving_steps_lt_1s_share", "reaction/latency_median_s")


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="crowd eval traversal", description=__doc__.split("\n\n")[0])
  p.add_argument("run", type=Path, help="a `crowd train traversal` run directory")
  p.add_argument("--checkpoint", type=Path, action="append",
                 help="model_<i>.pt to evaluate: a path or a name in the run (repeatable; default: --last)")
  p.add_argument("--last", type=int, default=1, help="evaluate the run's last N checkpoints (default: 1)")
  p.add_argument("--crowd", choices=(*CROWDS, "both"), default="aware")
  p.add_argument("--mode", choices=(*MODES, "both"), default="both")
  p.add_argument("--envs", type=int, default=4096)
  p.add_argument("--seconds", type=float, default=60.0, help="simulated seconds per rollout")
  p.add_argument("--episode-s", type=float, help="episode time-out (default: none within the rollout)")
  p.add_argument("--humans", type=int, help="humans per env (default: the run's)")
  p.add_argument("--behavior", help="encounter-design behavior config (default: the run's)")
  p.add_argument("--seed", type=int, default=0, help="crowd, goal and env seed")
  p.add_argument("--paired", action=argparse.BooleanOptionalAction, default=True,
                 help="draw goals, spawns and timers from per-env streams (default: on)")
  p.add_argument("--config", type=Path, help="JSON of TraversalEvalCfg scales (unknown keys are rejected)")
  p.add_argument("--attach", action=argparse.BooleanOptionalAction, default=True,
                 help="attach the results to the training run's W&B lineage (default: on)")
  p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  p.add_argument("--threads", type=int, help="torch CPU threads (default 1 on CPU)")
  p.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_eval_traversal_<run>)")
  p.add_argument("--planner-zero", type=float, choices=(1.0, 2.0), metavar="HZ",
                 help="P0: the released planner and frozen SONIC following the goal command at HZ (a zero "
                      "residual), in the run's env; no checkpoint, nothing attached")
  args = p.parse_args(argv)
  if args.device == "cpu" or args.threads:
    torch.set_num_threads(args.threads or 1)
  args.cfg = cfg_from_dict(load_json(args.config) if args.config else None)
  trained, plays_as, env_opts = run_policy(args.run)
  if args.planner_zero:
    if env_opts.get("no_avoidance"):
      p.error("--planner-zero needs a run whose env reads the crowd (not b0)")
    trained, plays_as = "planner_zero", "lidar"
    env_opts = {**env_opts, "planner_rate": args.planner_zero}
    args.attach, args.checkpoint, args.last = False, None, 1
  owned = run_checkpoints(args.run) if not args.planner_zero else [Path("model_0.pt")]
  if args.checkpoint:
    ckpts = [c if c.exists() else owned[-1].parent / c for c in args.checkpoint]
    for c in ckpts:
      if not c.exists():
        p.error(f"no checkpoint {c}")
  else:
    ckpts = owned[-max(args.last, 1):]
  crowds = CROWDS if args.crowd == "both" else (args.crowd,)
  modes = MODES if args.mode == "both" else (args.mode,)

  import mjlab.tasks  # noqa: F401  (populates the registry)
  from mjlab.rl import MjlabOnPolicyRunner

  from crowdmotion.sim.mjlab.traversal import traversal_rl_cfg

  agent = traversal_rl_cfg(teacher=plays_as == "teacher", planner_rate=env_opts.get("planner_rate") or 0.0,
                           navigation=env_opts.get("navigation"))

  protocol = {"run": str(args.run), "trained": trained, "plays_as": plays_as, "env": env_opts,
              "checkpoints": [str(c) for c in ckpts], "crowds": list(crowds), "modes": list(modes),
              "envs": args.envs, "seconds": args.seconds, "episode_s": args.episode_s,
              "humans": args.humans or env_opts["humans"], "behavior": args.behavior or env_opts["behavior"],
              "seed": args.seed, "paired": args.paired, "cfg": asdict(args.cfg)}
  results = []
  name = f"p0_{args.planner_zero:g}hz" if args.planner_zero else label(args.run)
  with RunDir.create(f"eval_traversal_{name}", out=args.out,
                     record={"kind": "eval_traversal", **protocol}) as run:
    for crowd in crowds:
      for mode in modes:
        env, wrapped = make_env(env_opts, crowd, mode, args, agent.clip_actions, args.device)
        steps = round(args.seconds / env.step_dt)
        if steps % getattr(wrapped, "substeps", 1):
          p.error(f"--seconds {args.seconds:g} is not a whole number of {env_opts['planner_rate']:g} Hz decisions")
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
        for ckpt in ckpts:
          if args.planner_zero:
            def policy(obs, n=env.num_envs):
              return torch.zeros(n, 3, device=args.device)
          else:
            runner.load(str(actor_checkpoint(ckpt)), load_cfg={"actor": True}, strict=True,
                        map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
          it = iteration_of(ckpt)
          print(f"{label(args.run)} {ckpt.name}: {crowd} crowd, {mode} mode, {args.envs} envs x {args.seconds:g} s",
                flush=True)
          metrics, tables = rollout(env, wrapped, policy, steps, args.cfg, args.seed)
          metrics = _clean(metrics)
          write_tables(run.path / "tables", f"{it}_{crowd}_{mode}", tables)
          entry = {"checkpoint": ckpt.name, "iteration": it, "crowd": crowd, "mode": mode, "metrics": metrics}
          if args.attach:  # queued now, published in one W&B session at the end
            scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            entry["wandb"] = queue_eval(args.run, run.path, iteration=it, kind=f"traversal/{crowd}/{mode}",
                                        scalars=scalars)
          results.append(entry)
          run.write_json("metrics.json", {"protocol": protocol, "results": results})
          for key in HEADLINE:
            value = metrics.get(key)
            print(f"  {key:34s} {value if value is None else round(value, 4)}", flush=True)
        env.close()
        del env, wrapped, runner
        torch.cuda.empty_cache()
    if args.attach and any(e.get("wandb") == "queued" for e in results):
      flushed = flush_outbox(args.run)  # empty: a live trainer holds the lock and publishes them
      for e in results:
        key = ledger_key(run.path, f"traversal/{e['crowd']}/{e['mode']}", e["iteration"])
        e["wandb"] = flushed.get(key, e["wandb"])
      run.write_json("metrics.json", {"protocol": protocol, "results": results})
    run.update(results=[{k: e[k] for k in ("checkpoint", "iteration", "crowd", "mode", "wandb") if k in e}
                        for e in results])
  print(json.dumps({"out": str(run.path), "rollouts": len(results)}), flush=True)
  return 0
