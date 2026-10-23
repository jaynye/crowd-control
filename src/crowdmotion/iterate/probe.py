"""Paired saved-graph turn-response probes through the shared controller and steering.

Each human is in its own empty plaza: this isolates reaction to a fixed 90/180-degree
request, including replan wait, steering, graph selection and execution. It does not
measure crowd spacing or the stochastic intent schedule. No motion prior is loaded.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

from crowdmotion.behavior.commands import CHANGE_T, ScenarioCommands
from crowdmotion.behavior.controller import CrowdController
from crowdmotion.behavior.kinematics import BufferVelocity
from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.behavior.sources import GraphMotionSource
from crowdmotion.behavior.steering import LocalSteering
from crowdmotion.core.geometry import wrap
from crowdmotion.core.paths import default_graph
from crowdmotion.core.provenance import file_sha256
from crowdmotion.core.rng import human_seeds
from crowdmotion.core.rundir import RunDir
from crowdmotion.graph import GraphRuntime, MotionGraph
from crowdmotion.iterate.evaluate import resolved_config
from crowdmotion.metrics.response import response_time, smoothed_velocity


class SteeredScenario:
  """A controlled intent request passed to exactly the plaza's local steering."""

  def __init__(self, commands, cfg):
    self.commands = commands
    self.cfg = dataclasses.replace(cfg, half_x=100., half_z=100., obstacles=())
    self.steering = LocalSteering(self.cfg, 1, "cpu")

  def command(self, t, kin, groups=None):
    speed, heading = self.commands(t)
    goal = torch.stack((speed * heading.sin(), speed * heading.cos()), -1)
    out_speed, out_heading = self.steering.command(
      kin.root_xz[:, None], kin.vel[:, None], kin.heading[:, None], goal[:, None],
      speed[:, None], (speed == 0)[:, None],
    )
    return out_speed[:, 0], out_heading[:, 0]

  def motion_options(self):
    return self.steering.motion_options()


def probe(graph, cfg, seed, scenario):
  fps, L, humans = graph.fps, graph.L, cfg["humans"]
  commands, xz = ScenarioCommands.draw(scenario, humans, seed, L / fps, True, "cpu")
  steering = SteeredScenario(commands, PlazaSteeringCfg(**cfg["steering"]))
  source = GraphMotionSource(GraphRuntime(graph))
  ctrl = CrowdController(steering, source, BufferVelocity(fps))
  ctrl.reset(human_seeds(seed, torch.arange(humans), "evaluator"), xz, commands.h0)
  roots, headings = [xz[:, None]], [commands.h0[:, None]]
  frames = round(cfg["seconds"] * fps)
  for frame in range(0, frames, L):
    step = ctrl.advance(frame / fps, limit=frames - frame)
    roots.append(step.motion.root[:, :step.take, [0, 2]])
    headings.append(step.motion.heading[:, :step.take])
  root, heading = torch.cat(roots, 1), torch.cat(headings, 1)
  velocity = smoothed_velocity(root, fps)
  speed = velocity.norm(dim=-1)
  target = commands.h0 + (math.pi / 2 if scenario == "turn90" else math.pi)
  onset = CHANGE_T + commands.offset
  start = torch.ceil(onset * fps).long()
  measures = {
    "root": (wrap(torch.atan2(velocity[..., 0], velocity[..., 1]) - target[:, None]).abs() < math.radians(20)) & (speed > .5),
    "body": wrap(heading - target[:, None]).abs() < math.radians(20),
  }
  summary = {"seed": seed, "scenario": scenario}
  arrays = {"root": root.numpy(), "heading": heading.numpy(), "onset_s": onset.numpy()}
  for name, ok in measures.items():
    delays = response_time(ok, start, fps) + start / fps - onset
    finite = delays[torch.isfinite(delays)]
    summary[name] = {"reached": len(finite), "humans": humans, "not_reached": humans - len(finite),
                     "delay_s_p50_p95": np.percentile(finite.numpy(), [50, 95]).tolist() if len(finite) else None}
    arrays[f"{name}_response_s"] = delays.numpy()
  summary["replan_wait_s_p50_p95"] = np.percentile(((torch.ceil(onset * fps / L) * L / fps) - onset).numpy(), [50, 95]).tolist()
  return summary, arrays


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--config", type=Path, action="append", required=True, help="repeat for paired settings")
  p.add_argument("--graph", type=Path)
  p.add_argument("--out", type=Path)
  args = p.parse_args(argv)
  configs = [resolved_config(json.loads(path.read_text())) for path in args.config]
  for cfg in configs:
    if cfg["seconds"] <= CHANGE_T + 1:
      raise ValueError("Turn response needs more than five seconds")
    if any(cfg[k] != configs[0][k] for k in ("humans", "seconds", "seeds")):
      raise ValueError("Probe comparisons require matching humans, duration and seed bank")
  torch.set_num_threads(1)
  graph_path = args.graph or default_graph()
  record = {"configs": configs, "config_paths": [str(p.resolve()) for p in args.config],
            "graph_path": str(graph_path.resolve()), "graph_sha256": file_sha256(graph_path),
            "scope": __doc__, "definition": "Within 20 deg for .25 s; root speed >.5 m/s. Delays include replan wait. "
                                             "Nonresponses are censored at the rollout end. These are diagnostic cutoffs, not pass limits."}
  with RunDir.create("behavior_probe", out=args.out, record=record) as run:
    results = []
    for i, cfg in enumerate(configs):
      graph = MotionGraph.load(graph_path, device="cpu", cfg_overrides=cfg["graph_overrides"])
      for seed in cfg["seeds"]:
        for scenario in ("turn90", "turn180"):
          summary, arrays = probe(graph, cfg, seed, scenario)
          summary["config_index"] = i
          np.savez_compressed(run.path / f"config_{i}_seed_{seed}_{scenario}.npz", **arrays)
          results.append(summary)
    run.write_json("summary.json", {"results": results, "definition": record["definition"]})
  print(run.path)


if __name__ == "__main__":
  main()
