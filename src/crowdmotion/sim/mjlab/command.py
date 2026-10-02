"""`GraphCrowd`: an mjlab command term whose humans are driven by a saved motion graph.

Each environment step it advances every environment's clock and, when an edge
(0.4 s) runs out, asks the `CrowdController` for the next edge: plaza intent and
local steering give a command, the planner a root path, and the graph motion
source the frames. The term then interpolates those frames to the current time,
converts them to MuJoCo Z-up, and writes them as capsule mocap poses.

Because the term lives inside the environment, any mjlab viewer (Viser, native,
offscreen `render()`) shows the crowd without extra plumbing. The graph comes
from `crowd graph build` (`MotionGraph.save`), so it loads on CPU without the
library or the kNN build.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from crowdmotion.behavior.controller import CrowdController
from crowdmotion.behavior.kinematics import BufferVelocity
from crowdmotion.behavior.plaza import PlazaBehavior, PlazaSteeringCfg
from crowdmotion.behavior.sources import GraphMotionSource
from crowdmotion.core.geometry import motion_to_mujoco
from crowdmotion.core.rng import human_seeds
from crowdmotion.graph import GraphRuntime, MotionGraph
from crowdmotion.sim.capsules import CapsuleSkeleton
from crowdmotion.sim.scene import COLORS

_GRAPHS: dict = {}


def load_graph(path: str, device, overrides: dict | None = None) -> MotionGraph:
  """Saved graph on `device`, cached per (path, device, overrides) within a process."""
  key = (str(path), str(device), tuple(sorted((overrides or {}).items())))
  if key not in _GRAPHS:
    _GRAPHS[key] = MotionGraph.load(path, device=device, cfg_overrides=overrides)
  return _GRAPHS[key]


@dataclass(kw_only=True)
class GraphCrowdCfg(CommandTermCfg):
  graph_path: str
  humans: int
  entity_name: str = "crowd"
  steering: PlazaSteeringCfg = field(default_factory=PlazaSteeringCfg)
  seed: int = 0
  graph_overrides: dict = field(default_factory=dict)
  resampling_time_range: tuple[float, float] = (1e9, 1e9)  # humans persist until reset

  def build(self, env) -> "GraphCrowd":
    return GraphCrowd(self, env)


class GraphCrowd(CommandTerm):
  cfg: GraphCrowdCfg

  def __init__(self, cfg: GraphCrowdCfg, env):
    super().__init__(cfg, env)
    self.graph = g = load_graph(cfg.graph_path, self.device, cfg.graph_overrides)
    self.skeleton = CapsuleSkeleton.from_rest(g.meta, g.rest_joints, self.device)
    E, M, L, J = self.num_envs, cfg.humans, g.L, g.joints_n
    self.M, self.L, self.fps, self.edge_s = M, L, g.fps, L / g.fps
    model = env.sim.mj_model
    ids = []
    for i in range(M):
      for bone in self.skeleton.names:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{cfg.entity_name}/h{i:02d}_{bone}")
        if body < 0:
          raise ValueError(f"missing capsule body {cfg.entity_name}/h{i:02d}_{bone}")
        ids.append(int(model.body_mocapid[body]))
    self.mocap = torch.tensor(ids, device=self.device)
    self.behavior = PlazaBehavior(E, M, cfg.steering, cfg.seed, self.device)
    self.source = GraphMotionSource(GraphRuntime(g))
    self.controller = CrowdController(self.behavior, self.source, BufferVelocity(g.fps), group=M)
    self.controller.allocate(E * M, self.device)
    dev = self.device
    self.frames = torch.zeros(E, M, L + 1, J, 3, device=dev)  # frame 0 = start of the edge
    self.edge_t = torch.zeros(E, device=dev)  # seconds into the current edge
    self.clock = torch.zeros(E, device=dev)  # seconds since reset
    self.resets = torch.zeros(E, dtype=torch.long, device=dev)
    self._command = torch.zeros(E, M, 2, device=dev)  # (speed, heading)
    self.metrics["natural_successor"] = torch.zeros(E, device=dev)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def state(self):
    """Graph state of every human, rows `env * M + i`."""
    return self.controller.state

  @property
  def vel(self) -> torch.Tensor:
    """`[E, M, 2]` root velocity estimate used for planning and steering."""
    return self.controller.kin.vel.view(self.num_envs, self.M, 2)

  # Reset ---------------------------------------------------------------------
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    env_ids = env_ids.to(self.device)
    e, M = len(env_ids), self.M
    xz, heading = self.behavior.reset(env_ids)
    rows = self.controller.rows(env_ids)
    seeds = human_seeds(self.cfg.seed, rows, "graphcrowd", episode=self.resets[env_ids].repeat_interleave(M))
    sub = self.controller.reset(seeds, xz.reshape(-1, 2), heading.reshape(-1), groups=env_ids)
    self.frames[env_ids] = self.source.current_joints(sub).view(e, M, -1, 3)[:, :, None]
    self.clock[env_ids] = 0.0
    self.resets[env_ids] += 1
    self._advance(env_ids)
    self.edge_t[env_ids] = 0.0

  # Stepping ------------------------------------------------------------------
  def _advance(self, env_ids: torch.Tensor) -> None:
    """Play the next edge for every human in `env_ids`."""
    e, M, L = len(env_ids), self.M, self.L
    step = self.controller.advance(self.clock[env_ids], groups=env_ids)
    self.frames[env_ids, :, 0] = self.frames[env_ids, :, -1]
    self.frames[env_ids, :, 1:] = step.motion.joints.view(e, M, L, -1, 3)
    self._command[env_ids] = torch.stack((step.speed.view(e, M), step.heading.view(e, M)), -1)
    self.metrics["natural_successor"][env_ids] = step.motion.extras["natural"].view(e, M).float().mean(1)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
      dt = self._env.step_dt
      self.clock += dt
      self.edge_t += dt
    due = env_ids[self.edge_t[env_ids] >= self.edge_s - 1e-6]
    while len(due):
      self._advance(due)
      self.edge_t[due] -= self.edge_s
      due = due[self.edge_t[due] >= self.edge_s - 1e-6]
    # Commands run after the env's forward(), so sensors and viewers see these
    # poses one step (20 ms) late. That lag is invisible here, and skipping an
    # extra forward() matters on CPU, where each call costs tens of ms.
    self._write_poses(env_ids)

  def _write_poses(self, env_ids: torch.Tensor) -> None:
    f = (self.edge_t[env_ids] * self.fps).clamp(0, self.L)
    lo = f.floor().long().clamp(max=self.L - 1)
    a = (f - lo)[:, None, None, None]
    rows = torch.arange(len(env_ids), device=self.device)
    frames = self.frames[env_ids]
    joints = frames[rows, :, lo] * (1 - a) + frames[rows, :, lo + 1] * a  # [e, M, J, 3]
    pos, quat = self.skeleton.poses(motion_to_mujoco(joints.flatten(0, 1)))
    e = len(env_ids)
    data = self._env.sim.data
    data.mocap_pos[env_ids[:, None], self.mocap[None]] = pos.view(e, -1, 3)
    data.mocap_quat[env_ids[:, None], self.mocap[None]] = quat.view(e, -1, 4)

  def _update_metrics(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer) -> None:
    """Goal markers and commanded velocity arrows (MuJoCo frame)."""
    M = self.M
    for env in visualizer.get_env_indices(self.num_envs):
      root = self.state.root_xz.view(-1, M, 2)[env].cpu()
      goal = self.behavior.goal[env].cpu()
      cmd = self._command[env].cpu()
      for i in range(M):
        color = COLORS[i % len(COLORS)]
        gx, gy = float(goal[i, 0]), -float(goal[i, 1])
        visualizer.add_cylinder(
          start=np.array([gx, gy, 0.0]), end=np.array([gx, gy, 0.02]), radius=0.12,
          color=(color[0], color[1], color[2], 0.6),
        )
        speed, head = float(cmd[i, 0]), float(cmd[i, 1])
        if speed > 0.05:
          x, y = float(root[i, 0]), -float(root[i, 1])
          dx, dy = speed * math.sin(head), -speed * math.cos(head)
          visualizer.add_arrow(
            start=np.array([x, y, 2.0]), end=np.array([x + dx, y + dy, 2.0]), color=color, width=0.02
          )
