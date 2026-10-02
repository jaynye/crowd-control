"""One crowd loop for every consumer: command, plan, step, then update kinematics.

Library generation, the quality benchmark, the behavior evaluator and the mjlab
`GraphCrowd` term all drive humans through `CrowdController.advance`, so a rule
added here (or in a command source, planner or motion source) reaches all of them.

Humans are rows `[H]`. With `group = M`, rows are environment-major
(`env * M + i`) and `advance(..., groups=env_ids)` steps only those environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crowdmotion.behavior.kinematics import Kinematics
from crowdmotion.behavior.planner import plan_path
from crowdmotion.behavior.sources import MotionStep


@dataclass
class ControllerStep:
  motion: MotionStep  # everything the source produced this step
  take: int  # frames executed (the first `take` of `motion`)
  path: torch.Tensor  # [h, path_frames, 2] requested root path
  speed: torch.Tensor  # [h] commanded speed
  heading: torch.Tensor  # [h] commanded heading


class CrowdController:
  def __init__(self, commands, source, velocity, planner=plan_path, group: int | None = None):
    if group is not None and not velocity.grouped:
      raise ValueError(f"{type(velocity).__name__} cannot step rows in groups; use BufferVelocity")
    self.commands, self.source, self.velocity, self.planner = commands, source, velocity, planner
    self.group = group
    self.ended = False  # a partial step of a source that cannot resume from it
    self.state = None
    self.kin: Kinematics | None = None
    self.vbuf: dict | None = None

  @property
  def fps(self) -> float:
    return self.source.fps

  def rows(self, groups: torch.Tensor | None) -> torch.Tensor | None:
    if groups is None:
      return None
    return (groups[:, None] * self.group + torch.arange(self.group, device=groups.device)).flatten()

  def allocate(self, humans: int, device) -> None:
    """Empty state for `humans` rows; environments are then reset by groups."""
    self.state = self.source.allocate(humans)
    z2 = torch.zeros(humans, 2, device=device)
    self.kin = Kinematics(z2.clone(), torch.zeros(humans, device=device), z2.clone())
    self.vbuf = self.velocity.reset(z2)

  def reset(self, seeds: torch.Tensor, root_xz: torch.Tensor, heading: torch.Tensor,
            groups: torch.Tensor | None = None):
    """Start humans (all rows, or the rows of `groups`) at `root_xz`, `heading`."""
    sub = self.source.reset(seeds, root_xz, heading)
    vbuf = self.velocity.reset(root_xz)
    vel = vbuf["vel"]
    if groups is None:
      # Owned copies: grouped advances update kinematics in place.
      self.state, self.vbuf = sub, vbuf
      self.kin = Kinematics(root_xz.clone(), heading.clone(), vel)
      return sub
    rows = self.rows(groups)
    self.source.put(self.state, rows, sub)
    for key, value in vbuf.items():
      self.vbuf[key][rows] = value
    self.kin.root_xz[rows], self.kin.heading[rows], self.kin.vel[rows] = root_xz, heading, vel
    return sub

  def advance(self, t, groups: torch.Tensor | None = None, take: int | None = None,
              limit: int | None = None) -> ControllerStep:
    """Command, plan and step the humans of `groups` (all when None) at time `t`.

    take: frames to execute from the step (default: all it produced); limit:
    an extra cap, e.g. frames left in a fixed-length rollout.
    """
    if self.ended:
      raise RuntimeError("the last step was partly executed and this source cannot resume from it")
    rows = self.rows(groups)
    kin = self.kin.rows(rows)
    speed, heading = self.commands.command(t, kin, groups)
    path = self.planner(kin.root_xz, kin.vel, speed, heading, self.source.path_frames, self.source.fps)
    state = self.state if rows is None else self.source.select(self.state, rows)
    motion = self.source.step(state, path)
    n = motion.frames if take is None else min(motion.frames, take)
    if limit is not None:
      n = min(n, limit)
    self.source.commit(state, motion, n)
    self.ended = n < motion.frames and not self.source.resumes_partial
    root_xz = motion.root[..., [0, 2]]
    if rows is None:
      self.velocity.update(self.vbuf, root_xz, n)
      self.kin = Kinematics(root_xz[:, n - 1], motion.heading[:, n - 1], self.vbuf["vel"])
    else:
      self.source.put(self.state, rows, state)
      sub = {k: v[rows] for k, v in self.vbuf.items()}
      self.velocity.update(sub, root_xz, n)
      for key, value in sub.items():
        self.vbuf[key][rows] = value
      self.kin.root_xz[rows] = root_xz[:, n - 1]
      self.kin.heading[rows] = motion.heading[:, n - 1]
      self.kin.vel[rows] = sub["vel"]
    return ControllerStep(motion, n, path, speed, heading)
