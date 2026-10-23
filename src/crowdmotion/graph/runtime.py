"""`GraphRuntime`: batched stepping of H humans through a `MotionGraph`.

One `step` plays one edge per human toward a requested world root path:

1. `select`: score every candidate edge at the human's boundary with the cost
   terms (`crowdmotion.graph.costs`) and sample one with per-human Gumbel noise
   drawn from `(seed, counter)`, so a human's choice never depends on the batch;
2. `correct`: optionally bend the chosen edge's root path toward the request
   (bounded yaw and speed correction);
3. `inertialize`: decay the offset between the current pose and the edge's
   start state over the edge;
4. diagnostics, and the state update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from crowdmotion.core.geometry import body_world, to_local, to_world, wrap
from crowdmotion.core.rng import hash_uniform
from crowdmotion.graph.costs import TERMS, CostTotal, candidate_context, total_cost
from crowdmotion.graph.posture import reset_posture, switch_posture
from crowdmotion.graph.tables import MotionGraph


@dataclass
class GraphState:
  boundary: torch.Tensor  # [H] library boundary the human is at
  root_xz: torch.Tensor  # [H, 2] world root of the last played frame
  heading: torch.Tensor  # [H] world heading of the last played frame
  pose: torch.Tensor  # [H, J, 3] last played body pose (body_local)
  seeds: torch.Tensor
  counters: torch.Tensor
  posture_hunched: torch.Tensor | None = None
  posture_seeds: torch.Tensor | None = None
  root_velocity: torch.Tensor | None = None  # [H, 2] last executed world displacement * fps


@dataclass
class Selection:
  edge: torch.Tensor  # [H] chosen edge ids
  pick: torch.Tensor  # [H] index of the choice among the candidates
  costs: CostTotal  # total and per-term values for all candidates
  ddist: torch.Tensor  # [H, k] descriptor distances of the candidates
  fallback: torch.Tensor | None = None  # no admissible finite candidate: hold current pose


class GraphRuntime:
  """Batched stepping of H humans through a MotionGraph."""

  name = "graph"

  def __init__(self, graph: MotionGraph, terms=None):
    self.graph = graph
    self.device = graph.device
    self.fps = graph.fps
    self.terms = tuple(TERMS) if terms is None else tuple(terms)  # registered terms, in order

  def skeleton_info(self) -> dict:
    return self.graph.skeleton_info()

  def reset(self, seeds: torch.Tensor, root_xz: torch.Tensor, heading: torch.Tensor) -> GraphState:
    g = self.graph
    seeds = seeds.to(self.device).long()
    hunched, posture_seeds = reset_posture(seeds, g.cfg.posture_initial_hunched)
    b = g.start_boundaries(seeds, posture_hunched=hunched if g.cfg.posture_enabled else None)
    return GraphState(
      boundary=b,
      root_xz=root_xz.to(self.device).float().clone(),
      heading=heading.to(self.device).float().clone(),
      pose=g.pred_pose[b].clone(),
      seeds=seeds,
      counters=torch.zeros_like(seeds),
      posture_hunched=hunched,
      posture_seeds=posture_seeds,
      root_velocity=torch.zeros_like(root_xz, device=self.device, dtype=torch.float32),
    )

  @property
  def path_frames(self) -> int:
    """Frames of requested path `step` expects: one edge, or two with lookahead."""
    return self.graph.L * (2 if self.graph.cfg.lookahead else 1)

  # Stages -------------------------------------------------------------------------
  def select(self, state: GraphState, path: torch.Tensor,
             orientation_weight: torch.Tensor | None = None,
             turn_target: torch.Tensor | None = None, turn_active: torch.Tensor | None = None,
             neighbors: tuple | None = None) -> Selection:
    """Score each human's candidates and sample one (greedy at temperature 0)."""
    cfg = self.graph.cfg
    ctx = candidate_context(self.graph, state, path)
    ctx.orientation_weight = orientation_weight
    ctx.turn_target, ctx.turn_active = turn_target, turn_active
    ctx.neighbors = neighbors
    costs = total_cost(ctx, cfg, self.terms)
    if cfg.temperature > 0:
      u = hash_uniform(state.seeds, state.counters, ctx.cand.shape[1]).clamp(1e-12, 1 - 1e-7)
      score = -costs.cost / cfg.temperature - torch.log(-torch.log(u))
    else:
      score = -costs.cost
    fallback = None
    if cfg.vertical_mask or cfg.twist_mask:
      allowed = self.graph.admissible_edges()[ctx.cand] & torch.isfinite(score)
      fallback = ~allowed.any(1)
      score = score.masked_fill(~allowed, -torch.inf)
    pick = score.argmax(1)
    rows = torch.arange(len(pick), device=self.device)
    return Selection(ctx.cand[rows, pick], pick, costs, ctx.ddist, fallback)

  def correct(self, state: GraphState, path: torch.Tensor, e: torch.Tensor,
              turn_active: torch.Tensor | None = None):
    """World root path `[H, L, 2]` and yaw `[H, L]` of edges `e` placed on each human,
    bent toward the request when correction is on. Also returns the yaw correction
    `phi [H]` and speed scale `sigma [H]` that were applied."""
    g, cfg, L = self.graph, self.graph.cfg, self.graph.L
    H = e.shape[0]
    root_l = g.edge_root[e]  # [H, L, 2] relative to the predecessor frame
    yaw_l = g.edge_yaw[e]
    frac = torch.arange(1, L + 1, device=self.device, dtype=torch.float32) / L
    phi = torch.zeros(H, device=self.device)
    sigma = torch.ones(H, device=self.device)
    if cfg.correct_yaw_deg > 0 or cfg.correct_speed > 0:
      want = to_local(path[:, L - 1] - state.root_xz, state.heading)
      have = root_l[:, -1]
      moving = (want.norm(dim=-1) > 0.05) & (have.norm(dim=-1) > 0.05)
      if cfg.correct_yaw_deg > 0:
        ang = wrap(torch.atan2(want[:, 0], want[:, 1]) - torch.atan2(have[:, 0], have[:, 1]))
        lim = math.radians(cfg.correct_yaw_deg)
        phi = torch.where(moving, ang.clamp(-lim, lim), phi)
        if cfg.turn_preserve_yaw and turn_active is not None:
          phi = phi.masked_fill(turn_active, 0.)
      if cfg.correct_speed > 0:
        ratio = want.norm(dim=-1) / have.norm(dim=-1).clamp(min=1e-6)
        ratio = ratio.clamp(1 - cfg.correct_speed, 1 + cfg.correct_speed)
        sigma = torch.where(moving, ratio, sigma)
      step = torch.diff(root_l, dim=1, prepend=torch.zeros_like(root_l[:, :1]))
      bend = phi[:, None] * frac[None]
      root_l = torch.cumsum(to_world(step, bend) * sigma[:, None, None], 1)  # rotate by +bend
      yaw_l = yaw_l + bend
    yaw_w = state.heading[:, None] + yaw_l
    root_w = state.root_xz[:, None] + to_world(root_l, state.heading[:, None])
    return root_w, yaw_w, phi, sigma

  def inertialize(self, state: GraphState, e: torch.Tensor):
    """Body-local poses `[H, L, J, 3]` of edges `e`, plus the decaying offset from each
    human's current pose when inertialization is on; also returns that offset."""
    g, cfg, L = self.graph, self.graph.cfg, self.graph.L
    pose = g.edge_pose[e]  # [H, L, J, 3]
    offset = state.pose - g.pred_pose[e]
    if cfg.inertialize:
      x = torch.arange(1, L + 1, device=self.device, dtype=torch.float32) / (L + 1)
      decay = 1 - x * x * (3 - 2 * x)
      pose = pose + offset[:, None] * decay[None, :, None, None]
    return pose, offset

  @torch.no_grad()
  def step(self, state: GraphState, path: torch.Tensor,
           *, orientation_weight: torch.Tensor | None = None,
           turn_target: torch.Tensor | None = None, turn_active: torch.Tensor | None = None,
           neighbors: tuple | None = None) -> dict:
    """Play one edge per human toward world path `[H, path_frames, 2]` (frames 1.. ahead).

    Returns world `joints [H, L, J, 3]`, `root [H, L, 3]`, `heading [H, L]`,
    `contacts [H, L, C]` and per-human diagnostics; updates `state` in place.
    """
    g = self.graph
    if g.cfg.posture_enabled:
      if state.posture_hunched is None:
        state.posture_hunched, state.posture_seeds = reset_posture(state.seeds, g.cfg.posture_initial_hunched)
      state.posture_hunched = switch_posture(state.posture_hunched, state.posture_seeds,
                                            state.counters, g.cfg, g.L / g.fps)
    sel = self.select(state, path, orientation_weight, turn_target, turn_active, neighbors)
    e = sel.edge
    root_w, yaw_w, phi, sigma = self.correct(state, path, e, turn_active)
    pose, offset = self.inertialize(state, e)
    if sel.fallback is not None:
      # Never turn an all-masked row into an arbitrary unsafe edge. Hold in place
      # until a reset or live mask change restores candidates; report every hold.
      hold = sel.fallback
      root_w = torch.where(hold[:, None, None], state.root_xz[:, None], root_w)
      yaw_w = torch.where(hold[:, None], state.heading[:, None], yaw_w)
      pose = torch.where(hold[:, None, None, None], state.pose[:, None], pose)
    joints = body_world(pose, root_w, yaw_w)
    root3 = torch.stack((root_w[..., 0], pose[:, :, 0, 1], root_w[..., 1]), -1)

    rows = torch.arange(e.shape[0], device=self.device)
    pick = sel.pick
    traj = sel.costs.values["trajectory"]
    end_vel_err = sel.costs.values["end_vel"]
    natural = e == state.boundary
    vel_jump = (g.pred_vel[state.boundary] - g.pred_vel[e]).norm(dim=-1).mean(-1)
    out = {
      "joints": joints,
      "root": root3,
      "heading": yaw_w,
      "contacts": g.edge_contacts[e],
      "edge": e,
      "traj_err_m": traj[rows, pick],
      "best_traj_err_m": traj.min(1).values,
      "end_vel_err_mps": end_vel_err[rows, pick],
      "best_end_vel_err_mps": end_vel_err.min(1).values,
      "desc_dist_m": sel.ddist[rows, pick],
      "pose_offset_m": offset.norm(dim=-1).mean(-1),
      "vel_jump_mps": torch.where(natural, torch.zeros_like(vel_jump), vel_jump),
      "natural": natural,
      "yaw_corr_rad": phi,
      "speed_scale": sigma,
    }
    if g.cfg.posture_enabled:
      out["posture_hunched"] = state.posture_hunched.clone()
      out["boundary"] = state.boundary.clone()
    if sel.fallback is not None:
      hold = sel.fallback
      out["vertical_fallback"] = hold
      out["boundary"] = state.boundary.clone()
      out["edge"] = e.masked_fill(hold, -1)  # -1 explicitly means no played library edge
      out["natural"] = natural & ~hold
      out["contacts"] = torch.where(hold[:, None, None],
        (state.pose[:, self.graph.foot_idx, 1] < .05)[:, None], out["contacts"])
      for key in ("desc_dist_m", "pose_offset_m", "vel_jump_mps", "yaw_corr_rad"):
        out[key] = out[key].masked_fill(hold, 0.)
      out["speed_scale"] = torch.where(hold, torch.ones_like(sigma), sigma)
      held_error = (path[:, :g.L] - state.root_xz[:, None]).norm(dim=-1).square().mean(1).sqrt()
      out["traj_err_m"] = torch.where(hold, held_error, out["traj_err_m"])
    state.boundary = e + 1
    if sel.fallback is not None:
      state.boundary = torch.where(sel.fallback, out["boundary"], state.boundary)
    previous_root = root_w[:, -2] if g.L > 1 else state.root_xz
    state.root_velocity = (root_w[:, -1] - previous_root) * g.fps
    state.root_xz = root_w[:, -1]
    state.heading = yaw_w[:, -1]
    state.pose = pose[:, -1]
    state.counters = state.counters + 1
    return out
