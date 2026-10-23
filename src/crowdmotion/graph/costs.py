"""Candidate cost terms: how a human scores the outgoing edges at its boundary.

Each term maps a `CandidateContext` to a cost `[H, k]` per candidate. The total
is the weighted sum in `TERMS` order:

  cost = traj + pose_weight * desc_dist + end_vel_weight * end_vel_err

`TrajectoryCost` is unweighted (meters); every other term reads its weight from
the `GraphConfig` field it names in `weight_key`. The runtime samples a candidate
from `-cost / temperature` plus per-human Gumbel noise.

Adding a term (for example a facing or turn-commitment cost) takes two edits:
1. in this module, a class decorated with `@cost_term`, with `name`, `weight_key`
   and `__call__(ctx) -> [H, k]`, and `skip_when_zero = True` so that a zero
   weight neither costs time nor changes any existing result;
2. its weight field (default 0.0) in `GraphConfig`.
Registration order is summation order; new terms register after the existing ones,
which keeps earlier results bit-identical. A term's raw value appears in
`CostTotal.values` under its `name` for diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch

from crowdmotion.core.geometry import to_world, wrap

END_VEL_BACK = 3  # frames for the end-of-edge velocity of edges and requested paths


@dataclass
class CandidateContext:
  """Everything terms may read for H humans and their k candidates."""

  graph: object  # MotionGraph
  state: object  # GraphState
  path: torch.Tensor  # [H, path_frames, 2] requested world path, frames 1.. ahead
  cand: torch.Tensor  # [H, k] candidate edge ids
  ddist: torch.Tensor  # [H, k] descriptor distance of each candidate (m, RMS per joint)
  head: torch.Tensor  # [H, 1, 1] current world heading
  local: torch.Tensor  # [H, k, F, 2] candidate root paths in the current heading frame
  pred: torch.Tensor  # [H, k, F, 2] candidate root paths placed in the world
  weight: torch.Tensor | None = None  # [H, k, F] lookahead frame weights (None: all equal)
  cache: dict = field(default_factory=dict)
  orientation_weight: torch.Tensor | None = None  # [H] priority of soft body-orientation preferences
  turn_target: torch.Tensor | None = None  # [H] retained body target independent of translating path
  turn_active: torch.Tensor | None = None  # [H] ordinary turn-before-walk phase
  # (positions [H, N, 2], velocities [H, N, 2], clearance distance [H]) of each human's nearest
  # neighbors, for the clearance term; pad missing neighbors far away.
  neighbors: tuple | None = None


def candidate_context(graph, state, path: torch.Tensor) -> CandidateContext:
  """Candidates of each human's boundary, with root paths (plus each candidate's own
  successor under `lookahead`) placed on its current root and heading."""
  g, cfg, L = graph, graph.cfg, graph.L
  cand = g.cand[state.boundary]  # [H, k]
  ddist = g.cand_dist[state.boundary]
  head = state.heading[:, None, None]
  local = g.edge_root[cand]  # [H, k, L, 2]
  weight = None
  if cfg.lookahead:
    # Append each candidate's own successor, composed on the candidate's last frame.
    succ = cand + 1
    tail = local[:, :, -1:] + to_world(g.edge_root[succ], g.edge_yaw[cand][:, :, -1:])
    local = torch.cat((local, tail), 2)
    weight = torch.ones(local.shape[:3], device=g.device)
    weight[:, :, L:] = g.valid[succ][..., None].float()
  pred = state.root_xz[:, None, None] + to_world(local, head)
  return CandidateContext(g, state, path, cand, ddist, head, local, pred, weight)


class CostTerm(Protocol):
  name: str
  weight_key: str | None  # GraphConfig field with the weight; None: unweighted
  skip_when_zero: bool

  def __call__(self, ctx: CandidateContext) -> torch.Tensor: ...


TERMS: list[CostTerm] = []  # registration order = summation order


def cost_term(cls):
  """Class decorator: register one instance of `cls` at the end of `TERMS`."""
  TERMS.append(cls())
  return cls


@cost_term
class TrajectoryCost:
  """RMS distance (m) between the candidate's placed root path and the request,
  over one edge, or two with `lookahead` (an invalid successor's frames weigh 0)."""

  name = "trajectory"
  weight_key = None
  skip_when_zero = False

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    pred = ctx.pred
    err2 = (pred - ctx.path[:, None, : pred.shape[2]]).norm(dim=-1).pow(2)
    if ctx.weight is not None:
      return ((err2 * ctx.weight).sum(-1) / ctx.weight.sum(-1)).sqrt()
    return err2.mean(-1).sqrt()  # [H, k]


@cost_term
class PoseDistanceCost:
  """Descriptor distance between the human's boundary and the candidate's start (m)."""

  name = "pose"
  weight_key = "pose_weight"
  skip_when_zero = False

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    return ctx.ddist


@cost_term
class EndVelocityCost:
  """End-of-edge root velocity error against the request's velocity (m/s)."""

  name = "end_vel"
  weight_key = "end_vel_weight"
  skip_when_zero = False

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    g, L, path, back = ctx.graph, ctx.graph.L, ctx.path, END_VEL_BACK
    want_vel = (path[:, L - 1] - path[:, L - 1 - back]) * g.fps / back
    return (to_world(g.edge_end_vel[ctx.cand], ctx.head[..., 0]) - want_vel[:, None]).norm(dim=-1)


@cost_term
class FacingCost:
  """Mean speed minus forward velocity (m/s), measured in the moving body frame.

  Each displacement is compared with the midpoint of its adjacent body headings,
  including the predecessor. A forward curve costs zero; sideways and backward
  travel cost one and two times speed. Slow motion is downweighted continuously.
  This is a soft preference, so path tracking can still select evasive steps.
  """

  name = "facing"
  weight_key = "facing_weight"
  skip_when_zero = True
  orientation_sensitive = True

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    g = ctx.graph
    yaw = g.edge_yaw[ctx.cand]
    if g.cfg.lookahead:
      yaw = torch.cat((yaw, yaw[..., -1:] + g.edge_yaw[ctx.cand + 1]), -1)
    prev = torch.cat((torch.zeros_like(yaw[..., :1]), yaw[..., :-1]), -1)
    midpoint = prev + 0.5 * wrap(yaw - prev)
    vel = torch.diff(ctx.local, dim=2, prepend=torch.zeros_like(ctx.local[:, :, :1])) * g.fps
    forward = vel[..., 0] * torch.sin(midpoint) + vel[..., 1] * torch.cos(midpoint)
    penalty = (vel.norm(dim=-1) - forward).clamp(min=0.0)
    if ctx.weight is not None:
      return (penalty * ctx.weight).sum(-1) / ctx.weight.sum(-1)
    return penalty.mean(-1)


@cost_term
class HeadingCost:
  """Prefer native body turns toward the requested path's terminal velocity.

  Speed-weighted angular error (m/s * radians). Unlike the travel-facing term,
  this rewards reorientation even when several candidates barely translate.
  Requests with no translation have no target direction and contribute zero.
  Scores the current edge's end heading; lookahead does not move the target.
  An optional angular deadband lets trajectory and pose costs choose between
  already aligned clips instead of rewarding every small heading correction.
  """

  name = "heading"
  weight_key = "heading_weight"
  skip_when_zero = True
  orientation_sensitive = True

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    g, L = ctx.graph, ctx.graph.L
    velocity = (ctx.path[:, L - 1] - ctx.path[:, L - 1 - END_VEL_BACK]) * g.fps / END_VEL_BACK
    target = torch.atan2(velocity[:, 0], velocity[:, 1])
    body = ctx.state.heading[:, None] + g.edge_yaw[ctx.cand][..., -1]
    error = wrap(body - target[:, None]).abs()
    if g.cfg.heading_deadband_deg > 0:
      error = (error - math.radians(g.cfg.heading_deadband_deg)).clamp(min=0.0)
    return error * velocity.norm(dim=-1)[:, None]


@cost_term
class PostureCost:
  """Soft pitch preference around each human's retained style; never filters turns.

  Targets lie on either side of the diagnostic cutoff. Twenty degrees of error
  costs one weight unit. This cost is independent of evasive orientation release.
  """

  name = "posture"
  weight_key = "posture_weight"
  skip_when_zero = True

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    g, c = ctx.graph, ctx.graph.cfg
    if not c.posture_enabled or ctx.state.posture_hunched is None:
      return torch.zeros_like(ctx.ddist)
    pitch = g.ensure_annotations().trunk_pitch_deg[ctx.cand]
    target = c.posture_cutoff_deg + torch.where(ctx.state.posture_hunched,
                                              c.posture_target_margin_deg, -c.posture_target_margin_deg)
    return ((pitch - target[:, None]) / 20.).square()


@cost_term
class TurnCost:
  """Native body reorientation toward retained intent, including zero-speed requests."""

  name = "turn"
  weight_key = "turn_weight"
  skip_when_zero = True

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    if ctx.turn_target is None or ctx.turn_active is None:
      return torch.zeros_like(ctx.ddist)
    body = ctx.state.heading[:, None] + ctx.graph.edge_yaw[ctx.cand][..., -1]
    return wrap(body - ctx.turn_target[:, None]).abs() * ctx.turn_active[:, None]


@cost_term
class StartVelocityCost:
  """Prefer candidate entry velocity continuous with the executed previous frame.

  This uses world planar velocity, before optional candidate root correction.
  It is separate from the requested end velocity and remains soft during evasion.
  """

  name = "start_vel"
  weight_key = "start_vel_weight"
  skip_when_zero = True

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    previous = ctx.state.root_velocity
    if previous is None:
      previous = torch.zeros_like(ctx.state.root_xz)
    velocity = to_world(ctx.graph.edge_root[ctx.cand, 0] * ctx.graph.fps, ctx.head[..., 0])
    return (velocity - previous[:, None]).norm(dim=-1)


@cost_term
class ClearanceCost:
  """Meters by which a candidate's root path closes on neighbors inside the clearance.

  Neighbors move on at their executed velocities over the candidate's frames. For each
  neighbor, the cost is how much deeper inside the clearance distance (body contact
  plus tolerance, from the command source) the candidate comes than standing still
  would, summed over neighbors. Moves that keep or open each gap cost nothing, even
  when already too close. Weighted high, it makes a
  command source's stop rule hold at motion selection, where clips that walk on or turn
  forward into a neighbor would otherwise win on other costs. Only humans with a
  neighbor in reach of one edge are evaluated.
  """

  name = "clearance"
  weight_key = "clearance_weight"
  skip_when_zero = True
  reach_m = 2.0  # neighbors further than clearance + this at the boundary are ignored

  def __call__(self, ctx: CandidateContext) -> torch.Tensor:
    out = torch.zeros_like(ctx.ddist)
    if ctx.neighbors is None:
      return out
    xz, vel, limit = ctx.neighbors
    here = ctx.state.root_xz
    rows = torch.nonzero(((xz - here[:, None]).norm(dim=-1).amin(-1) < limit + self.reach_m)).squeeze(1)
    if not len(rows):
      return out
    pred = ctx.pred[rows]  # [n, k, F, 2]
    t = torch.arange(1, pred.shape[2] + 1, device=pred.device, dtype=pred.dtype) / ctx.graph.fps
    future = xz[rows, :, None] + vel[rows, :, None] * t[None, None, :, None]  # [n, N, F, 2]
    gap = (pred[:, :, None] - future[:, None]).norm(dim=-1).amin(-1)  # [n, k, N]
    stay = (here[rows, None, None] - future).norm(dim=-1).amin(-1)  # [n, N]
    lim = limit[rows, None, None]
    deeper = (lim - gap).clamp(min=0.0) - (lim - stay[:, None]).clamp(min=0.0)
    out[rows] = deeper.clamp(min=0.0).sum(-1)  # each neighbor separately
    return out


@dataclass
class CostTotal:
  cost: torch.Tensor  # [H, k]
  values: dict[str, torch.Tensor]  # unweighted term values by name, [H, k]


def total_cost(ctx: CandidateContext, cfg, terms=None) -> CostTotal:
  """Weighted sum of `terms` (default: every registered term), left to right."""
  terms = TERMS if terms is None else terms
  cost = None
  values = {}
  for term in terms:
    weight = None if term.weight_key is None else getattr(cfg, term.weight_key)
    if weight == 0 and term.skip_when_zero:
      continue
    value = term(ctx)
    values[term.name] = value
    part = value if weight is None else weight * value
    if ctx.orientation_weight is not None and getattr(term, "orientation_sensitive", False):
      part = part * ctx.orientation_weight[:, None]
    cost = part if cost is None else cost + part
  return CostTotal(cost, values)
