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

from dataclasses import dataclass, field
from typing import Protocol

import torch

from crowdmotion.core.geometry import to_world

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
    cost = part if cost is None else cost + part
  return CostTotal(cost, values)
