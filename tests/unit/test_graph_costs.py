"""Graph cost terms, config compatibility and the saved-blob loader (CPU, synthetic)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from crowdmotion.core.geometry import to_world
from crowdmotion.graph import GraphConfig, GraphRuntime, GraphState, MotionGraph
from crowdmotion.graph.costs import TERMS, candidate_context, total_cost
from crowdmotion.graph.tables import FORMAT, read_blob
from synthetic import synthetic_library


def _random_state(graph: MotionGraph, n: int, seed: int):
  g = torch.Generator().manual_seed(seed)
  valid = torch.nonzero(graph.valid).squeeze(1)
  boundary = valid[torch.randint(len(valid), (n,), generator=g)]
  state = GraphState(
    boundary=boundary,
    root_xz=torch.randn(n, 2, generator=g),
    heading=torch.rand(n, generator=g) * 6 - 3,
    pose=graph.pred_pose[boundary].clone(),
    seeds=torch.arange(n),
    counters=torch.zeros(n, dtype=torch.long),
  )
  path = state.root_xz[:, None] + torch.cumsum(torch.randn(n, 2 * graph.L, 2, generator=g) * 0.05, 1)
  return state, path


def _reference_cost(graph: MotionGraph, state: GraphState, path: torch.Tensor) -> torch.Tensor:
  """The scoring expression as written before the split into cost terms."""
  g, cfg, L = graph, graph.cfg, graph.L
  cand = g.cand[state.boundary]
  ddist = g.cand_dist[state.boundary]
  head = state.heading[:, None, None]
  local = g.edge_root[cand]
  if cfg.lookahead:
    succ = cand + 1
    tail = local[:, :, -1:] + to_world(g.edge_root[succ], g.edge_yaw[cand][:, :, -1:])
    local = torch.cat((local, tail), 2)
    weight = torch.ones(local.shape[:3])
    weight[:, :, L:] = g.valid[succ][..., None].float()
  pred = state.root_xz[:, None, None] + to_world(local, head)
  err2 = (pred - path[:, None, : pred.shape[2]]).norm(dim=-1).pow(2)
  if cfg.lookahead:
    traj = ((err2 * weight).sum(-1) / weight.sum(-1)).sqrt()
  else:
    traj = err2.mean(-1).sqrt()
  back = 3
  want_vel = (path[:, L - 1] - path[:, L - 1 - back]) * g.fps / back
  end_vel_err = (to_world(g.edge_end_vel[cand], head[..., 0]) - want_vel[:, None]).norm(dim=-1)
  return traj + cfg.pose_weight * ddist + cfg.end_vel_weight * end_vel_err


@pytest.mark.parametrize("lookahead", [False, True])
@pytest.mark.parametrize("weights", [(1.0, 0.0), (0.3, 0.25), (0.0, 2.0)])
def test_terms_sum_to_the_original_total(lookahead, weights):
  cfg = GraphConfig(neighbors=4, pose_weight=weights[0], end_vel_weight=weights[1], lookahead=lookahead)
  graph = MotionGraph(synthetic_library(5), cfg, device="cpu")
  for seed in range(3):
    state, path = _random_state(graph, 7, seed)
    got = total_cost(candidate_context(graph, state, path), cfg).cost
    assert torch.equal(got, _reference_cost(graph, state, path))


@dataclasses.dataclass
class _FacingConfig(GraphConfig):
  facing_weight: float = 0.0


class _FacingCost:
  name = "facing"
  weight_key = "facing_weight"
  skip_when_zero = True

  def __call__(self, ctx):
    return ctx.graph.edge_yaw[ctx.cand][..., -1].abs()


def test_a_new_term_at_zero_weight_changes_nothing():
  base = GraphConfig(neighbors=4, end_vel_weight=0.25)
  graph = MotionGraph(synthetic_library(6), base, device="cpu")
  state, path = _random_state(graph, 5, 0)
  ctx = candidate_context(graph, state, path)
  ref = total_cost(ctx, base)
  terms = (*TERMS, _FacingCost())
  off = total_cost(ctx, _FacingConfig(**dataclasses.asdict(base)), terms)
  assert torch.equal(off.cost, ref.cost) and "facing" not in off.values
  on = total_cost(ctx, _FacingConfig(**{**dataclasses.asdict(base), "facing_weight": 0.5}), terms)
  assert torch.equal(on.cost, ref.cost + 0.5 * on.values["facing"])


def test_runtime_accepts_extra_terms():
  graph = MotionGraph(synthetic_library(6), _FacingConfig(neighbors=4, temperature=0.0), device="cpu")
  state, path = _random_state(graph, 4, 1)
  out = GraphRuntime(graph, terms=(*TERMS, _FacingCost())).step(state, path[:, : graph.L])
  assert out["edge"].shape == (4,)


def test_flat_keys_load_and_unknown_keys_are_ignored():
  cfg = GraphConfig.from_dict({"pose_weight": 0.5, "end_vel_weight": 0.25, "not_a_field": 3})
  assert (cfg.pose_weight, cfg.end_vel_weight) == (0.5, 0.25)


def test_pre_format_blob_loads_and_unknown_format_fails(tmp_path):
  lib = synthetic_library(7)
  graph = MotionGraph(lib, GraphConfig(neighbors=4, end_vel_weight=0.25), device="cpu")
  graph.save(tmp_path / "new.pt", rest_joints=lib["posed_joints"][0, 0])
  blob = torch.load(tmp_path / "new.pt", weights_only=False)
  assert blob["format"] == FORMAT
  del blob["format"]  # graphs saved before the format field existed
  torch.save(blob, tmp_path / "old.pt")
  old = MotionGraph.load(tmp_path / "old.pt", device="cpu", cfg_overrides={"pose_weight": 0.5})
  for name in MotionGraph._TABLES:
    assert torch.equal(getattr(old, name), getattr(MotionGraph.load(tmp_path / "new.pt", device="cpu"), name))
  assert old.cfg.pose_weight == 0.5 and old.cfg.end_vel_weight == 0.25
  assert torch.equal(old.rest_joints, lib["posed_joints"][0, 0].float())
  blob["format"] = FORMAT + 1
  torch.save(blob, tmp_path / "future.pt")
  with pytest.raises(ValueError, match="unsupported format"):
    read_blob(tmp_path / "future.pt")
  with pytest.raises(ValueError, match="fixed when the graph is built"):
    MotionGraph.load(tmp_path / "old.pt", device="cpu", cfg_overrides={"neighbors": 8})
