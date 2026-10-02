"""Motion-graph runtime: randomness, library reproduction, batch independence.

Runs on CPU with a synthetic library; no checkpoints needed.
"""

from __future__ import annotations

import math

import torch

from crowdmotion.core.rng import hash_uniform, mix64, splitmix64
from crowdmotion.graph import GraphConfig, GraphRuntime, GraphState, MotionGraph
from synthetic import L, synthetic_library


def test_mix64_matches_python_splitmix():
  values = [0, 1, 2**40 + 17, 2**63 - 1, 12345678901234567]
  ref = [splitmix64(v) for v in values]
  got = mix64(torch.tensor(values, dtype=torch.long)).tolist()
  assert [g & 0xFFFFFFFFFFFFFFFF for g in got] == ref


def test_hash_uniform_rows_are_independent_of_batch():
  a = hash_uniform(torch.tensor([3, 9, 27]), torch.tensor([0, 5, 2]), 16)
  b = hash_uniform(torch.tensor([27, 3]), torch.tensor([2, 0]), 16)
  assert torch.equal(a[2], b[0]) and torch.equal(a[0], b[1])
  assert ((a >= 0) & (a < 1)).all()


def _state_at(graph: MotionGraph, lib: dict, boundary: torch.Tensor, seeds) -> GraphState:
  s, k = boundary // graph.K, boundary % graph.K + 1
  p = k * graph.L - 1
  return GraphState(
    boundary=boundary.clone(),
    root_xz=lib["root_pos"][s, p][:, [0, 2]].clone(),
    heading=lib["heading"][s, p].clone(),
    pose=graph.pred_pose[boundary].clone(),
    seeds=torch.as_tensor(seeds),
    counters=torch.zeros(len(boundary), dtype=torch.long),
  )


def test_following_own_successor_reproduces_library():
  lib = synthetic_library()
  graph = MotionGraph(lib, GraphConfig(neighbors=4, temperature=0.0), device="cpu")
  runtime = GraphRuntime(graph)
  boundary = torch.tensor([1 * graph.K + 2, 4 * graph.K + 5])
  state = _state_at(graph, lib, boundary, [0, 1])
  s, k = boundary // graph.K, boundary % graph.K + 1
  frames = k[:, None] * L + torch.arange(L)
  path = lib["root_pos"][s[:, None], frames][..., [0, 2]]
  out = runtime.step(state, path)
  assert out["natural"].all()
  assert torch.allclose(out["joints"], lib["posed_joints"][s[:, None], frames], atol=1e-4)
  assert torch.equal(state.boundary, boundary + 1)


def test_step_is_independent_of_batch_composition():
  lib = synthetic_library(1)
  graph = MotionGraph(lib, GraphConfig(neighbors=6, temperature=0.05, correct_yaw_deg=5, correct_speed=0.1), device="cpu")
  runtime = GraphRuntime(graph)
  boundary = torch.tensor([3, graph.K + 4, 3 * graph.K + 1])
  path = torch.zeros(3, L, 2)
  path[..., 1] = torch.arange(1, L + 1) * 0.06
  full = runtime.step(_state_at(graph, lib, boundary, [10, 11, 12]), path + _state_at(graph, lib, boundary, [10, 11, 12]).root_xz[:, None])
  sub_state = _state_at(graph, lib, boundary[[2, 0]], [12, 10])
  sub = runtime.step(sub_state, path[[2, 0]] + sub_state.root_xz[:, None])
  assert torch.equal(full["edge"][[2, 0]], sub["edge"])
  assert torch.allclose(full["joints"][[2, 0]], sub["joints"])


def test_yaw_correction_is_bounded():
  lib = synthetic_library(2)
  graph = MotionGraph(lib, GraphConfig(neighbors=4, temperature=0.0, correct_yaw_deg=5), device="cpu")
  runtime = GraphRuntime(graph)
  boundary = torch.tensor([2, graph.K + 2])
  state = _state_at(graph, lib, boundary, [0, 1])
  # Ask for a path 90 degrees off the current heading.
  side = torch.stack((torch.cos(state.heading), -torch.sin(state.heading)), -1)
  path = state.root_xz[:, None] + side[:, None] * (torch.arange(1, L + 1) * 0.06)[None, :, None]
  out = runtime.step(state, path)
  assert (out["yaw_corr_rad"].abs() <= math.radians(5) + 1e-6).all()


def test_lookahead_and_end_velocity_still_follow_the_library():
  lib = synthetic_library(3)
  cfg = GraphConfig(neighbors=4, temperature=0.0, lookahead=True, end_vel_weight=0.25)
  graph = MotionGraph(lib, cfg, device="cpu")
  runtime = GraphRuntime(graph)
  assert runtime.path_frames == 2 * L
  boundary = torch.tensor([2 * graph.K + 1, 5 * graph.K + 3])
  state = _state_at(graph, lib, boundary, [0, 1])
  s, k = boundary // graph.K, boundary % graph.K + 1
  frames = k[:, None] * L + torch.arange(2 * L)
  path = lib["root_pos"][s[:, None], frames][..., [0, 2]]
  out = runtime.step(state, path)
  assert out["natural"].all()
  assert torch.allclose(out["traj_err_m"], torch.zeros(2), atol=1e-4)
  assert torch.allclose(out["end_vel_err_mps"], torch.zeros(2), atol=1e-3)
  assert torch.allclose(out["joints"], lib["posed_joints"][s[:, None], frames[:, :L]], atol=1e-4)


def test_save_and_load_reproduce_steps(tmp_path):
  lib = synthetic_library(4)
  cfg = GraphConfig(neighbors=4, temperature=0.05, end_vel_weight=0.25, correct_yaw_deg=5, correct_speed=0.1)
  graph = MotionGraph(lib, cfg, device="cpu")
  graph.save(tmp_path / "g.pt", rest_joints=lib["posed_joints"][0, 0])
  loaded = MotionGraph.load(tmp_path / "g.pt", device="cpu")
  assert torch.equal(graph.cand, loaded.cand)
  boundary = torch.tensor([1, graph.K + 3])
  path = torch.zeros(2, L, 2)
  path[..., 1] = torch.arange(1, L + 1) * 0.05
  a = GraphRuntime(graph).step(_state_at(graph, lib, boundary, [5, 6]), path)
  b = GraphRuntime(loaded).step(_state_at(loaded, lib, boundary, [5, 6]), path)
  assert torch.equal(a["edge"], b["edge"])
  assert torch.allclose(a["joints"], b["joints"], atol=5e-3)  # float16 tables
