"""Direction metrics, graph topology and iteration persistence on toy motion only."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

from crowdmotion.graph import GraphConfig, MotionGraph
from crowdmotion.graph.audit import audit_graph
from crowdmotion.iterate import compare as comparison
from crowdmotion.iterate import evaluate as evaluation
from crowdmotion.metrics.behavior import summarize_trace
from crowdmotion.metrics.locomotion import classify, motion_features
from synthetic import synthetic_library


def test_direction_uses_body_facing_and_is_rotation_invariant():
  # Identical +Z travel is forward/side/back for people with different facing.
  root = np.zeros((3, 3, 2))
  root[:, :, 1] = np.arange(3) * 0.05
  heading = np.broadcast_to(np.array([0, np.pi / 2, np.pi])[:, None], (3, 3)).copy()
  speed, angle, _ = motion_features(root, heading, 20)
  assert np.array_equal(classify(speed, angle), [[1, 1], [3, 3], [4, 4]])
  phi = 0.71
  rotated = np.stack(
    (
      root[..., 0] * np.cos(phi) + root[..., 1] * np.sin(phi),
      -root[..., 0] * np.sin(phi) + root[..., 1] * np.cos(phi),
    ),
    -1,
  )
  s2, a2, _ = motion_features(rotated, heading + phi, 20)
  assert np.allclose(angle, a2)
  assert np.allclose(speed, s2)


def test_heading_wrap_and_predecessor_are_handled():
  heading = np.radians([[179, -179]])
  root = np.array([[[0.0, 0.0], [0.0, -0.05]]])
  speed, angle, turn = motion_features(root, heading, 20)
  assert np.allclose(angle, 0)
  assert np.allclose(turn, 40)
  s, _, _ = motion_features(
    np.array([[[0.0, 0.05]]]), np.array([[0.0]]), 20, np.zeros((1, 2)), np.zeros(1)
  )
  assert np.allclose(s, 1)
  assert speed.shape == (
    1,
    1,
  )  # Missing predecessor does not create a fake still frame.


def test_curve_can_be_forward_and_still_masks_direction():
  head = np.radians([[0.0, 10.0, 20.0, 30.0]])
  mid = (head[:, 1:] + head[:, :-1]) / 2
  steps = 0.05 * np.stack((np.sin(mid), np.cos(mid)), -1)
  root = np.concatenate((np.zeros((1, 1, 2)), np.cumsum(steps, 1)), 1)
  speed, angle, turn = motion_features(root, head, 20)
  assert (classify(speed, angle) == 1).all()
  assert np.allclose(turn, 200)
  assert classify(np.array([0.01]), np.array([180.0]))[0] == 0


def test_gap_and_planted_foot_metrics_detect_known_motion():
  root = np.array(
    [[[0.0, 0.0], [0.0, 0.05], [0.0, 0.10]], [[0.5, 0.0], [0.5, 0.05], [0.5, 0.10]]]
  )
  joints = np.zeros((2, 3, 1, 3))
  joints[..., 0] = root[..., 0, None]
  joints[..., 2] = root[..., 1, None]
  trace = {
    "root": root,
    "heading": np.zeros((2, 3)),
    "joints": joints,
    "contacts": np.ones((2, 3, 1), dtype=bool),
    "foot_idx": np.array([0]),
  }
  out = summarize_trace(trace, 20)
  assert out["moving_forward_percent"] == 100
  assert out["min_pair_root_gap_m"] == 0.5
  assert out["root_gap_below_0_6m_frame_percent"] == 100
  assert np.allclose(out["planted_foot_speed_mps_p50_p95"], [1, 1])


def test_audit_transition_counts_leave_edge_end_not_start(tmp_path):
  lib = synthetic_library()
  graph = MotionGraph(lib, GraphConfig(neighbors=3), device="cpu")
  path = tmp_path / "toy.pt"
  graph.save(path, lib["posed_joints"][0, 0])
  blob = torch.load(path, weights_only=False)
  out = audit_graph(blob)
  valid = np.flatnonzero(graph.valid.numpy())
  assert sum(r["count"] for r in out["edge_composition"]) == len(valid)
  assert sum(r["count"] for r in out["frame_composition"]) == len(valid) * graph.L
  assert np.array(out["transition_counts"]).sum() == len(valid) * 3
  reachable = np.unique(graph.cand[torch.from_numpy(valid) + 1].numpy() + 1)
  assert out["reachable_end_boundaries"] == len(reachable)


def test_config_rejects_mistyped_fields_and_duplicate_seeds():
  with pytest.raises(ValueError, match="Unknown graph"):
    evaluation.resolved_config({"graph_overrides": {"temperatur": 0.1}})
  with pytest.raises(ValueError, match="distinct"):
    evaluation.resolved_config({"seeds": [0, 0]})


def test_toy_evaluation_trace_and_paired_comparison(tmp_path, monkeypatch):
  monkeypatch.setattr(
    "crowdmotion.core.rundir.environment", lambda: {"host": "toy-test", "gpu": None}
  )
  lib = synthetic_library()
  graph = MotionGraph(lib, GraphConfig(neighbors=3), device="cpu")
  graph_path = tmp_path / "toy.pt"
  graph.save(graph_path, lib["posed_joints"][0, 0])
  config = tmp_path / "toy.json"
  config.write_text(json.dumps({"humans": 2, "seconds": 0.45, "seeds": [0]}))
  run = tmp_path / "run"
  args = argparse.Namespace(
    config=config, out=run, graph=graph_path, device="cpu", threads=1, standoffs=True
  )
  evaluation.evaluate(args)
  trace = np.load(run / "seed_0.npz")
  assert trace["joints"].shape[:2] == (2, 10)  # Initial pose + 9 executed frames.
  assert trace["edge"].shape == (2, 3)  # A partial final edge is trimmed in playback.
  assert np.isfinite(trace["root"]).all()
  trace.close()
  assert json.loads((run / "run.json").read_text())["status"] == "completed"
  compare = tmp_path / "compare"
  comparison.compare(argparse.Namespace(baseline=run, candidate=run, out=compare))
  rows = json.loads((compare / "comparison.json").read_text())["paired_metrics"]
  assert any(r["metric"] == "standoffs.pair_standoffs.per_human_minute" for r in rows)
  assert all(r["delta"] is None or np.allclose(r["delta"], 0) for r in rows)
  assert (compare / "index.html").exists()
  with pytest.raises(FileExistsError):
    evaluation.evaluate(args)
