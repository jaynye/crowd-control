"""Absolute-height annotations, live eligibility and explicit masked dead ends."""

import numpy as np
import pytest
import torch

from crowdmotion.core.geometry import body_local, body_world
from crowdmotion.graph import GraphConfig, GraphRuntime, MotionGraph
from crowdmotion.graph.annotations import annotate_edges
from crowdmotion.metrics.posture import vertical_metrics
from synthetic import synthetic_library


def annotated_graph():
  lib = synthetic_library(4)
  lib["meta"]["joint_names"] = ["Hips", "LeftFoot", "RightFoot", "Head", "Neck"]
  g = MotionGraph(lib, GraphConfig(neighbors=4), device="cpu")
  # Named synthetic anatomy in body coordinates, making known ground heights.
  g.edge_pose[..., 0, :] = torch.tensor([0., .9, 0.])
  g.edge_pose[..., 4, :] = torch.tensor([0., 1.4, .1])
  g.edge_pose[..., 3, :] = torch.tensor([0., 1.7, .15])
  g.edge_pose[..., 1:3, 1] = .02
  g.pred_pose.copy_(g.edge_pose[:, 0])
  return g


def test_height_is_absolute_and_pitch_excludes_root_translation_yaw():
  g = annotated_graph()
  p = g.pred_pose[:2]
  root, heading = torch.tensor([[3., 5.], [-7., 2.]]), torch.tensor([1., -2.])
  world = body_world(p, root, heading)
  assert torch.equal(world[..., 1], p[..., 1])
  assert torch.allclose(body_local(world, root, heading), p, atol=1e-6)
  a = annotate_edges(g)
  assert a.trunk_pitch_deg[0] == pytest.approx(np.degrees(np.arctan(.2)), abs=1e-5)
  assert a.lowest_foot_min_m[0] == pytest.approx(.02)
  # The predecessor counts too, even if every executed frame has constant height.
  g.pred_pose[0, 0, 1] = .6
  assert annotate_edges(g).pelvis_range_m[0] == pytest.approx(.3)


@pytest.mark.parametrize("temperature", [0., .01])
def test_all_masked_holds_finite_pose_without_playing_an_edge(temperature):
  g = annotated_graph()
  runtime = GraphRuntime(g)
  state = runtime.reset(torch.tensor([1, 2]), torch.zeros(2, 2), torch.zeros(2))
  old = {k: getattr(state, k).clone() for k in ("pose", "heading", "root_xz", "boundary")}
  g.cfg.vertical_mask, g.cfg.temperature = True, temperature
  g.cfg.vertical_foot_max_m = 0.  # Deliberate all-masked live setting.
  out = runtime.step(state, torch.ones(2, g.L, 2))
  assert out["vertical_fallback"].all() and (out["edge"] == -1).all()
  for key, value in old.items():
    assert torch.equal(getattr(state, key), value)
  for key in ("joints", "root", "heading", "traj_err_m"):
    assert torch.isfinite(out[key]).all()
  assert (state.counters == 1).all()
  with pytest.raises(ValueError, match="No admissible"):
    runtime.reset(torch.tensor([1]), torch.zeros(1, 2), torch.zeros(1))
  g.cfg.vertical_foot_max_m = .6
  assert not runtime.step(state, torch.ones(2, g.L, 2))["vertical_fallback"].any()


def test_mask_filters_candidates_and_starts_without_changing_tables():
  g = annotated_graph()
  original = g.valid.clone()
  g.edge_pose[0, :, 1:3, 1] = .3
  g.pred_pose[0, 1:3, 1] = .3
  g.cfg.vertical_mask = True
  assert not g.admissible_edges()[0]
  assert torch.equal(g.valid, original)
  assert (g.start_boundaries(torch.arange(100)) != 0).all()
  g.cfg.vertical_mask = False
  assert torch.equal(g.admissible_edges(), original)


def test_executed_vertical_metrics_include_transitions():
  joints = np.zeros((1, 5, 3, 3))
  joints[..., 0, 1] = [.9, .95, 1., 1.1, 1.2]
  joints[..., 1:, 1] = .2
  out = vertical_metrics({"joints": joints, "foot_idx": [1, 2]}, 20., 4)
  assert out["pelvis_range_per_executed_edge_m"]["100"] == pytest.approx(.3)
  assert out["lowest_foot_above018_percent"] == 100


def test_optional_peak_lower_foot_mask_catches_brief_airborne_edges():
  g = annotated_graph()
  g.edge_pose[0, 3:5, 1:3, 1] = .2
  g.cfg.vertical_mask = True
  assert g.admissible_edges()[0]  # The minimum is still grounded.
  g.cfg.vertical_lowest_foot_max_m = .12
  assert not g.admissible_edges()[0]
  assert g.valid[0]
