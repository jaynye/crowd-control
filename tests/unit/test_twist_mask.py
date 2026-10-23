"""Planted twists are an opt-in saved-table mask, including standing edges."""

import math

import pytest
import torch
from test_turn_library import turning_library

from crowdmotion.graph import GraphConfig, GraphRuntime, MotionGraph
from crowdmotion.graph.annotations import annotate_twist
from crowdmotion.graph.turns import foot_yaw_deg


def twist_graph():
  g = MotionGraph(turning_library(humans=4, duration=8.), GraphConfig(neighbors=4), device="cpu")
  g.edge_contacts[:] = True
  set_yaw(g, slice(None), 0.)
  return g


def set_yaw(g, edges, degrees):
  names = g.meta["joint_names"]
  for side in ("Left", "Right"):
    foot, toe = names.index(side + "Foot"), names.index(side + "ToeBase")
    angle = math.radians(degrees)
    g.edge_pose[edges, :, toe] = g.edge_pose[edges, :, foot] + torch.tensor(
      [.12 * math.sin(angle), -.03, .12 * math.cos(angle)])
  g.twist_annotations = None


def test_standing_and_turning_twists_removed_but_stepping_kept():
  g = twist_graph()
  ids = g.valid.nonzero().flatten()[:4]
  set_yaw(g, ids, 60.)
  g.edge_yaw[ids[0]] = 0.  # standing twist must be covered
  g.edge_yaw[ids[1]] = torch.linspace(0., .8, g.L)
  g.edge_contacts[ids[2], 0, 0] = False  # conservative all-labels planted definition
  set_yaw(g, ids[3:4], 15.)
  baked = g.valid.clone()
  assert torch.equal(g.admissible_edges(), baked)  # defaults off
  g.cfg.twist_mask = True
  allowed = g.admissible_edges()
  assert allowed[ids].tolist() == [False, False, True, True]
  assert torch.equal(g.valid, baked)
  assert not torch.isin(g.start_boundaries(torch.arange(100)), ids[:2]).any()
  assert torch.equal(torch.from_numpy(foot_yaw_deg(g)), annotate_twist(g).foot_yaw_deg)
  g.cfg.twist_max_foot_yaw_deg = 65.
  assert torch.equal(g.admissible_edges(), baked)


@pytest.mark.parametrize("temperature", [0., .01])
def test_twist_only_dead_end_holds_and_can_resume(temperature):
  g = twist_graph()
  g.cfg.temperature = temperature
  runtime = GraphRuntime(g)
  state = runtime.reset(torch.arange(2), torch.zeros(2, 2), torch.zeros(2))
  old = {k: getattr(state, k).clone() for k in ("pose", "heading", "root_xz", "boundary")}
  set_yaw(g, slice(None), 60.)
  g.cfg.twist_mask = True
  out = runtime.step(state, torch.ones(2, g.L, 2))
  assert out["vertical_fallback"].all() and (out["edge"] == -1).all()
  for key, value in old.items():
    assert torch.equal(getattr(state, key), value)
  assert torch.isfinite(out["joints"]).all()
  with pytest.raises(ValueError, match="No admissible"):
    runtime.reset(torch.arange(2), torch.zeros(2, 2), torch.zeros(2))
  g.cfg.twist_mask = False
  assert (runtime.step(state, torch.ones(2, g.L, 2))["edge"] >= 0).all()


def test_loading_masks_without_persisting_annotations_or_mutating_tables(tmp_path):
  g = twist_graph()
  set_yaw(g, 0, 60.)
  path = tmp_path / "graph.pt"
  g.save(path)
  original = path.read_bytes()
  loaded = MotionGraph.load(path, device="cpu", cfg_overrides={"twist_mask": True})
  assert loaded.twist_annotations is not None and not loaded.admissible_edges()[0]
  assert torch.equal(loaded.valid, g.valid)
  assert path.read_bytes() == original
  blob = torch.load(path, weights_only=False)
  assert blob["format"] == 1 and "twist_annotations" not in blob
  assert set(blob["tables"]) == set(MotionGraph._TABLES)


def test_bad_threshold_and_missing_anatomy_raise_only_when_enabled():
  g = twist_graph()
  g.cfg.twist_max_foot_yaw_deg = float("nan")
  assert torch.equal(g.admissible_edges(), g.valid)
  g.cfg.twist_mask = True
  with pytest.raises(ValueError, match="twist_max"):
    g.admissible_edges()
  g.cfg.twist_max_foot_yaw_deg = 21.4
  g.meta["joint_names"] = ["unknown"] * g.joints_n
  with pytest.raises(ValueError, match="named joints"):
    g.admissible_edges()


def test_live_twist_controls_build_annotations_and_reject_empty_masks():
  from types import SimpleNamespace

  from test_live_crowd_tuning import Gui

  from crowdmotion.behavior.plaza import PlazaSteeringCfg
  from crowdmotion.sim.mjlab.tuning import LiveCrowdTuning

  g = twist_graph()
  ids = g.valid.nonzero().flatten()[:2]
  set_yaw(g, ids, 60.)
  tuning = LiveCrowdTuning(g, PlazaSteeringCfg())
  tuning.create_gui(SimpleNamespace(gui=Gui()))
  tuning.handles[("graph", "twist_mask")].change(True)
  assert tuning.apply_pending() == 1 and g.cfg.twist_mask
  assert g.twist_annotations is not None and not g.admissible_edges()[ids].any()
  set_yaw(g, slice(None), 15.)  # every edge slightly twisted: admitted at 21.4 deg
  g.ensure_twist_annotations()
  assert torch.equal(g.admissible_edges(), g.valid)
  tuning.handles[("graph", "twist_max_foot_yaw_deg")].change(10.)  # would admit none
  assert tuning.apply_pending() == 0 and g.cfg.twist_max_foot_yaw_deg == 21.4
  assert tuning.handles[("graph", "twist_max_foot_yaw_deg")].value == 21.4
  tuning.handles[("graph", "twist_max_foot_yaw_deg")].change(30.)
  assert tuning.apply_pending() == 1 and torch.equal(g.admissible_edges(), g.valid)
