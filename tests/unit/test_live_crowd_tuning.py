"""Headless GUI callbacks use the same mutable configs that runtime reads."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_posture_style import two_style_graph

from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.behavior.steering import LocalSteering
from crowdmotion.graph import GraphRuntime
from crowdmotion.sim.mjlab.tuning import CONTROLS, LiveCrowdTuning


class Handle:
  def __init__(self, value=None):
    self.value = value

  def on_update(self, callback):
    self.callback = callback

  def change(self, value):
    self.value = value
    self.callback(None)


class Gui:
  def add_folder(self, name):
    return nullcontext()

  def add_number(self, label, *, initial_value, **kwargs):
    return Handle(initial_value)

  add_checkbox = add_number

  def add_markdown(self, text):
    return SimpleNamespace(content=text)


def test_gui_callbacks_queue_edits_and_initial_share_waits_for_reset():
  graph = two_style_graph()
  steering = PlazaSteeringCfg()
  tuning = LiveCrowdTuning(graph, steering)
  tuning.create_gui(SimpleNamespace(gui=Gui()))
  assert len(tuning.handles) == len(CONTROLS)
  runtime = GraphRuntime(graph)
  state = runtime.reset(torch.arange(32), torch.zeros(32, 2), torch.zeros(32))
  latent = state.posture_hunched.clone()
  for owner, field, value in [
    ("graph", "temperature", .05), ("graph", "facing_weight", 1.5),
    ("graph", "posture_initial_hunched", 1.),
    ("graph", "posture_upright_to_hunched_rate", .1),
    ("steering", "forward_steering", True), ("steering", "forward_lead_deg", 20.),
    ("steering", "turn_before_walk", True), ("steering", "redirect_speed_mps", .8),
  ]:
    tuning.handles[(owner, field)].change(value)
  assert graph.cfg.temperature == .01 and not steering.forward_steering
  assert tuning.apply_pending() == 8
  assert graph.cfg.temperature == .05 and steering.forward_steering
  assert steering.redirect_speed_mps == .8
  assert graph.cfg.posture_upright_to_hunched_rate == .1
  assert torch.equal(latent, state.posture_hunched)
  assert runtime.reset(torch.arange(32), torch.zeros(32, 2), torch.zeros(32)).posture_hunched.all()


def test_invalid_pair_rejected_without_partial_mutation():
  tuning = LiveCrowdTuning(two_style_graph(), PlazaSteeringCfg())
  tuning.submit("steering", "redirect_finish_deg", 60.)
  assert tuning.apply_pending() == 0
  assert tuning.steering.redirect_finish_deg == 15.
  assert "Rejected" in tuning.last_result
  for value in [float("nan"), -1., float("inf")]:
    with pytest.raises(ValueError):
      tuning.submit("graph", "temperature", value)
  with pytest.raises(KeyError):
    tuning.submit("graph", "hunch_ratio", .9)


def test_live_mask_toggle_is_finite_and_does_not_modify_saved_validity():
  g = two_style_graph()
  valid = g.valid.clone()
  runtime = GraphRuntime(g)
  state = runtime.reset(torch.arange(4), torch.zeros(4, 2), torch.zeros(4))
  tuning = LiveCrowdTuning(g, PlazaSteeringCfg())
  tuning.submit("graph", "vertical_mask", True)
  assert tuning.apply_pending() == 1
  result = runtime.step(state, torch.zeros(4, g.L, 2))
  assert torch.isfinite(result["joints"]).all()
  assert torch.equal(g.valid, valid)
  tuning.submit("graph", "vertical_mask", False)
  assert tuning.apply_pending() == 1
  assert not g.cfg.vertical_mask


def test_cached_tables_have_independent_runtime_configs(monkeypatch):
  from crowdmotion.sim.mjlab import command
  graph = two_style_graph()
  monkeypatch.setattr(command, "_GRAPHS", {})
  monkeypatch.setattr(command.MotionGraph, "load", lambda *a, **k: graph)
  a = command.load_graph("mock", "cpu")
  b = command.load_graph("mock", "cpu")
  a.cfg.temperature = .123
  assert b.cfg.temperature == .01 and graph.cfg.temperature == .01
  assert a.edge_pose is b.edge_pose


def test_live_temperature_weights_and_steering_are_read_without_recreation():
  class RankCost:
    name, weight_key, skip_when_zero = "test_rank", "facing_weight", False

    def __call__(self, ctx):
      return .02 * torch.arange(ctx.cand.shape[1]).expand_as(ctx.ddist)

  graph = two_style_graph()
  steering = PlazaSteeringCfg(forward_steering=True)
  tuning = LiveCrowdTuning(graph, steering)
  runtime = GraphRuntime(graph, terms=[RankCost()])
  state = runtime.reset(torch.arange(64), torch.zeros(64, 2), torch.zeros(64))
  path = torch.zeros(64, graph.L, 2)
  tuning.submit("graph", "temperature", 0.)
  tuning.submit("graph", "facing_weight", 1.)
  tuning.apply_pending()
  cold = runtime.select(state, path)
  assert not cold.pick.any()
  tuning.submit("graph", "temperature", .2)
  tuning.submit("graph", "facing_weight", 2.)
  tuning.apply_pending()
  warm = runtime.select(state, path)
  assert warm.pick.any()
  assert torch.equal(warm.costs.cost, cold.costs.cost * 2)
  local = LocalSteering(steering, 1, "cpu")
  def command():
    return local.command(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1),
                         torch.tensor([[[1., 0.]]]), torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.bool))
  before = command()[1]
  tuning.submit("steering", "forward_lead_deg", 60.)
  tuning.apply_pending()
  assert torch.allclose(command()[1], before * 2)


def test_demo_config_drives_geometry_and_preserves_cli_population(monkeypatch, tmp_path):
  from crowdmotion.sim.mjlab import demo, plaza

  captured = {}
  def make(graph, **kwargs):
    captured.update(graph=graph, **kwargs)
    return SimpleNamespace(commands={"crowd": SimpleNamespace()})
  monkeypatch.setattr(plaza, "make_plaza_env_cfg", make)
  path = tmp_path / "behavior.json"
  path.write_text(json.dumps({"humans": 16, "seeds": [1], "steering": {"half_x": 12., "forward_steering": True},
                             "graph_overrides": {"facing_weight": 1.5, "vertical_mask": True}}))
  args = SimpleNamespace(config=path, graph="test.pt", humans=4, seed=53, geometry="rigid")
  cfg = demo._make_env_cfg(args, num_envs=2)
  assert captured["humans"] == 4 and captured["seed"] == 53 and captured["num_envs"] == 2
  assert captured["human_geometry"] == "rigid"
  assert captured["steering"].half_x == 12. and captured["steering"].forward_steering
  assert cfg.commands["crowd"].graph_overrides == {"facing_weight": 1.5, "vertical_mask": True}
  path.write_text(json.dumps({"graph_overrides": {"facing_weigth": 1.5}}))
  with pytest.raises(ValueError, match="Unknown"):
    demo._make_env_cfg(args)
