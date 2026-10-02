"""Backend configs from flat benchmark cases (no checkpoints needed)."""

from __future__ import annotations

from crowdmotion.backends.registry import ArdyConfig, KimodoConfig, config_from_case


def test_case_keys_map_onto_typed_configs():
  ardy = config_from_case({"backend": "ardy", "future": 8, "steps": 5, "humans": 32, "cfg_weight": [1, 2]},
                          precision="tf32")
  assert ardy == ArdyConfig(future_frames=8, steps=5, precision="tf32", cfg_weight=(1, 2))
  kimodo = config_from_case({"backend": "kimodo", "history": 7, "new_frames": 20}, steps=50)
  assert kimodo == KimodoConfig(transition_frames=7, new_frames=20, steps=50)


def test_case_keys_override_caller_defaults():
  cfg = config_from_case({"backend": "ardy", "precision": "fp32"}, precision="tf32", cfg_type="nocfg")
  assert cfg.precision == "fp32" and cfg.cfg_type == "nocfg"
