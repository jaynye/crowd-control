"""Every config under configs/ passes its loader's key checks; typos are rejected."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crowdmotion.bench import body_geometry
from crowdmotion.bench.quality import QUALITY_CASE_KEYS
from crowdmotion.bench.throughput import THROUGHPUT_CASE_KEYS
from crowdmotion.core.config import check_keys
from crowdmotion.graph import GraphConfig
from crowdmotion.iterate.evaluate import resolved_config
from crowdmotion.library.generate import CONFIG_KEYS

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _cases(config: dict) -> list[dict]:
  return config.get("cases", []) + config.get("ardy", []) + config.get("kimodo", [])


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*/*.json")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shipped_configs_load(path):
  cfg = json.loads(path.read_text())
  kind = path.parent.name
  if kind == "behavior":
    resolved_config(cfg)
  elif kind == "graph":
    assert set(cfg) <= set(GraphConfig.__dataclass_fields__)
  elif kind == "library":
    check_keys(cfg, CONFIG_KEYS, "library config")
  elif path.name.startswith("body_geometry"):
    body_geometry.validate(cfg)
  else:
    keys = QUALITY_CASE_KEYS if "scenarios" in json.dumps(cfg) else THROUGHPUT_CASE_KEYS
    for case in _cases(cfg):
      check_keys(case, keys, path.name)


def test_typos_are_rejected():
  with pytest.raises(ValueError, match="precison"):
    check_keys({"precison": "bf16"}, CONFIG_KEYS, "library config")
