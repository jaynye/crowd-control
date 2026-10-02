"""core: seed schemes, strict configs, traces, run directories."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from crowdmotion.core.config import from_dict
from crowdmotion.core.rng import human_seeds, per_human_noise, row_generator_seed, stream_seed
from crowdmotion.core.rundir import RunDir
from crowdmotion.core.trace import Trace


def test_seed_schemes_match_the_historical_formulas():
  ids = torch.arange(5)
  assert torch.equal(human_seeds(3, ids, "quality"), ids + 30_000)
  assert torch.equal(human_seeds(2, ids, "stream", offset=7), ids + 2_000_007)
  assert torch.equal(human_seeds(4, ids, "evaluator"), ids + 4 * 1_000_003)
  episode = torch.tensor([0, 0, 1, 1, 2])
  assert torch.equal(human_seeds(1, ids, "graphcrowd", episode=episode), 1_000_003 + episode * 1_000_000_007 + ids)
  with pytest.raises(ValueError):
    human_seeds(0, ids, "nope")
  assert row_generator_seed(1, 2) == 1_000_005


def test_per_human_noise_rows_ignore_the_batch():
  seeds, counters = torch.tensor([9, 4, 7]), torch.tensor([0, 2, 1])
  full = per_human_noise(seeds, counters, (3,), "cpu")
  solo = per_human_noise(seeds[1:2], counters[1:2], (3,), "cpu")
  assert torch.equal(full[1:2], solo)
  assert stream_seed(9, 0) != stream_seed(9, 1)


@dataclass
class _Cfg:
  a: float = 1.0
  pair: tuple[float, float] = (0.0, 1.0)
  bands: tuple[tuple[float, float], ...] = ((0.1, 0.2),)


def test_from_dict_is_strict_and_tuples_lists():
  cfg = from_dict(_Cfg, {"pair": [2, 3], "bands": [[1, 2], [3, 4]]})
  assert cfg.pair == (2, 3) and cfg.bands == ((1, 2), (3, 4))
  with pytest.raises(ValueError, match="Unknown thing fields"):
    from_dict(_Cfg, {"aa": 1}, what="thing")
  assert from_dict(_Cfg, {"aa": 1}, strict=False) == _Cfg()


def test_trace_round_trip_and_quality_clip_loading(tmp_path):
  rng = np.random.default_rng(0)
  trace = Trace(rng.normal(size=(2, 5, 3, 3)), rng.normal(size=(2, 5, 2)), rng.normal(size=(2, 5)), 20.0,
                rng.random((2, 5, 4)) > 0.5, np.array([1, 2]), {"edge": np.arange(4).reshape(2, 2)})
  trace.save(tmp_path / "seed_0.npz")
  back = Trace.load(tmp_path / "seed_0.npz", fps=20.0)
  for key, value in trace.arrays().items():
    assert np.array_equal(back.arrays()[key], value)
  np.savez(tmp_path / "clips.npz", joints=trace.joints, contacts=trace.contacts, fps=20.0, foot_idx=np.array([1]))
  clip = Trace.load(tmp_path / "clips.npz")
  assert clip.root.shape == (2, 5, 2) and "derived" in clip.extras


def test_rundir_records_completion_and_failure(tmp_path, monkeypatch):
  monkeypatch.setattr("crowdmotion.core.rundir.environment", lambda: {"host": "test"})
  with RunDir.create("x", out=tmp_path / "ok", record={"config": {"a": 1}}) as run:
    run.write_json("out.json", {"b": 2})
  assert json.loads((tmp_path / "ok" / "run.json").read_text())["status"] == "completed"
  with pytest.raises(RuntimeError), RunDir.create("x", out=tmp_path / "bad"):
    raise RuntimeError("boom")
  record = json.loads((tmp_path / "bad" / "run.json").read_text())
  assert record["status"] == "failed" and "boom" in record["error"]
  with pytest.raises(FileExistsError):
    RunDir.create("x", out=tmp_path / "ok")


def test_default_run_names_do_not_collide(tmp_path, monkeypatch):
  monkeypatch.setattr("crowdmotion.core.rundir.RUNS", tmp_path)
  monkeypatch.setattr("crowdmotion.core.rundir.utc_stamp", lambda: "20260101T000000Z")
  a = RunDir.create("eval", env=False)
  b = RunDir.create("eval", env=False)
  assert a.path.name == "20260101T000000Z_eval" and b.path.name == "20260101T000000Z_eval_2"
