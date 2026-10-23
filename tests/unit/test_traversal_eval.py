"""`crowd eval traversal` helpers that need no simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from crowdmotion.sim.mjlab.evaluate import _clean, iteration_of, run_checkpoints, write_tables


def test_checkpoints_come_from_the_run_itself_in_iteration_order(tmp_path):
  logs = tmp_path / "logs" / "g1_crowd_traversal"
  for folder, iters in (("2026-10-18_a", (0, 100, 200)), ("2026-10-19_b", (200, 300, 1000)), ("resume_x", (5000,))):
    (logs / folder).mkdir(parents=True)
    for i in iters:
      (logs / folder / f"model_{i}.pt").write_text("")
  found = run_checkpoints(tmp_path)
  assert [iteration_of(c) for c in found] == [0, 100, 200, 300, 1000]
  assert found[2].parent.name == "2026-10-19_b"  # the later attempt's copy of an iteration
  with pytest.raises(FileNotFoundError):
    run_checkpoints(tmp_path / "empty")


def test_results_are_json_safe():
  assert _clean({"a": np.float32(1.5), "b": math.nan, "c": {"d": np.int64(3), "e": math.inf}, "f": None}) == {
    "a": 1.5, "b": None, "c": {"d": 3, "e": None}, "f": None}


def test_tables_are_written_as_csv_with_headers(tmp_path):
  write_tables(tmp_path, "100_aware_end", {"legs": {"env": np.array([0.0, 1.0]), "spl": np.array([0.5, 1.0])},
                                           "falls": {"env": np.zeros(0), "near": np.zeros(0)}})
  legs = (tmp_path / "100_aware_end_legs.csv").read_text().splitlines()
  assert legs == ["env,spl", "0,0.5", "1,1"]
  assert (tmp_path / "100_aware_end_falls.csv").read_text().strip() == "env,near"
