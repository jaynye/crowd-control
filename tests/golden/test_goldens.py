"""Bit-exact golden outputs through `tests/golden/harness.py` (fast variants).

Cases that need the saved motion graph skip without it (`$CROWD_GRAPH` or the
`graph-default` artifact). `-m slow` runs the real ARDY and Kimodo CPU cases,
which need `scripts/env.sh` (ARDY on the path) and the checkpoints.
"""

from __future__ import annotations

import pytest

from golden import harness

FAST = [name for name, (_, need) in harness.CASES.items() if need != "slow"]
SLOW = [name for name, (_, need) in harness.CASES.items() if need == "slow"]


def _run(case: str, tmp_path, fast: bool) -> int:
  need = harness.CASES[case][1]
  graph = harness.default_graph()
  if need == "graph" and (graph is None or not graph.exists()):
    pytest.skip("no saved motion graph (CROWD_GRAPH or `scripts/fetch_artifacts.sh`)")
  if case in ("mjlab_plaza", "corridor"):
    pytest.importorskip("mjlab")
  argv = ["check", "--cases", case, "--out", str(tmp_path)]
  if fast:
    argv.append("--fast")
  if graph is not None:
    argv += ["--graph", str(graph)]
  return harness.main(argv)


@pytest.mark.golden
@pytest.mark.parametrize("case", FAST)
def test_golden_fast(case, tmp_path):
  assert _run(case, tmp_path, fast=True) == 0


@pytest.mark.golden
@pytest.mark.slow
@pytest.mark.parametrize("case", SLOW)
def test_golden_slow(case, tmp_path):
  pytest.importorskip("ardy" if case == "ardy_cpu" else "kimodo")
  assert _run(case, tmp_path, fast=False) == 0
