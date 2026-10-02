"""Shared test setup: markers, CPU defaults and the synthetic motion library.

Unit tests run on CPU without checkpoints, `scripts/env.sh` or a saved graph.
`gpu` tests skip without CUDA and the checkpoints; `slow` tests (real ARDY and
Kimodo goldens on CPU) run only with `-m slow`.
"""

from __future__ import annotations

import os

import pytest
import torch

from synthetic import synthetic_library  # noqa: F401  (re-exported for tests)


def pytest_configure(config):
  torch.set_num_threads(int(os.environ.get("CROWD_TEST_THREADS", "1")))


def _checkpoints_present() -> bool:
  dirs = [os.environ.get("CHECKPOINTS_DIR"), os.environ.get("CHECKPOINT_DIR")]
  return all(d and os.path.isdir(d) and os.listdir(d) for d in dirs)


def pytest_collection_modifyitems(config, items):
  reason = None
  if not torch.cuda.is_available():
    reason = "needs CUDA"
  elif not _checkpoints_present():
    reason = "needs checkpoints (source scripts/env.sh; scripts/fetch_artifacts.sh --checkpoints)"
  if reason is None:
    return
  skip = pytest.mark.skip(reason=reason)
  for item in items:
    if "gpu" in item.keywords:
      item.add_marker(skip)


@pytest.fixture
def toy_library():
  return synthetic_library()
