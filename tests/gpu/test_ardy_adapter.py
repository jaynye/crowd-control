"""ARDY adapter parity with upstream (GPU if present; CPU is very slow).

  source scripts/env.sh && .venv/bin/python -m pytest tests/gpu/test_ardy_adapter.py -q
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

pytest.importorskip("ardy")

from crowdmotion.backends.base import MotionRequest, injected_randn  # noqa: E402
from crowdmotion.backends.registry import ArdyConfig, make_backend  # noqa: E402
from crowdmotion.core.rng import per_human_noise  # noqa: E402

MODEL = "ARDY-Core-RP-20FPS-Horizon8"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def backend():
  return make_backend(ArdyConfig(MODEL, device=DEVICE, future_frames=8, timing=False))


def _request(backend, batch, history=None, seed0=0):
  horizon = backend.horizon + backend.future_frames
  t = torch.arange(horizon, dtype=torch.float32, device=DEVICE) / backend.fps
  speed = torch.linspace(0.6, 1.4, batch, device=DEVICE)[:, None]
  path = torch.stack((0.2 * t.expand(batch, -1), speed * t), -1)
  return MotionRequest(
    text_feat=backend.empty_text(batch),
    seeds=torch.arange(seed0, seed0 + batch),
    counters=torch.zeros(batch, dtype=torch.long),
    history=history,
    root_path=path,
  )


def test_conditions_match_library(backend):
  req = _request(backend, 3)
  num_frames = 4 + backend.horizon + backend.future_frames
  backend.conditions_mode = "library"
  lib = backend.conditions(req.root_path, 4, num_frames, 3)
  backend.conditions_mode = "vectorized"
  vec = backend.conditions(req.root_path, 4, num_frames, 3)
  torch.testing.assert_close(vec[0], lib[0])
  torch.testing.assert_close(vec[1], lib[1])


def test_generate_matches_upstream_step(backend):
  fresh = backend.generate(_request(backend, 2))
  assert fresh.frames == backend.horizon
  history = fresh.features[:, -4:]
  req = _request(backend, 2, history=history)
  ours = backend.generate(req)
  assert ours.frames == backend.horizon  # history is not returned

  m = backend.model
  num_frames = 4 + backend.horizon + backend.future_frames
  mask, obs = backend.conditions(req.root_path, 4, num_frames, 2)
  noise = per_human_noise(
    req.seeds, req.counters, (backend.horizon // 4, backend.token_dim), DEVICE
  )
  with injected_randn(backend._module, noise):
    upstream = m.autoregressive_step(
      num_frames=num_frames,
      num_denoising_steps=backend.steps,
      motion_mask=mask,
      observed_motion=obs,
      cfg_weight=backend.cfg_weight,
      text_feat=req.text_feat,
      text_pad_mask=torch.ones(2, 1, dtype=torch.bool, device=DEVICE),
      init_history_sequence=history,
    )
  assert upstream.shape[1] == 4 + backend.horizon
  torch.testing.assert_close(ours.features, upstream[:, 4:])


def test_noise_independent_of_batch_order():
  seeds = torch.tensor([11, 5, 7])
  counters = torch.tensor([0, 3, 1])
  a = per_human_noise(seeds, counters, (2, 5), "cpu")
  perm = torch.tensor([2, 0, 1])
  b = per_human_noise(seeds[perm], counters[perm], (2, 5), "cpu")
  torch.testing.assert_close(a[perm], b)
  c = per_human_noise(seeds[:1], counters[:1], (2, 5), "cpu")
  torch.testing.assert_close(a[:1], c)
  d = per_human_noise(seeds[:1], counters[:1] + 1, (2, 5), "cpu")
  assert not torch.allclose(c, d)


def test_generation_independent_of_batch_composition(backend):
  full = backend.generate(_request(backend, 3))
  solo = backend.generate(_request(backend, 3))
  torch.testing.assert_close(full.features, solo.features)
  req = _request(backend, 3)
  one = MotionRequest(
    text_feat=req.text_feat[1:2], seeds=req.seeds[1:2], counters=req.counters[1:2],
    root_path=req.root_path[1:2],
  )
  torch.testing.assert_close(backend.generate(one).features, full.features[1:2], atol=1e-4, rtol=1e-4)
