"""Kimodo adapter continuation parity with upstream multi-prompt generation (GPU).

  sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/gpu_pytest.sbatch tests/gpu/test_kimodo_adapter.py

Upstream `_multiprompt` returns seg1[:-T] + cross-faded transition + seg2[T:].
With identical noise, the adapter's fresh call must equal seg1 outside the
cross-faded frames, and its continuation (seam blending off) must equal seg2[T:].
"""

import contextlib

import pytest
import torch

pytestmark = pytest.mark.gpu

pytest.importorskip("kimodo")
if not torch.cuda.is_available():
  pytest.skip("needs a GPU", allow_module_level=True)

from crowdmotion.backends.base import MotionRequest  # noqa: E402
from crowdmotion.backends.registry import KimodoConfig, make_backend  # noqa: E402
from crowdmotion.core.rng import per_human_noise  # noqa: E402

B, F, T, STEPS = 2, 30, 5, 5


@contextlib.contextmanager
def sequential_randn(module, noises):
  real = module.torch
  queue = list(noises)

  class _Proxy:
    def __getattr__(self, name):
      return getattr(real, name)

    @staticmethod
    def randn(*size, **kwargs):
      out = queue.pop(0)
      shape = tuple(size[0]) if len(size) == 1 and not isinstance(size[0], int) else tuple(size)
      assert shape == tuple(out.shape), (shape, out.shape)
      return out

  module.torch = _Proxy()
  try:
    yield queue
  finally:
    module.torch = real


@pytest.fixture(scope="module")
def backend():
  return make_backend(KimodoConfig(device="cuda", steps=STEPS, precision="fp32", timing=False, new_frames=F))


def test_continuation_matches_multiprompt(backend):
  dim = backend.model.motion_rep.motion_rep_dim
  seeds = torch.arange(B)
  c0, c1 = torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)
  text = backend.empty_text(B)
  zeros2 = torch.zeros(B, 2, device="cuda")
  ours1 = backend.generate(
    MotionRequest(text_feat=text, seeds=seeds, counters=c0, start_xz=zeros2,
                  start_heading=torch.zeros(B, device="cuda")))
  ours2 = backend.generate(
    MotionRequest(text_feat=text, seeds=seeds, counters=c1, history=ours1.features[:, -T:]))

  n1 = per_human_noise(seeds, c0, (F, dim), "cuda")
  n2 = per_human_noise(seeds, c1, (F + T, dim), "cuda")
  with sequential_randn(backend._module, [n1, n2]) as left:
    up = backend.model(
      ["", ""], [F, F], num_denoising_steps=STEPS, multi_prompt=True, num_samples=B,
      cfg_weight=backend.cfg_weight, first_heading_angle=torch.zeros(B, device="cuda"),
      num_transition_frames=T, post_processing=False, progress_bar=lambda x: x,
    )
  assert not left
  root = up["root_positions"]
  assert root.shape[1] == 2 * F
  torch.testing.assert_close(ours1.root_pos[:, : F - T], root[:, : F - T], atol=1e-4, rtol=1e-4)
  torch.testing.assert_close(ours2.root_pos, root[:, F:], atol=1e-3, rtol=1e-3)
