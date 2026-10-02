"""Text conditioning stand-ins while the gated Llama-3 encoder is unavailable.

Both ARDY and Kimodo pool a prompt to one 4096-d token, so denoiser cost does not
depend on prompt content; empty-prompt features are zeros of that shape.
"""

from __future__ import annotations

import torch

TEXT_DIM = 4096


def empty_text(batch: int, device) -> torch.Tensor:
  """`[B, 1, 4096]` empty-prompt features."""
  return torch.zeros(batch, 1, TEXT_DIM, device=device)


class ZeroTextEncoder:
  """Stands in for LLM2Vec (gated Llama-3) in upstream loaders: empty-prompt features."""

  def __call__(self, texts):
    texts = [texts] if isinstance(texts, str) else texts
    return torch.zeros(len(texts), 1, TEXT_DIM), [0] * len(texts)
