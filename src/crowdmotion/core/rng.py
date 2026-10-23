"""Counter-based per-human randomness, and the seed formulas existing runs used.

Every human owns a stream id `seed`. Draw `counter` of that stream depends on
nothing else, so results never depend on batch size, order or composition.

- `stream_seed` (Python ints) and `mix64` / `hash_uniform` (int64 tensors) are
  the same SplitMix64 hash.
- `per_human_noise` seeds one torch generator per row from `stream_seed`.
- `human_seeds` builds per-human stream ids. Its named schemes are the formulas
  saved outputs were made with; they stay so those outputs reproduce.
"""

from __future__ import annotations

import torch

MASK63 = (1 << 63) - 1
_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15 - (1 << 64)
_MUL1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_MUL2 = 0x94D049BB133111EB - (1 << 64)


def splitmix64(x: int) -> int:
  x = (x + 0x9E3779B97F4A7C15) & _MASK64
  x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
  x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
  return x ^ (x >> 31)


def stream_seed(seed: int, counter: int) -> int:
  """Seed for draw `counter` of per-human stream `seed`."""
  return splitmix64(splitmix64(seed) ^ counter) & MASK63


def _lsr(x: torch.Tensor, k: int) -> torch.Tensor:
  return (x >> k) & ((1 << (64 - k)) - 1)


def mix64(x: torch.Tensor) -> torch.Tensor:
  """SplitMix64 finalizer on int64 tensors (two's-complement wraparound)."""
  x = x + _GOLDEN
  x = (x ^ _lsr(x, 30)) * _MUL1
  x = (x ^ _lsr(x, 27)) * _MUL2
  return x ^ _lsr(x, 31)


def hash_uniform(seeds: torch.Tensor, counters: torch.Tensor, slots: int) -> torch.Tensor:
  """`[H, slots]` uniforms in [0, 1); row h depends only on (seeds[h], counters[h])."""
  base = mix64(mix64(seeds.long()) ^ counters.long())
  j = torch.arange(slots, device=seeds.device, dtype=torch.long)
  bits = _lsr(mix64(base[:, None] + j * _GOLDEN), 11)
  return bits.double().mul_(2.0**-53).float()


def per_human_noise(
  seeds: torch.Tensor, counters: torch.Tensor, shape: tuple[int, ...], device
) -> torch.Tensor:
  """Standard normal noise `[B, *shape]`, row b drawn only from (seeds[b], counters[b]).

  One generator launch per row, so a row never depends on the batch.
  """
  out = torch.empty((len(seeds), *shape), device=device)
  gen = torch.Generator(device=device)
  for row, (seed, counter) in enumerate(zip(seeds.tolist(), counters.tolist())):
    gen.manual_seed(stream_seed(seed, counter))
    out[row].normal_(generator=gen)
  return out


def row_generator_seed(seed: int, counter: int) -> int:
  """Torch generator seed for draw `counter` of stream `seed` in the random-arc paths.

  A legacy linear formula (not hashed); kept so benchmark paths reproduce.
  """
  return (seed * 1_000_003 + counter) & MASK63


# Per-human stream ids ----------------------------------------------------------
SEED_SCHEMES = ("quality", "stream", "evaluator", "graphcrowd", "plaza_turn", "graph_posture", "encounter",
                "traversal_goals", "crowd_flow")


def human_seeds(
  base: int,
  ids: torch.Tensor,
  scheme: str,
  *,
  offset: int = 0,
  episode: torch.Tensor | int = 0,
) -> torch.Tensor:
  """Stream ids for humans `ids` (int64 tensor) of a run seeded with `base`.

  quality:    base * 10_000 + ids                     (quality harness)
  stream:     base * 1_000_000 + offset + ids         (library generation, throughput)
  evaluator:  base * 1_000_003 + ids                  (behavior evaluator)
  graphcrowd: base * 1_000_003 + episode * 1_000_000_007 + ids   (mjlab GraphCrowd;
              `ids` are global human indices env * M + i, `episode` the env's reset count)
  plaza_turn: graphcrowd formula + 2**61 (optional committed-turn decisions;
              does not consume or replace the legacy plaza generator)
  graph_posture: domain-separated hash of already constructed motion stream ids;
              callers pass base=0, ids=motion_seeds. Does not consume motion RNG.
  encounter:  domain-separated hash of the graphcrowd formula (encounter behavior
              decisions, `behavior.encounter`; `ids` are global human indices)
  traversal_goals: domain-separated hash of the graphcrowd formula over env indices
              (paired evaluation goals, spawns and timers of the traversal command)
  """
  if scheme == "quality":
    return ids + base * 10_000
  if scheme == "stream":
    return ids + base * 1_000_000 + offset
  if scheme == "evaluator":
    return ids + base * 1_000_003
  if scheme == "graphcrowd":
    return base * 1_000_003 + episode * 1_000_000_007 + ids
  if scheme == "plaza_turn":
    return base * 1_000_003 + episode * 1_000_000_007 + ids + 2**61
  if scheme == "graph_posture":
    return mix64(ids ^ (base * 1_000_003) ^ 0x50_057) & MASK63
  if scheme == "encounter":
    return mix64((base * 1_000_003 + episode * 1_000_000_007 + ids) ^ 0xE7C0_07E5) & MASK63
  if scheme == "traversal_goals":
    return mix64((base * 1_000_003 + episode * 1_000_000_007 + ids) ^ 0x60A1_5EED) & MASK63
  if scheme == "crowd_flow":
    return mix64((base * 1_000_003 + episode * 1_000_000_007 + ids) ^ 0xF10A_2026) & MASK63
  raise ValueError(f"unknown seed scheme {scheme!r}; choose from {SEED_SCHEMES}")
