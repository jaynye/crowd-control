"""Committed turns finish on executed facing and retain targets through avoidance."""

import math

import pytest
import torch

from crowdmotion.behavior.intent import CommittedTurnIntent
from crowdmotion.behavior.plaza import PlazaBehavior, PlazaSteeringCfg


def _intent(envs=1, humans=3, **kw):
  cfg = PlazaSteeringCfg(committed_turns=True, turn_walk_s=(1., 1.), wait_probability=0.,
                         turn_angles_deg=(90.,), turn_weights=(1.,), turn_jitter_deg=0., **kw)
  return CommittedTurnIntent(cfg, envs, humans, torch.Generator().manual_seed(3), "cpu", 17)


def test_target_is_sampled_once_and_completion_uses_executed_body_heading():
  intent = _intent()
  ids = torch.tensor([0])
  xz, heading = intent.reset(ids)
  intent.preferred_velocity(torch.tensor([1.]), xz, ids, heading)
  target = intent.target.clone()
  assert torch.all(intent.mode == intent.TURN)
  assert torch.allclose((target - heading).abs(), torch.full_like(heading, math.pi / 2))
  counters = intent.counters.clone()
  # Time alone (or avoidance in the opposite direction) must not finish/resample.
  intent.preferred_velocity(torch.tensor([100.]), xz, ids, heading - .2)
  assert torch.equal(intent.target, target)
  assert torch.equal(intent.counters, counters)
  assert torch.all(intent.mode == intent.TURN)
  intent.preferred_velocity(torch.tensor([101.]), xz, ids, target)
  assert torch.all(intent.mode == intent.WALK)
  assert torch.all(intent.until == 102)


def test_turn_draws_do_not_consume_legacy_rng_and_survive_group_reordering():
  a, b = _intent(envs=2), _intent(envs=2)
  ids = torch.tensor([0, 1])
  xz, heading = a.reset(ids)
  b.reset(ids)
  rng_before = a.gen.get_state().clone()
  a.preferred_velocity(torch.tensor([1., 1.]), xz, ids, heading)
  b.preferred_velocity(torch.tensor([1.]), xz[1:], ids[1:], heading[1:])
  b.preferred_velocity(torch.tensor([1.]), xz[:1], ids[:1], heading[:1])
  assert torch.equal(a.target, b.target)
  assert torch.equal(a.gen.get_state(), rng_before)


def test_default_and_optional_intent_spawn_identically_and_reset_only_selected_envs():
  base = PlazaBehavior(2, 3, PlazaSteeringCfg(), 4, "cpu")
  turns = PlazaBehavior(2, 3, PlazaSteeringCfg(committed_turns=True), 4, "cpu")
  ids = torch.tensor([0, 1])
  for x, y in zip(base.reset(ids), turns.reset(ids)):
    assert torch.equal(x, y)
  other = turns.intent.target[1].clone()
  old_seeds = turns.intent.seeds.clone()
  turns.reset(ids[:1])
  assert torch.equal(turns.intent.target[1], other)
  assert torch.equal(turns.intent.seeds[1], old_seeds[1])
  assert not torch.equal(turns.intent.seeds[0], old_seeds[0])


def test_preferred_turn_velocity_is_forward_with_bounded_lead():
  intent = _intent()
  ids = torch.tensor([0])
  xz, heading = intent.reset(ids)
  vel, speed, waiting = intent.preferred_velocity(torch.tensor([1.]), xz, ids, heading)
  facing = torch.stack((heading.sin(), heading.cos()), -1)
  assert torch.allclose((vel * facing).sum(-1) / vel.norm(dim=-1), torch.full_like(heading, math.cos(math.radians(25))))
  assert torch.allclose(vel.norm(dim=-1), speed * .7)
  assert not waiting.any()


@pytest.mark.parametrize("kwargs", [{"turn_weights": (1.,)}, {"turn_walk_s": (0., 1.)},
                                    {"turn_finish_deg": 30.}, {"turn_speed_scale": 0.}])
def test_bad_turn_settings_are_rejected(kwargs):
  with pytest.raises(ValueError):
    PlazaSteeringCfg(committed_turns=True, **kwargs)
