import torch
from tests.unit.test_external_agents import walk

from crowdmotion.behavior.encounter import EncounterBehavior, EncounterCfg


def test_independent_humans_cross_each_other_and_ignore_the_robot():
  cfg = EncounterCfg(reactive=False, conditioning="velocity", starts=((-2., 0., 90.), (2., 0., -90.)),
                     velocities=((1., 90.), (1., -90.)))
  b = EncounterBehavior(2, 2, cfg, 3, "cpu")
  pos, heading = b.reset(torch.arange(2))
  b.set_external(torch.zeros(2, 1, 2))
  track, out = walk(b, pos, heading, 10)
  assert (track[5, :, 0] - track[5, :, 1]).norm(dim=-1).max() < .01
  assert (track[-1, :, 0, 0] > 1.5).all()
  assert not b.stopped.any() and not b.separating.any()
  assert not b.robot_perceives.any() and b._neighbors is None


def test_flow_reset_and_commands_are_independent_of_batch_order():
  cfg = EncounterCfg(reactive=False, flow_fraction=.9, flow_spread_deg=0.)
  a, b = (EncounterBehavior(20, 8, cfg, 7, "cpu") for _ in range(2))
  a.reset(torch.arange(20))
  b.reset(torch.tensor([9, 3]))
  for field in ("flow_member", "flow_heading", "goal", "speed", "seeds"):
    assert torch.equal(getattr(a, field)[[9, 3]], getattr(b, field)[[9, 3]])
  assert .8 < a.flow_member.float().mean() < 1.
