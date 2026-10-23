"""Encounter humans perceiving external agents (the robot), with their own traits."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from crowdmotion.behavior.encounter import CRUISE, HIDDEN_M, EncounterBehavior, EncounterCfg


def walk(b: EncounterBehavior, xz, heading, steps: int, dt: float = 0.4):
  """Kinematic rollout that executes each command at once; returns the root track and
  every step's commands, modes and switch flags."""
  ids = torch.arange(b.E)
  vel = torch.zeros_like(xz)
  track, out = [xz.clone()], []
  for k in range(steps):
    speed, travel = b.encounter_command(torch.full((b.E,), k * dt), xz, vel, heading, ids)
    out.append((speed.clone(), travel.clone(), b.mode.clone(), b.switched.clone()))
    vel = torch.stack((torch.sin(travel), torch.cos(travel)), -1) * speed[..., None]
    xz = xz + vel * dt
    heading = torch.where(speed > 0, travel, heading)
    track.append(xz.clone())
  return torch.stack(track), out


def toward_robot(envs: int, **kw):
  """One human walking at 1 m/s straight at a robot standing 4 m ahead."""
  cfg = EncounterCfg(conditioning="velocity", starts=((-4.0, 0.0, 90.0),), velocities=((1.0, 90.0),), **kw)
  b = EncounterBehavior(envs, 1, cfg, seed=5, device="cpu")
  xz, heading = b.reset(torch.arange(envs))
  b.set_external(torch.zeros(envs, 1, 2))
  return b, xz, heading


def test_unperceived_external_agents_change_no_decision():
  cfg = dict(stubborn_fraction=.1, minimal_avoidance=True, stall_recovery_s=.8, pass_commitment_s=3.,
             pass_commitment_jitter=.75)
  plain = EncounterBehavior(4, 8, EncounterCfg(**cfg), seed=11, device="cpu")
  blind = EncounterBehavior(4, 8, EncounterCfg(robot_oblivious_fraction=1.0, **cfg), seed=11, device="cpu")
  xz, heading = plain.reset(torch.arange(4))
  xz2, heading2 = blind.reset(torch.arange(4))
  assert torch.equal(xz, xz2) and torch.equal(heading, heading2)
  blind.set_external(torch.tensor([[[0.5, 0.5]]]).expand(4, 1, 2), torch.tensor([[[0.3, 0.0]]]))
  _, a = walk(plain, xz, heading, 40)
  _, b = walk(blind, xz2, heading2, 40)
  for (sa, ta, ma, wa), (sb, tb, mb, wb) in zip(a, b):
    assert torch.equal(ma, mb) and torch.equal(wa, wb)
    assert torch.allclose(sa, sb, atol=1e-5) and torch.allclose(ta, tb, atol=1e-5)


def test_cooperative_humans_avoid_a_standing_robot_and_oblivious_humans_walk_through():
  b, xz, heading = toward_robot(64, robot_assertiveness_range=(0.0, 0.0))
  track, out = walk(b, xz, heading, 30)
  gap = track.norm(dim=-1).amin(0)  # closest root distance to the robot at the origin
  assert (gap >= 0.35 + 0.3 + 0.01 - 0.02).all()
  assert torch.stack([w for *_, w in out]).any(0).float().mean() > 0.9  # nearly all intervene
  assert (track[-1, :, 0, 0] > 0.5).float().mean() > 0.9  # and get past it

  b, xz, heading = toward_robot(8, robot_oblivious_fraction=1.0)
  track, out = walk(b, xz, heading, 30)
  assert (track.norm(dim=-1).amin(0) < 0.3).all()  # straight through: the robot is not there
  assert not torch.stack([w for *_, w in out]).any()


def test_robot_stubborn_humans_stop_but_never_intervene_for_the_robot():
  b, xz, heading = toward_robot(32, robot_stubborn_fraction=1.0)
  track, out = walk(b, xz, heading, 30)
  assert not torch.stack([w for *_, w in out]).any()
  assert all((m == CRUISE).all() for *_, m, _ in out)
  gap = track.norm(dim=-1).amin(0)
  assert (gap >= 0.35 + 0.3 + 0.01 - 0.02).all()  # the stop rule still holds them off
  assert b.stopped.all()


def test_a_bigger_robot_is_kept_further_away():
  gaps = []
  for radius in (0.2, 0.6):
    b, xz, heading = toward_robot(16, robot_stubborn_fraction=1.0, robot_radius_m=radius)
    track, _ = walk(b, xz, heading, 30)
    gaps.append(track.norm(dim=-1).amin().item())
  assert gaps[1] - gaps[0] == pytest.approx(0.4, abs=0.05)


def test_robot_traits_follow_their_fractions_and_leave_human_traits_unchanged():
  base = dict(stubborn_fraction=.05)
  a = EncounterBehavior(400, 8, EncounterCfg(**base), seed=2, device="cpu")
  b = EncounterBehavior(400, 8, EncounterCfg(robot_oblivious_fraction=.1, robot_stubborn_fraction=.2,
                                             robot_assertiveness_range=(.2, .4), **base), seed=2, device="cpu")
  a.reset(torch.arange(400))
  b.reset(torch.arange(400))
  for name in ("stubborn", "stubborn_patience", "assertiveness", "goal", "speed"):
    assert torch.equal(getattr(a, name), getattr(b, name))
  oblivious = ~b.robot_perceives
  stubborn = b.robot_perceives & ~b.robot_decides
  assert abs(oblivious.float().mean() - .1) < .02 and abs(stubborn.float().mean() - .2) < .02
  assert not (oblivious & b.robot_decides).any()
  r = b.robot_assertiveness
  assert r.min() >= .2 and r.max() <= .4
  # Independent of batch order and composition.
  c = EncounterBehavior(400, 8, b.cfg, seed=2, device="cpu")
  c.reset(torch.tensor([7, 3]))
  for name in ("robot_perceives", "robot_decides", "robot_assertiveness"):
    assert torch.equal(getattr(c, name)[[7, 3]], getattr(b, name)[[7, 3]])


def test_robot_config_is_validated():
  with pytest.raises(ValueError, match="must not exceed 1"):
    EncounterCfg(robot_oblivious_fraction=.6, robot_stubborn_fraction=.5)
  with pytest.raises(ValueError, match="robot_radius_m"):
    EncounterCfg(robot_radius_m=0.0)
  b = EncounterBehavior(2, 3, EncounterCfg(), seed=0, device="cpu")
  with pytest.raises(ValueError, match="external positions"):
    b.set_external(torch.zeros(3, 1, 2))
  b.set_external(torch.zeros(2, 1, 2))
  b.set_external(None)
  assert b.external_xz is None


def test_graph_neighbors_carry_the_robot_with_its_own_clearance():
  from crowdmotion.graph.costs import CandidateContext, ClearanceCost

  b, xz, heading = toward_robot(1, robot_radius_m=0.6)
  b.set_external(torch.tensor([[[0.0, 0.0]]]))
  b.encounter_command(torch.zeros(1), torch.tensor([[[-2.0, 0.0]]]), torch.zeros(1, 1, 2), heading,
                      torch.arange(1))
  points, speeds, limit = b._neighbors
  assert points.shape == (1, 3, 2) and limit.shape == (1, 3)  # two wall mirrors, the robot
  threshold = 2 * 0.35 + 0.01
  assert limit[0, :2].tolist() == pytest.approx([threshold] * 2)
  assert limit[0, 2].item() == pytest.approx(threshold + 0.6 - 0.35)
  b.robot_perceives[:] = False
  b.encounter_command(torch.full((1,), .4), torch.tensor([[[-2.0, 0.0]]]), torch.zeros(1, 1, 2), heading,
                      torch.arange(1))
  assert b._neighbors[0][0, 2, 0].item() >= HIDDEN_M - 10

  # Per-neighbor limits reproduce one shared limit, and a larger one costs more.
  frames = torch.arange(1, 9, dtype=torch.float32)[:, None] / 8
  pred = torch.stack((torch.zeros(8, 2), frames * torch.tensor([0.0, 0.3])))[None]

  def cost(limit):
    ctx = CandidateContext(graph=SimpleNamespace(fps=20.0), state=SimpleNamespace(root_xz=torch.zeros(1, 2)),
                           path=None, cand=None, ddist=torch.zeros(1, 2), head=None, local=None, pred=pred)
    ctx.neighbors = (torch.tensor([[[0.0, -0.3], [0.0, 0.8]]]), torch.zeros(1, 2, 2), limit)
    return ClearanceCost()(ctx)[0]

  assert torch.allclose(cost(torch.tensor([0.51])), cost(torch.tensor([[0.51, 0.51]])))
  assert cost(torch.tensor([[0.51, 0.61]]))[1].item() == pytest.approx(0.11, abs=1e-6)
  assert math.isfinite(cost(torch.tensor([[0.51, 0.61]])).sum().item())
