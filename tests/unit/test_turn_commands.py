"""Turning command source, heading targets through requests, and ARDY heading conditions."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crowdmotion.backends.ardy import ArdyBackend
from crowdmotion.backends.base import MotionChunk, MotionRequest
from crowdmotion.behavior.commands import ARC, IN_PLACE, STAND, WALK, TurnCommands, TurnRanges
from crowdmotion.behavior.kinematics import Kinematics
from crowdmotion.behavior.sources import OnlineMotionSource
from crowdmotion.core.geometry import wrap
from crowdmotion.library import generate

FPS = 20.0


def kin(n, heading=None):
  h = torch.zeros(n) if heading is None else heading
  return Kinematics(torch.zeros(n, 2), h, torch.zeros(n, 2))


def rollout(seeds, steps=400, dt=0.4, ranges=None):
  """Commands read every `dt` with the executed heading equal to the command."""
  cmds = TurnCommands(seeds, torch.zeros(len(seeds)), ranges or TurnRanges())
  heading = torch.zeros(len(seeds))
  rows = []
  for k in range(steps):
    speed, heading = cmds.command(k * dt, kin(len(seeds), heading))
    st = cmds.state()
    rows.append({key: st[key].copy() for key in ("mode", "segment", "rate", "angle", "radius")}
                | {"speed": speed.numpy().copy(), "heading": heading.numpy().copy()})
  return cmds, rows


def test_turn_ranges_are_strict():
  with pytest.raises((TypeError, ValueError, KeyError)):
    TurnRanges.from_dict({"in_place_degs": [10, 20]})
  assert TurnRanges.from_dict({"walk_heading": False}).walk_heading is False


def test_segment_statistics_follow_ranges():
  r = TurnRanges()
  _, rows = rollout(torch.arange(64))
  mode = np.stack([row["mode"] for row in rows])
  seg = np.stack([row["segment"] for row in rows])
  first = np.concatenate([np.ones((1, mode.shape[1]), bool), seg[1:] != seg[:-1]])
  kinds = mode[first]
  assert set(np.unique(kinds)) == {WALK, STAND, IN_PLACE, ARC}
  rate = np.stack([row["rate"] for row in rows])[first]
  angle = np.stack([row["angle"] for row in rows])[first]
  radius = np.stack([row["radius"] for row in rows])[first]
  speed = np.stack([row["speed"] for row in rows])[first]
  ip, arc = kinds == IN_PLACE, kinds == ARC
  deg = np.degrees(np.abs(angle))
  assert (deg[ip] >= r.in_place_deg[0] - 1e-6).all() and (deg[ip] <= r.in_place_deg[1] + 1e-6).all()
  assert (np.degrees(np.abs(rate[ip])) >= r.in_place_rate_deg[0] - 1e-6).all()
  assert (np.degrees(np.abs(rate[ip])) <= r.in_place_rate_deg[1] + 1e-6).all()
  assert (speed[ip | (kinds == STAND)] == 0).all()
  assert (deg[arc] >= r.arc_deg[0] - 1e-6).all() and (deg[arc] <= r.arc_deg[1] + 1e-6).all()
  assert (radius[arc] >= r.arc_radius_m[0]).all() and (radius[arc] <= r.arc_radius_m[1]).all()
  assert np.allclose(np.abs(rate[arc]) * radius[arc], speed[arc], rtol=1e-5)  # v = |w| r
  assert (np.degrees(np.abs(rate[arc])) <= r.arc_max_rate_deg + 1e-4).all()
  assert np.isnan(radius[~arc]).all() and (rate[kinds == WALK] == 0).all()
  # Both turn directions occur, and in-place turns mostly follow a stand.
  assert (angle[ip] > 0).any() and (angle[ip] < 0).any()


def test_turn_heading_reaches_the_segment_angle():
  _, rows = rollout(torch.arange(16), steps=200, dt=0.05)
  mode = np.stack([row["mode"] for row in rows])
  seg = np.stack([row["segment"] for row in rows])
  heading = np.stack([row["heading"] for row in rows])
  angle = np.stack([row["angle"] for row in rows])
  rate = np.stack([row["rate"] for row in rows])
  checked = 0
  for i in range(mode.shape[1]):
    for s in np.unique(seg[:, i]):
      idx = np.flatnonzero(seg[:, i] == s)
      if mode[idx[0], i] in (IN_PLACE, ARC) and idx[-1] + 1 < len(seg):
        # The last read of a segment is within one read interval of its end; in-place
        # turns settle after turning, so they reach their angle exactly.
        tol = 1e-4 if mode[idx[0], i] == IN_PLACE else abs(rate[idx[0], i]) * 0.05 + 1e-4
        assert heading[idx[-1], i] - heading[idx[0], i] == pytest.approx(angle[idx[0], i], abs=tol)
        checked += 1
  assert checked > 5


def test_segments_ignore_batch_composition():
  seeds = torch.tensor([3, 11, 2**33 + 5, 40])
  _, all_rows = rollout(seeds, steps=60)
  _, sub_rows = rollout(seeds[[2, 0]], steps=60)
  for a, b in zip(all_rows, sub_rows):
    for key in ("mode", "segment", "rate", "angle", "speed"):
      np.testing.assert_array_equal(a[key][[2, 0]], b[key])


def test_plan_targets_shapes_and_switches():
  seeds = torch.arange(32)
  cmds, _ = rollout(seeds, steps=7)
  n, frames = len(seeds), 40
  speed, heading = cmds.command(7 * 0.4, kin(n))
  path = cmds.plan(torch.zeros(n, 2), torch.zeros(n, 2), speed, heading, frames, FPS)
  assert path.shape == (n, frames, 2) and cmds.targets.shape == (n, frames)
  assert torch.isfinite(cmds.targets).all()
  assert set(cmds.motion_options()) == {"root_heading"}
  free = TurnCommands(seeds, torch.zeros(n), TurnRanges(walk_heading=False, stand_heading=False))
  speed, heading = free.command(0.0, kin(n))
  free.plan(torch.zeros(n, 2), torch.zeros(n, 2), speed, heading, frames, FPS)
  mode = torch.as_tensor(free.mode)
  nan_rows = torch.isnan(free.targets).all(1)
  assert torch.equal(nan_rows, (mode == WALK) | (mode == STAND))
  assert TurnCommands(seeds, torch.zeros(n), TurnRanges(heading_targets=False)).motion_options() == {}


def test_in_place_plan_turns_without_moving():
  r = TurnRanges(start=(0.0, 0.0, 1.0, 0.0), in_place_deg=(90.0, 90.0), in_place_rate_deg=(90.0, 90.0))
  cmds = TurnCommands(torch.arange(8), torch.zeros(8), r)
  h0 = torch.linspace(-3, 3, 8)
  speed, heading = cmds.command(0.0, kin(8, h0))
  assert torch.allclose(heading, h0)
  path = cmds.plan(torch.ones(8, 2), torch.zeros(8, 2), speed, heading, 40, FPS)
  assert torch.allclose(path, torch.ones(8, 40, 2))
  sign = torch.as_tensor(np.sign(cmds.angle), dtype=torch.float32)
  # 90 deg at 90 deg/s: the target reaches h0 + 90 deg after 1 s (frame 20) and holds.
  assert torch.allclose(cmds.targets[:, 19:], (h0 + sign * math.pi / 2)[:, None].expand(8, 21), atol=1e-5)
  assert torch.allclose(cmds.targets[:, 0], h0 + sign * math.radians(90) / FPS, atol=1e-5)


def test_targets_start_from_executed_heading():
  r = TurnRanges(start=(1.0, 0.0, 0.0, 0.0))
  cmds = TurnCommands(torch.arange(4), torch.zeros(4), r)
  speed, heading = cmds.command(0.0, kin(4, torch.zeros(4)))
  # A later read whose executed heading lags the command by 0.3 rad.
  executed = heading - 0.3
  speed, heading = cmds.command(0.1, kin(4, executed))
  cmds.plan(torch.zeros(4, 2), torch.zeros(4, 2), torch.zeros(4), heading, 40, FPS)
  lag = wrap(cmds.targets - heading[:, None])
  assert torch.allclose(lag[:, 0], torch.full((4,), -0.3 * 0.9), atol=1e-5)  # first frame: 10% blended
  assert torch.allclose(lag[:, 10:], torch.zeros(4, 30), atol=1e-6)


def test_request_root_heading_defaults_to_none():
  backend = SimpleNamespace(fps=FPS, path_frames=8, empty_text=lambda b: torch.zeros(b, 1, 4))
  src = OnlineMotionSource(backend, history_frames=4)
  state = src.reset(torch.arange(3), torch.zeros(3, 2), torch.zeros(3))
  path = torch.zeros(3, 8, 2)
  assert src.request(state, path).root_heading is None
  h = torch.ones(3, 8)
  assert src.request(state, path, root_heading=h).root_heading is h
  sent = []
  src.generate = lambda req: sent.append(req)
  src.step(state, path)
  src.step(state, path, root_heading=h)
  assert sent[0].root_heading is None and sent[1].root_heading is h


class FakeRep:
  """The parts of an ARDY motion rep `conditions` reads (root_pos then heading)."""

  def __init__(self):
    self.slice_dict = {"root_pos": slice(0, 3), "global_root_heading": slice(3, 5), "body": slice(5, 9)}
    self.motion_rep_dim = 9
    self.mean = torch.linspace(-0.5, 0.5, 9)
    self.std = torch.linspace(0.5, 2.0, 9)

  def normalize(self, x):
    return (x - self.mean) / self.std


def fake_ardy(mode="vectorized"):
  return SimpleNamespace(model=SimpleNamespace(motion_rep=FakeRep()), conditions_mode=mode,
                         device=torch.device("cpu"))


def test_ardy_vectorized_heading_conditions():
  self = fake_ardy()
  g = torch.Generator().manual_seed(0)
  path = torch.randn(2, 12, 2, generator=g)
  mask0, obs0 = ArdyBackend.conditions(self, path, 4, 16, 2)
  mask1, obs1 = ArdyBackend.conditions(self, path, 4, 16, 2, None)
  assert torch.equal(mask0, mask1) and torch.equal(obs0, obs1)
  assert mask0[..., 3:].sum() == 0 and mask0[:, 4:16, [0, 2]].all() and mask0[:, :4].sum() == 0
  heading = torch.randn(2, 12, generator=g)
  heading[0, 3] = float("nan")
  mask, obs = ArdyBackend.conditions(self, path, 4, 16, 2, heading)
  assert torch.isfinite(obs).all()
  assert torch.equal(mask[..., :3], mask0[..., :3]) and torch.equal(obs[..., :3], obs0[..., :3])
  rep = self.model.motion_rep
  ok = torch.isfinite(heading)
  assert torch.equal(mask[:, 4:16, 3].bool(), ok) and torch.equal(mask[:, 4:16, 4].bool(), ok)
  raw = obs[:, 4:16, 3:5] * rep.std[3:5] + rep.mean[3:5]
  expect = torch.stack((torch.cos(heading), torch.sin(heading)), -1)
  assert torch.allclose(raw[ok], expect[ok], atol=1e-6)
  assert (obs[:, 4:16, 3:5][~ok] == 0).all()
  with pytest.raises(ValueError):
    ArdyBackend.conditions(self, path, 4, 16, 2, heading[:, :5])


def test_kimodo_rejects_heading_targets():
  from crowdmotion.backends.kimodo import KimodoBackend

  req = MotionRequest(text_feat=torch.zeros(1, 1, 4), seeds=torch.zeros(1, dtype=torch.long),
                      counters=torch.zeros(1, dtype=torch.long), root_path=torch.zeros(1, 4, 2),
                      root_heading=torch.zeros(1, 4))
  with pytest.raises(NotImplementedError):
    KimodoBackend.generate(SimpleNamespace(), req)


class PathFollower:
  """A fake ARDY that walks the requested path exactly, facing the heading targets
  (or the path tangent without them). Joints: root plus two rotated offsets."""

  device = torch.device("cpu")
  fps = FPS
  frames_per_token = 4
  path_frames = 16
  joint_names = ["Hips", "LeftFoot", "RightFoot"]

  def __init__(self):
    self.requests = []

  def empty_text(self, batch):
    return torch.zeros(batch, 1, 4)

  def generate(self, req):
    self.requests.append(req)
    path = req.root_path
    b, f = path.shape[:2]
    if req.root_heading is not None:
      heading = torch.nan_to_num(req.root_heading, nan=0.0)
    else:
      d = torch.diff(path, dim=1, prepend=path[:, :1])
      heading = torch.atan2(d[..., 0], d[..., 1])
    root = torch.stack((path[..., 0], torch.full_like(path[..., 0], 0.9), path[..., 1]), -1)
    side = torch.stack((torch.cos(heading), torch.zeros_like(heading), -torch.sin(heading)), -1) * 0.1
    joints = torch.stack((root, root + side - torch.tensor([0, 0.85, 0]), root - side - torch.tensor([0, 0.85, 0])), 2)
    return MotionChunk(
      features=torch.zeros(b, f, 6), root_pos=root, local_rot=torch.eye(3).expand(b, f, 3, 3, 3).clone(),
      posed_joints=joints, contacts=torch.zeros(b, f, 2, dtype=torch.bool), heading=heading, fps=FPS)


def test_generate_streams_turns_records_targets():
  cfg = {"edge_frames": 8, "history": 4, "duration": 6.0, "seed": 1, "command_source": "turns"}
  backend = PathFollower()
  out = generate.generate_streams(backend, cfg, 0, 5)
  edges = int(6.0 * FPS) // 8
  assert out["heading_target"].shape == (5, edges * 8)
  for key in ("cmd_rate", "cmd_angle", "cmd_radius", "cmd_segment", "cmd_mode"):
    assert out[key].shape == (5, edges)
  assert all(req.root_heading is not None and req.root_heading.shape == (5, 16) for req in backend.requests)
  # The fake follows the targets, so the executed heading is the target on every frame.
  assert torch.allclose(wrap(out["heading"] - out["heading_target"]), torch.zeros(5, edges * 8), atol=1e-5)
  off = PathFollower()
  cfg_off = dict(cfg, commands={"heading_targets": False})
  out_off = generate.generate_streams(off, cfg_off, 0, 5)
  assert all(req.root_heading is None for req in off.requests)
  assert torch.equal(out_off["cmd_mode"], out["cmd_mode"])  # same segment draws either way
  rand = generate.generate_streams(PathFollower(), dict(cfg, command_source="random"), 0, 5)
  assert set(rand) == {"features", "posed_joints", "root_pos", "heading", "contacts", "local_rot", "cmd_speed",
                       "cmd_heading", "cmd_mode", "seeds"}
  with pytest.raises(ValueError):
    generate.generate_streams(PathFollower(), dict(cfg, command_source="spin"), 0, 5)


def test_library_configs_parse_for_their_command_source():
  import json
  from pathlib import Path

  from crowdmotion.behavior.commands import CommandRanges
  from crowdmotion.core.config import check_keys

  root = Path(__file__).resolve().parents[2]
  for path in sorted((root / "configs/library").glob("*.json")):
    cfg = json.loads(path.read_text())
    check_keys(cfg, generate.CONFIG_KEYS, "library config")
    source = cfg.get("command_source", "random")
    assert source in generate.COMMAND_SOURCES, path
    (TurnRanges if source == "turns" else CommandRanges).from_dict(cfg.get("commands"))
