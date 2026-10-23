"""The torch port of SONIC's tracker against the release's ONNX files (skipped without them)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crowdmotion.backends import sonic

ROOT = sonic.release_dir()
needs_weights = pytest.mark.skipif(not (ROOT / "model_decoder.onnx").exists(), reason="GEAR-SONIC weights not fetched")


@pytest.fixture(scope="module")
def tracker():
  if not (ROOT / "model_decoder.onnx").exists():
    pytest.skip("GEAR-SONIC weights not fetched")
  torch.set_num_threads(2)
  return sonic.SonicTracker()


def session(name: str):
  import onnxruntime as ort

  opts = ort.SessionOptions()
  opts.intra_op_num_threads = opts.inter_op_num_threads = 1
  return ort.InferenceSession(str(ROOT / name), opts, providers=["CPUExecutionProvider"])


def rotations(b: int, g: torch.Generator) -> torch.Tensor:
  q = torch.randn(b, sonic.FRAMES, 4, generator=g)
  q = q / q.norm(dim=-1, keepdim=True)
  w, x, y, z = q.unbind(-1)
  m = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(b, sonic.FRAMES, 3, 3)
  return m[..., :2].reshape(b, sonic.FRAMES, 6)


def test_batched_encode_and_decode_match_single_row_onnx(tracker):
  g = torch.Generator().manual_seed(0)
  b = 6
  pos = torch.randn(b, sonic.FRAMES, sonic.NUM_JOINTS, generator=g) * 0.5
  vel = torch.randn(b, sonic.FRAMES, sonic.NUM_JOINTS, generator=g) * 2.0
  ori = rotations(b, g)
  token = tracker.encode(pos, vel, ori)
  enc = session("model_encoder.onnx")
  obs = sonic.encoder_obs(pos, vel, ori).numpy()
  ref = np.concatenate([enc.run(None, {"obs_dict": obs[i:i + 1]})[0] for i in range(b)])
  assert token.shape == (b, sonic.TOKEN_DIM)
  assert np.abs(token.numpy() - ref).max() <= 1.0 / 16 + 1e-6  # at most one quantization step
  assert (np.abs(token.numpy() - ref) > 1e-5).mean() < 0.01  # rounding ties aside, equal

  hist = [torch.randn(b, sonic.HISTORY, d, generator=g) for d in (3, 29, 29, 29, 3)]
  action = tracker.decode(token, *hist)
  dec = session("model_decoder.onnx")
  obs = torch.cat([token] + [h.reshape(b, -1) for h in hist], 1).numpy()
  ref = np.concatenate([dec.run(None, {"obs_dict": obs[i:i + 1]})[0].reshape(1, -1) for i in range(b)])
  assert action.shape == (b, sonic.NUM_JOINTS)
  np.testing.assert_allclose(action.numpy(), ref, atol=2e-4, rtol=1e-4)


def test_joint_maps_and_targets(tracker):
  assert sorted(sonic.TO_MJCF) == list(range(29))
  assert all(sonic.TO_MJCF[sonic.TO_ISAACLAB[i]] == i for i in range(29))
  assert sonic.TO_ISAACLAB[:3] == (0, 6, 12)  # IsaacLab: left hip pitch, right hip pitch, waist yaw
  zero = tracker.targets(torch.zeros(1, 29))
  np.testing.assert_allclose(zero[0].numpy(), sonic.DEFAULT_ANGLES, atol=1e-7)
  # IsaacLab index 0 and 1 are the left and right hip pitch (MJCF 0 and 6).
  one = torch.zeros(1, 29)
  one[0, 1] = 1.0
  moved = (tracker.targets(one) - zero)[0]
  assert moved.nonzero().flatten().tolist() == [6]


def ramp_plan(b: int, frames: int, slope: float, start: float = 0.0) -> torch.Tensor:
  """A 30 Hz plan whose root x and every joint grow linearly; identity root rotation."""
  qpos = torch.zeros(b, 64, 36)
  t = torch.arange(64).float()
  qpos[..., 0] = start + slope * t
  qpos[..., 3] = 1.0
  qpos[..., 7:] = (start + slope * t)[None, :, None]
  return qpos


def test_plan_buffer_resamples_blends_and_samples_context():
  buf = sonic.PlanBuffer(2, "cpu")
  ids = torch.arange(2)
  n30 = torch.tensor([44, 30])
  buf.install(ids, ramp_plan(2, 64, 0.03), n30, torch.tensor([True, True]))
  assert buf.length.tolist() == [73, 50]  # floor(44 / 30 * 50), floor(30 / 30 * 50)
  # 50 Hz frame f samples 30 Hz frame 0.6 f: value 0.03 * 0.6 f, velocity 0.03 * 30 = 0.9 / s.
  assert torch.allclose(buf.joint_pos[0, 10], torch.full((29,), 0.03 * 6.0))
  assert torch.allclose(buf.joint_vel[0, :71], torch.full((71, 29), 0.9), atol=1e-4)
  assert torch.allclose(buf.joint_vel[0, 71], torch.full((29,), 0.6), atol=1e-4)  # frame 72 clamps to 30 Hz frame 43
  assert torch.allclose(buf.joint_vel[0, 72], buf.joint_vel[0, 71])  # last frame keeps the one before
  for _ in range(20):
    buf.advance()
  assert buf.frame.tolist() == [20, 20]
  # Context: 4 frames at 30 Hz from 2 frames ahead (frame 22 = 30 Hz frame 13.2), MuJoCo order.
  ctx = buf.context(ids)
  assert ctx.shape == (2, 4, 36)
  assert torch.allclose(ctx[0, :, 7], 0.03 * (13.2 + torch.arange(4) * 5 / 3 * 0.6), atol=1e-5)
  # A replan blends in: the first LOOKAHEAD frames stay, then 8 frames fade, then the new plan.
  old = buf.joint_pos[0, 20:40, 0].clone()
  buf.install(ids, ramp_plan(2, 64, 0.0, start=5.0), n30, torch.tensor([False, False]))
  assert buf.frame.tolist() == [0, 0] and buf.length[0] == 2 + 73
  assert torch.allclose(buf.joint_pos[0, :3, 0], old[:3])  # weight 0 for frames 0-2
  assert torch.allclose(buf.joint_pos[0, 10:, 0][:5], torch.full((5,), 5.0))  # weight 1 from frame 10
  mid = buf.joint_pos[0, 6, 0]
  assert torch.isclose(mid, 0.5 * old[6] + 0.5 * 5.0)
  for _ in range(200):
    buf.advance()
  assert buf.frame[0] == buf.length[0] - 1  # holds the last frame
  jp, jv, q = buf.window()
  assert jp.shape == (2, sonic.FRAMES, 29) and torch.allclose(q[..., 0], torch.ones(2, sonic.FRAMES))


def test_anchor_6d_of_equal_rotations_is_identity():
  q = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
  assert torch.allclose(sonic.anchor_6d(q, q), torch.tensor([1.0, 0, 0, 1, 0, 0]).expand(5, 6), atol=1e-6)
  yaw = torch.tensor([0.0, 0.0, 0.0, 1.0])  # 180 degrees about z
  r = sonic.anchor_6d(torch.tensor([[1.0, 0, 0, 0]]), yaw[None])
  assert torch.allclose(r, torch.tensor([[-1.0, 0, 0, -1, 0, 0]]), atol=1e-6)
