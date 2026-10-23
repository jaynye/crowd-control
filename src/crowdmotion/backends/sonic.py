"""NVIDIA SONIC's universal-token tracker for the Unitree G1 (GEAR-SONIC "Default"),
frozen and batched in torch.

The weights load from the release's ONNX files (`<checkpoints>/GEAR-SONIC/model_encoder.onnx`
and `model_decoder.onnx`), and the ops repeat the exported graphs, so a batch of B rows
gives the same numbers as B single-row ONNX calls (`tests/unit/test_sonic_tracker.py`).
Only the G1 motion encoder is ported; the teleop and SMPL encoders are not.

Inputs follow the release's deploy layout (`observation_config.yaml`, gathered as in
`gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp`). Joints are in IsaacLab order
(`TO_MJCF` and `TO_ISAACLAB` convert); quaternions are wxyz.

- `encode`: the reference at the current frame and every 5th 50 Hz frame after it (10
  frames, 0.9 s): joint positions and velocities, and the reference root orientation
  relative to the robot's base orientation as the first two columns of its rotation
  matrix (row-major, 6 numbers). Returns the 64-number token (2 × 32, finite scalar
  quantized).
- `decode`: the token and 10 frames of history, oldest first: base angular velocity (body
  frame), joint positions minus `DEFAULT_ANGLES`, joint velocities, the previous raw actions,
  and gravity in the base frame. Returns 29 raw actions; the joint target is
  `DEFAULT_ANGLES + action × ACTION_SCALE` (`targets`).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

FRAMES, STEP = 10, 5  # reference frames the encoder reads, and their spacing in 50 Hz frames
HISTORY = 10  # proprioceptive frames the decoder reads
NUM_JOINTS = 29
TOKEN_DIM = 64
ENCODER_OBS, DECODER_OBS = 1762, 994

# Gather tables: `x_mjcf = x_isaaclab[..., TO_MJCF]` and `x_isaaclab = x_mjcf[..., TO_ISAACLAB]`
# (deploy `isaaclab_to_mujoco` and `mujoco_to_isaaclab` in `policy_parameters.hpp`).
TO_MJCF = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28)
TO_ISAACLAB = tuple(TO_MJCF.index(i) for i in range(NUM_JOINTS))

# MJCF order (`policy_parameters.hpp`; equal to mjlab's G1 knees-bent keyframe and action scale).
DEFAULT_ANGLES = (-0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0,
                  0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0)
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_MOTORS = ("7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020") * 2 + ("7520_14", "5020", "5020") + (
  "5020", "5020", "5020", "5020", "5020", "4010", "4010") * 2
_OMEGA = 10 * 2.0 * 3.1415926535
ACTION_SCALE = tuple(0.25 * _EFFORT[m] / (_ARMATURE[m] * _OMEGA ** 2) for m in _MOTORS)


def release_dir(root: Path | None = None) -> Path:
  from crowdmotion.core.paths import checkpoints_dir
  return Path(root) if root is not None else checkpoints_dir() / "GEAR-SONIC"


def _initializers(path: Path) -> tuple[dict, dict]:
  import onnx
  from onnx import numpy_helper

  model = onnx.load(str(path))
  weights = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in model.graph.initializer}
  consts = {n.output[0]: torch.from_numpy(numpy_helper.to_array(n.attribute[0].t).copy())
            for n in model.graph.node if n.op_type == "Constant"}
  return weights, consts


def _mlp(widths: list[int]) -> nn.Sequential:
  layers: list[nn.Module] = []
  for i, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
    layers.append(nn.Linear(a, b))
    if i < len(widths) - 2:
      layers.append(nn.SiLU())
  return nn.Sequential(*layers)


class SonicTracker(nn.Module):
  """The G1 encoder, its quantizer and the decoder (module docstring)."""

  def __init__(self, root: Path | None = None):
    super().__init__()
    root = release_dir(root)
    weights, consts = _initializers(root / "model_encoder.onnx")
    self.encoder = _mlp([640, 2048, 1024, 512, 512, TOKEN_DIM])
    linears = [m for m in self.encoder if isinstance(m, nn.Linear)]
    with torch.no_grad():
      for j, layer in zip((0, 2, 4, 6, 8), linears, strict=True):
        layer.weight.copy_(weights[f"module.encoders.g1.module.{j}.weight"])
        layer.bias.copy_(weights[f"module.encoders.g1.module.{j}.bias"])
    # Finite scalar quantization: round(tanh(z + shift) * scale - offset) / half_width.
    for name, key in (("fsq_shift", 1), ("fsq_scale", 2), ("fsq_offset", 3), ("fsq_half", 4)):
      self.register_buffer(name, consts[f"/quantizer/Constant_{key}_output_0"].float())

    weights, _ = _initializers(root / "model_decoder.onnx")
    mats = sorted((k for k, v in weights.items() if v.ndim == 2), key=lambda k: int(k.rsplit("_", 1)[1]))
    biases = [weights[f"module.decoders.g1_dyn.module.{j}.bias"] for j in (0, 2, 4, 6, 8, 10, 12)]
    self.decoder = _mlp([DECODER_OBS, 2048, 2048, 1024, 1024, 512, 512, NUM_JOINTS])
    linears = [m for m in self.decoder if isinstance(m, nn.Linear)]
    with torch.no_grad():
      for layer, mat, bias in zip(linears, mats, biases, strict=True):
        layer.weight.copy_(weights[mat].T)
        layer.bias.copy_(bias)
    self.register_buffer("default_angles", torch.tensor(DEFAULT_ANGLES))
    self.register_buffer("action_scale", torch.tensor(ACTION_SCALE))
    self.register_buffer("to_mjcf", torch.tensor(TO_MJCF))
    self.register_buffer("to_isaaclab", torch.tensor(TO_ISAACLAB))
    self.requires_grad_(False)
    self.eval()

  def encode(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, anchor_ori: torch.Tensor) -> torch.Tensor:
    """`[B, 10, 29]`, `[B, 10, 29]`, `[B, 10, 6]` -> token `[B, 64]`."""
    b = joint_pos.shape[0]
    # As exported: the 290 positions then the 290 velocities, reshaped to [10, 58].
    motion = torch.cat([joint_pos.reshape(b, -1), joint_vel.reshape(b, -1)], dim=1).reshape(b, FRAMES, 58)
    z = self.encoder(torch.cat([motion, anchor_ori], dim=-1).reshape(b, 640)).reshape(b, 2, 32)
    q = torch.tanh(z + self.fsq_shift) * self.fsq_scale - self.fsq_offset
    return (torch.round(q) / self.fsq_half).reshape(b, TOKEN_DIM)

  def decode(self, token: torch.Tensor, ang_vel: torch.Tensor, joint_pos_rel: torch.Tensor, joint_vel: torch.Tensor,
             last_action: torch.Tensor, gravity: torch.Tensor) -> torch.Tensor:
    """Token `[B, 64]` and histories `[B, 10, 3 | 29]` (oldest first) -> raw actions `[B, 29]`."""
    b = token.shape[0]
    obs = torch.cat([token, ang_vel.reshape(b, -1), joint_pos_rel.reshape(b, -1), joint_vel.reshape(b, -1),
                     last_action.reshape(b, -1), gravity.reshape(b, -1)], dim=1)
    return self.decoder(obs)

  def targets(self, action: torch.Tensor) -> torch.Tensor:
    """Raw actions (IsaacLab order) -> joint position targets in MJCF order."""
    return self.default_angles + action[:, self.to_mjcf] * self.action_scale


def encoder_obs(joint_pos: torch.Tensor, joint_vel: torch.Tensor, anchor_ori: torch.Tensor) -> torch.Tensor:
  """The full deploy encoder input `[B, 1762]` in G1 mode (mode 0; the other modes' terms zero),
  for checking `SonicTracker.encode` against `model_encoder.onnx`."""
  b = joint_pos.shape[0]
  obs = joint_pos.new_zeros(b, ENCODER_OBS)
  obs[:, 4:294] = joint_pos.reshape(b, -1)
  obs[:, 294:584] = joint_vel.reshape(b, -1)
  obs[:, 601:661] = anchor_ori.reshape(b, -1)
  return obs


# Quaternions (wxyz) for the plan buffer and the anchor orientation.

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def heading_quat(q: torch.Tensor) -> torch.Tensor:
  """The yaw part of `q` (deploy `calc_heading_quat_d`)."""
  w, x, y, z = q.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  zero = torch.zeros_like(yaw)
  return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], -1)


def quat_slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Shortest-path slerp; `t` broadcasts against the leading dimensions."""
  t = t.unsqueeze(-1)
  dot = (a * b).sum(-1, keepdim=True)
  b = torch.where(dot < 0, -b, b)
  dot = dot.abs().clamp(max=1.0)
  theta = torch.acos(dot)
  sin = torch.sin(theta)
  near = sin < 1e-6
  wa = torch.where(near, 1 - t, torch.sin((1 - t) * theta) / sin.clamp(min=1e-6))
  wb = torch.where(near, t, torch.sin(t * theta) / sin.clamp(min=1e-6))
  out = wa * a + wb * b
  return out / out.norm(dim=-1, keepdim=True)


def anchor_6d(base: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
  """`conj(base) * ref` as the first two columns of its rotation matrix, row-major `[..., 6]`."""
  w, x, y, z = quat_mul(quat_conj(base), ref).unbind(-1)
  return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                      2 * (x * z - w * y), 2 * (y * z + w * x)], -1)


PLAN_FPS, CONTROL_FPS = 30, 50
BLEND_FRAMES = 8  # cross-fade of a new plan into the playing one
LOOKAHEAD = 2  # a replan's context starts this many control frames ahead
DEFAULT_HEIGHT = 0.788740  # planner context height at initialization
WALK_TOKENS = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)  # deploy default: 6-11 tokens (0.8-1.47 s)


class PlanBuffer:
  """Per-env planned motion at the control rate, as the release's deploy keeps it: each plan
  (30 Hz MuJoCo qpos) is resampled to 50 Hz (joints and position linearly, the root quaternion
  by slerp; joints in IsaacLab order), joint velocities are forward differences, and a new plan
  generated `LOOKAHEAD` frames ahead cross-fades in over `BLEND_FRAMES`. Playback advances one
  frame per control step and holds the last frame."""

  def __init__(self, envs: int, device, frames: int = 128):
    z = lambda *s: torch.zeros(envs, frames, *s, device=device)  # noqa: E731
    self.joint_pos, self.joint_vel, self.pos, self.quat = z(NUM_JOINTS), z(NUM_JOINTS), z(3), z(4)
    self.quat[..., 0] = 1.0
    self.length = torch.zeros(envs, dtype=torch.long, device=device)
    self.frame = torch.zeros(envs, dtype=torch.long, device=device)
    self.frames = frames
    self._to_isaaclab = torch.tensor(TO_ISAACLAB, device=device)
    self._to_mjcf = torch.tensor(TO_MJCF, device=device)

  def _sample(self, ids: torch.Tensor, f: torch.Tensor):
    """Fractional frames `f` `[B, K]` of envs `ids` -> joint_pos, pos, quat (clamped to each length)."""
    last = (self.length[ids] - 1).clamp(min=0)[:, None]
    f0 = f.floor().long().clamp(max=last).clamp(min=0)
    f1 = (f0 + 1).clamp(max=last)
    w1 = (f - f0).clamp(0, 1)
    rows = ids[:, None]
    lerp = lambda x: x[rows, f0] * (1 - w1[..., None]) + x[rows, f1] * w1[..., None]  # noqa: E731
    return lerp(self.joint_pos), lerp(self.pos), quat_slerp(self.quat[rows, f0], self.quat[rows, f1], w1)

  def context(self, ids: torch.Tensor) -> torch.Tensor:
    """The planner context of envs `ids`: 4 frames at 30 Hz from `LOOKAHEAD` frames ahead,
    as MuJoCo qpos `[B, 4, 36]` (deploy `UpdateContextFromMotion`)."""
    t = (self.frame[ids, None] + LOOKAHEAD).float() / CONTROL_FPS + torch.arange(4, device=ids.device) / PLAN_FPS
    joints, pos, quat = self._sample(ids, t * CONTROL_FPS)
    return torch.cat([pos, quat, joints[..., self._to_mjcf]], -1)

  def install(self, ids: torch.Tensor, qpos: torch.Tensor, num_frames: torch.Tensor, first: torch.Tensor):
    """New plans for envs `ids`: qpos `[B, F, 36]` at 30 Hz, valid for `num_frames` `[B]`. Envs
    with `first` set take the plan as it is; the others blend it in `LOOKAHEAD` frames ahead."""
    b, dev = ids.shape[0], ids.device
    n50 = torch.floor(num_frames.float() / PLAN_FPS * CONTROL_FPS).long().clamp(min=2)
    f = torch.arange(self.frames, device=dev)[None].expand(b, -1)
    f30 = f.float() / CONTROL_FPS * PLAN_FPS
    f0 = f30.floor().long().clamp(max=num_frames[:, None] - 1)
    f1 = (f0 + 1).clamp(max=num_frames[:, None] - 1)
    w1 = (f30 - f0).clamp(0, 1)
    rows = torch.arange(b, device=dev)[:, None]
    lerp = lambda x: x[rows, f0] * (1 - w1[..., None]) + x[rows, f1] * w1[..., None]  # noqa: E731
    new_pos = lerp(qpos[..., :3])
    new_joints = lerp(qpos[..., 7:])[..., self._to_isaaclab]
    new_quat = quat_slerp(qpos[rows, f0, 3:7], qpos[rows, f1, 3:7], w1)
    nxt = torch.cat([new_joints[:, 1:], new_joints[:, -1:]], 1)
    new_vel = (nxt - new_joints) * CONTROL_FPS
    last_vel = new_vel[rows[:, 0], (n50 - 2).clamp(min=0)]
    new_vel = torch.where((f >= n50[:, None] - 1)[..., None], last_vel[:, None], new_vel)

    # Blend (deploy `CurrentFrameAdvancement`): frame k of the result is the playing plan's
    # frame current + k faded into the new plan's frame k - LOOKAHEAD.
    gen = torch.where(first, 0, LOOKAHEAD)
    length = (gen + n50).clamp(max=self.frames)
    f_old = (f + self.frame[ids, None]).clamp(max=(self.length[ids] - 1).clamp(min=0)[:, None])
    f_new = (f - gen[:, None]).clamp(min=0, max=self.frames - 1).clamp(max=(n50 - 1)[:, None])
    w_new = ((f - gen[:, None]).float() / BLEND_FRAMES).clamp(0, 1)
    w_new = torch.where(first[:, None], 1.0, w_new)
    w_old = 1 - w_new
    old = lambda x: x[ids[:, None], f_old]  # noqa: E731
    new = lambda x: x[rows, f_new]  # noqa: E731
    jp = old(self.joint_pos) * w_old[..., None] + new(new_joints) * w_new[..., None]
    jv = old(self.joint_vel) * w_old[..., None] + new(new_vel) * w_new[..., None]
    p = old(self.pos) * w_old[..., None] + new(new_pos) * w_new[..., None]
    q = quat_slerp(old(self.quat), new(new_quat), w_new)
    self.joint_pos[ids], self.joint_vel[ids], self.pos[ids], self.quat[ids] = jp, jv, p, q
    self.length[ids] = length
    self.frame[ids] = 0

  def window(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The encoder's reference for every env: joint positions, velocities and root quaternions
    at the current frame and every `STEP`-th frame after it, `[E, FRAMES, ...]`."""
    f = (self.frame[:, None] + STEP * torch.arange(FRAMES, device=self.frame.device)).clamp(
      max=(self.length - 1).clamp(min=0)[:, None])
    rows = torch.arange(f.shape[0], device=f.device)[:, None]
    return self.joint_pos[rows, f], self.joint_vel[rows, f], self.quat[rows, f]

  def advance(self) -> None:
    self.frame = torch.minimum(self.frame + 1, (self.length - 1).clamp(min=0))


def example_planner_inputs() -> dict:
  """One request at batch 1 (numpy, as `planner_sonic.onnx` takes it): standing at the
  initialization height, walking forward at 0.8 m/s. `OnnxPort` evaluates the graph's
  shape subgraphs on it; no values are frozen."""
  import numpy as np

  q = np.zeros((1, 4, 36), np.float32)
  q[:, :, 2] = DEFAULT_HEIGHT
  q[:, :, 3] = 1.0
  q[:, :, 7:] = DEFAULT_ANGLES
  return dict(
    context_mujoco_qpos=q, target_vel=np.array([0.8], np.float32), mode=np.array([2], np.int64),
    movement_direction=np.array([[1.0, 0.0, 0.0]], np.float32), facing_direction=np.array([[1.0, 0.0, 0.0]], np.float32),
    random_seed=np.array([1234], np.int64), has_specific_target=np.array([[0]], np.int64),
    specific_target_positions=np.zeros((1, 4, 3), np.float32), specific_target_headings=np.zeros((1, 4), np.float32),
    allowed_pred_num_tokens=np.array([WALK_TOKENS], np.int64), height=np.array([-1.0], np.float32),
  )


class SonicPlanner:
  """The released kinematic planner (`planner_sonic.onnx`, FP32) batched over a leading
  dimension: `OnnxPort` under `torch.vmap`. Call with the ONNX inputs, each `[B, *shape]`
  where `shape` is the graph's batch-1 shape; returns 30 Hz MuJoCo qpos `[B, 64, 36]` and the
  valid frame counts `[B]`. `stats` counts calls, plans and seconds (synchronized)."""

  def __init__(self, root: Path | None = None, device: str = "cuda"):
    import onnx

    from crowdmotion.backends.onnx_port import OnnxPort

    model = onnx.load(str(release_dir(root) / "planner_sonic.onnx"))
    with torch.inference_mode():
      self.port = OnnxPort(model, example_planner_inputs(), device=device)
    del model
    self._batched = torch.vmap(self.port)
    self.names = tuple(self.port.names)
    self.stats = {"calls": 0, "plans": 0, "seconds": 0.0}

  def __call__(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    import time

    start = time.perf_counter()
    b = inputs[self.names[0]].shape[0]
    with torch.inference_mode():
      qpos, frames = self._batched(*(inputs[k] for k in self.names))
    if qpos.is_cuda:
      torch.cuda.synchronize(qpos.device)
    self.stats["calls"] += 1
    self.stats["plans"] += b
    self.stats["seconds"] += time.perf_counter() - start
    return qpos.reshape(b, -1, 36), frames.reshape(b)
