"""ARDY adapter: one batched autoregressive window per call.

Mirrors `Ardy.autoregressive_step` (ardy/ardy/model/ardy_model.py) using the
model's own window, encode and decode methods, with three additions: per-human
initial noise, per-stage CUDA timing, and only the new frames returned (the
upstream call returns history + horizon).

The window seen by the model is `history + horizon + future` frames; root-path
constraints may cover the horizon and the future context. History, horizon and
future must be multiples of `frames_per_token` (4).
"""

from __future__ import annotations

import torch

from crowdmotion.backends.base import (
  MotionChunk,
  MotionRequest,
  check_precision,
  denoise_autocast,
  identity,
  injected_randn,
  tf32_matmul,
)
from crowdmotion.backends.registry import ArdyConfig
from crowdmotion.backends.text import empty_text
from crowdmotion.core.rng import per_human_noise
from crowdmotion.core.skeleton import SkeletonInfo
from crowdmotion.core.timing import StageTimer


class ArdyBackend:
  name = "ardy"

  def __init__(self, cfg: ArdyConfig | None = None):
    import ardy.model.ardy_model as ardy_model_module
    from ardy.model.cfg import AutoLatentClassifierFreeGuidedModel
    from ardy.model.load_model import load_model

    cfg = cfg or ArdyConfig()
    self._module = ardy_model_module
    self.model = load_model(cfg.model, device=cfg.device, text_encoder=False)
    inner = self.model.denoiser.model
    self.model.denoiser = AutoLatentClassifierFreeGuidedModel(inner, cfg_type=cfg.cfg_type)
    self.model_name = cfg.model
    self.device = torch.device(cfg.device)
    self.cfg_type = cfg.cfg_type
    self.token_dim = inner.nframe_root_dim + inner.latent_embedding_dim
    self.configure(cfg)

  def configure(self, cfg: ArdyConfig) -> None:
    """Apply `cfg` to the loaded model (same checkpoint and device)."""
    if cfg.model != self.model_name or torch.device(cfg.device) != self.device:
      raise ValueError("configure() cannot change the model or device; make a new backend")
    check_precision(cfg.precision)
    if cfg.conditions not in ("vectorized", "library"):
      raise ValueError("conditions must be 'vectorized' or 'library'")
    nfpt = self.model.num_frames_per_token
    if cfg.future_frames % nfpt:
      raise ValueError(f"future_frames must be a multiple of {nfpt}")
    if cfg.cfg_type != self.cfg_type:
      self.set_cfg_type(cfg.cfg_type)
    self.cfg = cfg
    self.cfg_weight = tuple(cfg.cfg_weight)
    self.steps = cfg.steps or self.model.diffusion.num_base_steps
    self.precision = cfg.precision
    self.conditions_mode = cfg.conditions
    self.timing = cfg.timing
    self.future_frames = cfg.future_frames

  def set_cfg_type(self, cfg_type: str) -> None:
    """Swap guidance: separated (3 passes), regular (2) or nocfg (1)."""
    from ardy.model.cfg import AutoLatentClassifierFreeGuidedModel

    self.model.denoiser = AutoLatentClassifierFreeGuidedModel(self.model.denoiser.model, cfg_type=cfg_type)
    self.cfg_type = cfg_type

  # Native metadata ----------------------------------------------------------
  @property
  def fps(self) -> float:
    return float(self.model.motion_rep.fps)

  @property
  def horizon(self) -> int:
    return self.model.gen_horizon_len

  @property
  def frames_per_token(self) -> int:
    return self.model.num_frames_per_token

  @property
  def path_frames(self) -> int:
    """Root-path frames a request may constrain: the horizon plus future context."""
    return self.horizon + self.future_frames

  @property
  def frames_per_call(self) -> int:
    return self.horizon

  @property
  def joint_names(self) -> list[str]:
    return list(self.model.skeleton.bone_order_names)

  def skeleton_info(self) -> dict:
    return SkeletonInfo.from_upstream(self.model.skeleton, self.joint_names).as_dict()

  def empty_text(self, batch: int) -> torch.Tensor:
    return empty_text(batch, self.device)

  # Conditions ---------------------------------------------------------------
  def conditions(self, root_path, history_frames: int, num_frames: int, batch: int):
    """Normalized `(motion_mask, observed_motion)` for a world XZ root path."""
    if root_path is None:
      return None, None
    rep = self.model.motion_rep
    n = min(root_path.shape[1], num_frames - history_frames)
    if self.conditions_mode == "library":
      from ardy.constraints import Root2DConstraintSet

      frames = torch.arange(history_frames, history_frames + n)
      masks, observed = [], []
      for b in range(batch):
        constraint = Root2DConstraintSet(self.model.skeleton, frames, root_path[b, :n])
        obs, mask = rep.create_conditions_from_constraints(
          [constraint], length=num_frames, to_normalize=False, device=self.device
        )
        observed.append(rep.normalize(obs) * mask)
        masks.append(mask)
      return torch.stack(masks), torch.stack(observed)
    root = rep.slice_dict["root_pos"]
    x, z = root.start, root.start + 2
    obs = torch.zeros(batch, num_frames, rep.motion_rep_dim, device=self.device)
    mask = torch.zeros_like(obs)
    window = slice(history_frames, history_frames + n)
    obs[:, window, x] = root_path[:, :n, 0]
    obs[:, window, z] = root_path[:, :n, 1]
    mask[:, window, x] = 1.0
    mask[:, window, z] = 1.0
    return mask, rep.normalize(obs) * mask

  # Generation ---------------------------------------------------------------
  @torch.no_grad()
  def generate(self, req: MotionRequest) -> MotionChunk:
    m = self.model
    rep = m.motion_rep
    batch = req.batch
    nfpt = m.num_frames_per_token
    horizon = m.gen_horizon_len
    hist_len = 0 if req.history is None else req.history.shape[1]
    if hist_len % nfpt:
      raise ValueError(f"history length must be a multiple of {nfpt}")
    num_frames = hist_len + horizon + self.future_frames
    timer = StageTimer(self.timing)
    with tf32_matmul(self.precision == "tf32"):
      with timer.stage("conditions"):
        motion_mask, observed = self.conditions(req.root_path, hist_len, num_frames, batch)
      with timer.stage("noise"):
        noise = per_human_noise(
          req.seeds, req.counters, (horizon // nfpt, self.token_dim), self.device
        )
      text_feat = req.text_feat
      text_pad_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=self.device)
      steps_t = torch.tensor([self.steps], device=self.device)
      m.diffusion.calc_diffusion_vars(m.diffusion.space_timesteps(steps_t[0])[0])
      with timer.stage("encode_history"):
        if req.history is not None:
          hist_seq, global_transl, first_heading = m._encode_init_history(req.history, batch)
        else:
          hist_seq = None
          global_transl = torch.zeros(batch, 3, device=self.device)
          if req.start_xz is not None:
            global_transl[:, 0] = req.start_xz[:, 0]
            global_transl[:, 2] = req.start_xz[:, 1]
          first_heading = (
            req.start_heading.to(self.device).float()
            if req.start_heading is not None
            else torch.zeros(batch, device=self.device)
          )
      with timer.stage("denoise"), denoise_autocast(self.precision), injected_randn(self._module, noise):
        hist_seq = m._generate_window(
          hist_seq,
          global_transl,
          0,
          hist_len,
          num_frames,
          text_feat,
          text_pad_mask,
          first_heading,
          motion_mask,
          observed,
          steps_t,
          self.cfg_weight,
          list(range(self.steps))[::-1],
          progress_bar=identity,
        )
      with timer.stage("decode"):
        root_motion, latent_body = m.hybrid.get_root_and_latent_body_motion_from_hybrid(hist_seq)
        root_motion = self._module.translate_normalized_root_motion(root_motion, global_transl, rep)
        if m.autoencoder.encode_with_quantization:
          latent_body = m.autoencoder.requantize(latent_body)
        hist_seq = m.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(root_motion, latent_body)
        out_len = hist_len + horizon
        features = m.hybrid.get_explicit_motion_from_hybrid(
          hist_seq,
          torch.ones(batch, out_len, dtype=torch.bool, device=self.device),
          torch.full((batch,), out_len, dtype=torch.long, device=self.device),
          motion_mask=motion_mask,
        )[:, hist_len:]
      with timer.stage("kinematics"):
        unnorm = rep.unnormalize(features)
        dec = rep.inverse(unnorm, is_normalized=False)
        heading = rep.get_root_heading_angle(unnorm)
    chunk = MotionChunk(
      features=features,
      root_pos=dec["root_positions"],
      local_rot=dec["local_rot_mats"],
      posed_joints=dec["posed_joints"],
      contacts=dec["foot_contacts"],
      heading=heading,
      fps=self.fps,
      timer=timer,
    )
    return chunk
