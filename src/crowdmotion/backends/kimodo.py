"""Kimodo adapter: full-window diffusion, continued from executed frames.

Continuation follows Kimodo's own multi-prompt transition (`Kimodo._multiprompt`
in kimodo/kimodo/model/kimodo_model.py): the last `transition_frames` executed
frames become full-body and end-effector constraints at the start of a new
window, the window is translated so its first frame sits at the origin, and
the rest of the window is generated. Only the frames after the transition are
returned. `multi_prompt` stays off: batch entries are independent humans.

Constraint objects are built per human with Kimodo's own classes, which is a
Python loop over the batch; its cost is timed as the `conditions` stage.
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
from crowdmotion.backends.registry import KimodoConfig
from crowdmotion.backends.text import ZeroTextEncoder, empty_text
from crowdmotion.core.rng import per_human_noise
from crowdmotion.core.skeleton import SkeletonInfo
from crowdmotion.core.timing import StageTimer


class KimodoBackend:
  name = "kimodo"

  def __init__(self, cfg: KimodoConfig | None = None):
    import kimodo.model.kimodo_model as kimodo_model_module
    from kimodo.model.load_model import load_model

    cfg = cfg or KimodoConfig()
    check_precision(cfg.precision)
    self._module = kimodo_model_module
    self.model = load_model(cfg.model, device=cfg.device, text_encoder=ZeroTextEncoder())
    self.model_name = cfg.model
    self.device = torch.device(cfg.device)
    self.configure(cfg)

  def configure(self, cfg: KimodoConfig) -> None:
    """Apply `cfg` to the loaded model (same checkpoint and device)."""
    if cfg.model != self.model_name or torch.device(cfg.device) != self.device:
      raise ValueError("configure() cannot change the model or device; make a new backend")
    check_precision(cfg.precision)
    self.cfg = cfg
    self.steps = cfg.steps
    self.cfg_weight = list(cfg.cfg_weight)
    self.transition_frames = cfg.transition_frames
    self.new_frames = cfg.new_frames
    self.precision = cfg.precision
    self.post_processing = cfg.post_processing
    self.to_soma77 = cfg.to_soma77
    # Online stand-in for Kimodo's multi-prompt cross-fade, which rewrites the
    # previous segment's last frames (already executed online). Instead, the
    # offset between the last executed frame and the regenerated transition's
    # last frame decays linearly over the first `seam_blend_frames` new frames.
    self.seam_blend_frames = cfg.seam_blend_frames
    self.timing = cfg.timing

  @property
  def fps(self) -> float:
    return float(self.model.motion_rep.fps)

  @property
  def path_frames(self) -> int:
    """Root-path frames a request may constrain: the new frames."""
    return self.new_frames

  @property
  def frames_per_call(self) -> int:
    return self.new_frames

  @property
  def joint_names(self) -> list[str]:
    return list(self.model.skeleton.bone_order_names)

  def skeleton_info(self) -> dict:
    return SkeletonInfo.from_upstream(self.model.skeleton, self.joint_names).as_dict()

  def empty_text(self, batch: int) -> torch.Tensor:
    return empty_text(batch, self.device)

  def _constraints(self, hist_out, root_path, new_frames: int, batch: int):
    """Per-human Kimodo constraint lists over one window (world coordinates)."""
    from kimodo.constraints import (
      EndEffectorConstraintSet,
      FullBodyConstraintSet,
      Root2DConstraintSet,
    )

    skel = self.model.skeleton
    n = 0 if hist_out is None else self.transition_frames
    lists = []
    for b in range(batch):
      items = []
      if hist_out is not None:
        frames = torch.arange(n)
        root2d = hist_out["smooth_root_pos"][b, :, [0, 2]]
        items.append(
          FullBodyConstraintSet(
            skel, frames, hist_out["posed_joints"][b], hist_out["global_rot_mats"][b], root2d
          )
        )
        items.append(
          EndEffectorConstraintSet(
            skel,
            frames,
            hist_out["posed_joints"][b],
            hist_out["global_rot_mats"][b],
            root2d,
            joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
          )
        )
      if root_path is not None:
        k = min(root_path.shape[1], new_frames)
        items.append(Root2DConstraintSet(skel, torch.arange(n, n + k), root_path[b, :k]))
      lists.append(items)
    return lists

  @torch.no_grad()
  def generate(self, req: MotionRequest) -> MotionChunk:
    m = self.model
    rep = m.motion_rep
    batch = req.batch
    new_frames = self.new_frames
    timer = StageTimer(self.timing)
    with tf32_matmul(self.precision == "tf32"):
      with timer.stage("encode_history"):
        hist_out = None
        if req.history is not None:
          hist = req.history[:, -self.transition_frames :]
          hist_out = rep.inverse(hist, is_normalized=True, return_numpy=False)
          origin = hist_out["smooth_root_pos"][:, 0, [0, 2]]
          first_heading = self._module.compute_heading_angle(hist_out["posed_joints"], m.skeleton)[:, 0]
        else:
          origin = (
            req.start_xz.to(self.device).float()
            if req.start_xz is not None
            else torch.zeros(batch, 2, device=self.device)
          )
          first_heading = (
            req.start_heading.to(self.device).float()
            if req.start_heading is not None
            else torch.zeros(batch, device=self.device)
          )
      n_hist = 0 if hist_out is None else self.transition_frames
      window = n_hist + new_frames
      with timer.stage("conditions"):
        constraint_lst = self._constraints(hist_out, req.root_path, new_frames, batch)
        lengths = torch.full((batch,), window, device=self.device)
        motion_mask, observed = None, None
        if any(constraint_lst):
          observed, motion_mask = rep.create_conditions_from_constraints_batched(
            constraint_lst, lengths, to_normalize=False, device=self.device
          )
          observed = rep.translate_2d(observed, -origin) * motion_mask
          observed = rep.normalize(observed)
      with timer.stage("noise"):
        noise = per_human_noise(req.seeds, req.counters, (window, rep.motion_rep_dim), self.device)
      pad_mask = torch.ones(batch, window, dtype=torch.bool, device=self.device)
      text_pad_mask = torch.zeros(req.text_feat.shape[:2], dtype=torch.bool, device=self.device)
      with timer.stage("denoise"), denoise_autocast(self.precision), injected_randn(self._module, noise):
        motion = m._generate(
          None,
          window,
          num_denoising_steps=self.steps,
          pad_mask=pad_mask,
          first_heading_angle=first_heading,
          motion_mask=motion_mask,
          observed_motion=observed,
          cfg_weight=self.cfg_weight,
          text_feat=req.text_feat,
          text_pad_mask=text_pad_mask,
          progress_bar=identity,
        )
      with timer.stage("kinematics"):
        motion = rep.translate_2d(rep.unnormalize(motion.float()), origin)
        k = min(self.seam_blend_frames, new_frames)
        if hist_out is not None and k > 0:
          executed_last = rep.unnormalize(req.history[:, -1])
          delta = executed_last - motion[:, n_hist - 1]
          ramp = 1 - torch.arange(1, k + 1, device=self.device, dtype=motion.dtype) / (k + 1)
          motion = motion.clone()
          motion[:, n_hist : n_hist + k] += ramp[None, :, None] * delta[:, None]
        out = rep.inverse(motion, is_normalized=False, return_numpy=False)
      if self.post_processing:
        with timer.stage("postprocess"):
          corrected = self._module.post_process_motion(
            out["local_rot_mats"],
            out["root_positions"],
            out["foot_contacts"],
            m.skeleton,
            constraint_lst if any(constraint_lst) else None,
          )
          out.update(corrected)
          motion = rep(
            local_joint_rots=out["local_rot_mats"],
            root_positions=out["root_positions"],
            to_normalize=False,
            lengths=torch.full((batch,), window, device=self.device),
          )
      new = slice(n_hist, window)
      if self.to_soma77:
        with timer.stage("to_soma77"):
          out = m.skeleton.output_to_SOMASkeleton77(out)
    return MotionChunk(
      features=rep.normalize(motion[:, new]),
      root_pos=out["root_positions"][:, new],
      local_rot=out["local_rot_mats"][:, new],
      posed_joints=out["posed_joints"][:, new],
      contacts=out["foot_contacts"][:, new],
      heading=out["global_root_heading"][:, new],
      fps=self.fps,
      timer=timer,
    )
