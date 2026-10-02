"""`MotionGraph`: boundary descriptors, edge tables and candidates, and their persistence.

Tables (B = streams x K boundaries; edge id = id of its start boundary):

- `pred_pose [B, J, 3]`, `pred_vel [B, J, 3]`: body-local pose and velocity at
  each boundary (the descriptor is `pred_pose` plus `vel_weight * pred_vel`);
- `edge_root [B, L, 2]`, `edge_yaw [B, L]`, `edge_pose [B, L, J, 3]`,
  `edge_contacts [B, L, C]`, `edge_end_vel [B, 2]`: each edge's frames relative to
  its predecessor boundary;
- `valid [B]` (finite and not hunched), `root_speed [B]` (at the boundary);
- `cand [B, k]`, `cand_dist [B, k]`: candidate edges and descriptor distances
  (`crowdmotion.graph.build`).

`save` writes the runtime tables (not descriptors) and `load` reads them back
without the library or the kNN build. Blobs written before the format field
existed (all graphs saved up to 2026-10-01) load as format 1.
"""

from __future__ import annotations

import torch

from crowdmotion.core.geometry import body_local, to_local, wrap
from crowdmotion.core.rng import hash_uniform
from crowdmotion.graph.build import build_candidates
from crowdmotion.graph.config import BAKED, GraphConfig

FORMAT = 1


class MotionGraph:
  """Edge tables and the candidate table, all on one device."""

  # Persisted tables, and how they are stored (float16 positions, int32 ids).
  _TABLES = (
    "pred_pose", "pred_vel", "edge_root", "edge_yaw", "edge_pose", "edge_contacts",
    "edge_end_vel", "valid", "root_speed", "cand", "cand_dist",
  )
  _HALF = ("pred_pose", "edge_pose", "cand_dist")
  _SCALARS = ("meta", "fps", "L", "streams", "K", "joints_n", "stats", "rest_joints")

  def __init__(self, lib: dict, cfg: GraphConfig, device="cuda"):
    """Build every table from a library dict (`library.pt` layout) and run the kNN build."""
    self.cfg = cfg
    self.device = torch.device(device)
    meta = lib["meta"]
    self.fps = float(meta["fps"])
    self.L = L = int(meta["edge_frames"])
    self.meta = meta
    joints = lib["posed_joints"].to(self.device).float()
    root = lib["root_pos"].to(self.device).float()
    heading = lib["heading"].to(self.device).float()
    contacts = lib["contacts"].to(self.device).bool()
    n, t = joints.shape[:2]
    self.streams, self.K = n, t // L
    K = self.K
    joints, root, heading, contacts = joints[:, : K * L], root[:, : K * L], heading[:, : K * L], contacts[:, : K * L]

    # Boundary descriptors at p = k*L - 1, k = 1..K.
    p = torch.arange(1, K + 1, device=self.device) * L - 1
    jp, jq = joints[:, p], joints[:, p - 1]
    hp, rp = heading[:, p], root[:, p][..., [0, 2]]
    pos = body_local(jp, rp, hp)
    vel = torch.cat(
      (to_local((jp - jq)[..., [0, 2]], hp[..., None]), (jp - jq)[..., 1:2]), -1
    )[..., [0, 2, 1]] * self.fps
    self.pred_pose = pos.reshape(n * K, -1, 3)  # [B, J, 3]
    self.pred_vel = vel.reshape(n * K, -1, 3)
    self.joints_n = self.pred_pose.shape[1]
    self.desc = torch.cat((self.pred_pose.flatten(1), cfg.vel_weight * self.pred_vel.flatten(1)), 1)

    # Edge frames relative to the predecessor boundary (edge id = start boundary id).
    k_idx = torch.arange(1, K, device=self.device)
    f = (k_idx[:, None] * L + torch.arange(L, device=self.device)).reshape(-1)  # [(K-1)*L]
    ej = joints[:, f].reshape(n, K - 1, L, -1, 3)
    er = root[:, f].reshape(n, K - 1, L, 3)
    eh = heading[:, f].reshape(n, K - 1, L)
    hp_e, rp_e = hp[:, :-1], rp[:, :-1]  # predecessor of edge k is boundary k
    pad = lambda x: torch.cat((x, torch.zeros_like(x[:, :1])), 1).flatten(0, 1)  # noqa: E731
    self.edge_root = pad(to_local(er[..., [0, 2]] - rp_e[:, :, None], hp_e[..., None]))  # [B, L, 2]
    self.edge_yaw = pad(wrap(eh - hp_e[..., None]))  # [B, L]
    self.edge_pose = pad(body_local(ej, er[..., [0, 2]], eh))  # [B, L, J, 3]
    self.edge_contacts = pad(contacts[:, f].reshape(n, K - 1, L, -1))
    back = 3  # frames for the end-of-edge root velocity
    self.edge_end_vel = (self.edge_root[:, L - 1] - self.edge_root[:, L - 1 - back]) * self.fps / back

    valid = torch.zeros(n, K, dtype=torch.bool, device=self.device)
    valid[:, :-1] = torch.isfinite(ej).all(dim=(-1, -2, -3))
    self.stats = {"edges_total": int(valid.sum())}
    if cfg.hunch_ratio > 0:
      head = ej[..., meta["head_idx"], 1].mean(-1)
      ok = head >= cfg.hunch_ratio * meta["neutral_head_height"]
      valid[:, :-1] &= ok
      self.stats["edges_dropped_hunched"] = int((~ok).sum())
    self.valid = valid.flatten()
    self.stats["edges_valid"] = int(self.valid.sum())
    self.root_speed = (
      (root[:, p][..., [0, 2]] - root[:, p - 1][..., [0, 2]]).norm(dim=-1) * self.fps
    ).flatten()
    self.rest_joints = None
    build_candidates(self)

  # Persistence ------------------------------------------------------------------
  def save(self, path, rest_joints: torch.Tensor | None = None) -> None:
    """Save the runtime tables (not descriptors), so loading skips the kNN build.

    Large position tables are stored in float16 (sub-millimeter error at human
    scale) and candidate ids in int32. `rest_joints [J, 3]` (any posed frame)
    lets `CapsuleSkeleton` recover bone lengths without the library.
    """
    tables = {}
    for name in self._TABLES:
      t = getattr(self, name).detach().cpu().contiguous().clone()  # never save a view
      if name in self._HALF:
        t = t.half()
      elif name == "cand":
        t = t.int()
      tables[name] = t
    torch.save(
      {
        "format": FORMAT,
        "tables": tables,
        "cfg": self.cfg.__dict__,
        "meta": self.meta,
        "fps": self.fps,
        "L": self.L,
        "streams": self.streams,
        "K": self.K,
        "joints_n": self.joints_n,
        "stats": self.stats,
        "rest_joints": None if rest_joints is None else rest_joints.detach().cpu().float().clone(),
      },
      path,
    )

  @classmethod
  def load(cls, path, device="cuda", cfg_overrides: dict | None = None) -> "MotionGraph":
    """Load saved tables. Overrides may change scoring and correction, but not
    `neighbors`, `vel_weight` or `hunch_ratio`, which are baked into the tables."""
    return cls.from_blob(read_blob(path), device, cfg_overrides)

  @classmethod
  def from_blob(cls, blob: dict, device="cuda", cfg_overrides: dict | None = None) -> "MotionGraph":
    cfg = GraphConfig.from_dict({**blob["cfg"], **(cfg_overrides or {})})
    if any(getattr(cfg, k) != blob["cfg"][k] for k in BAKED):
      raise ValueError(f"{BAKED} are fixed when the graph is built")
    self = cls.__new__(cls)
    self.cfg, self.device = cfg, torch.device(device)
    for key in cls._SCALARS:
      setattr(self, key, blob[key])
    for name, t in blob["tables"].items():
      t = t.float() if name in cls._HALF else (t.long() if name == "cand" else t)
      setattr(self, name, t.to(self.device))
    return self

  # Skeleton and starts ----------------------------------------------------------
  @property
  def foot_idx(self) -> list[int]:
    return list(self.meta["foot_idx"])

  def skeleton_info(self) -> dict:
    return {
      "parents": list(self.meta["parents"]),
      "foot_idx": self.foot_idx,
      "joint_names": list(self.meta["joint_names"]),
      "head_idx": self.meta["head_idx"],
      "neutral_head_height": self.meta["neutral_head_height"],
    }

  def start_boundaries(self, seeds: torch.Tensor, max_speed: float = 0.15) -> torch.Tensor:
    """A near-standing boundary per human, drawn from its own stream."""
    pool = torch.nonzero(self.valid & (self.root_speed < max_speed)).squeeze(1)
    if pool.numel() == 0:
      pool = torch.nonzero(self.valid).squeeze(1)
    u = hash_uniform(seeds.to(self.device), torch.full_like(seeds, -1).to(self.device), 1)[:, 0]
    return pool[(u * pool.numel()).long().clamp(max=pool.numel() - 1)]


def read_blob(path) -> dict:
  """A saved graph as a dict in the current format (CPU tensors, nothing converted).

  Format 1 is the only format so far; blobs without a `format` key predate the
  field and are format 1. Later formats add a branch here, so old files keep loading.
  """
  blob = torch.load(path, map_location="cpu", weights_only=False)
  version = blob.get("format", 1)
  if version != FORMAT:
    raise ValueError(f"graph {path}: unsupported format {version} (this code reads {FORMAT})")
  missing = [k for k in ("tables", "cfg", *MotionGraph._SCALARS) if k not in blob]
  if missing:
    raise ValueError(f"graph {path}: missing {missing}")
  return blob
