"""LiDAR cost and fidelity of human body geometries at training scale.

  crowd bench body-geometry --config configs/bench/body_geometry_a40.json [--out runs/<UTC>_body_geometry]

Every environment is the plaza (floor, walls, scanner mast) with `humans` people
posed from the offline library and a 360 x 16 LiDAR at the scanner. Humans play
library clips in place (root XZ removed every frame) at random positions and
headings; the pose bank is drawn once per run (`pose_bank.pt`).

Geometries:
- `none`: room and scanner only (the floor of the cost).
- `capsules`: today's 25 capsules per human (`sim.capsules`).
- `rigid`: Core-skin pieces per bone as mesh geoms on mocap bodies (`sim.body`),
  ray cast by mujoco_warp like any mesh. `variants > 1` adds that many scaled
  copies of the pieces and assigns one per human through per-world `geom_dataid`,
  `geom_pos` and `geom_quat` tables (a timing stand-in for distinct morphologies).
- `pieces`: the same pieces as static Warp meshes in their bone frames, ray cast
  by a custom kernel: each ray tests its world's per-human boxes, then queries
  that human's pieces with the ray moved into each piece frame.
- `skinned`: the whole Core skin deformed by LBS every step (a Warp kernel) in
  one Warp mesh grouped per human and refit every step; each ray tests the same
  per-human boxes, then queries that human's subtree.

`pieces` and `skinned` merge their human hits with mujoco_warp's hits on the
static scene, and differ only in refitting a deformed surface versus querying
rigid pieces.

`capsules` and `rigid` go through mujoco_warp's scene BVH like mjlab's
`RayCastSensor`. `build` picks the pose that BVH is built from before it is only
refit: `compile` (mjlab today: the SAH tree over every body at its spec position,
which groups each human's bones together), or an in-place rebuild (Warp's GPU
`rebuild` is LBVH only) from the timed placement (`current`) or from another
placement (`independent`, as after humans move).

Each case runs in its own process (an OOM ends only that case) and appends a row
to `rows.jsonl`. Timings are CUDA-event medians over `steps` steps after
`warmup`: `pose` (torch: placement plus mocap poses or skinning matrices),
`kinematics` (mujoco_warp mocap to geoms), the sense stages one by one, and
`sense_graph`: the sense stages replayed as one CUDA graph, as mjlab does.
`fidelity` compares per-ray ranges against a full-resolution skinned reference
on the same poses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from crowdmotion.core.config import check_keys, load_json
from crowdmotion.core.paths import ROOT, artifact_path
from crowdmotion.core.rundir import RunDir, write_json

TOP_KEYS = {"name", "description", "library", "envs", "warmup", "steps", "seed", "bank", "lidar",
            "reference", "cases", "fidelity"}
CASE_KEYS = {"name", "geometry", "humans", "faces", "build", "variants", "envs", "constructor"}
FIDELITY_KEYS = {"envs", "humans", "steps", "reference_faces", "cases"}
GEOMETRIES = ("none", "capsules", "rigid", "pieces", "skinned")
BUILDS = ("compile", "current", "independent")
HALF_X, HALF_Y = 7.0, 5.0  # plaza half extents (MuJoCo XY)
SCANNER = (-6.0, 0.0, 1.15)
MAX_RANGE = 20.0


# Pose bank ----------------------------------------------------------------------
def build_bank(library: Path, sequences: int, frames: int, seed: int) -> dict:
  """`sequences` random library windows of `frames` frames as Y-up joint transforms,
  root XZ moved to the origin every frame (in-place playback)."""
  from crowdmotion.sim.body import joint_transforms

  lib = torch.load(library, map_location="cpu", mmap=True, weights_only=False)
  streams, length = lib["posed_joints"].shape[:2]
  gen = torch.Generator().manual_seed(seed)
  sid = torch.randint(streams, (sequences,), generator=gen)
  start = torch.randint(length - frames, (sequences,), generator=gen)
  parents = lib["meta"]["parents"]
  out = []
  for s, f in zip(sid.tolist(), start.tolist()):
    joints = lib["posed_joints"][s, f:f + frames].clone()
    joints[..., [0, 2]] -= joints[:, :1, [0, 2]]
    out.append(joint_transforms(lib["local_rot"][s, f:f + frames], joints, parents))
  return {
    "transforms": torch.stack(out),  # [B, T, J, 4, 4]
    "joint_names": list(lib["meta"]["joint_names"]),
    "parents": list(parents),
    "streams": sid, "starts": start,
  }


def placements(envs: int, humans: int, seed: int, spacing: float = 0.8, scanner_gap: float = 1.0):
  """Per-human MuJoCo `(x, y, yaw)` `[E, M, 3]`: uniform in the plaza, `spacing` apart."""
  rng = np.random.default_rng(seed)
  out = np.zeros((envs, humans, 3))
  lo, hi = np.array([-HALF_X + 0.5, -HALF_Y + 0.5]), np.array([HALF_X - 0.5, HALF_Y - 0.5])
  for e in range(envs):
    placed = []
    while len(placed) < humans:
      p = rng.uniform(lo, hi)
      if np.hypot(p[0] - SCANNER[0], p[1] - SCANNER[1]) < scanner_gap:
        continue
      if all(np.hypot(*(p - q)) >= spacing for q in placed):
        placed.append(p)
    out[e, :, :2] = placed
  out[..., 2] = rng.uniform(-math.pi, math.pi, (envs, humans))
  return out


def placement_matrices(place: np.ndarray, device) -> torch.Tensor:
  """`[E*M, 4, 4]`: Y-up library frame to a MuJoCo world pose at `(x, y, yaw)`."""
  from crowdmotion.sim.body import YUP_TO_MUJOCO

  p = torch.as_tensor(place.reshape(-1, 3), dtype=torch.float32)
  c, s = p[:, 2].cos(), p[:, 2].sin()
  m = torch.zeros(len(p), 4, 4)
  m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = c, -s, s, c
  m[:, 2, 2] = m[:, 3, 3] = 1
  m[:, 0, 3], m[:, 1, 3] = p[:, 0], p[:, 1]
  conv = torch.eye(4)
  conv[:3, :3] = torch.as_tensor(YUP_TO_MUJOCO, dtype=torch.float32)
  return (m @ conv).to(device)


def box_bounds(skin, parents, geometry: str, scales) -> tuple[list[int], list[float]]:
  """`(joint, radius)` pairs whose spheres bound a posed human: the box of
  `joints[joint] +- radius` holds the surface for any pose.

  Every surface point a bone moves stays within `r` of a segment from its joint to one
  of its child joints (both rigid with the bone); an LBS vertex is a weighted mean of
  such points. `r` adds scaling about the joint by `scales` (variants) and 5 cm slack
  for the bind skeleton vs the library skeleton (they differ by up to 2 cm)."""
  from crowdmotion.sim.body import rigid_pieces

  joint = skin.bind[:, :3, 3]
  if geometry == "pieces":
    sets = {p.bone: p.vertices @ skin.bind[p.bone, :3, :3].T + joint[p.bone] for p in rigid_pieces(skin)}
  else:
    sets = {j: skin.vertices[((skin.bones == j) & (skin.weights > 0)).any(1)].astype(np.float64)
            for j in range(len(joint))}
  grow = float(max(abs(sc - 1) for sc in scales))
  pairs, radius = [], []
  for j, x in sets.items():
    if not len(x):
      continue
    ends = [c for c, par in enumerate(parents) if par == j] or [j]
    d = np.full(len(x), np.inf)
    for c in ends:  # distance to the segment joint j -> child c
      a, ab = joint[j], joint[c] - joint[j]
      t = np.clip((x - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
      d = np.minimum(d, np.linalg.norm(x - (a + t[:, None] * ab), axis=1))
    r = float((d + grow * np.linalg.norm(x - joint[j], axis=1)).max()) + 0.05
    for c in sorted({j, *ends}):
      pairs.append(c)
      radius.append(r)
  return pairs, radius


# Scene ----------------------------------------------------------------------------
def scene_spec(geometry: str, humans: int, bank: dict, skin, variants: int, device="cpu"):
  """Plaza, scanner and humans as an MjSpec; returns `(spec, body)` where `body` poses humans."""
  import mujoco

  from crowdmotion.sim.body import RigidBody, rigid_pieces
  from crowdmotion.sim.capsules import CapsuleSkeleton
  from crowdmotion.sim.scene import COLORS, room_boxes

  spec = mujoco.MjSpec()
  for name, pos, size, rgba in room_boxes(HALF_X, HALF_Y):
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=rgba,
                            group=0 if name == "floor" else 2)
  scanner = spec.worldbody.add_body(name="scanner", pos=SCANNER)
  scanner.add_geom(name="mast", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.04, 0.55, 0), pos=(0, 0, -0.6), group=2)
  scanner.add_geom(name="housing", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.08, 0, 0), group=2)
  body = None
  if geometry == "capsules":
    rest = bank["transforms"][0, 0, :, :3, 3]
    body = CapsuleSkeleton.from_rest({"joint_names": bank["joint_names"], "parents": bank["parents"]}, rest, device)
    body.add_to_spec(spec, humans, COLORS)
  elif geometry == "rigid":
    body = RigidBody(rigid_pieces(skin), device)
    scales = np.linspace(0.85, 1.15, variants) if variants > 1 else (1.0,)
    body.add_to_spec(spec, humans, COLORS, scales=scales)
  return spec, body


# Warp kernels for the skinned surface and per-piece rigid meshes ------------------
_KERNELS: dict = {}


def kernels():
  if _KERNELS:
    return _KERNELS
  import warp as wp

  @wp.kernel
  def skin_points(
    mats: wp.array2d(dtype=wp.mat44), rest: wp.array2d(dtype=wp.vec3), variant: wp.array(dtype=int),
    bones: wp.array2d(dtype=int), weights: wp.array2d(dtype=float), nvert: int, points: wp.array(dtype=wp.vec3),
  ):
    h, v = wp.tid()
    p = rest[variant[h], v]
    out = wp.vec3(0.0, 0.0, 0.0)
    for k in range(bones.shape[1]):
      w = weights[v, k]
      if w > 0.0:
        out += w * wp.transform_point(mats[h, bones[v, k]], p)
    points[h * nvert + v] = out

  @wp.func
  def box_hit(o: wp.vec3, d: wp.vec3, lo: wp.vec3, hi: wp.vec3, tmax: float) -> bool:
    """Slab test: does the ray `o + t d`, `0 <= t <= tmax`, meet the box?"""
    t0 = float(0.0)
    t1 = tmax
    for a in range(3):
      if wp.abs(d[a]) < 1.0e-9:
        if o[a] < lo[a] or o[a] > hi[a]:
          return False
      else:
        inv = 1.0 / d[a]
        ta = (lo[a] - o[a]) * inv
        tb = (hi[a] - o[a]) * inv
        t0 = wp.max(t0, wp.min(ta, tb))
        t1 = wp.min(t1, wp.max(ta, tb))
    return t0 <= t1

  @wp.kernel
  def group_roots(mesh: wp.uint64, roots: wp.array(dtype=int)):
    i = wp.tid()
    roots[i] = wp.mesh_get_group_root(mesh, i)

  @wp.kernel
  def human_rays(
    mesh: wp.uint64, roots: wp.array(dtype=int), lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3),
    humans: int, max_range: float,
    pnt: wp.array2d(dtype=wp.vec3), vec: wp.array2d(dtype=wp.vec3), scene: wp.array2d(dtype=float),
    dist: wp.array2d(dtype=float), human: wp.array2d(dtype=int),
  ):
    w, r = wp.tid()
    o = pnt[w, r]
    d = vec[w, r]
    best = scene[w, r]
    if best < 0.0:
      best = max_range
    hit = int(-1)
    for i in range(humans):
      h = w * humans + i
      if box_hit(o, d, lo[h], hi[h], best):
        q = wp.mesh_query_ray(mesh, o, d, best, roots[h])
        if q.result and q.t < best:
          best = q.t
          hit = i
    if hit < 0 and scene[w, r] < 0.0:
      best = -1.0
    dist[w, r] = best
    human[w, r] = hit

  @wp.kernel
  def piece_rays(
    meshes: wp.array2d(dtype=wp.uint64), variant: wp.array(dtype=int), inv: wp.array2d(dtype=wp.mat44),
    lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3), humans: int, max_range: float,
    pnt: wp.array2d(dtype=wp.vec3), vec: wp.array2d(dtype=wp.vec3), scene: wp.array2d(dtype=float),
    dist: wp.array2d(dtype=float), human: wp.array2d(dtype=int),
  ):
    w, r = wp.tid()
    o = pnt[w, r]
    d = vec[w, r]
    best = scene[w, r]
    if best < 0.0:
      best = max_range
    hit = int(-1)
    for i in range(humans):
      h = w * humans + i
      if box_hit(o, d, lo[h], hi[h], best):
        for p in range(meshes.shape[1]):
          m = inv[h, p]
          # Rigid transforms preserve distance along the ray.
          q = wp.mesh_query_ray(meshes[variant[h], p], wp.transform_point(m, o), wp.transform_vector(m, d), best)
          if q.result and q.t < best:
            best = q.t
            hit = i
    if hit < 0 and scene[w, r] < 0.0:
      best = -1.0
    dist[w, r] = best
    human[w, r] = hit

  _KERNELS.update(skin_points=skin_points, group_roots=group_roots, human_rays=human_rays, piece_rays=piece_rays)
  return _KERNELS


# One case -------------------------------------------------------------------------
@dataclass
class Case:
  name: str
  geometry: str
  humans: int
  envs: int
  faces: int = 0
  build: str = "compile"
  variants: int = 1
  constructor: str = "lbvh"

  @classmethod
  def parse(cls, d: dict, envs: int) -> "Case":
    check_keys(d, CASE_KEYS, "case")
    case = cls(**{"envs": envs, **d})
    if case.geometry not in GEOMETRIES or case.build not in BUILDS:
      raise ValueError(f"case {case.name}: geometry in {GEOMETRIES}, build in {BUILDS}")
    if case.geometry in ("rigid", "pieces", "skinned") and case.faces <= 0:
      raise ValueError(f"case {case.name}: {case.geometry} needs faces")
    return case


class Crowd:
  """One batched scene: mujoco_warp model and data, LiDAR, pose source, sense stages."""

  def __init__(self, case: Case, bank: dict, lidar: dict, seed: int, device: str = "cuda"):
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp
    from mjlab.sensor import LidarPatternCfg

    from crowdmotion.sim.body import decimate, load_core_skin

    self.case, self.device = case, device
    self.parents = list(bank["parents"])
    E, M = case.envs, case.humans
    self.E, self.M, self.H = E, M, E * M
    self.skin = decimate(load_core_skin(), case.faces) if case.faces else None
    spec, self.body = scene_spec(case.geometry, M, bank, self.skin, case.variants, device)
    self.mjm = mjm = spec.compile()
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    self.m = mjwarp.put_model(mjm)
    self.d = mjwarp.put_data(mjm, mjd, nworld=E, nconmax=4, njmax=4)
    if case.geometry == "rigid" and case.variants > 1:
      self._assign_variants(seed)
    self.rc = mjwarp.create_render_context(
      mjm, nworld=E, use_textures=False, use_shadows=False, enabled_geom_groups=[0, 2],
      cam_active=[False] * mjm.ncam,
    )
    if mjm.nmesh:
      # mujoco_warp bounds a mesh geom by a box centred on the geom origin with half the
      # vertex extent, but MuJoCo centres meshes on their centre of mass, so body pieces
      # poke out by up to ~3 cm and grazing rays are culled. Use max(|min|, |max|).
      half = np.stack([np.abs(mjm.mesh_vert[a:a + n]).max(0) for a, n in zip(mjm.mesh_vertadr, mjm.mesh_vertnum)])
      self.rc.mesh_bounds_size.assign(half.astype(np.float32))
    # Bank on the device; per-human clip, start frame and placement.
    self.bank = bank["transforms"].to(device)
    B, T = self.bank.shape[:2]
    gen = torch.Generator().manual_seed(seed)
    self.seq = torch.randint(B, (self.H,), generator=gen).to(device)
    self.off = torch.randint(T, (self.H,), generator=gen).to(device)
    self.place = placement_matrices(placements(E, M, seed), device)
    # LiDAR: one origin per world, the mjlab pattern.
    _, dirs = LidarPatternCfg(azimuth_samples=lidar["azimuth"], elevation_samples=lidar["elevation"]
                              ).generate_rays(None, device)
    self.nray = N = len(dirs)
    origin = torch.tensor(SCANNER, device=device).expand(E, N, 3).contiguous()
    self.pnt = wp.from_torch(origin, dtype=wp.vec3)
    self.vec = wp.from_torch(dirs.expand(E, N, 3).contiguous(), dtype=wp.vec3)
    self.dist = wp.zeros((E, N), dtype=float)
    self.geomid = wp.zeros((E, N), dtype=int)
    self.normal = wp.zeros((E, N), dtype=wp.vec3)
    scanner = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "scanner")
    self.exclude = wp.array(np.full(N, scanner), dtype=int)
    # mujoco_warp's group mask: -1 includes a group, 0 excludes it (LiDAR groups 0 and 2).
    self.groups = wp.types.vector(length=6, dtype=float)(-1, 0, -1, 0, 0, 0)
    human_body = np.array([mjm.body(int(b)).name.startswith("h") for b in mjm.geom_bodyid])
    self.human_geom = torch.as_tensor(human_body, device=device)
    self.mocap_pos = wp.to_torch(self.d.mocap_pos)
    self.mocap_quat = wp.to_torch(self.d.mocap_quat)
    self.step = 0
    if case.geometry in ("pieces", "skinned"):
      self._init_boxes()
    if case.geometry == "skinned":
      self._init_skinned(seed)
    if case.geometry == "pieces":
      self._init_pieces(seed)

  # Variants: per-world mesh, position and orientation tables for every human piece.
  def _assign_variants(self, seed: int) -> None:
    import mujoco
    import warp as wp

    mjm, E, M, V = self.mjm, self.E, self.M, self.case.variants
    pieces = self.body.names
    mesh_id = np.array([[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MESH, f"body_v{v:02d}_{p}") for p in pieces]
                        for v in range(V)])
    geoms = np.array([[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, f"h{i:02d}_{p}_geom") for p in pieces]
                      for i in range(M)])
    pick = np.random.default_rng(seed + 7).integers(V, size=(E, M))
    did = np.tile(mjm.geom_dataid, (E, 1))
    did[:, geoms] = mesh_id[pick][:, :, :]  # [E, M, P]
    pos = np.tile(mjm.geom_pos, (E, 1, 1))
    quat = np.tile(mjm.geom_quat, (E, 1, 1))
    pos[:, geoms] = mjm.mesh_pos[did[:, geoms]]
    quat[:, geoms] = mjm.mesh_quat[did[:, geoms]]
    self.m.geom_dataid = wp.array(did, dtype=int)
    self.m.geom_pos = wp.array(pos.astype(np.float32), dtype=wp.vec3)
    self.m.geom_quat = wp.array(quat.astype(np.float32), dtype=wp.quat)

  def _init_boxes(self) -> None:
    """Per-human boxes for the custom ray kernels. Every surface point a bone moves
    stays within `r_j` of a segment from its joint to one of its child joints (rigid
    with the bone), so the posed surface lies in the bounds of those joints grown by
    `r_j`; an LBS vertex is a weighted mean of such points and lies there too. `r_j`
    adds the variant scaling about the joint and 5 cm slack (bind vs library skeleton)."""
    import warp as wp

    pairs, radius = box_bounds(self.skin, self.parents, self.case.geometry, self.scales())
    self.box_joint = torch.as_tensor(pairs, device=self.device)
    self.box_radius = torch.as_tensor(radius, dtype=torch.float32, device=self.device)[:, None]
    self.lo_t = torch.zeros(self.H, 3, device=self.device)
    self.hi_t = torch.zeros(self.H, 3, device=self.device)
    self.lo = wp.from_torch(self.lo_t, dtype=wp.vec3)
    self.hi = wp.from_torch(self.hi_t, dtype=wp.vec3)
    self.hdist = wp.zeros((self.E, self.nray), dtype=float)
    self.hhuman = wp.zeros((self.E, self.nray), dtype=int)

  def scales(self):
    """Variant scales of the body about each joint (bone-local coordinates)."""
    V = self.case.variants
    return np.linspace(0.85, 1.15, V) if V > 1 else (1.0,)

  def variant_pick(self, seed: int) -> np.ndarray:
    """`[H]` variant index per human (the same draw as the rigid geom tables)."""
    return np.random.default_rng(seed + 7).integers(self.case.variants, size=(self.E, self.M)).reshape(-1)

  def _init_pieces(self, seed: int) -> None:
    """Static per-(variant, piece) Warp meshes in bone frames; per-human inverse piece transforms."""
    import warp as wp

    from crowdmotion.sim.body import RigidBody, rigid_pieces

    self.body = RigidBody(rigid_pieces(self.skin), self.device)
    scales = self.scales()
    self.piece_meshes = [
      [wp.Mesh(points=wp.array((p.vertices * sc).astype(np.float32), dtype=wp.vec3),
               indices=wp.array(p.faces.astype(np.int32).ravel(), dtype=wp.int32),
               bvh_constructor=self.case.constructor)
       for p in self.body.pieces]
      for sc in scales
    ]
    self.mesh_ids = wp.array([[m.id for m in row] for row in self.piece_meshes], dtype=wp.uint64)
    self.variant = wp.array(self.variant_pick(seed).astype(np.int32), dtype=int)
    self.inv_t = torch.zeros(self.H, len(self.body.pieces), 4, 4, device=self.device)
    self.inv_t[..., 3, 3] = 1
    self.inv = wp.from_torch(self.inv_t, dtype=wp.mat44)

  def _init_skinned(self, seed: int) -> None:
    import warp as wp

    s, H = self.skin, self.H
    self.nvert = V = len(s.vertices)
    # Variants scale the body about each joint, like the rigid pieces: rest vertices
    # and bind translations by the same factor, so bone-local coordinates scale.
    binds, rests = [], []
    for sc in self.scales():
      b = s.bind.copy()
      b[:, :3, 3] *= sc
      binds.append(np.linalg.inv(b))
      rests.append(s.vertices * sc)
    inv_bind = torch.as_tensor(np.stack(binds), dtype=torch.float32, device=self.device)  # [Vv, J, 4, 4]
    pick = self.variant_pick(seed)
    self.inv_bind = inv_bind[0] if len(binds) == 1 else inv_bind[torch.as_tensor(pick, device=self.device)]
    self.variant = wp.array(pick.astype(np.int32) if len(binds) > 1 else np.zeros(H, np.int32), dtype=int)
    self.mats_t = torch.zeros(H, len(s.bind), 4, 4, device=self.device)
    self.mats = wp.from_torch(self.mats_t, dtype=wp.mat44)
    self.rest = wp.array(np.stack(rests).astype(np.float32), dtype=wp.vec3)
    self.bones = wp.array(s.bones.astype(np.int32), dtype=int)
    self.weights = wp.array(s.weights, dtype=float)
    self.points = wp.zeros(H * V, dtype=wp.vec3)
    faces = torch.as_tensor(s.faces, dtype=torch.int32, device=self.device)
    offs = (torch.arange(H, device=self.device, dtype=torch.int32) * V)[:, None, None]
    self.indices = wp.from_torch((faces[None] + offs).reshape(-1).contiguous(), dtype=wp.int32)
    groups = torch.arange(H, device=self.device, dtype=torch.int32).repeat_interleave(len(s.faces))
    self.tri_groups = wp.from_torch(groups.contiguous(), dtype=wp.int32)
    # Build from a walking pose at the timed placement, not the bind pose.
    torch.matmul(self.transforms(0), self.inv_bind, out=self.mats_t)
    self._skin()
    wp.synchronize()
    t0 = time.perf_counter()
    self.mesh = wp.Mesh(points=self.points, indices=self.indices, groups=self.tri_groups,
                        bvh_constructor=self.case.constructor)
    self.roots = wp.zeros(H, dtype=int)
    wp.launch(kernels()["group_roots"], dim=H, inputs=[self.mesh.id], outputs=[self.roots])
    wp.synchronize()
    self.mesh_build_s = time.perf_counter() - t0

  # Stages ------------------------------------------------------------------------------
  def transforms(self, k: int, place: torch.Tensor | None = None) -> torch.Tensor:
    """MuJoCo-frame joint transforms `[H, J, 4, 4]` at step `k`."""
    T = self.bank.shape[1]
    frames = self.bank[self.seq, (self.off + k) % T]
    return (self.place if place is None else place)[:, None] @ frames

  def pose(self, k: int, place: torch.Tensor | None = None) -> None:
    if self.case.geometry != "none":
      self.apply(self.transforms(k, place))
      self.boxes()

  def boxes(self) -> None:
    """Per-human boxes from the last applied joint transforms (custom ray kernels only)."""
    if self.case.geometry in ("pieces", "skinned"):
      joints = self._t[:, self.box_joint, :3, 3]
      torch.amin(joints - self.box_radius, 1, out=self.lo_t)
      torch.amax(joints + self.box_radius, 1, out=self.hi_t)

  def apply(self, t: torch.Tensor) -> None:
    """Geometry inputs from joint transforms: mocap poses, or skinning matrices."""
    geometry = self.case.geometry
    if geometry == "none":
      return
    self._t = t
    if geometry == "pieces":
      tb = t[:, self.body.bone]
      r = tb[..., :3, :3].transpose(-1, -2)
      self.inv_t[..., :3, :3] = r
      self.inv_t[..., :3, 3] = -(r @ tb[..., :3, 3:])[..., 0]
      return
    if geometry == "skinned":
      torch.matmul(t, self.inv_bind, out=self.mats_t)
      return
    if geometry == "capsules":
      pos, quat = self.body.poses(t[..., :3, 3])
    else:
      pos, quat = self.body.poses(t)
    self.mocap_pos.copy_(pos.reshape(self.E, -1, 3))
    self.mocap_quat.copy_(quat.reshape(self.E, -1, 4))

  def kinematics(self) -> None:
    import mujoco_warp as mjwarp

    if self.case.geometry in ("capsules", "rigid"):
      mjwarp.kinematics(self.m, self.d)

  def refit(self) -> None:
    import mujoco_warp as mjwarp

    mjwarp.refit_bvh(self.m, self.d, self.rc)

  def rays(self) -> None:
    import mujoco_warp as mjwarp

    mjwarp.rays(self.m, self.d, self.pnt, self.vec, self.groups, True, self.exclude,
                self.dist, self.geomid, self.normal, self.rc)

  def _skin(self) -> None:
    import warp as wp

    wp.launch(kernels()["skin_points"], dim=(self.H, self.nvert),
              inputs=[self.mats, self.rest, self.variant, self.bones, self.weights, self.nvert],
              outputs=[self.points])

  def skin_refit(self) -> None:
    self.mesh.refit()

  def human_rays(self) -> None:
    import warp as wp

    wp.launch(kernels()["human_rays"], dim=(self.E, self.nray),
              inputs=[self.mesh.id, self.roots, self.lo, self.hi, self.M, MAX_RANGE, self.pnt, self.vec, self.dist],
              outputs=[self.hdist, self.hhuman])

  def piece_rays(self) -> None:
    import warp as wp

    wp.launch(kernels()["piece_rays"], dim=(self.E, self.nray),
              inputs=[self.mesh_ids, self.variant, self.inv, self.lo, self.hi, self.M, MAX_RANGE,
                      self.pnt, self.vec, self.dist],
              outputs=[self.hdist, self.hhuman])

  def stages(self) -> list[tuple[str, callable]]:
    out = [("refit", self.refit), ("rays", self.rays)]
    if self.case.geometry == "skinned":
      out += [("skin", self._skin), ("skin_refit", self.skin_refit), ("human_rays", self.human_rays)]
    if self.case.geometry == "pieces":
      out += [("human_rays", self.piece_rays)]
    return out

  def sense(self) -> None:
    for _, fn in self.stages():
      fn()

  def ranges(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-ray range (-1: no hit) and whether a human was hit, `[E, N]`."""
    import warp as wp

    if self.case.geometry in ("pieces", "skinned"):
      return wp.to_torch(self.hdist).clone(), wp.to_torch(self.hhuman) >= 0
    geom = wp.to_torch(self.geomid)
    human = self.human_geom[geom.clamp(min=0)] & (geom >= 0)
    return wp.to_torch(self.dist).clone(), human

  def rebuild_bvh(self) -> float:
    """Rebuild the scene BVH from the current refit bounds; returns seconds."""
    import warp as wp
    from mujoco_warp._src.bvh import compute_bvh_group_roots

    wp.synchronize()
    t0 = time.perf_counter()
    self.rc.bvh.rebuild()
    wp.launch(compute_bvh_group_roots, dim=self.E, inputs=[self.rc.bvh.id], outputs=[self.rc.group_root])
    wp.synchronize()
    return time.perf_counter() - t0


def _ms(events) -> list[float]:
  torch.cuda.synchronize()
  return [a.elapsed_time(b) for a, b in events]


def _stats(xs: list[float]) -> dict:
  xs = sorted(xs)
  return {"median": xs[len(xs) // 2], "p90": xs[int(0.9 * (len(xs) - 1))], "min": xs[0]}


def run_case(case: Case, bank: dict, cfg: dict) -> dict:
  import warp as wp

  torch.cuda.init()
  wp.init()
  free0, total = torch.cuda.mem_get_info()
  t0 = time.perf_counter()
  # A dedicated stream shared by torch and Warp: CUDA graphs cannot capture the legacy default stream.
  tstream = torch.cuda.Stream()
  stream = wp.stream_from_torch(tstream)
  row = {"case": case.__dict__}
  with torch.cuda.stream(tstream), wp.ScopedStream(stream):
    crowd = Crowd(case, bank, cfg["lidar"], cfg.get("seed", 0))
    row["setup_s"] = time.perf_counter() - t0
    row["geoms"] = int(crowd.mjm.ngeom)
    row["meshes"] = int(crowd.mjm.nmesh)
    if crowd.skin is not None:
      row["skin_faces"] = len(crowd.skin.faces)
      row["skin_vertices"] = len(crowd.skin.vertices)
    if case.geometry in ("rigid", "pieces"):
      row["piece_faces"] = crowd.body.faces
      row["pieces"] = len(crowd.body.pieces)
    if case.geometry == "skinned":
      row["mesh_build_s"] = crowd.mesh_build_s
    # Scene BVH topology.
    if case.geometry in ("capsules", "rigid") and case.build != "compile":
      place = None
      if case.build == "independent":
        place = placement_matrices(placements(case.envs, case.humans, cfg.get("seed", 0) + 101), crowd.device)
      crowd.pose(17, place)
      crowd.kinematics()
      crowd.refit()
      row["rebuild_s"] = crowd.rebuild_bvh()
    elif case.geometry in ("capsules", "rigid"):
      # Price a rebuild without keeping it (the timed run uses the compile-pose tree).
      pass

    warmup, steps = cfg["warmup"], cfg["steps"]
    # `synth`: the harness's own pose source (bank gather and placement), standing in
    # for the graph; `pose`: geometry inputs from those transforms.
    names = ["synth", "pose", "boxes", "kinematics"] + [n for n, _ in crowd.stages()]
    per = {n: [] for n in names}
    held = {}
    for k in range(warmup + steps):
      ev = {n: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for n in names}
      fns = [("synth", lambda k=k: held.__setitem__("t", crowd.transforms(k))),
             ("pose", lambda: crowd.apply(held["t"])), ("boxes", crowd.boxes), ("kinematics", crowd.kinematics),
             *crowd.stages()]
      for n, fn in fns:
        ev[n][0].record()
        fn()
        ev[n][1].record()
      if k >= warmup:
        for n, ms in zip(names, _ms([ev[n] for n in names])):
          per[n].append(ms)
    row["stages_ms"] = {n: _stats(v) for n, v in per.items()}
    dist, human = crowd.ranges()
    row["human_hit_frac"] = float(human.float().mean())
    row["hit_frac"] = float((dist >= 0).float().mean())

    # The same sense stages as one CUDA graph (mjlab captures refit + rays this way).
    with wp.ScopedCapture(stream=stream) as cap:
      crowd.sense()
    graph, total_ms = [], []
    for k in range(warmup + steps):
      a, b, c = (torch.cuda.Event(enable_timing=True) for _ in range(3))
      a.record()
      crowd.pose(k)
      crowd.kinematics()
      b.record()
      wp.capture_launch(cap.graph, stream=stream)
      c.record()
      if k >= warmup:
        torch.cuda.synchronize()
        graph.append(b.elapsed_time(c))
        total_ms.append(a.elapsed_time(c))
    row["sense_graph_ms"] = _stats(graph)
    row["total_ms"] = _stats(total_ms)
    if case.geometry in ("capsules", "rigid") and case.build == "compile":
      crowd.refit()
      row["rebuild_s"] = crowd.rebuild_bvh()  # cost of one rebuild, measured after timing
  free1, _ = torch.cuda.mem_get_info()
  row["device_used_gb"] = (free0 - free1) / 1e9
  row["torch_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
  row["device_total_gb"] = total / 1e9
  return row


# Fidelity ---------------------------------------------------------------------------
def run_fidelity(fid: dict, bank: dict, cfg: dict) -> dict:
  """Per-ray ranges of each geometry against a full-resolution skinned reference."""
  import warp as wp

  wp.init()
  E, M, S = fid["envs"], fid["humans"], fid["steps"]
  frames = [4 * k for k in range(S)]
  tstream = torch.cuda.Stream()
  stream = wp.stream_from_torch(tstream)
  out = {"envs": E, "humans": M, "frames": frames}

  def sweep(case: Case):
    crowd = Crowd(case, bank, cfg["lidar"], cfg.get("seed", 0))
    if case.geometry in ("capsules", "rigid"):
      crowd.pose(0)
      crowd.kinematics()
      crowd.refit()
      crowd.rebuild_bvh()
    dists, humans = [], []
    for k in frames:
      crowd.pose(k)
      crowd.kinematics()
      crowd.sense()
      d, h = crowd.ranges()
      dists.append(d)
      humans.append(h)
    del crowd
    return torch.stack(dists), torch.stack(humans)

  with torch.cuda.stream(tstream), wp.ScopedStream(stream):
    ref_d, ref_h = sweep(Case("reference", "skinned", M, E, faces=fid["reference_faces"]))
    out["reference_human_frac"] = float(ref_h.float().mean())
    rows = []
    for d in fid["cases"]:
      case = Case.parse({**d, "humans": M}, E)
      cd, ch = sweep(case)
      both = ref_h & ch
      err = (cd - ref_d)[both]
      either = ref_h | ch
      rows.append({
        "case": case.__dict__,
        "human_frac": float(ch.float().mean()),
        "recall": float(both.sum() / ref_h.sum().clamp(min=1)),
        "precision": float(both.sum() / ch.sum().clamp(min=1)),
        "abs_err_cm": {q: float(err.abs().quantile(v) * 100) for q, v in (("median", 0.5), ("p90", 0.9), ("p99", 0.99))}
        if len(err) else None,
        "mean_err_cm": float(err.mean() * 100) if len(err) else None,
        # Rays touching a human in either set whose range differs by more than 5 cm (or hit/miss).
        "disagree_5cm": float(((cd - ref_d).abs() > 0.05)[either].float().mean()),
      })
      print(json.dumps(rows[-1]), flush=True)
    out["cases"] = rows
  return out


# Driver ------------------------------------------------------------------------------
def validate(cfg: dict) -> list[Case]:
  """Check a config's keys and cases (fidelity cases too); returns the timed cases."""
  check_keys(cfg, TOP_KEYS, "body-geometry config")
  check_keys(cfg.get("fidelity"), FIDELITY_KEYS, "fidelity")
  for d in (cfg.get("fidelity") or {}).get("cases", []):
    Case.parse({"humans": 1, **d}, 1)
  return [Case.parse(d, cfg["envs"]) for d in cfg["cases"]]


def worker(args) -> None:
  cfg = load_json(args.run / "config.json")
  bank = torch.load(args.run / "pose_bank.pt", weights_only=False)
  if args.worker == "fidelity":
    write_json(args.run / "fidelity.json", run_fidelity(cfg["fidelity"], bank, cfg))
    return
  case = Case.parse(json.loads(args.worker), cfg["envs"])
  row = run_case(case, bank, cfg)
  with open(args.run / "rows.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
  print(json.dumps(row), flush=True)


def summarize(run: Path, cfg: dict) -> str:
  rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()] if (run / "rows.jsonl").exists() else []
  ref = (cfg.get("reference") or {}).get("step_ms")
  head = ("| case | geometry | M | faces/human | build | sense graph ms | synth ms | pose ms | kin. ms | total − synth ms |"
          + (" total / ref |" if ref else "") + " human hits | device GB | rebuild s |")
  lines = [head, "|" + "---|" * (head.count("|") - 1)]
  for r in rows:
    c = r["case"]
    faces = r.get("piece_faces") if c["geometry"] in ("rigid", "pieces") else r.get("skin_faces", "")
    st = r["stages_ms"]
    # Sensing cost at runtime: everything but the harness's pose source.
    total = r["total_ms"]["median"] - st.get("synth", {}).get("median", 0)
    lines.append(
      f"| {c['name']} | {c['geometry']} | {c['humans']} | {faces} | {c['build'] if c['geometry'] in ('capsules', 'rigid') else ''}"
      f" | {r['sense_graph_ms']['median']:.2f} | {st.get('synth', {}).get('median', 0):.2f}"
      f" | {st['pose']['median']:.2f} | {st['kinematics']['median']:.2f}"
      f" | {total:.2f} |" + (f" {total / ref:.2f} |" if ref else "")
      + f" {100 * r['human_hit_frac']:.2f}% | {r['device_used_gb']:.1f} |"
      + (f" {r['rebuild_s']:.3f} |" if "rebuild_s" in r else " |")
    )
  return "\n".join(lines) + "\n"


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--config", type=Path)
  parser.add_argument("--out", type=Path, help="run directory (default: runs/<UTC>_body_geometry)")
  parser.add_argument("--only", nargs="+", help="case names to run (default: all)")
  parser.add_argument("--no-fidelity", action="store_true")
  parser.add_argument("--worker", help=argparse.SUPPRESS)
  parser.add_argument("--run", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args(argv)
  if args.worker:
    return worker(args)
  cfg = load_json(args.config)
  cases = validate(cfg)
  if args.only:
    cases = [c for c in cases if c.name in args.only]
  library = Path(cfg["library"])
  if not library.exists():
    library = artifact_path(cfg["library"])
  import mujoco
  import mujoco_warp
  import warp

  versions = {"warp": warp.__version__, "mujoco": mujoco.__version__,
              "mujoco_warp": getattr(mujoco_warp, "__version__", "?")}
  with RunDir.create("body_geometry", out=args.out, record={"config": cfg, "versions": versions}) as run:
    write_json(run.path / "config.json", cfg)
    bank_cfg = cfg["bank"]
    torch.save(build_bank(library, bank_cfg["sequences"], bank_cfg["frames"], cfg.get("seed", 0)),
               run.path / "pose_bank.pt")
    failed = []
    env = {**os.environ, "CROWD_ROOT": str(ROOT)}
    for case in cases:
      cmd = [sys.executable, "-m", "crowdmotion.bench.body_geometry", "--run", str(run.path),
             "--worker", json.dumps({k: v for k, v in case.__dict__.items() if k != "envs"} | {"envs": case.envs})]
      print(f"== {case.name}", flush=True)
      proc = subprocess.run(cmd, env=env)
      if proc.returncode:
        failed.append({"case": case.name, "returncode": proc.returncode})
        with open(run.path / "failed.jsonl", "a") as f:
          f.write(json.dumps(failed[-1]) + "\n")
      (run.path / "summary.md").write_text(summarize(run.path, cfg))
    if cfg.get("fidelity") and not args.no_fidelity:
      print("== fidelity", flush=True)
      proc = subprocess.run([sys.executable, "-m", "crowdmotion.bench.body_geometry", "--run", str(run.path),
                             "--worker", "fidelity"], env=env)
      if proc.returncode:
        failed.append({"case": "fidelity", "returncode": proc.returncode})
    run.update(failed=failed)
    print((run.path / "summary.md").read_text() if (run.path / "summary.md").exists() else "", flush=True)


if __name__ == "__main__":
  main()
