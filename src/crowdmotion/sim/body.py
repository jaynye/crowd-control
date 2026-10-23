"""Mesh humans from ARDY's Core skin: poses, skinning, decimation and rigid pieces.

The Core skeleton (27 joints) ships one skinned body, `skin_standard.npz` in ARDY's
`cskel27` assets: 9,084 vertices, 18,152 faces and up to five bone weights per
vertex, in the motion library's Y-up frame. A pose is one rigid transform per joint
(`joint_transforms`: library joint positions plus FK of the local rotations).
Two runtime geometries come from it:

- **Skinned:** linear blend skinning of the whole surface (`lbs`).
- **Rigid pieces:** each face goes to the bone carrying most of its vertices'
  weight; each piece's holes are capped so rays cannot enter it, and each piece
  rides its bone as a MuJoCo mesh geom on a mocap body (`RigidBody`).

`decimate` reduces the face count by quadric collapse and copies each kept
vertex's weights from its nearest original vertex. Transforms passed to
`RigidBody.poses` are in MuJoCo's Z-up frame (`to_mujoco`).

The simulated crowd uses the full-resolution pieces (`core_pieces`). Its poses come
from graph joint positions plus saved joint rotations (`crowdmotion.graph.rotations`),
blended with the quaternion helpers here; `swing_to_children` then turns each
single-child bone onto its posed child, so pieces stay attached through blended
transitions. `crowdmotion.sim.rays` casts rays against the posed pieces.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from crowdmotion.core.paths import ROOT

CORE_SKIN = ROOT / "third_party/ardy/ardy/assets/skeletons/cskel27/skin_standard.npz"
# Library Y-up -> MuJoCo Z-up, the rotation `core.geometry.motion_to_mujoco` applies.
YUP_TO_MUJOCO = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@dataclass
class Skin:
  """A skinned surface on the Core skeleton (bind pose, Y-up metres)."""

  vertices: np.ndarray  # [V, 3] float32
  faces: np.ndarray  # [F, 3] int64, counter-clockwise seen from outside
  bones: np.ndarray  # [V, K] int64 joint indices
  weights: np.ndarray  # [V, K] float32, rows sum to 1
  bind: np.ndarray  # [J, 4, 4] bind-pose joint transforms
  joint_names: list[str]

  @property
  def dominant(self) -> np.ndarray:
    """[V] the bone with the largest weight on each vertex."""
    return self.bones[np.arange(len(self.bones)), self.weights.argmax(1)]

  def volume(self) -> float:
    """Signed volume (positive when faces wind outward); holes make it approximate."""
    v = self.vertices.astype(np.float64)[self.faces]
    return float(np.einsum("fi,fi->f", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6)


def load_core_skin(path: Path | str = CORE_SKIN) -> Skin:
  z = np.load(path)
  return Skin(
    vertices=z["bind_vertices"].astype(np.float32),
    faces=z["faces"].astype(np.int64),
    bones=z["lbs_indices"].astype(np.int64),
    weights=z["lbs_weights"].astype(np.float32),
    bind=z["bind_rig_transform"].astype(np.float64),
    joint_names=[str(n) for n in z["rig_joint_names"]],
  )


def decimate(skin: Skin, faces: int, aggression: float = 7.0) -> Skin:
  """Quadric decimation to about `faces` triangles; weights from the nearest original vertex."""
  import fast_simplification
  from scipy.spatial import cKDTree

  if faces >= len(skin.faces):
    return skin
  points, tris = fast_simplification.simplify(
    skin.vertices.astype(np.float64), skin.faces.astype(np.int32), target_count=faces, agg=aggression
  )
  _, near = cKDTree(skin.vertices).query(points)
  return replace(
    skin, vertices=points.astype(np.float32), faces=tris.astype(np.int64),
    bones=skin.bones[near], weights=skin.weights[near],
  )


# Poses ------------------------------------------------------------------------
def joint_transforms(local_rot: torch.Tensor, joints: torch.Tensor, parents) -> torch.Tensor:
  """Per-joint rigid transforms `[..., J, 4, 4]` (Y-up) from library local rotations
  `[..., J, 3, 3]` and joint positions `[..., J, 3]`. Rotations are re-orthonormalized
  first (the library stores them in fp16)."""
  u, _, vh = torch.linalg.svd(local_rot.float())
  local = u @ vh
  glob: list[torch.Tensor] = []
  for j, p in enumerate(parents):
    glob.append(local[..., j, :, :] if p < 0 else glob[p] @ local[..., j, :, :])
  out = torch.zeros(*joints.shape[:-1], 4, 4, dtype=torch.float32, device=joints.device)
  out[..., :3, :3] = torch.stack(glob, -3)
  out[..., :3, 3] = joints.float()
  out[..., 3, 3] = 1
  return out


def to_mujoco(transforms: torch.Tensor) -> torch.Tensor:
  """Y-up transforms `[..., 4, 4]` to MuJoCo Z-up (left-multiplied frame change)."""
  c = torch.eye(4, dtype=transforms.dtype, device=transforms.device)
  c[:3, :3] = torch.as_tensor(YUP_TO_MUJOCO, dtype=transforms.dtype)
  return c @ transforms


def skin_matrices(skin: Skin, transforms: torch.Tensor) -> torch.Tensor:
  """`[..., J, 4, 4]` skinning matrices `T_j @ inverse(bind_j)` for posed joint transforms."""
  inv = torch.as_tensor(np.linalg.inv(skin.bind), dtype=transforms.dtype, device=transforms.device)
  return transforms @ inv


def lbs(skin: Skin, transforms: torch.Tensor) -> torch.Tensor:
  """Linear blend skinning: posed vertices `[..., V, 3]` in the frame of `transforms`."""
  mats = skin_matrices(skin, transforms)[..., :3, :]  # [..., J, 3, 4]
  dev = transforms.device
  rest = torch.as_tensor(skin.vertices, device=dev)
  rest = torch.cat((rest, torch.ones_like(rest[:, :1])), -1)  # [V, 4]
  bones = torch.as_tensor(skin.bones, device=dev)
  weights = torch.as_tensor(skin.weights, device=dev)
  out = 0
  for k in range(bones.shape[1]):
    out = out + weights[:, k, None] * (mats[..., bones[:, k], :, :] @ rest[:, :, None])[..., 0]
  return out


def quat_from_matrix(m: torch.Tensor) -> torch.Tensor:
  """Rotation matrices `[..., 3, 3]` to unit quaternions `[..., 4]` (wxyz, w >= 0)."""
  # Shepperd's method: solve from the largest of w, x, y, z, so 180-degree turns are exact.
  m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
  big = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
  s = 2 * big.clamp(min=1e-12).sqrt()  # 4 * the solved component
  a21, a02, a10 = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
  s01, s02, s12 = m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1]
  cand = torch.stack((
    torch.stack((big[..., 0], a21, a02, a10), -1),
    torch.stack((a21, big[..., 1], s01, s02), -1),
    torch.stack((a02, s01, big[..., 2], s12), -1),
    torch.stack((a10, s02, s12, big[..., 3]), -1),
  ), -2) / s[..., :, None]  # [..., 4 choices, 4]
  pick = big.argmax(-1)
  q = cand.gather(-2, pick[..., None, None].expand(*pick.shape, 1, 4))[..., 0, :]
  q = torch.where(q[..., :1] < 0, -q, q)
  return q / q.norm(dim=-1, keepdim=True)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
  """Unit quaternions `[..., 4]` (wxyz) to rotation matrices `[..., 3, 3]`."""
  w, x, y, z = q.float().unbind(-1)
  return torch.stack((
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ), -1).unflatten(-1, (3, 3))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Hamilton product `a * b` of wxyz quaternions (rotate by `b`, then by `a`)."""
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack((
    aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw,
  ), -1)


def quat_yaw(heading: torch.Tensor) -> torch.Tensor:
  """Quaternions `[..., 4]` turning by `heading` about the library's +Y axis, the
  rotation `core.geometry.to_world` applies."""
  half = heading.float() / 2
  zero = torch.zeros_like(half)
  return torch.stack((half.cos(), zero, half.sin(), zero), -1)


def quat_nlerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Normalized linear blend from `a` to `b` by `t` (broadcast), along the shorter arc."""
  b = torch.where(((a * b).sum(-1, keepdim=True) < 0), -b, b)
  q = a + (b - a) * t
  return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def single_children(parents) -> torch.Tensor:
  """`[J]` each joint's only child, or -1 for leaves and branching joints."""
  child = [-1] * len(parents)
  count = [0] * len(parents)
  for j, p in enumerate(parents):
    if p >= 0:
      count[p] += 1
      child[p] = j
  return torch.tensor([c if n == 1 else -1 for c, n in zip(child, count)], dtype=torch.long)


def swing_to_children(rot: torch.Tensor, joints: torch.Tensor, offsets: torch.Tensor,
                      child: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Turn each single-child joint's rotation by the shortest arc that carries its rest
  offset to the child onto the posed child direction, so rigid pieces span the posed
  joints exactly (twist is kept).

  `rot [..., J, 3, 3]` world rotations, `joints [..., J, 3]` world positions,
  `offsets [J, 3]` each joint's rest offset in its parent's frame, `child [J]` from
  `single_children`. Returns the rotations and the turn of each corrected joint
  `[..., S]` in radians (diagnostics: about zero where rotations and positions come
  from the same frame).
  """
  sel = torch.nonzero(child >= 0).squeeze(-1).to(rot.device)
  c = child.to(rot.device)[sel]
  u = (rot[..., sel, :, :] @ offsets.to(rot)[c][..., None])[..., 0]
  v = joints[..., c, :] - joints[..., sel, :]
  u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-9)
  v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
  k = torch.cross(u, v, dim=-1)
  cos = (u * v).sum(-1)
  zero = torch.zeros_like(cos)
  skew = torch.stack((zero, -k[..., 2], k[..., 1], k[..., 2], zero, -k[..., 0], -k[..., 1], k[..., 0], zero),
                     -1).unflatten(-1, (3, 3))
  eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
  turn = eye + skew + skew @ skew / (1 + cos).clamp(min=1e-6)[..., None, None]
  turn = torch.where((cos > -0.999)[..., None, None], turn, eye)  # opposite: leave as is
  out = rot.clone()
  out[..., sel, :, :] = turn @ rot[..., sel, :, :]
  return out, torch.atan2(k.norm(dim=-1), cos)


# Rigid pieces -------------------------------------------------------------------
@dataclass
class Piece:
  bone: int
  name: str
  vertices: np.ndarray  # [n, 3] in the bone's bind frame
  faces: np.ndarray  # [m, 3], closed by caps


def cap_holes(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Close every boundary loop with a fan around its centroid, keeping the winding."""
  edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
  present = set(map(tuple, edges.tolist()))
  nxt: dict[int, list[int]] = defaultdict(list)
  for a, b in edges.tolist():
    if (b, a) not in present:
      nxt[a].append(b)
  verts, new_faces = list(vertices), []
  while nxt:
    start = next(iter(nxt))
    loop, a = [start], start
    while a in nxt:
      b = nxt[a].pop()
      if not nxt[a]:
        del nxt[a]
      if b == start:
        break
      loop.append(b)
      a = b
    if len(loop) < 3:
      continue
    c = len(verts)
    verts.append(vertices[loop].mean(0))
    # Boundary edge a->b belongs to a face (a, b, x); the cap face runs b->a.
    new_faces += [(loop[(i + 1) % len(loop)], loop[i], c) for i in range(len(loop))]
  out = np.concatenate((faces, np.array(new_faces, dtype=faces.dtype).reshape(-1, 3)))
  return np.asarray(verts, dtype=vertices.dtype), out


def rigid_pieces(skin: Skin) -> list[Piece]:
  """Split `skin` by bone (each face to the bone with the most weight over its three
  vertices), cap each piece and express it in its bone's bind frame."""
  joints = len(skin.bind)
  score = np.zeros((len(skin.faces), joints))
  for corner in range(3):
    v = skin.faces[:, corner]
    for k in range(skin.bones.shape[1]):
      np.add.at(score, (np.arange(len(v)), skin.bones[v, k]), skin.weights[v, k])
  owner = score.argmax(1)
  inv = np.linalg.inv(skin.bind)
  pieces = []
  for bone in np.unique(owner):
    faces = skin.faces[owner == bone]
    used, local = np.unique(faces, return_inverse=True)
    verts, faces = cap_holes(skin.vertices[used].astype(np.float64), local.reshape(-1, 3))
    verts = verts @ inv[bone, :3, :3].T + inv[bone, :3, 3]
    pieces.append(Piece(int(bone), skin.joint_names[bone], verts.astype(np.float32), faces))
  return pieces


@functools.lru_cache(maxsize=None)
def core_pieces() -> tuple[Piece, ...]:
  """Full-resolution rigid pieces of the Core skin (computed once per process)."""
  if not CORE_SKIN.exists():
    raise FileNotFoundError(f"rigid humans need ARDY's Core skin at {CORE_SKIN}; run scripts/bootstrap.sh")
  return tuple(rigid_pieces(load_core_skin()))


def piece_spheres(pieces) -> tuple[np.ndarray, np.ndarray]:
  """Bounding sphere of each piece in its bone frame: centres `[P, 3]`, radii `[P]`."""
  centers = np.stack([(p.vertices.min(0) + p.vertices.max(0)) / 2 for p in pieces])
  radii = np.array([np.linalg.norm(p.vertices - c, axis=1).max() for p, c in zip(pieces, centers)])
  return centers.astype(np.float32), radii.astype(np.float32)


class RigidBody:
  """Rigid per-bone mesh pieces as MuJoCo mocap bodies, one set per human.

  `variants` scale copies of the pieces about their joints (for per-world mesh
  assignment tests); variant 0 is the unscaled set the geoms reference.
  """

  def __init__(self, pieces: list[Piece], device="cpu"):
    self.pieces = pieces
    self.bone = torch.tensor([p.bone for p in pieces], dtype=torch.long, device=device)

  @property
  def names(self) -> list[str]:
    return [p.name for p in self.pieces]

  @property
  def faces(self) -> int:
    return sum(len(p.faces) for p in self.pieces)

  def add_meshes(self, spec: mujoco.MjSpec, prefix: str = "body", scales=(1.0,)) -> list[list[str]]:
    """Add every piece once per scale; returns mesh names `[scale][piece]`."""
    names = []
    for s, scale in enumerate(scales):
      row = []
      for p in self.pieces:
        mesh = spec.add_mesh(name=f"{prefix}_v{s:02d}_{p.name}")
        mesh.uservert = (p.vertices * scale).astype(np.float64).ravel().tolist()
        mesh.userface = p.faces.astype(np.int32).ravel().tolist()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        row.append(mesh.name)
      names.append(row)
    return names

  def add_to_spec(self, spec: mujoco.MjSpec, humans: int, colors, group: int = 2,
                  prefix: str = "body", scales=(1.0,), tag: str = "") -> list[list[str]]:
    """Add the meshes and `humans x pieces` mocap bodies `h<i>_<tag><piece>`; returns body names."""
    meshes = self.add_meshes(spec, prefix, scales)
    out = []
    for i in range(humans):
      rgba = colors[i % len(colors)]
      row = []
      for p, mesh in zip(self.pieces, meshes[0]):
        body_name = f"h{i:02d}_{tag}{p.name}"
        body = spec.worldbody.add_body(name=body_name, mocap=True, pos=(0, 0, -5.0 - i))
        body.add_geom(
          name=f"{body_name}_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
          rgba=rgba, group=group, contype=0, conaffinity=0,
        )
        row.append(body_name)
      out.append(row)
    return out

  def poses(self, transforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mocap `pos [H, P, 3]` and `quat [H, P, 4]` from MuJoCo-frame joint transforms `[H, J, 4, 4]`."""
    t = transforms[:, self.bone]
    return t[..., :3, 3], quat_from_matrix(t[..., :3, :3])
