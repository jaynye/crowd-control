"""Ray casting against rigid mesh humans, one Warp mesh per body piece shared by all humans.

`RigidRays` holds a Warp mesh for each rigid piece (`crowdmotion.sim.body`, in its bone
frame) and, for every human, each piece's world-to-bone transform and a box around the
posed body. Calling it casts a batch of rays:

1. per ray, test each human's box (`humans` per world);
2. inside a box, carry the ray into each piece's bone frame and query that piece's mesh
   (rigid transforms keep distances along the ray);
3. overwrite the ray's distance, geom id and world normal only where a piece is nearer
   than the hit already stored (`dist < 0` means none yet).

The call matches mjlab's ray-target signature (`RayCastSensor.add_ray_target`) without
importing mjlab. `update` writes new poses into fixed buffers, so a captured CUDA graph
that launches the kernel stays valid; `enabled` switches the kernel on and off the
same way. Boxes come from each piece's bounding sphere, so they hold the posed pieces
whatever the pose.
"""

from __future__ import annotations

import numpy as np
import torch

from crowdmotion.sim.body import piece_spheres

_KERNELS: dict = {}


def _kernels():
  if _KERNELS:
    return _KERNELS
  import warp as wp

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
  def piece_rays(
    meshes: wp.array(dtype=wp.uint64), inv: wp.array2d(dtype=wp.mat44),
    lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3), geom: wp.array2d(dtype=int),
    humans: int, enabled: wp.array(dtype=int), max_range: float,
    pnt: wp.array2d(dtype=wp.vec3), vec: wp.array2d(dtype=wp.vec3),
    dist: wp.array2d(dtype=float), geomid: wp.array2d(dtype=int), normal: wp.array2d(dtype=wp.vec3),
  ):
    w, r = wp.tid()
    if enabled[0] == 0:
      return
    o = pnt[w, r]
    d = vec[w, r]
    best = dist[w, r]
    if best < 0.0:
      best = max_range
    hit_i = int(-1)
    hit_p = int(-1)
    n = wp.vec3(0.0, 0.0, 0.0)
    for i in range(humans):
      h = w * humans + i
      if box_hit(o, d, lo[h], hi[h], best):
        for p in range(meshes.shape[0]):
          m = inv[h, p]
          q = wp.mesh_query_ray(meshes[p], wp.transform_point(m, o), wp.transform_vector(m, d), best)
          if q.result and q.t < best:
            best = q.t
            hit_i = i
            hit_p = p
            n = q.normal
    if hit_i >= 0:
      m = inv[w * humans + hit_i, hit_p]
      # The inverse transform's rotation is R^T, so R n is its transpose applied to n.
      world = wp.vec3(m[0, 0] * n[0] + m[1, 0] * n[1] + m[2, 0] * n[2],
                      m[0, 1] * n[0] + m[1, 1] * n[1] + m[2, 1] * n[2],
                      m[0, 2] * n[0] + m[1, 2] * n[1] + m[2, 2] * n[2])
      dist[w, r] = best
      geomid[w, r] = geom[hit_i, hit_p]
      normal[w, r] = wp.normalize(world)

  @wp.func
  def wxyz_matrix(q: wp.vec4) -> wp.mat33:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return wp.mat33(
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
      2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    )

  @wp.kernel
  def pose_pieces(
    env_ids: wp.array(dtype=int), lo: wp.array(dtype=int), alpha: wp.array(dtype=float),
    joints: wp.array3d(dtype=wp.vec3), rots: wp.array4d(dtype=wp.vec4),
    bone: wp.array(dtype=int), child: wp.array(dtype=int), offsets: wp.array(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3), frame: wp.mat33, mocap_id: wp.array2d(dtype=int), humans: int,
    mocap_pos: wp.array2d(dtype=wp.vec3), mocap_quat: wp.array2d(dtype=wp.quat),
    inv: wp.array2d(dtype=wp.mat44), world_centers: wp.array2d(dtype=wp.vec3), turn: wp.array3d(dtype=float),
  ):
    """One thread per (listed env, human, piece): blend the bone's rotation between two
    stored frames, swing it onto the posed child joint, then write the piece's mocap
    pose, its world-to-bone transform and its bounding-sphere centre."""
    k, i, p = wp.tid()
    env = env_ids[k]
    j = bone[p]
    q0 = rots[env, i, lo[k], j]
    q1 = rots[env, i, lo[k] + 1, j]
    if wp.dot(q0, q1) < 0.0:
      q1 = -q1
    a = alpha[k]
    r = wxyz_matrix(wp.normalize(q0 * (1.0 - a) + q1 * a))
    t = joints[k, i, j]
    c = child[j]
    angle = float(0.0)
    if c >= 0:
      u = wp.normalize(r * offsets[c])
      v = wp.normalize(joints[k, i, c] - t)
      axis = wp.cross(u, v)
      cs = wp.dot(u, v)
      angle = wp.atan2(wp.length(axis), cs)
      if cs > -0.999:
        s = wp.skew(axis)
        r = (wp.identity(n=3, dtype=float) + s + s * s * (1.0 / (1.0 + cs))) * r
    rm = frame * r
    tm = frame * t
    q = wp.quat_from_matrix(rm)  # Warp order: x, y, z, w
    sign = float(1.0)
    if q[3] < 0.0:
      sign = -1.0
    mocap = mocap_id[i, p]
    mocap_pos[env, mocap] = tm
    # mujoco_warp keeps MuJoCo's w, x, y, z order in `wp.quat` slots.
    mocap_quat[env, mocap] = wp.quat(sign * q[3], sign * q[0], sign * q[1], sign * q[2])
    rt = wp.transpose(rm)
    b = -(rt * tm)
    h = env * humans + i
    inv[h, p] = wp.mat44(
      rt[0, 0], rt[0, 1], rt[0, 2], b[0], rt[1, 0], rt[1, 1], rt[1, 2], b[1],
      rt[2, 0], rt[2, 1], rt[2, 2], b[2], 0.0, 0.0, 0.0, 1.0,
    )
    world_centers[h, p] = rm * centers[p] + tm
    turn[k, i, p] = angle

  @wp.kernel
  def human_boxes(
    env_ids: wp.array(dtype=int), humans: int, world_centers: wp.array2d(dtype=wp.vec3),
    radii: wp.array(dtype=float), lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3),
  ):
    k, i = wp.tid()
    h = env_ids[k] * humans + i
    a = wp.vec3(1.0e9, 1.0e9, 1.0e9)
    b = wp.vec3(-1.0e9, -1.0e9, -1.0e9)
    for p in range(radii.shape[0]):
      r = wp.vec3(radii[p], radii[p], radii[p])
      a = wp.min(a, world_centers[h, p] - r)
      b = wp.max(b, world_centers[h, p] + r)
    lo[h] = a
    hi[h] = b

  _KERNELS.update(piece_rays=piece_rays, pose_pieces=pose_pieces, human_boxes=human_boxes)
  return _KERNELS


class RigidRays:
  """Rays against `humans` rigid mesh humans in each of `worlds` worlds.

  `geom_ids [humans, P]`: the id reported for a hit on piece `P` of human `i` (the
  visual MuJoCo geom, so sensor geom ids keep naming the body part). `max_range`:
  how far to search when a ray has no hit yet.
  """

  def __init__(self, pieces, worlds: int, humans: int, geom_ids, device, max_range: float):
    import warp as wp

    wp.init()
    self.device = torch.device(device)
    wp_device = wp.device_from_torch(self.device)
    self.worlds, self.humans, self.max_range = worlds, humans, float(max_range)
    ctor = "cubql" if self.device.type == "cuda" else "sah"
    # Keep the meshes referenced: the kernel only holds their ids.
    self._meshes = [
      wp.Mesh(points=wp.array(p.vertices.astype(np.float32), dtype=wp.vec3, device=wp_device),
              indices=wp.array(p.faces.astype(np.int32).ravel(), dtype=wp.int32, device=wp_device),
              bvh_constructor=ctor)
      for p in pieces
    ]
    self._mesh_ids = wp.array([m.id for m in self._meshes], dtype=wp.uint64, device=wp_device)
    # Compile now: the first launch may happen inside mjlab's sense-graph capture.
    _kernels()
    wp.load_module(module=__name__, device=wp_device)
    P, H = len(pieces), worlds * humans
    centers, radii = piece_spheres(pieces)
    self.centers = torch.as_tensor(centers, device=self.device)
    self.radii = torch.as_tensor(radii, device=self.device)
    self.inv = torch.eye(4, device=self.device).repeat(H, P, 1, 1)
    # Empty boxes far below the scene until the first update.
    self.lo = torch.full((H, 3), -1e4, device=self.device)
    self.hi = torch.full((H, 3), -1e4 + 1e-3, device=self.device)
    self.enabled = torch.ones(1, dtype=torch.int32, device=self.device)
    self._wp = {
      "inv": wp.from_torch(self.inv, dtype=wp.mat44),
      "lo": wp.from_torch(self.lo, dtype=wp.vec3),
      "hi": wp.from_torch(self.hi, dtype=wp.vec3),
      "enabled": wp.from_torch(self.enabled, dtype=wp.int32),
      "geom": wp.array(np.asarray(geom_ids, dtype=np.int32), dtype=wp.int32, device=wp_device),
    }

  def update(self, rot: torch.Tensor, pos: torch.Tensor, humans: torch.Tensor | None = None) -> None:
    """Pose humans (rows `world * humans + i`; default all) from each piece's bone pose
    in the world: rotations `[n, P, 3, 3]` and positions `[n, P, 3]`. Writes in place."""
    rows = slice(None) if humans is None else humans
    rot_t = rot.transpose(-1, -2)
    self.inv[rows, :, :3, :3] = rot_t
    self.inv[rows, :, :3, 3] = -(rot_t @ pos[..., None])[..., 0]
    centers = (rot @ self.centers[:, :, None])[..., 0] + pos  # [n, P, 3]
    self.lo[rows] = (centers - self.radii[:, None]).amin(1)
    self.hi[rows] = (centers + self.radii[:, None]).amax(1)

  def set_skeleton(self, bones, child, offsets, mocap_ids, frame) -> None:
    """Enable `pose`: each piece's bone `[P]`, each joint's only child or -1 `[J]`
    (`body.single_children`), rest offsets in the parent frame `[J, 3]`, the mocap id of
    each human's piece `[humans, P]`, and the 3x3 frame change from the joints' frame
    to the world (`body.YUP_TO_MUJOCO`)."""
    import warp as wp

    dev = wp.device_from_torch(self.device)
    as_int = lambda x: wp.array(np.asarray(torch.as_tensor(x).cpu(), dtype=np.int32), dtype=int, device=dev)  # noqa: E731
    self.world_centers = torch.zeros(self.worlds * self.humans, len(self._meshes), 3, device=self.device)
    self._pose = {
      "bone": as_int(bones), "child": as_int(child), "mocap": as_int(mocap_ids),
      "offsets": wp.array(np.asarray(torch.as_tensor(offsets).cpu(), dtype=np.float32), dtype=wp.vec3, device=dev),
      "centers": wp.from_torch(self.centers, dtype=wp.vec3),
      "radii": wp.from_torch(self.radii),
      "frame": wp.mat33(*np.asarray(frame, dtype=np.float32).ravel().tolist()),
      "world_centers": wp.from_torch(self.world_centers, dtype=wp.vec3),
    }

  def pose(self, joints: torch.Tensor, rots: torch.Tensor, env_ids: torch.Tensor, lo: torch.Tensor,
           alpha: torch.Tensor, mocap_pos, mocap_quat) -> torch.Tensor:
    """Pose the humans of worlds `env_ids` in one fused pass (needs `set_skeleton`).

    `joints [e, M, J, 3]`: their joint positions (joints' frame). `rots [worlds, M, F, J, 4]`:
    stored wxyz rotations per frame; each listed world blends frames `lo[k]` and
    `lo[k] + 1` by `alpha[k]`. Writes the pieces' poses into the Warp arrays `mocap_pos
    [worlds, nmocap]` and `mocap_quat`, and this target's transforms and boxes. Returns
    the swing turn of each piece's bone `[e, M, P]` (radians).
    """
    import warp as wp

    k = _kernels()
    e, M = joints.shape[:2]
    P = len(self._meshes)
    turn = torch.empty(e, M, P, device=self.device)
    w, s = self._wp, self._pose
    env = wp.from_torch(env_ids.to(torch.int32))
    wp.launch(
      k["pose_pieces"], dim=(e, M, P),
      inputs=[env, wp.from_torch(lo.to(torch.int32)), wp.from_torch(alpha.float().contiguous()),
              wp.from_torch(joints.float().contiguous(), dtype=wp.vec3), wp.from_torch(rots, dtype=wp.vec4),
              s["bone"], s["child"], s["offsets"], s["centers"], s["frame"], s["mocap"], self.humans],
      outputs=[mocap_pos, mocap_quat, w["inv"], s["world_centers"], wp.from_torch(turn)],
      device=env.device,
    )
    wp.launch(k["human_boxes"], dim=(e, M), inputs=[env, self.humans, s["world_centers"], s["radii"]],
              outputs=[w["lo"], w["hi"]], device=env.device)
    return turn

  def set_enabled(self, on: bool) -> None:
    self.enabled.fill_(1 if on else 0)

  def __call__(self, pnt, vec, dist, geomid, normal) -> None:
    """Cast `[worlds, rays]` Warp rays `pnt + t vec`; keep the nearer of each stored hit."""
    import warp as wp

    w = self._wp
    wp.launch(
      _kernels()["piece_rays"], dim=(self.worlds, pnt.shape[1]),
      inputs=[self._mesh_ids, w["inv"], w["lo"], w["hi"], w["geom"], self.humans, w["enabled"], self.max_range,
              pnt, vec],
      outputs=[dist, geomid, normal], device=pnt.device,
    )
