"""Plaza scene pieces shared by the mjlab environment and the replay renderer.

Both draw the same walled plaza: library XZ maps to MuJoCo XY as `(x, -z)`, the
floor is a box, and four walls of half-height `WALL_HALF_HEIGHT` sit just outside
`half_x` by `half_z`. Two presets exist because the two consumers differ:

- `sim_room_spec`: a named mocap room body for mjlab (walls in geom group 2 so the
  LiDAR can include them), a key and a fill light.
- `replay_spec`: unnamed world geoms, steering obstacles drawn as cylinders, one
  light, and the capsule humans. The mjlab room draws only the scanner mast,
  so obstacles appear in replay but not to the LiDAR (a known difference).
"""

from __future__ import annotations

import mujoco

COLORS = (
  (0.90, 0.33, 0.25, 1), (0.25, 0.55, 0.85, 1), (0.30, 0.70, 0.40, 1), (0.85, 0.65, 0.20, 1),
  (0.60, 0.40, 0.75, 1), (0.20, 0.70, 0.70, 1), (0.85, 0.45, 0.65, 1), (0.55, 0.55, 0.55, 1),
)
FLOOR_RGBA = (0.42, 0.45, 0.48, 1)
WALL_RGBA = (0.72, 0.74, 0.77, 1)
OBSTACLE_RGBA = (0.3, 0.35, 0.4, 1)
WALL_HALF_HEIGHT = 0.6

# Overview camera of the plaza, and the replay renderer's alternatives.
CAMERAS = {
  "overview": {"lookat": (0.0, 0.0, 0.6), "distance": 15.5, "azimuth": 115.0, "elevation": -38.0},
  "top": {"lookat": (0.0, 0.0, 0.6), "distance": 18.0, "azimuth": 90.0, "elevation": -89.0},
  "follow": {"lookat": (0.0, 0.0, 0.6), "distance": 4.5, "azimuth": 115.0, "elevation": -20.0},
}


def room_boxes(hx: float, hz: float):
  """`(name, pos, size, rgba)` of the floor and the +x, -x, +y, -y walls (MuJoCo frame)."""
  h = WALL_HALF_HEIGHT
  return (
    ("floor", (0, 0, -0.1), (hx + 0.2, hz + 0.2, 0.1), FLOOR_RGBA),
    ("wall_px", (hx + 0.1, 0, h), (0.1, hz + 0.2, h), WALL_RGBA),
    ("wall_nx", (-hx - 0.1, 0, h), (0.1, hz + 0.2, h), WALL_RGBA),
    ("wall_py", (0, hz + 0.1, h), (hx, 0.1, h), WALL_RGBA),
    ("wall_ny", (0, -hz - 0.1, h), (hx, 0.1, h), WALL_RGBA),
  )


def _directional(spec: mujoco.MjSpec, direction, diffuse, ambient=None, shadow=True):
  light = spec.worldbody.add_light(pos=(0, 0, 8.0), dir=direction)
  light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  light.diffuse = diffuse
  if ambient is not None:
    light.ambient = ambient
  light.castshadow = shadow
  return light


def sim_room_spec(hx: float, hz: float) -> mujoco.MjSpec:
  """Room entity for mjlab: floor in group 0, walls in group 2, key and fill lights."""
  spec = mujoco.MjSpec()
  room = spec.worldbody.add_body(name="room", mocap=True)
  for name, pos, size, rgba in room_boxes(hx, hz):
    group = 0 if name == "floor" else 2
    room.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, group=group, rgba=rgba)
  _directional(spec, (0.3, -0.4, -1.0), (0.55, 0.55, 0.55), ambient=(0.22, 0.22, 0.25), shadow=True)
  _directional(spec, (-0.5, 0.6, -1.0), (0.3, 0.3, 0.32), shadow=False)
  return spec


def replay_spec(hx: float, hz: float, obstacles, skeleton, humans: int) -> mujoco.MjSpec:
  """Replay scene: room, obstacle cylinders `(x, z, radius)`, one light, capsule humans."""
  spec = mujoco.MjSpec()
  world = spec.worldbody
  for _name, pos, size, rgba in room_boxes(hx, hz):
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=rgba)
  for x, z, radius in obstacles:
    if radius > 0:
      world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=(x, -z, WALL_HALF_HEIGHT),
        size=(radius, WALL_HALF_HEIGHT, 0), rgba=OBSTACLE_RGBA,
      )
  light = world.add_light(pos=(0, 0, 8), dir=(0.3, -0.4, -1))
  light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  light.ambient, light.diffuse = (0.25, 0.25, 0.25), (0.65, 0.65, 0.65)
  skeleton.add_to_spec(spec, humans, COLORS)
  return spec


def set_camera(camera: mujoco.MjvCamera, view: str) -> None:
  preset = CAMERAS[view]
  camera.lookat[:] = preset["lookat"]
  camera.distance, camera.azimuth, camera.elevation = preset["distance"], preset["azimuth"], preset["elevation"]
