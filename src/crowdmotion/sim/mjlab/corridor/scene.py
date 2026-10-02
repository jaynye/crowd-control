"""Enclosed corridor fixture for primitive-human LiDAR development."""

import math

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  LidarPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from crowdmotion.sim.mjlab.corridor.humans import PrimitiveHumanCfg
from crowdmotion.sim.mjlab.corridor.walking import CorridorWalkCfg


def make_room_spec() -> mujoco.MjSpec:
  """12 x 4 x 2.8 m interior. Group 4 hides only the ceiling in the viewer."""
  spec = mujoco.MjSpec()
  room = spec.worldbody.add_body(name="room", mocap=True)
  boxes = (
    ("floor", (0, 0, -0.1), (6.2, 2.2, 0.1), 0),
    ("wall_front", (6.1, 0, 1.4), (0.1, 2.2, 1.4), 2),
    ("wall_back", (-6.1, 0, 1.4), (0.1, 2.2, 1.4), 2),
    ("wall_left", (0, 2.1, 1.4), (6, 0.1, 1.4), 2),
    ("wall_right", (0, -2.1, 1.4), (6, 0.1, 1.4), 2),
    ("ceiling", (0, 0, 2.9), (6.2, 2.2, 0.1), 4),
  )
  for name, pos, size, group in boxes:
    room.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
      group=group,
      rgba=(0.7, 0.72, 0.75, 1.0) if group else (0.35, 0.4, 0.45, 1.0),
      contype=1,
      conaffinity=1,
    )
  spec.worldbody.add_light(pos=(0, 0, 2.6), dir=(0, 0, -1))
  return spec


def make_scanner_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="scanner", mocap=True)
  body.add_geom(
    name="housing",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(0.07, 0, 0),
    rgba=(1.0, 0.5, 0.05, 1.0),
    group=2,
  )
  body.add_site(name="lidar", size=(0.02, 0, 0))
  return spec


def make_crowd_env_cfg(
  num_envs: int = 1,
  num_humans: int = 4,
  seed: int = 0,
  moving: bool = True,
  azimuth_samples: int = 180,
  elevation_samples: int = 16,
) -> ManagerBasedRlEnvCfg:
  """No checkpoint required: a fixed sensor rig observes physical pedestrians.

  The sensor pose (-5, 0, 1.15) is a test fixture, not G1 mount calibration.
  Worlds use identical local coordinates but independent MuJoCo-Warp state.
  """
  if num_envs < 1 or not 0 <= num_humans <= 8:
    raise ValueError("Require num_envs >= 1 and num_humans in [0, 8]")
  names = tuple(f"human_{i}" for i in range(num_humans))
  entities = {
    "room": EntityCfg(spec_fn=make_room_spec),
    "scanner": EntityCfg(
      spec_fn=make_scanner_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(-5.0, 0.0, 1.15)),
    ),
  }
  for i, name in enumerate(names):
    xy = (-3.0 + 2.0 * (i // 2), -0.8 if i % 2 == 0 else 0.8)
    entities[name] = PrimitiveHumanCfg(
      height=1.65 + 0.1 * (i % 3),
      rgba=(0.25 + 0.12 * (i % 3), 0.55, 0.85 - 0.15 * (i % 3), 1.0),
    ).entity_cfg(xy=xy, yaw=0.0 if i % 2 == 0 else math.pi)
  lidar = RayCastSensorCfg(
    name="lidar",
    frame=ObjRef(type="site", name="lidar", entity="scanner"),
    pattern=LidarPatternCfg(
      azimuth_samples=azimuth_samples, elevation_samples=elevation_samples
    ),
    ray_alignment="base",
    min_distance=0.1,
    max_distance=20.0,
    include_geom_groups=(0, 1, 2, 4),
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(hit_sphere_radius=0.08, stride=8),
  )
  contacts = tuple(
    ContactSensorCfg(
      name=f"{name}_contacts",
      primary=ContactMatch(mode="subtree", pattern="body", entity=name),
      fields=("found", "force", "dist", "pos", "normal"),
      reduce="maxforce",
      history_length=4,
    )
    for name in names
  )
  return ManagerBasedRlEnvCfg(
    seed=seed,
    decimation=4,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=0.0,
      entities=entities,
      sensors=(lidar, *contacts),
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005),
      njmax=256,
      nconmax=256,
      contact_sensor_maxmatch=128,
    ),
    commands={
      name: CorridorWalkCfg(
        entity_name=name,
        neighbor_names=tuple(n for n in names if n != name),
        seed=seed + i * 65537,
        resampling_time_range=(2.0, 5.0),
        debug_vis=True,
      )
      for i, name in enumerate(names)
    }
    if moving
    else {},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.0, 0.0, 0.8),
      distance=11.0,
      azimuth=120.0,
      elevation=-55.0,
      width=960,
      height=640,
      max_extra_envs=0,
    ),
  )
