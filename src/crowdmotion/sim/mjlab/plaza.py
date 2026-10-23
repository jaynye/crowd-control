"""A walled plaza with a LiDAR rig and graph-driven humans, as an mjlab env config.

Humans are rigid Core-skin meshes by default (`human_geometry`); capsules remain
available, alone or beside the meshes for the Viser toggle (`GraphCrowd`).
"""

from __future__ import annotations

import mujoco
import torch

from crowdmotion.behavior.encounter import EncounterCfg
from crowdmotion.behavior.plaza import PlazaSteeringCfg
from crowdmotion.sim.capsules import CapsuleSkeleton
from crowdmotion.sim.mjlab.command import MESH_TAG, GraphCrowdCfg
from crowdmotion.sim.scene import CAMERAS, COLORS, sim_room_spec


def make_plaza_env_cfg(
  graph_path: str,
  humans: int = 16,
  num_envs: int = 1,
  seed: int = 0,
  steering: PlazaSteeringCfg | None = None,
  scanner_xz: tuple[float, float] = (-6.0, 0.0),
  lidar_azimuth: int = 360,
  lidar_elevation: int = 16,
  lidar_stride: int = 2,
  width: int = 1280,
  height: int = 720,
  step_dt: float = 0.02,
  decimation: int = 1,
  human_geometry: str | tuple[str, ...] = ("rigid",),
  rotations_path: str | None = None,
  design: str = "plaza",
  encounter: EncounterCfg | None = None,
):
  """A walled plaza (library XZ -> MuJoCo XY as (x, -z)), a LiDAR rig, and
  `humans` graph-driven humans per environment.

  `human_geometry`: `"rigid"` (full-resolution Core-skin pieces, the default),
  `"capsules"`, or both as a tuple whose first entry starts active (the Viser demo
  switches between them). Rigid pieces are visual geoms in group 1; the LiDAR traces
  groups 0 and 2 and sees them through `GraphCrowd`'s ray target. Capsules sit in
  group 2. `rotations_path`: the graph's joint rotations (rigid only; default
  artifact `graph-rotations`). `design`: the behavior design, `plaza` (`steering`) or
  `encounter` (`encounter`); the active design's section sets the room size.

  Everything is kinematic, so contacts are disabled and one physics step per
  environment step suffices (`decimation`); both matter on CPU.
  """
  from mjlab.entity import EntityCfg
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.scene import SceneCfg
  from mjlab.sensor import LidarPatternCfg, ObjRef, RayCastSensorCfg
  from mjlab.sim import MujocoCfg, SimulationCfg
  from mjlab.viewer import ViewerConfig

  steering = steering or PlazaSteeringCfg(obstacles=((scanner_xz[0], scanner_xz[1], 0.3),))
  encounter = encounter or EncounterCfg(obstacles=((scanner_xz[0], scanner_xz[1], 0.3),))
  scene = encounter if design == "encounter" else steering
  hx, hz = scene.half_x, scene.half_z
  geometry = (human_geometry,) if isinstance(human_geometry, str) else tuple(human_geometry)
  skeleton = body = None
  if "capsules" in geometry:
    blob = torch.load(graph_path, map_location="cpu", weights_only=False)
    skeleton = CapsuleSkeleton.from_rest(blob["meta"], blob["rest_joints"])
    del blob
  if "rigid" in geometry:
    from crowdmotion.sim.body import RigidBody, core_pieces

    body = RigidBody(list(core_pieces()))

  def room_spec() -> mujoco.MjSpec:
    return sim_room_spec(hx, hz)

  def scanner_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="scanner", mocap=True)
    body.add_geom(name="mast", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.04, 0.55, 0), pos=(0, 0, -0.6),
                  rgba=(0.2, 0.2, 0.2, 1), group=2)
    body.add_geom(name="housing", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.08, 0, 0),
                  rgba=(1.0, 0.5, 0.05, 1.0), group=2)
    body.add_site(name="lidar", size=(0.02, 0, 0))
    return spec

  def crowd_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    if skeleton is not None:
      skeleton.add_to_spec(spec, humans, COLORS)
    if body is not None:
      body.add_to_spec(spec, humans, COLORS, group=1, prefix="core", tag=MESH_TAG)
    return spec

  lidar = RayCastSensorCfg(
    name="lidar",
    frame=ObjRef(type="site", name="lidar", entity="scanner"),
    pattern=LidarPatternCfg(azimuth_samples=lidar_azimuth, elevation_samples=lidar_elevation),
    ray_alignment="base",
    min_distance=0.1,
    max_distance=20.0,
    include_geom_groups=(0, 2),
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(hit_sphere_radius=0.5, stride=lidar_stride),
  )
  view = CAMERAS["overview"]
  return ManagerBasedRlEnvCfg(
    seed=seed,
    decimation=decimation,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=0.0,
      entities={
        "room": EntityCfg(spec_fn=room_spec),
        "scanner": EntityCfg(
          spec_fn=scanner_spec,
          init_state=EntityCfg.InitialStateCfg(pos=(scanner_xz[0], -scanner_xz[1], 1.15)),
        ),
        "crowd": EntityCfg(spec_fn=crowd_spec),
      },
      sensors=(lidar,),
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=step_dt / decimation, disableflags=("contact",)), njmax=64, nconmax=64
    ),
    commands={
      "crowd": GraphCrowdCfg(
        graph_path=str(graph_path), humans=humans, steering=steering, design=design, encounter=encounter,
        seed=seed, debug_vis=True,
        human_geometry=geometry, rotations_path=rotations_path,
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=view["lookat"],
      distance=view["distance"],
      azimuth=view["azimuth"],
      elevation=view["elevation"],
      width=width,
      height=height,
      max_extra_envs=0,
    ),
  )


def set_hit_sphere_radius(env, meters: float = 0.035) -> None:
  """LiDAR hit spheres are sized as a multiple of the model's mean size."""
  sensor = env.scene["lidar"]
  sensor.cfg.viz.hit_sphere_radius = meters / float(env.sim.mj_model.stat.meansize)
