"""Finite-mass upright pedestrians with shared sensing/collision geometry."""

from dataclasses import dataclass
from functools import partial

import mujoco

from mjlab.actuator import BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


@dataclass(frozen=True)
class PrimitiveHumanCfg:
  """A rigid human silhouette on planar x/y/yaw joints.

  This proxy has no balance or gait controller. Its floor support is a joint
  constraint; the velocity actuators have finite effort and contacts affect its
  motion. All lengths scale with height. Geometry uses group 1.
  """

  height: float = 1.75
  mass: float = 70.0
  rgba: tuple[float, float, float, float] = (0.25, 0.55, 0.85, 1.0)

  def __post_init__(self) -> None:
    if not 0.5 <= self.height <= 2.5 or not 1.0 <= self.mass <= 300.0:
      raise ValueError("Require human height in [0.5, 2.5] and mass in [1, 300]")

  def build_spec(self, xy: tuple[float, float] = (0.0, 0.0)) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    origin = spec.worldbody.add_body(name="origin", mocap=True)
    # Matching body offsets and joint references keep qpos equal to world x/y
    # while preventing all humans overlapping in the compiler's default pose.
    body = origin.add_body(name="body", pos=(xy[0], xy[1], 0.0))
    body.add_joint(
      name="x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(1, 0, 0), ref=xy[0]
    )
    body.add_joint(
      name="y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(0, 1, 0), ref=xy[1]
    )
    body.add_joint(name="yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1))
    scale = self.height / 1.75
    # Name, shape, center, size, mass fraction. Capsules are vertical.
    parts = (
      ("head", "sphere", (0, 0, 1.625), (0.125, 0, 0), 0.08),
      ("torso", "ellipsoid", (0, 0, 1.22), (0.14, 0.23, 0.28), 0.38),
      ("pelvis", "ellipsoid", (0, 0, 0.92), (0.14, 0.19, 0.16), 0.14),
      ("left_arm", "capsule", (0, 0.29, 1.10), (0.065, 0.25, 0), 0.05),
      ("right_arm", "capsule", (0, -0.29, 1.10), (0.065, 0.25, 0), 0.05),
      ("left_leg", "capsule", (0, 0.11, 0.46), (0.085, 0.36, 0), 0.15),
      ("right_leg", "capsule", (0, -0.11, 0.46), (0.085, 0.36, 0), 0.15),
    )
    types = {
      "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
      "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
      "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    }
    for name, shape, center, size, fraction in parts:
      body.add_geom(
        name=name,
        type=types[shape],
        pos=tuple(v * scale for v in center),
        size=tuple(v * scale for v in size),
        mass=self.mass * fraction,
        rgba=self.rgba,
        group=1,
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(0.5, 0.005, 0.0001),
      )
    return spec

  def entity_cfg(
    self, xy: tuple[float, float] = (0.0, 0.0), yaw: float = 0.0
  ) -> EntityCfg:
    return EntityCfg(
      spec_fn=partial(self.build_spec, xy=xy),
      init_state=EntityCfg.InitialStateCfg(
        joint_pos={"x": xy[0], "y": xy[1], "yaw": yaw}
      ),
      articulation=EntityArticulationInfoCfg(
        actuators=(
          BuiltinVelocityActuatorCfg(
            target_names_expr=("x", "y"), damping=200.0, effort_limit=140.0
          ),
          BuiltinVelocityActuatorCfg(
            target_names_expr=("yaw",), damping=30.0, effort_limit=25.0
          ),
        )
      ),
    )
