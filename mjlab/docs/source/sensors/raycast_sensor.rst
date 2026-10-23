.. _raycast_sensor:

RayCast Sensor
==============

``RayCastSensor`` provides GPU-accelerated raycasting for terrain
scanning, obstacle detection, and depth sensing. Rays are emitted from
a frame attached to a body, site, or geom in the scene, and the sensor
reports hit distances, world-space hit positions, and surface normals.

.. raw:: html

   <video controls style="display: block; margin: 0 auto; max-width: 100%; height: auto;">
     <source src="../../_static/raycast_demo.mp4" type="video/mp4">
   </video>


Quick start
-----------

.. code-block:: python

    from mjlab.sensor import RayCastSensorCfg, GridPatternCfg, ObjRef

    # Downward-facing grid for terrain height scanning.
    raycast_cfg = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(size=(1.0, 1.0), resolution=0.1),
        max_distance=5.0,
    )

    scene_cfg = SceneCfg(
        entities={"robot": robot_cfg},
        sensors=(raycast_cfg,),
    )

    # Access at runtime.
    data = env.scene["terrain_scan"].data
    data.distances      # [B, N] distance to hit, -1 if miss
    data.hit_pos_w      # [B, N, 3] world-space hit positions
    data.normals_w      # [B, N, 3] surface normals


Ray patterns
------------

Ray patterns define the spatial distribution and direction of rays
emitted from the sensor frame.

.. grid:: 2

   .. grid-item-card:: Grid pattern

      Parallel rays in a 2D grid with fixed spatial resolution. The
      ground footprint does not change with sensor height because ray
      spacing is defined in world units (meters). The natural choice for
      height maps and terrain scanning.

      .. raw:: html

         <video autoplay loop muted playsinline style="width: 100%; height: auto;">
           <source src="../../_static/pattern_grid.mp4" type="video/mp4">
         </video>

   .. grid-item-card:: Pinhole camera pattern

      Diverging rays emitted from a single origin, analogous to a depth
      camera. The ground coverage increases with sensor
      height because the field of view is fixed in angular units.

      .. raw:: html

         <video autoplay loop muted playsinline style="width: 100%; height: auto;">
           <source src="../../_static/pattern_pinhole.mp4" type="video/mp4">
         </video>

.. code-block:: python

    from mjlab.sensor import GridPatternCfg, PinholeCameraPatternCfg

    # Parallel grid: fixed footprint, height-invariant.
    grid = GridPatternCfg(
        size=(1.0, 1.0),              # Grid dimensions in meters
        resolution=0.1,               # Spacing between rays
        direction=(0.0, 0.0, -1.0),   # Ray direction (down)
    )

    # Pinhole: perspective projection, diverging rays.
    pinhole = PinholeCameraPatternCfg(
        width=16,
        height=12,
        fovy=45.0,  # Vertical FOV in degrees
    )

    # Pinhole from a MuJoCo camera definition.
    pinhole = PinholeCameraPatternCfg.from_mujoco_camera("robot/depth_cam")

    # Pinhole from an intrinsic matrix.
    pinhole = PinholeCameraPatternCfg.from_intrinsic_matrix(
        intrinsic_matrix=[500, 0, 320, 0, 500, 240, 0, 0, 1],
        width=640,
        height=480,
    )


Angular LiDAR
^^^^^^^^^^^^^

``LidarPatternCfg`` emits rays from one origin, with +X forward and +Z up.
Angles are in degrees. Full 360-degree sweeps exclude the duplicate endpoint;
partial sweeps include both endpoints. Output is ordered by elevation, then
azimuth. A dimension with one sample uses the lower angle bound.

.. code-block:: python

    from mjlab.sensor import LidarPatternCfg

    lidar = RayCastSensorCfg(
        name="lidar",
        frame=ObjRef(type="site", name="lidar", entity="scanner"),
        pattern=LidarPatternCfg(
            azimuth_range=(-180.0, 180.0),
            elevation_range=(-7.0, 52.0),
            azimuth_samples=180,
            elevation_samples=16,
        ),
        min_distance=0.1,
        max_distance=20.0,
    )

This is an instantaneous angular raster. It does not reproduce a physical
LiDAR's temporal scan pattern or noise. Rays see moving primitives through the
existing per-scan BVH refit.


Pattern comparison
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Grid
     - Pinhole
   * - Ray direction
     - Parallel
     - Diverging
   * - Spacing unit
     - Meters
     - Degrees (FOV)
   * - Height affects coverage
     - No
     - Yes
   * - Projection model
     - Orthographic
     - Perspective


Frame attachment
----------------

Rays are emitted from a frame in the scene specified via ``ObjRef``.
The frame can be a body, site, or geom on any entity.

.. code-block:: python

    frame = ObjRef(type="body", name="base", entity="robot")
    frame = ObjRef(type="site", name="scan_site", entity="robot")
    frame = ObjRef(type="geom", name="sensor_mount", entity="robot")

``exclude_parent_body`` (default ``True``) prevents rays from hitting
the body to which the sensor is attached.


Ray alignment
-------------

The ``ray_alignment`` setting controls how rays orient relative to the
attached frame when the body rotates.

.. raw:: html

   <video autoplay loop muted playsinline
          style="display: block; margin: 0 auto; max-width: 100%; height: auto;">
     <source src="../../_static/ray_alignment_comparison.mp4" type="video/mp4">
   </video>

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Mode
     - Description
     - Use case
   * - ``"base"``
     - Full position and rotation tracking
     - Body-mounted sensors
   * - ``"yaw"``
     - Follows yaw, ignores pitch and roll
     - Terrain height maps
   * - ``"world"``
     - Fixed world-frame direction
     - Gravity-aligned sensing

.. code-block:: python

    RayCastSensorCfg(
        name="height_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(size=(1.0, 1.0), resolution=0.1),
        ray_alignment="yaw",
    )


Geom group filtering
--------------------

MuJoCo assigns geoms to groups 0 through 5. Use
``include_geom_groups`` to restrict which geoms rays can hit. This is
useful for ignoring visual-only geoms or isolating terrain geometry.

.. code-block:: python

    RayCastSensorCfg(
        name="terrain_only",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(),
        include_geom_groups=(0, 1),
    )


Ray targets
-----------

``RayCastSensor.add_ray_target(target)`` adds geometry that is not in the MuJoCo
model, for example many rigid meshes posed by user code. After the MuJoCo raycast,
inside the same sense graph, the sensor calls
``target(pnt, vec, dist, geomid, normal)`` on its ``[B, N]`` Warp buffers. The target
launches Warp kernels that overwrite a ray only where they find a nearer hit
(``dist < 0`` means no hit yet), so ``min_distance``, ``max_distance`` and
``geom_ids`` keep their meaning. Its buffers must stay at fixed addresses. On CUDA the
sense graph is captured when the simulation starts, so call
``Simulation.create_graph()`` after registering.


Output
------

``RayCastData`` is a dataclass with shape annotations relative to
``B`` (number of environments) and ``N`` (number of rays).

.. code-block:: python

    @dataclass
    class RayCastData:
        distances: Tensor   # [B, N] distance to hit, -1 if miss
        hit_pos_w: Tensor   # [B, N, 3] world-space hit positions
        normals_w: Tensor   # [B, N, 3] surface normals
        pos_w: Tensor       # [B, 3] sensor frame position
        quat_w: Tensor      # [B, 4] sensor frame orientation (w, x, y, z)
        geom_ids: Tensor    # [B, N] hit geom IDs, -1 for invalid returns

``geom_ids`` is simulation metadata for debugging and validation, not a
deployable sensor observation. Distances outside ``[min_distance, max_distance]``
are reported as -1, with zero normals and hit positions at the ray origin. A
surface inside the minimum range still occludes surfaces behind it.

.. note::

   Set ``debug_vis=True`` on the config to visualize ray hits at
   runtime. ``viz.stride`` draws every Nth ray while retaining all sensor data.


Examples
--------

.. code-block:: python

    from mjlab.sensor import (
        RayCastSensorCfg, GridPatternCfg, PinholeCameraPatternCfg, ObjRef,
    )

    # Dense height map for terrain-aware locomotion.
    height_scan = RayCastSensorCfg(
        name="height_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(
            size=(1.6, 1.0),
            resolution=0.1,
            direction=(0.0, 0.0, -1.0),
        ),
        ray_alignment="yaw",
        max_distance=2.0,
    )

    # Simulated depth camera using pinhole projection.
    depth_cam = RayCastSensorCfg(
        name="depth",
        frame=ObjRef(type="site", name="camera_site", entity="robot"),
        pattern=PinholeCameraPatternCfg.from_mujoco_camera("robot/depth_cam"),
        max_distance=10.0,
    )

    # Forward-facing obstacle scan.
    obstacle_scan = RayCastSensorCfg(
        name="obstacle",
        frame=ObjRef(type="body", name="head", entity="robot"),
        pattern=GridPatternCfg(
            size=(0.5, 0.3),
            resolution=0.1,
            direction=(-1.0, 0.0, 0.0),
        ),
        max_distance=3.0,
        include_geom_groups=(0,),
    )


TerrainHeightSensor
-------------------

``TerrainHeightSensor`` is a thin ``RayCastSensor`` subclass that adds
per-frame vertical clearance to the sensor data. It computes
``frame_z - hit_z`` for each ray, replaces misses with ``max_distance``,
and reduces across rays per frame.

.. code-block:: python

    from mjlab.sensor import TerrainHeightSensorCfg, RingPatternCfg, ObjRef

    cfg = TerrainHeightSensorCfg(
        name="foot_height",
        frame=(
            ObjRef(type="site", name="left_foot", entity="robot"),
            ObjRef(type="site", name="right_foot", entity="robot"),
        ),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
        max_distance=1.0,
        include_geom_groups=(0,),
    )

    # At runtime:
    sensor = env.scene["foot_height"]
    sensor.data.heights    # [B, F] vertical clearance per foot
    sensor.data.distances  # [B, N] raw ray distances (inherited)

The ``reduction`` config field controls how rays are aggregated within
each frame: ``"min"`` (default), ``"max"``, or ``"mean"``.
