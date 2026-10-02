# mjlab (our copy)

mjlab is the simulator: MuJoCo-Warp environments with managers for commands, sensors,
rewards and viewers. Since 2026-10-01 it is **first-party** (ledger D13): its source is
tracked in this repository under `mjlab/`, with our changes. The upstream base commit,
`c2e1e0640` of `mujocolab/mjlab` ("Judge the terrain curriculum against the
distance the episode commanded (#1188)"), is recorded in `third_party.toml` for diffing
and merging upstream releases. Inspected at that base plus our diff.

## Our changes to the simulator

Simulator features, general enough to offer upstream. They are listed in
`mjlab/docs/source/changelog.rst` ("Upcoming version") and documented in
`mjlab/docs/source/sensors/raycast_sensor.rst`.

- **`LidarPatternCfg`** (`mjlab/src/mjlab/sensor/raycast_sensor.py`): an angular raster
  from one origin, +X forward and +Z up, angles in degrees. Defaults cover the MID-360
  field of view: azimuth −180…180°, elevation −7…52°, 180 × 16 rays. Rays are
  elevation-major, then increasing azimuth. A full 360° sweep drops the duplicate
  endpoint; partial sweeps include both ends; one sample uses the lower bound. It is an
  instantaneous raster, not the device's non-repeating temporal pattern, and adds no
  noise.
- **`RayCastSensorCfg.min_distance`** (default 0): returns nearer than it report −1, and
  a surface inside the minimum range still occludes what is behind it. `build()`
  requires `0 <= min_distance < max_distance`.
- **`RayCastData.geom_ids`** `[B, N]`: hit geom ID, −1 for invalid returns. Simulation
  metadata for debugging and validation (for example counting rays that hit humans); it
  must never become a policy observation.
- **`RayCastSensorCfg.VizCfg.stride`**: draw every Nth ray without subsampling the data.
  The crowd demos use it to keep Viser responsive.

Tests: `mjlab/tests/test_lidar_pattern.py` (ray angles, invalid patterns). The scene
tests that exercise the sensor against moving geometry (occlusion, minimum range,
CPU-ray agreement, floor/ceiling/wall distances) use the corridor fixture and live in
`tests/unit/test_corridor.py`. Together they replace the former
`mjlab/tests/test_crowd_lidar.py`.

The full diff against the base is five files (+125/−6): `.gitignore`, the changelog
bullet, `docs/source/sensors/raycast_sensor.rst`, `src/mjlab/sensor/__init__.py` and
`src/mjlab/sensor/raycast_sensor.py`, plus the new `tests/test_lidar_pattern.py`. How to
merge an upstream release on top: [third-party sources](../../docs/third-party.md#merging-an-upstream-release).

## Corridor fixture inherited from earlier sessions (2026-09-26 to 09-28)

The agent Astra and earlier sessions built a checkpoint-free **corridor fixture** inside
the mjlab clone (`src/mjlab/tasks/crowd/`, `asset_zoo/humans/primitives.py`,
`scripts/demos/crowd.py`). On 2026-10-01 it moved out of mjlab, unchanged in behavior
(checked by the `corridor` golden), to `src/crowdmotion/sim/mjlab/corridor/`
(`scene.make_crowd_env_cfg`, `walking.CorridorWalk`, `humans.PrimitiveHumanCfg`, the demo
and scan plot behind `crowd demo corridor`), so that our mjlab diff holds only simulator
features. The fixture is
a 12 × 4 × 2.8 m room, a fixed scanner at (−5, 0, 1.15) m, and up to 8 **primitive
humans** (head sphere, torso and pelvis ellipsoids, four limb capsules; finite mass,
x/y/yaw joints, bounded velocity actuators). `CorridorWalk` is a seeded waypoint intent
with separate random state per human and world: resetting one world never advances
another's stream. Limbs are rigid and feet do not step; local repulsion can deadlock in
dense crowds. It is a perception and contact baseline, not a human model, and was
superseded for body motion by the motion graph
([crowd architecture](../core/design/crowd-architecture.md)).

Usage: [corridor](../../docs/usage/corridor.md). History: the
[G1 crowd simulation plan](../archive/2026-09-28-mjlab-g1-crowd-simulation-plan.md) and
the [Kimodo stochastic humans proposal](../archive/2026-09-28-mjlab-kimodo-stochastic-humans.md).

## Viewing on a cluster

mjlab's **Viser** viewer serves the scene over HTTP/WebSocket; the browser renders it,
so no VNC, VirtualGL or X display is needed. On Slurm, run the viewer in the job and
forward its port with `ssh -N -L <local>:<node>:<port> <login host>`. The listener
defaults to `0.0.0.0:8080` but can move if the port is busy: read the actual port from
the job log. Viser has no authentication; keep it behind the tunnel. Viser does not
support mouse body perturbation (the native viewer does). Details and the agent control
plane proposal: [archived note](../archive/2026-09-26-mjlab-slurm-viser-and-agentic-crowds.md).

## Traps

- `ManagerBasedRlEnv` on CPU uses MuJoCo-Warp's CPU backend; the first run compiles
  kernels (minutes), later runs load them from the Warp cache.
- Run `uv sync` only from the repository root: inside `mjlab/` it does an exact sync of
  the member and uninstalls the root's other dependencies (`AGENTS.md`).
- Commands run after the env's `forward()`, so sensors and viewers see poses written by a
  command term one step late.
