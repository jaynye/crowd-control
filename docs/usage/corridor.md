# Corridor fixture: primitive humans and LiDAR

`crowd demo corridor` runs a checkpoint-free corridor scene with primitive pedestrians
and an angular LiDAR. Use it to inspect geometry, occlusion, motion and physical contacts
without the motion graph or a robot. Unlike the [plaza](demos.md), its humans are
physical bodies. Code: `src/crowdmotion/sim/mjlab/corridor/` (`scene.make_crowd_env_cfg`,
`walking.CorridorWalk`, `humans.PrimitiveHumanCfg`); tests:
`tests/unit/test_corridor.py`.

## Interactive inspection

```bash
crowd demo corridor --viewer viser
crowd demo corridor --viewer native
```

The viewer shows scan hits and desired-velocity arrows; enable contact rendering in the
viewer to inspect interactions. The ceiling is hidden in the default rendering groups,
but remains collidable and visible to the LiDAR. Use the viewer's pause, step and
environment selection controls to inspect individual worlds.

| Option | Meaning |
|---|---|
| `--num-envs` | Independent worlds (1) |
| `--num-humans` | 0 to 8 (4) |
| `--seed` | Reproducible initial runs (0) |
| `--moving False` | Disable commanded motion while keeping physical dynamics |
| `--device` | `cpu` or `cuda:0` (default: CUDA if available) |

Waypoint durations, speeds, pauses and destinations use separate random state per human
and world. Resetting one world does not advance another world's stream. Resetting a
world continues its stream; start a fresh process with the same seed to replay a run.

## Headless runs and exports

```bash
crowd demo corridor --viewer none --device cpu --steps 250 --capture-every 25 \
    --output-dir runs/<UTC>_corridor
crowd demo corridor plot --scan-file runs/<UTC>_corridor/scans.npz
MUJOCO_GL=egl crowd demo corridor --viewer none --steps 250 \
    --output-dir runs/<UTC>_corridor_frames --render True
```

A bounded run records environment 0 at time zero, at each capture interval and at the
final step:

- `scans.npz`: timestamps, ranges, hit positions, hit geom IDs and names, sensor
  positions, human x/y/yaw states and contact-force histories (the most recent four
  physics substeps at each capture);
- `summary.json`: the configuration, and the peak per-human contact force across the
  whole run, including steps between captures. Contact forces use the sensor's
  `maxforce` reduction: the strongest individual contact, not a sum. Force vectors use
  the contact frame. Contacts include walls and other humans;
- `scene/`: the compiled scene and its assets.

`plot` writes `scan.png` with XY and XZ views of the returned points, human trajectories
and headings, without a simulator or OpenGL context. With `--render True`, rendered
`frame_*.png` files include the sensor and behavior overlays. Hit IDs are simulation
metadata for debugging and must never become policy observations.

## The model

- **Humans:** a head sphere, torso and pelvis ellipsoids and four limb capsules. The
  same surfaces take part in raycasting and physical collision. Bodies have finite mass
  and x/y/yaw joints; bounded velocity actuators move them between sampled waypoints,
  with occasional pauses and turns. Local repulsion gives basic yielding to other
  pedestrians and keeps them off the walls.
- **Limits:** the controller handles only the empty rectangular corridor; it is not a
  route planner and can deadlock in dense crowds. People stay upright on a level plane;
  limbs are rigid, feet do not step, and nobody falls or climbs stairs. It is a
  perception and contact baseline, not a biomechanical human model.
- **Room:** 12 × 4 × 2.8 m interior.
- **Scanner:** fixed at (−5, 0, 1.15) m, a test fixture rather than a calibrated G1
  mount. Its 180 × 16 rays cover 360° in azimuth and −7° to 52° in elevation, with a
  0.1 m minimum range and a 20 m fixture-specific maximum, as an instantaneous uniform
  raster every 20 ms. This approximates the Livox MID-360 field of view, not its
  non-repeating scan pattern, acquisition timing, noise, reflectivity or hardware range
  envelope. Dynamic geometry is refitted before each scan; returns inside the minimum
  range are invalidated after nearest-hit selection, so close occluders do not expose
  geometry behind them.
