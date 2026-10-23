# Demos: the mjlab plaza

A walled 14 × 10 m plaza in mjlab where graph-driven humans walk to random goals
while a fixed LiDAR rig scans them. It is the integration check between the behavior
layer, the motion graph and the simulator (one environment by default; not a training
setup). Both demos read the saved graph: `--graph`, else `$CROWD_GRAPH`, else the
`graph-default` artifact.

Both demos accept `--config <behavior.json>` from the behavior evaluator. Its
`steering` and `graph_overrides` are loaded, including the plaza half sizes;
`--humans`, `--seed`, `--seconds` (video) and `--num-envs` (Viser) remain CLI settings.

## The scene

`crowdmotion.sim.mjlab.plaza.make_plaza_env_cfg(graph_path, humans, ...)` returns an
mjlab `ManagerBasedRlEnvCfg` with:

- the plaza floor and walls (`sim.scene`; walls in geom group 2, visible to the LiDAR);
- a scanner on a mast at (−6, 0) with an mjlab `RayCastSensor` using our
  `LidarPatternCfg`: 360 × 16 rays, 0.1–20 m;
- `humans` humans per environment, in one or both body geometries (`human_geometry`,
  `--geometry`):
  - **rigid mesh** (the default): ARDY's Core skin at full resolution (18,152 faces),
    cut into 22 rigid pieces, one per bone (`sim.body`). Each piece is a visual mesh
    geom on a mocap body in geom group 1, which the LiDAR does not trace. The LiDAR
    hits the pieces through a ray target that `GraphCrowd` registers with the sensor
    (`sim.rays`; mjlab's `RayCastSensor.add_ray_target`), so hit geom ids still name
    the body part.
  - **capsules**: one mocap body per Core bone plus a head cap (`sim.capsules`), in
    geom group 2, traced by MuJoCo.

  Rigid humans need the graph's joint rotations, the `graph-rotations` artifact
  (`$CROWD_GRAPH_ROTATIONS`; built by `crowd graph rotations`), and ARDY's Core skin
  from the `third_party/ardy` clone. Without them the plaza fails with a message
  naming the missing file; it never falls back to capsules.

Everything is kinematic: contacts are disabled, and there is one physics step per 20 ms
environment step.

The humans are driven by the **`GraphCrowd` command term** (`sim.mjlab.command`). Each
environment step it advances the clock; at each 0.4 s edge it asks its
`CrowdController` for the next edge (plaza intent and steering, root path, graph step);
then it interpolates the 20 fps frames to the current time, converts them to MuJoCo Z-up
`(x, -z, y)` and writes mocap poses. Rigid pieces also take bone rotations: the saved
rotation of each frame, turned by the edge's heading, blended between frames and
swung onto the posed joints so pieces stay attached through blended transitions.
Poses are written after the environment's `forward()`, so viewers and MuJoCo-traced
geometry (capsules) appear to the LiDAR one step (20 ms) late; the rigid ray target
reads the poses written this step. A reset draws new
start states and goals. Debug overlays show goal discs, commanded-velocity arrows and
LiDAR hits. Because the crowd is a command term, every mjlab viewer shows it.

## Video

```bash
MUJOCO_GL=egl crowd demo video --humans 16 --seconds 20 --out runs/<UTC>_mjlab_graph_demo
```

Renders offscreen and writes `crowd.mp4` (50 fps), `summary.json` (minimum pairwise root
gap, mean number of LiDAR rays hitting humans, natural-successor share, step and render
times) and `run.json`. A GPU with EGL is the normal setting; on CPU use
`--device cpu` and `MUJOCO_GL=osmesa` if OSMesa is installed. On the cluster:
`scripts/slurm/mjlab_graph_demo.sbatch` ([Slurm](../slurm.md)).

## Viser (interactive, in a browser)

```bash
crowd demo viser --port 8080                       # GPU if available
crowd demo viser --device cpu --port 8090          # CPU
```

This serves mjlab's `ViserPlayViewer`: play/pause, single step, speed, reset (a new
random crowd) and overlay toggles. The browser renders the 3D scene; the process prints
`VISER host=<host> port=<port>`.

- **Locally:** open `http://localhost:<port>`.
- **On a remote machine** (a workstation, a login node or a Slurm job), forward the port
  through SSH and open `http://localhost:18080` on your computer:
  ```bash
  ssh -N -L 18080:localhost:<port> <user>@<remote host>          # process on that host
  ssh -N -L 18080:<compute node>:<port> <user>@<login host>      # Slurm job on a compute node
  ```
  Viser listens on all interfaces without authentication; keep it behind the tunnel
  rather than exposing the port. If the port is taken, Viser picks another: use the
  port it prints.

Performance notes:

- **Defaults.** On CPU the demo uses one PyTorch thread, draws every 16th LiDAR hit and
  targets 30 viewer FPS; on GPU every 8th hit and 60 FPS. Override with `--threads`,
  `--lidar-stride` and `--frame-rate`; `--verbose` prints the viewer FPS and real-time
  factor.
- **Drawing LiDAR hits** (per-frame Python) costs more than the simulation, so a sparser
  `--lidar-stride` is the main speed knob.
- **Measured:** about 1× real time at 50 FPS on a GPU node with every 8th hit; 1.0× at
  30 FPS on a 4-core CPU-only job; 0.4–0.7× on a heavily loaded shared login node.
- **Threads.** With PyTorch's default thread count on a busy shared host, one 20 ms step
  took 500–800 ms. Keep one thread on shared machines.
- **First launch** compiles Warp kernels (about a minute on CPU); later launches use the
  cache.

On the cluster, `scripts/slurm/viser_graph_demo.sbatch` (GPU) and
`viser_graph_demo_cpu.sbatch` (4 CPU cores) start the viewer in a job; read the
`VISER host=... port=...` line from the job log, then tunnel as above. Use the
non-preemptible partition for live sessions. `scripts/slurm/viser_play.sbatch` serves
any mjlab task in Viser the same way.

## Switching body geometry

The Viser demo builds both geometries, starting with `--geometry` (default `rigid`).
Under **Commands → Crowd body**, the **Geometry** dropdown switches the humans shown
and the geometry the LiDAR hits. The switch is queued and applied on the next
simulation step: the other geometry is parked below the floor, and the rigid ray
target is turned off for capsules. The video demo builds only the `--geometry` it is
given.

## Live behavior controls

```bash
crowd demo viser --config <candidate.json> --humans 16 --seed 0 --port 8080
```

Under **Commands → Crowd behavior**, edit sampling temperature, pose/entry-velocity/end-velocity/
facing/heading costs, forward-steering lead and speed floor, evasive orientation
release, turn-before-walk settings, native-yaw preservation, posture cost and
switching rates, initial hunched share, the flat-ground mask toggle, and the
planted-twist mask with its foot yaw limit. Higher
forward preference makes ordinary lateral/backward selections less likely; it is
a soft cost, not a ban. Posture settings and diagnostic definitions are described
in [behavior iteration](behavior-iteration.md#posture-and-flat-ground-transitions).

A config with `"design": "encounter"` runs the
[encounter design](behavior-iteration.md#encounter-design). Its panel keeps the graph
controls but replaces the plaza steering controls with encounter controls:

- bubble radius, passing margin and look-ahead;
- switch rate and urgency exponent;
- the assertiveness range, which applies at the next reset;
- patience;
- strategy weights, far and near;
- veer radii, head-on side preference, and in-place probability and radius range for
  ordinary turns;
- body radius, stop clearance, stop look-ahead and the waypoint wait probability.

Tuple settings appear as one control per element. An edit that would make the
settings invalid, such as a bubble no larger than body contact, or a mask setting that
leaves no admissible edge, is rejected with a status message.

On the opt-in turning graph, rigid humans need that graph's rotations as well
(`CROWD_GRAPH_ROTATIONS`; [data](../data.md)):

```bash
G=$(.venv/bin/python -c "from crowdmotion.core.paths import artifact_path as a; print(a('graph-turns'))")
R=$(.venv/bin/python -c "from crowdmotion.core.paths import artifact_path as a; print(a('graph-turns-rotations'))")
CROWD_GRAPH_ROTATIONS=$R crowd demo viser --graph $G --config configs/behavior/encounter_turns.json --port 8080
# on the cluster:
sbatch --partition=rl2-lab --qos=short --export=ALL,CROWD_GRAPH_ROTATIONS=$R \
    scripts/slurm/viser_graph_demo.sbatch $G --config configs/behavior/encounter_turns.json
```

Edits are queued from the GUI thread and applied on the simulation thread at the
next 0.4 s motion edge, or before reset. While paused, advance or reset to apply
pending changes. Controls affect all environments in that viewer. The initial
posture distribution is used only at the next reset; rates change existing styles
stochastically, without redrawing them. Enabling style midway through an episode
cannot retroactively change its start pose. Masking a boundary with no admissible
continuation holds the current pose until settings or reset restore one.

Each crowd has a private runtime configuration even when saved graph tables are
shared. The panel does not edit the config file or saved graph. Invalid angle pairs
are rejected with a status message. Geometry, the baked `hunch_ratio`, graph
neighbors and latent class calibration are not live controls. The bindings are
testable headlessly; visual browser interaction still needs a human review.
