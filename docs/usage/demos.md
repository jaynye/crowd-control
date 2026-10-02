# Demos: the mjlab plaza

A walled 14 × 10 m plaza in mjlab where graph-driven capsule humans walk to random goals
while a fixed LiDAR rig scans them. It is the integration check between the behavior
layer, the motion graph and the simulator (one environment by default; not a training
setup). Both demos read the saved graph: `--graph`, else `$CROWD_GRAPH`, else the
`graph-default` artifact.

## The scene

`crowdmotion.sim.mjlab.plaza.make_plaza_env_cfg(graph_path, humans, ...)` returns an
mjlab `ManagerBasedRlEnvCfg` with:

- the plaza floor and walls (`sim.scene`; walls in geom group 2, visible to the LiDAR);
- a scanner on a mast at (−6, 0) with an mjlab `RayCastSensor` using our
  `LidarPatternCfg`: 360 × 16 rays, 0.1–20 m;
- `humans` capsule humans per environment: one mocap body per ARDY Core bone plus a
  head cap (`sim.capsules`), in geom group 2.

Everything is kinematic: contacts are disabled, and there is one physics step per 20 ms
environment step.

The humans are driven by the **`GraphCrowd` command term** (`sim.mjlab.command`). Each
environment step it advances the clock; at each 0.4 s edge it asks its
`CrowdController` for the next edge (plaza intent and steering, root path, graph step);
then it interpolates the 20 fps frames to the current time, converts them to MuJoCo Z-up
`(x, -z, y)` and writes mocap poses. Poses are written after the environment's
`forward()`, so sensors and viewers see them one step (20 ms) late. A reset draws new
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
