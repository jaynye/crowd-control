# Slurm viewer and agentic crowd simulation

> Archived from `mjlab/memory/2026-09-26-slurm-viser-and-agentic-crowds.md` on 2026-10-01; historical. Links were adjusted to the archive location.

**Update 2026-09-28:** the checkout moved to `~/flash/crowd-control/mjlab`.
The uv environment is now project-wide (`crowd-control/pyproject.toml`,
`uv.lock`, `.venv`, `.uv-python`; mjlab is a workspace member and
`mjlab/.venv` is a symlink to `../.venv`). Run `uv sync --locked` from
`crowd-control/`, not from `mjlab/`. Paths below predate the move.

Date: 2026-09-26. This is an implementation note for the mjlab checkout at
`~/flash/mjlab` (`~/flash` resolves to `/coc/flash7/jni66`).

## Decision

Use mjlab's existing **Viser** viewer in a one-GPU Slurm job, and forward its
HTTP/WebSocket port through SSH to a browser on the user's computer. The browser
renders the 3D scene; no VNC server, VirtualGL, or X display is needed. Keep a
`uv` project environment on the shared filesystem. Use a container only
if a compute-node dependency or driver mismatch makes the uv setup unreliable.

The runnable job file is `scripts/slurm/viser_play.sbatch`. It invokes `play`
with `--viewer viser`, one environment by default, and optional normal `play`
arguments. It does not require a checkpoint for `--agent zero` or
`--agent random`.

## What this checkout already does

- `docs/source/viewers.rst` documents native, Viser, and offscreen viewers.
  Viser has play/pause, one-step, reset, speed, environment selection, command
  controls, reward and metric charts, contacts, and live camera sensor panels.
- `src/mjlab/scripts/play.py` creates a **new** environment and starts
  `ViserPlayViewer` in that same process. `--viewer auto` chooses Viser on a
  headless host; specifying `--viewer viser` makes the Slurm job predictable.
- Viser's default listener is `0.0.0.0:8080`; it can choose another port if
  occupied, so use the **actual** port in the job log. mjlab's `play` CLI
  currently has no public host/port flags.
- The Viser viewer is interactive for playback and command controls, but it
  does **not** offer mouse body perturbations. It does not reflect every
  per-world visual domain-randomization change. The native viewer does both.
- `play` is **not** an attachment to an existing training rollout. It is a
  separate inference rollout from a checkpoint. Its checkpoint selector can
  reload later checkpoints. To observe the exact training process, add a
  state-streaming bridge or integrate Viser into that process.
- No crowd or pedestrian task was found under `src/mjlab/tasks` as of this
  note. Crowd navigation will need its own scene, human behavior layer, robot
  task, reward and termination definitions, and evaluation scenarios.

## Setup and launch

From the login host, install once in the shared checkout. The Python version
requested by `.python-version` is 3.13. The managed interpreter is stored in
`.uv-python/` and the environment in `.venv/`; both are ignored by Git.

```bash
cd ~/flash/mjlab
UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" uv sync --locked --no-dev
mkdir -p logs/slurm
sbatch scripts/slurm/viser_play.sbatch \
  Mjlab-Velocity-Flat-Unitree-G1 --agent zero
```

The job requests one GPU, six CPUs, 24 GiB of memory, and four hours. Adjust
these via `sbatch` options before the script name, for example
`--partition=<your-partition> --time=02:00:00`. The cluster's default partition
is currently `overcap`, which reports preemption/requeue; a non-preemptible
partition is preferable for an interactive session if available. To view more
than one environment, submit with `MJLAB_NUM_ENVS=4` exported to the job, for
example `sbatch --export=ALL,MJLAB_NUM_ENVS=4 ...`.

For a trained policy, replace `--agent zero` with either
`--checkpoint-file /shared/path/model.pt` or
`--wandb-run-path entity/project/run_id`. Tracking tasks may also need
`--motion-file`. Keep all referenced files on a path visible to the compute
node. Viser's checkpoint selector can load other checkpoints from the same
local directory or W&B run while the viewer stays open.

```bash
squeue -u "$USER" -o '%.18i %.2t %N'
tail -f logs/slurm/mjlab-viser-<JOBID>.out
```

Read the compute hostname and Viser listener port in that log. On the **local
computer** (replace the login hostname if SSH uses a different gateway):

```bash
ssh -N -L 18080:<COMPUTE_NODE>:<VISER_PORT> \
  jni66@sky2.cc.gatech.edu
```

Open `http://localhost:18080` in the local browser. The login host must be able
to reach the compute node's port. The viewer binds all compute-node interfaces
by default and has no mjlab authentication layer; keep it on the cluster's
internal network and use the SSH tunnel rather than publishing a public URL.
If direct SSH to allocated compute nodes is allowed, a future `--viser-host
127.0.0.1` option would permit a narrower bind with `ProxyJump`. Do not assume
that direct compute-node SSH is enabled here. Cancel with `scancel <JOBID>`;
the browser disconnects when the allocation ends.

`MUJOCO_GL=egl` is set in the job for headless GPU camera rendering. The Viser
3D viewport itself is rendered by the browser. `--video` records trained-policy
playback for later review and is useful for reproducibility, but does not
provide live control; `play` disables video with dummy agents. Camera sensor
panels appear only when the environment config defines
`CameraSensor` instances. A global camera or saved PNG sequence is a useful
fallback artifact, not the interactive viewer.

## uv versus containers

The repo already locks dependencies with `uv.lock`, uses `uv run`, and has a
Dockerfile for CUDA 12.8 and EGL. For this cluster, `uv` is the shortest path:
one checkout, one shared `.venv`, and no additional container runtime. Docker
being installed on the login host does not establish that a daemon or Docker
jobs are allowed on compute nodes. Neither `apptainer` nor `singularity` was
found on this login host. If Apptainer is offered on compute nodes, a SIF built
from the repo image and `apptainer exec --nv` can be tested, with the shared
checkout/checkpoints bound in. Driver, CUDA, graphics library, and cluster
policy compatibility still need a compute-node smoke test. Containerizing does
not remove the need for an SSH tunnel to the Viser port.

## Agent control plane for GPT-6 Astra

Keep the **simulator and low-level controllers in the allocated GPU process**.
Expose a small, typed session API from that process. Put an MCP adapter near
the agent (on the local computer or an approved service host), connected to
the worker through an authenticated SSH tunnel or cluster-approved transport.
The OpenAI Responses API supports `gpt-6-astra` and remote MCP; the official
Secure MCP Tunnel is an option when a private MCP server must be reachable by
the hosted API. If compute nodes cannot call outside APIs, run the model client
outside Slurm and send only compact commands/observations over the bridge.

Proposed MCP tools:

| Tool | Contract |
| --- | --- |
| `start_session` / `stop_session` | Manage a bounded Slurm job. |
| `get_status` / `get_scene` | Return job and scene state. |
| `set_human_intent` | Set bounded waypoints, speed, and duration. |
| `advance` | Step a fixed interval and return the observation. |
| `reset` | Reset to a specified scenario and seed. |
| `get_camera_frame` | Return RGB, timestamp, and camera pose. |
| `save_episode` | Flush a versioned trajectory and scenario. |

All simulation mutations should execute on one simulation thread. Queue tool
requests into that thread; do not let HTTP/MCP handler threads call the GPU
simulation concurrently. Give each command a session ID and sequence number;
return step count and timestamp so observations cannot be confused across
resets. Bound action magnitude, duration, agent count, and tool rate. Use
explicit approval for starting expensive jobs or saving/publishing external
artifacts in a deployed agent, according to that application's policy.

Use state vectors and camera-sensor frames as agent observations. Viser's
browser viewport is for humans; `ClientHandle.get_render()` depends on a
connected browser client and is not a dependable headless sensor. GPT-6 Astra
can inspect images via the Responses API, but sending every physics frame to
the model would be slow and costly. Let the model issue sparse human intentions
or trajectories (for example, every 0.5-2 simulated seconds); let deterministic
controllers interpolate and handle collision avoidance at the physics rate.

The first research milestone is a seeded crowd scene with simple baseline
pedestrian policies and one humanoid traversal task. Then collect model-guided
episodes with human intents, observations, actions, collisions, near misses,
robot progress, and full seed/config/checkpoint provenance. Compare against
baseline pedestrians before using those episodes as training or evaluation
data. A live Viser session is a debugging and inspection surface, while saved
trajectories are the reproducible training artifact.

## Validation and open items

At the time of this note, Slurm reports `overcap` as the default partition,
with GPU types including A40 and L40S. The locked Python 3.13 environment was
installed successfully in `.venv/` and imports mjlab and Viser. The `play`
help output confirms the launcher flags. The job file passed `bash -n` and
`sbatch --test-only`; that Slurm check submits no job. No
GPU allocation, tunnel, or live viewer was launched during this investigation,
so compute-node CUDA/EGL behavior and login-to-compute port reachability remain
to be checked with a short real job. If the Viser page loads but camera panels
fail, test EGL separately; the browser-based 3D scene can still work without
offscreen camera rendering. If the page is unreachable, inspect the actual
listener port in the log and whether the login host can route to that node.

## Sources

- Local `docs/source/viewers.rst`, `src/mjlab/scripts/play.py`, and
  `src/mjlab/viewer/viser/viewer.py`.
- Viser server defaults and port behavior:
  https://viser.studio/main/_modules/viser/_viser/
- Slurm `sbatch` and GPU GRES:
  https://slurm.schedmd.com/sbatch.html
- uv project environment path:
  https://docs.astral.sh/uv/concepts/projects/config/
- Apptainer NVIDIA GPU support:
  https://apptainer.org/docs/user/main/gpu.html
- Official OpenAI Docs for GPT-6 Astra and MCP:
  https://developers.openai.com/api/docs/models/gpt-6-astra
  https://developers.openai.com/api/docs/guides/tools-connectors-mcp
- Official OpenAI Docs for image input:
  https://developers.openai.com/api/docs/guides/images-vision
