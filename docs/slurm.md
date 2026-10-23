# Slurm (cluster only)

Everything in this repository runs on a local Linux GPU machine. On the lab cluster, GPU
work goes through Slurm with the jobs in `scripts/slurm/`. Each job file sources
`scripts/env.sh`, runs a `crowd` command or a `scripts/smoke/*.py` script, and documents
its arguments in its header comment.

## Conventions

- **Submit from the repository root**, after creating the log folder once:
  `mkdir -p logs/slurm`. Jobs log to `logs/slurm/<job name>-<job id>.out` (`logs/` is not
  tracked).
- **Partition and QoS:** `--partition=rl2-lab,overcap --qos=short`. `rl2-lab` is not
  preemptible and caps jobs at 4 h; `overcap` is preemptible. Use `rl2-lab` alone for
  live sessions (Viser) that must not be preempted.
- **GPUs:** prefer L40S, A40 or A5000; never Titan X or 2080 Ti. Exclude the node
  `bishop` (the job files do).
- **Compare speeds only within one GPU type**, and record the type; the benchmark
  configs are named for the A40.
- **Memory is not a scheduled resource** (`--mem` is not enforced). Check `FreeMem`
  (`scontrol show node <node>`) before stacking jobs on one node.
- Never launch generator calls inside a training loop; ARDY and Kimodo run offline only.

## Jobs

| Job file | Runs | Typical submission arguments |
|---|---|---|
| `behavior_eval.sbatch` | `crowd behavior evaluate`, then renders seed 0 (overview and follow) | `<graph.pt> <config.json> runs/<UTC>_<name>` |
| `iterate_worker.sbatch` | `crowd iterate worker`: drains an iteration session's queue | submitted by `crowd iterate start --executor slurm`, not by hand |
| `iterate_codex.sbatch` | `crowd iterate agent`: starts Codex and a local iteration session once the allocation begins | `[name] [--hours 2.75] [--prompt <task.txt>] [--model gpt-6-astra] [--effort xhigh]`; defaults to three hours on `rl2-lab` |
| `library_pipeline.sbatch` | `crowd library generate`, then graph quality and graph speed on the new library | `configs/library/<cfg>.json configs/bench/<graph cfg>.json <UTC stamp>` |
| `graph_eval.sbatch` | Graph quality and graph speed on an existing library | `<library.pt> <quality cfg> <graph cfg> <UTC stamp>` |
| `bench_throughput.sbatch` | `crowd bench throughput` | `configs/bench/<cfg>.json runs/<UTC>_<name>` |
| `bench_quality.sbatch` | `crowd bench quality` | `configs/bench/<cfg>.json runs/<UTC>_<name> [--group ...]` |
| `sim_speed.sbatch` | `crowd bench sim-speed` | `runs/<UTC>_sim_speed` |
| `bench_body_geometry.sbatch` | `crowd bench body-geometry` | `configs/bench/<cfg>.json runs/<UTC>_body_geometry [--only ...]` |
| `mjlab_graph_demo.sbatch` | `crowd demo video` (EGL) | `<graph.pt> runs/<UTC>_mjlab_graph_demo [demo args]` |
| `viser_graph_demo.sbatch` | `crowd demo viser` on a GPU node | `<graph.pt> [demo args]`; partition `rl2-lab` |
| `viser_graph_demo_cpu.sbatch` | `crowd demo viser --device cpu` on 4 CPU cores, no GPU | `<graph.pt> [demo args]`; partition `rl2-lab` |
| `viser_play.sbatch` | mjlab's `play` for any registered mjlab task in Viser | `TASK [play options]` |
| `gpu_pytest.sbatch` | `pytest` on a GPU (default `tests/gpu`) | `[pytest args]` |
| `ardy_smoke.sbatch`, `kimodo_smoke.sbatch` | Model load, generation and latency smoke tests | `[script args]` |
| `ardy_heading.sbatch` | ARDY body-heading targets: condition parity, angle convention and following (`scripts/smoke/ardy_heading.py`) | `[--out runs/<UTC>_ardy_heading]` |
| `profile_ardy.sbatch` | ARDY profiler table and `torch.compile` comparison | `runs/<UTC>_ardy_profile` |

Example:

```bash
mkdir -p logs/slurm
sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/behavior_eval.sbatch \
    <graph.pt> configs/behavior/baseline.json runs/<UTC>_behavior_baseline
squeue -u "$USER"
```

## Viser from a compute node

The Viser jobs print `VISER host=<node> port=<port>` in their log. Forward the port
through the login node and open `http://localhost:18080`:

```bash
grep VISER logs/slurm/viser-crowd-<JOBID>.out
ssh -N -L 18080:<node>:<port> <user>@<login host>
```

Viser has no authentication; keep it behind the tunnel. Cancel with `scancel <JOBID>`.
More in [demos](usage/demos.md#viser-interactive-in-a-browser).

## Iteration sessions on the cluster

`crowd iterate start <name> --executor slurm` submits `iterate_worker.sbatch` and records
the job ID in the session. Queue requests with `crowd iterate run`, and keep
`crowd iterate watch <session>` running on the login node: it journals results and
cancels only the session's job when the GPU-wait limit or the deadline passes
([behavior iteration](usage/behavior-iteration.md#iteration-sessions)).

## Start Codex automatically in a batch job

`iterate_codex.sbatch` launches the agent itself on the compute node with
`codex exec`, then uses local iteration requests within that allocation. No terminal
or login-node watcher is needed. The session is created after allocation, so queue
waiting does not consume its iteration budget or trigger the worker's five-minute
GPU-wait cutoff. The initial batch submission can still queue.

Before submission, install Codex in the shared home and sign in there. Confirm
`codex login status` succeeds. The job reuses that saved login; never put credentials
in the script or prompt. The compute node must have outbound access to OpenAI.
The launcher runs `codex exec --approve-for-me` (Codex 0.160.0), which selects the
workspace-write sandbox and routes approval requests to Codex's automatic reviewer.
That flag rejects an explicit `--sandbox`. `codex exec` runs directly in the batch
process; it does not attach the interactive TUI to a shared daemon.

On sky1 (Ubuntu 24.04, `kernel.apparmor_restrict_unprivileged_userns=1`) Codex's
bubblewrap sandbox cannot start (`bwrap: loopback: Failed RTM_NEWADDR` or
`setting up uid map: Permission denied`). Commands then run outside the sandbox
only after the automatic reviewer approves an escalation, so in practice the agent's
commands are reviewed but not sandboxed. Compute nodes have not been checked
separately.

Before each `codex exec` launch, `crowd iterate agent` checks the requested model and
reasoning effort against `codex debug models`. It then runs the exact launch arguments
with empty stdin: Codex parses them and stops with "No prompt provided" before any
model request. Any other outcome stops the launcher before it opens a run. Run the same
check on the login node before submitting:

```bash
mkdir -p logs/slurm
crowd iterate agent --check            # or with the --model/--effort you will submit
sbatch scripts/slurm/iterate_codex.sbatch
```

Defaults: one A40, four CPUs, three-hour allocation, 2.75-hour agent/session deadline,
`rl2-lab` account and partition, `short` QoS, `bishop` excluded. This wrapper uses
`rl2-lab` alone to avoid preemption during code editing. To use the four-hour limit:

```bash
sbatch --time=04:00:00 scripts/slurm/iterate_codex.sbatch --hours 3.75
```

The prompt is `configs/behavior/iterate_codex.txt`; review it before submitting. Pass
`--prompt <task.txt>`, `--model <slug>`, `--effort <level>` or `--graph <graph.pt>`
after the script name to override them, for example
`sbatch scripts/slurm/iterate_codex.sbatch --model gpt-6-sol --effort max`. The
default is `gpt-6-astra` at `xhigh` reasoning effort, independent of
`~/.codex/config.toml`; `codex debug models` lists the slugs and the efforts each
supports. `run.json` records the model, effort and exact argv.

`configs/behavior/iterate_codex_encounter.txt` is a second prompt, for the
[encounter design](usage/behavior-iteration.md#encounter-design) on the opt-in
`graph-turns` graph. `--graph` only records the path and names it in the prompt, so
also export the graph and its rotations; otherwise any command run without `--graph`,
including the rigid-mesh mjlab check, falls back to `graph-default`:

```bash
G=$(.venv/bin/python -c "from crowdmotion.core.paths import artifact_path as a; print(a('graph-turns'))")
R=$(.venv/bin/python -c "from crowdmotion.core.paths import artifact_path as a; print(a('graph-turns-rotations'))")
sbatch --time=04:00:00 --export=ALL,CROWD_GRAPH=$G,CROWD_GRAPH_ROTATIONS=$R \
    scripts/slurm/iterate_codex.sbatch encounter_turns --hours 3.75 \
    --prompt configs/behavior/iterate_codex_encounter.txt --graph $G
```

`configs/behavior/iterate_codex_decisive.txt` is a third prompt for the same graph and
exports: it targets standoffs and decisive avoidance in the encounter design, with a
tunable stubborn population. Pass it as `--prompt` in the command above.

The goldens also read `CROWD_GRAPH`, so inside such a run they must be run with
`env -u CROWD_GRAPH -u CROWD_GRAPH_ROTATIONS`; the prompt says so. Keep `--hours`
at least fifteen minutes below the allocation's `--time`. Only one agent should edit
a checkout at a time; use separate checkouts for simultaneous sessions.

Monitor the job and read the run path printed in its Slurm log:

```bash
squeue -u "$USER"
tail -f logs/slurm/crowd-codex-<JOBID>.out
```

The fresh run holds `run.json`, `agent_prompt.txt`, `agent.jsonl`,
`agent.stderr.log`, `final.md` (when Codex finishes), `changes.diff` and `session/`.
Within `session/`, the usual journal, request logs and experimental outputs remain
available. `run.json` distinguishes successful agent exit, failure, deadline and
interruption; a successful exit does not by itself establish a behavior improvement.

Stop gracefully with `crowd iterate stop <run>/session --reason user_stop`, or cancel
the allocation with `scancel <JOBID>`. The runner closes the session and terminates
the agent on STOP, deadline or SIGTERM. Slurm sends a batch-process SIGTERM two
minutes before its hard cutoff. Outputs already written remain, but an interrupted
agent may leave unfinished requests and no final report. The job is not automatically
requeued. After a failure, inspect the logs, working tree and run-log entry before
submitting a fresh session; `scripts/index_runs.py` drafts missing run-log rows.

This launcher has CPU lifecycle tests with a fake Codex executable that rejects the
same argument conflict as the real CLI, and its `--check` runs against the real CLI
without inference. Authentication, network access, EGL and real refinement still
require a compute-node run.
