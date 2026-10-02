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
| `library_pipeline.sbatch` | `crowd library generate`, then graph quality and graph speed on the new library | `configs/library/<cfg>.json configs/bench/<graph cfg>.json <UTC stamp>` |
| `graph_eval.sbatch` | Graph quality and graph speed on an existing library | `<library.pt> <quality cfg> <graph cfg> <UTC stamp>` |
| `bench_throughput.sbatch` | `crowd bench throughput` | `configs/bench/<cfg>.json runs/<UTC>_<name>` |
| `bench_quality.sbatch` | `crowd bench quality` | `configs/bench/<cfg>.json runs/<UTC>_<name> [--group ...]` |
| `sim_speed.sbatch` | `crowd bench sim-speed` | `runs/<UTC>_sim_speed` |
| `mjlab_graph_demo.sbatch` | `crowd demo video` (EGL) | `<graph.pt> runs/<UTC>_mjlab_graph_demo [demo args]` |
| `viser_graph_demo.sbatch` | `crowd demo viser` on a GPU node | `<graph.pt> [demo args]`; partition `rl2-lab` |
| `viser_graph_demo_cpu.sbatch` | `crowd demo viser --device cpu` on 4 CPU cores, no GPU | `<graph.pt> [demo args]`; partition `rl2-lab` |
| `viser_play.sbatch` | mjlab's `play` for any registered mjlab task in Viser | `TASK [play options]` |
| `gpu_pytest.sbatch` | `pytest` on a GPU (default `tests/gpu`) | `[pytest args]` |
| `ardy_smoke.sbatch`, `kimodo_smoke.sbatch` | Model load, generation and latency smoke tests | `[script args]` |
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
