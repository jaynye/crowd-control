#!/usr/bin/env bash
# CPU-only Viser view of the graph-driven plaza crowd (no GPU).
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab --qos=short scripts/slurm/viser_graph_demo_cpu.sbatch <graph.pt> [demo args]
#   Tunnel as for viser_graph_demo.sbatch (logs/slurm/viser-crowd-cpu-<JOBID>.out).
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h; overcap is preemptible. Compare
# speeds only within one GPU type. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=viser-crowd-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:55:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=""
graph="$1"; shift
exec .venv/bin/crowd demo viser --graph "$graph" --device cpu "$@"
