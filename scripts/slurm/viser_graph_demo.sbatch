#!/usr/bin/env bash
# Interactive Viser view of the graph-driven plaza crowd on a GPU node.
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab --qos=short scripts/slurm/viser_graph_demo.sbatch <graph.pt> [demo args]
#   (rl2-lab only: overcap may preempt a live demo.) Read the VISER host/port line in
#   logs/slurm/viser-crowd-<JOBID>.out, then on your computer:
#     ssh -N -L 18080:<host>:<port> <user>@sky1.cc.gatech.edu   and open http://localhost:18080
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h; overcap is preemptible. Compare
# speeds only within one GPU type. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=viser-crowd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:55:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
export MUJOCO_GL=egl
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
graph="$1"; shift
exec .venv/bin/crowd demo viser --graph "$graph" --device cuda:0 "$@"
