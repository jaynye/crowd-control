#!/usr/bin/env bash
# Render the mjlab plaza (graph-driven crowd, LiDAR) to MP4.
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/mjlab_graph_demo.sbatch <graph.pt> runs/<UTC>_mjlab_graph_demo [demo args]
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h; overcap is preemptible. Compare
# speeds only within one GPU type. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=mjlab-graph-demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
export MUJOCO_GL=egl
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
graph="$1"; out="$2"; shift 2
exec .venv/bin/crowd demo video --graph "$graph" --out "$out" --device cuda:0 "$@"
