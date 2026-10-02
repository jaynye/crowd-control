#!/usr/bin/env bash
# Evaluate a behavior config on a saved graph and render seed 0 (overview, follow).
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/behavior_eval.sbatch <graph.pt> <config.json> runs/<UTC>_behavior_candidate
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h; overcap is preemptible. Compare
# speeds only within one GPU type. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=crowd-behavior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
export MUJOCO_GL=egl
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
.venv/bin/crowd behavior evaluate --graph "$1" --config "$2" --out "$3" --device cuda:0
.venv/bin/crowd behavior render --run "$3" --seed 0 --view overview
.venv/bin/crowd behavior render --run "$3" --seed 0 --view follow
