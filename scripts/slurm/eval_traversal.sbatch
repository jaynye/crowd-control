#!/usr/bin/env bash
# Measure trained traversal policies on a GPU node: `crowd eval traversal` (inference only).
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/eval_traversal.sbatch runs/<UTC>_train_traversal_<name> [eval args]
#   e.g. [--last 3 --crowd both --mode both --envs 4096 --seconds 60]
# The job writes runs/<UTC>_eval_traversal_<name>/ (metrics.json, tables/). A preempted
# evaluation is cheap to repeat, so overcap is fine.
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=eval-traversal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec .venv/bin/crowd eval traversal "$@" --device cuda:0
