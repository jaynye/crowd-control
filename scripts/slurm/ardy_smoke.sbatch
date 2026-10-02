#!/usr/bin/env bash
# ARDY Core H8/H40 load, autoregressive step and skin smoke test.
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/ardy_smoke.sbatch [--out runs/<UTC>_ardy_smoke/smoke.json]
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h; overcap is preemptible. Compare
# speeds only within one GPU type. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=ardy-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec .venv/bin/python scripts/smoke/ardy_smoke.py "$@"
