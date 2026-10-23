#!/usr/bin/env bash
# LiDAR cost and fidelity of human body geometries (capsules, rigid mesh pieces, skinned mesh).
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/bench_body_geometry.sbatch \
#       configs/bench/body_geometry_a40.json runs/<UTC>_body_geometry [--only ...] [--no-fidelity]
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h; overcap is preemptible (rows already
# written stay in rows.jsonl). Compare speeds only within one GPU type. Logs go to
# logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=body-geometry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
config="$1"
out="$2"
shift 2
exec .venv/bin/crowd bench body-geometry --config "$config" --out "$out" "$@"
