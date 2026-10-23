#!/usr/bin/env bash
# Train the crowd traversal policy: `crowd train traversal` on one GPU.
# Submit from the repository root (logs/slurm must already exist):
#   sbatch scripts/slurm/train_traversal.sbatch [name] [crowd train traversal options] [-- mjlab TrainConfig options]
# Continue a run in a later job with `--resume runs/<UTC>_<name>`; each job writes a
# fresh run directory. Export CROWD_GRAPH and CROWD_GRAPH_ROTATIONS for the graph.
#SBATCH --job-name=crowd-train
#SBATCH --account=rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --exclude=bishop
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
# CROWD_VENV: a virtualenv elsewhere (for a source snapshot without its own .venv,
# with PYTHONPATH pointing at the snapshot's src/ and mjlab/src).
PY="${CROWD_VENV:-$PWD/.venv}/bin/python"
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec "$PY" -m crowdmotion.cli train traversal "$@"
