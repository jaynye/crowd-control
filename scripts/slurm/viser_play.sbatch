#!/usr/bin/env bash
# Serve a registered mjlab task in a Viser browser viewer on a GPU node.
#
#   mkdir -p logs/slurm
#   sbatch --partition=rl2-lab --qos=short --exclude=bishop \
#       scripts/slurm/viser_play.sbatch TASK [play options]
#
# Submit from the crowd-control root. Tunnel the printed port (usually 8080) from
# the compute node, e.g. `ssh -N -L 8080:<node>:8080 <login host>`. For the
# graph-driven plaza crowd use scripts/slurm/viser_graph_demo.sbatch instead.
# Lab GPUs: prefer L40S, A40 or A5000; never Titan X or 2080 Ti.
#SBATCH --job-name=mjlab-viser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: sbatch scripts/slurm/viser_play.sbatch TASK [play options]" >&2
  exit 2
fi
task_id=$1
shift

cd "${SLURM_SUBMIT_DIR:?Submit this job from the crowd-control root}"
if [[ ! -x .venv/bin/play ]]; then
  echo "Run scripts/bootstrap.sh (or uv sync) in $(pwd) first." >&2
  exit 1
fi
source scripts/env.sh
unset DISPLAY WAYLAND_DISPLAY
export MUJOCO_GL=egl

echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Compute node: $(hostname -f)"
echo "Viser will print its actual listening port below (usually 8080)."

exec .venv/bin/play "$task_id" --viewer viser --num-envs "${MJLAB_NUM_ENVS:-1}" "$@"
