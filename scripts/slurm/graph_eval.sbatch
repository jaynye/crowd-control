#!/usr/bin/env bash
# Score and time a motion graph on an existing library.
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/graph_eval.sbatch <library.pt> <quality config> <graph config> <UTC stamp>
#   Writes runs/<stamp>_graph_quality_<config name> and runs/<stamp>_graph_speed.
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h; overcap is preemptible. Compare
# speeds only within one GPU type. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=graph-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
library="$1"; quality_cfg="$2"; graph_cfg="$3"; stamp="$4"
name=$(basename "$quality_cfg" .json)
.venv/bin/crowd bench quality --config "$quality_cfg" \
  --out "runs/${stamp}_graph_quality_${name}" --library "$library"
.venv/bin/crowd graph speed --library "$library" \
  --out "runs/${stamp}_graph_speed" --graph-config "$graph_cfg" --neighbors 32 128
