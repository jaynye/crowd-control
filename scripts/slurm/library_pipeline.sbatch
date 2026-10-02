#!/usr/bin/env bash
# Generate an ARDY library, score the motion graph against online ARDY, time the graph.
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab,overcap --qos=short scripts/slurm/library_pipeline.sbatch \
#       configs/library/ardy_a40.json configs/bench/graph_a40.json <UTC stamp> [graph speed args]
#   Writes runs/<stamp>_ardy_library, runs/<stamp>_graph_quality, runs/<stamp>_graph_speed.
#   Then build the saved graph: crowd graph build --library runs/<stamp>_ardy_library/library.pt \
#       --out runs/<UTC>_motion_graph/graph.pt (GPU recommended).
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h; overcap is preemptible. Compare
# speeds only within one GPU type. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=motion-library
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:55:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
library_cfg="$1"; quality_cfg="$2"; stamp="$3"; shift 3
lib="runs/${stamp}_ardy_library"
.venv/bin/crowd library generate --config "$library_cfg" --out "$lib"
.venv/bin/crowd bench quality --config "$quality_cfg" --out "runs/${stamp}_graph_quality" \
  --library "$lib/library.pt"
.venv/bin/crowd graph speed --library "$lib/library.pt" \
  --out "runs/${stamp}_graph_speed" "$@"
