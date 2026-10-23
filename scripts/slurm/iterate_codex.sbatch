#!/usr/bin/env bash
# Start the Codex behavior-refinement process after the GPU allocation begins.
# Submit from the repository root (logs/slurm must already exist):
#   sbatch scripts/slurm/iterate_codex.sbatch [crowd iterate agent arguments]
# Example: --hours 2.75 --model gpt-6-astra --effort xhigh (the defaults)
# Check first on the login node: crowd iterate agent --check [--model ... --effort ...]
# Saved Codex authentication and outbound OpenAI access must work on the node.
# Use one writer per checkout. No automatic requeue: a later run needs a new review
# of the working tree and a fresh output folder. See docs/slurm.md.
#SBATCH --job-name=crowd-codex
#SBATCH --account=rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --exclude=bishop
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
export PATH="$PWD/.venv/bin:$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
command -v codex
codex login status
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# exec makes the runner receive Slurm's batch-shell SIGTERM directly. It launches
# codex exec as a subprocess, closes the local session and saves the run on exit.
exec .venv/bin/crowd iterate agent "$@"
