#!/usr/bin/env bash
# A trained traversal policy in the crowd on a GPU node: the live Viser viewer (switch
# policy and crowd from the browser), or an MP4 with --video.
#
# Submit from the crowd-control root:
#   sbatch --partition=rl2-lab --qos=short scripts/slurm/play_traversal.sbatch runs/<UTC>_train_traversal_<name> [play args]
#   (rl2-lab only: overcap may preempt a live viewer; add --gres=gpu:l40s:1 before the script
#   when the lab's A40s are taken.) For Viser, read the VISER host/port
#   line in logs/slurm/play-traversal-<JOBID>.out, then on your computer:
#     ssh -N -L 18080:<host>:<port> <user>@sky1.cc.gatech.edu   and open http://localhost:18080
#   With --video the job writes runs/<UTC>_play_traversal_<name>/play.mp4 and exits;
#   shorten it with --time=00:20:00, and add overcap (--partition=rl2-lab,overcap): a
#   preempted render is cheap to repeat.
# Lab rules: prefer L40S, A40 or A5000 (never Titan X or 2080 Ti); exclude bishop.
# rl2-lab is not preemptible and caps jobs at 4 h. Logs go to logs/slurm (mkdir -p logs/slurm).
#SBATCH --job-name=play-traversal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:55:00
#SBATCH --exclude=bishop
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
export MUJOCO_GL=egl
echo "node $(hostname -f) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec .venv/bin/crowd play traversal "$@" --device cuda:0
