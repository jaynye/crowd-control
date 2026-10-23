#!/usr/bin/env bash
# Train the crowd traversal policy: one lineage of `crowd train traversal` on one GPU,
# requeued until it reaches its target. Submit from the repository root (logs/slurm must
# already exist), with the name first:
#   sbatch scripts/slurm/train_traversal.sbatch [name] [crowd train traversal options] [-- mjlab TrainConfig options]
# The lineage's directory is `--run` if given, else runs/<UTC>_<name> chosen at the first
# start and kept in logs/slurm/<job name>-<job id>.run, so a requeued job reruns the same
# command line on the same directory and resumes it in place. Slurm sends TERM 300 s
# before the time limit; the trainer checkpoints after the current iteration and exits 75
# (`checkpointed`), and the job requeues itself only then, and only when the signal came
# at the time limit. A preemption on overcap requeues through Slurm (PreemptMode=REQUEUE);
# `scancel` does not requeue. After a completed lineage, `crowd wandb sync` publishes
# anything the live W&B writer missed (nothing for a lineage with W&B disabled).
# `--resume`, `--distill` and `--finetune` start a new lineage from another run.
# Export CROWD_GRAPH and CROWD_GRAPH_ROTATIONS for the graph.
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
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
unset DISPLAY WAYLAND_DISPLAY
source scripts/env.sh
# CROWD_VENV: a virtualenv elsewhere (for a source snapshot without its own .venv,
# with PYTHONPATH pointing at the snapshot's src/ and mjlab/src).
PY="${CROWD_VENV:-$PWD/.venv}/bin/python"
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# The --signal lead time (Slurm may send it up to 60 s early). Submitting with another
# --signal=B:TERM@<s>? Export CROWD_SIGNAL_LEAD_S=<s> to match.
LEAD=${CROWD_SIGNAL_LEAD_S:-300}
CHECKPOINTED=75  # crowd train traversal's exit status after a checkpoint at a signal
echo "node $(hostname -f) job ${SLURM_JOB_ID:-} restart ${SLURM_RESTART_COUNT:-0} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

RUN=""
prev=""
for arg in "$@"; do  # an explicit --run before `--`
  [[ $arg == -- ]] && break
  [[ $prev == --run ]] && RUN=$arg
  [[ $arg == --run=* ]] && RUN=${arg#--run=}
  prev=$arg
done
if [[ -z $RUN ]]; then
  NAME=train_traversal
  [[ $# -gt 0 && $1 != -* ]] && NAME=$1
  RUNFILE="logs/slurm/${SLURM_JOB_NAME:-crowd-train}-${SLURM_JOB_ID:?}.run"
  [[ -s $RUNFILE ]] || echo "runs/$(date -u +%Y%m%dT%H%M%SZ)_${NAME}" > "$RUNFILE"
  RUN=$(<"$RUNFILE")
  set -- --run "$RUN" "$@"
fi
echo "lineage $RUN"

time_left() {  # seconds until the job's time limit (squeue %L: [D-]HH:MM:SS, MM:SS or SS)
  local t d=0 h=0 m=0 s=0
  t=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null) || { echo 999999; return; }
  [[ $t =~ ^[0-9:-]+$ ]] || { echo 999999; return; }
  if [[ $t == *-* ]]; then d=${t%%-*}; t=${t#*-}; fi
  IFS=: read -r -a p <<< "$t"
  case ${#p[@]} in
    3) h=${p[0]} m=${p[1]} s=${p[2]} ;;
    2) m=${p[0]} s=${p[1]} ;;
    1) s=${p[0]} ;;
  esac
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

AT_LIMIT=0
forward() {
  local left
  left=$(time_left)
  (( left <= LEAD + 90 )) && AT_LIMIT=1
  echo "TERM with ${left} s left: forwarding to the trainer (at the time limit: $AT_LIMIT)"
  kill -TERM "$PID" 2>/dev/null || true
}

"$PY" -m crowdmotion.cli train traversal "$@" &
PID=$!
trap forward TERM
set +e
while :; do  # `wait` returns early when the trap runs; keep waiting for the trainer
  wait "$PID"
  kill -0 "$PID" 2>/dev/null || break
done
wait "$PID"
STATUS=$?
set -e
echo "trainer exit status $STATUS $(date -u +%FT%TZ)"
if [[ $STATUS -eq $CHECKPOINTED ]]; then
  if [[ $AT_LIMIT -eq 1 && -n ${SLURM_JOB_ID:-} ]]; then
    echo "requeue $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID" || true
    exit 0
  fi
  echo "checkpointed away from the time limit (preemption requeues through Slurm; scancel does not)"
elif [[ $STATUS -eq 0 ]]; then
  "$PY" -m crowdmotion.cli wandb sync "$RUN" || true
fi
exit "$STATUS"
