#!/usr/bin/env bash
# Array worker submitted by `crowd train navigation-ablations --submit`.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the crowd-control root}"
source scripts/env.sh
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CAMPAIGN=$1
shift
export CROWD_ROOT="$PWD"
export PYTHONPATH="$CAMPAIGN/source/src:$CAMPAIGN/source/mjlab/src:${PYTHONPATH:-}"
exec .venv/bin/python -c 'from crowdmotion.sim.mjlab.ablate_navigation import dispatch; import sys; sys.exit(dispatch(sys.argv[1:]))' \
  "$CAMPAIGN" --index "${SLURM_ARRAY_TASK_ID:?}" "$@"
