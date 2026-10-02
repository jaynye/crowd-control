# Source from the crowd-control root: `source scripts/env.sh`.
# ARDY from source (third_party/ardy is not installed; see pyproject.toml), offline
# weights for Kimodo (CHECKPOINT_DIR) and ARDY (CHECKPOINTS_DIR) under
# $CROWD_CHECKPOINTS (default: <root>/checkpoints), and the managed Python.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
_ckpt="${CROWD_CHECKPOINTS:-${_root}/checkpoints}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${_ckpt}/kimodo}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${_ckpt}/ARDY}"
export UV_PYTHON_INSTALL_DIR="${_root}/.uv-python"
export PYTHONPATH="${_root}/third_party/ardy${PYTHONPATH:+:${PYTHONPATH}}"
unset _root _ckpt
