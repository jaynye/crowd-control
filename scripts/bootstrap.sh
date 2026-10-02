#!/usr/bin/env bash
# Fetch the pinned third-party sources (third_party.toml), then build the environment.
#
#   scripts/bootstrap.sh                    # clone kimodo and ardy, uv sync, download SOMA-X
#   scripts/bootstrap.sh --no-sync          # clones only (uv sync needs third_party/kimodo)
#   scripts/bootstrap.sh --reinstall-kimodo # rebuild Kimodo after changing its clone or pin
#
# Safe to rerun: existing clones are checked against their pinned commit and left alone.
# Requires git, uv, a C++ compiler and make (e.g. build-essential). Kimodo builds a CMake
# extension; without a system cmake, the PyPI cmake is used through uvx.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
sync=1
reinstall=()
case "${1:-}" in
  --no-sync) sync=0 ;;
  --reinstall-kimodo) reinstall=(--reinstall-package kimodo) ;;
  "") ;;
  *) echo "usage: scripts/bootstrap.sh [--no-sync|--reinstall-kimodo]" >&2; exit 2 ;;
esac

# pin <section> <key>: value of `key = "..."` inside `[section]` of third_party.toml.
pin() {
  awk -v section="[$1]" -v key="$2" '
    /^\[/ { inside = ($0 == section) }
    inside && $1 == key { sub(/^[^"]*"/, ""); sub(/".*$/, ""); print; exit }
  ' third_party.toml
}

clone() {
  local name="$1" url commit path
  url="$(pin "$name" url)"
  commit="$(pin "$name" commit)"
  path="$(pin "$name" path)"
  if [[ -d "$path/.git" ]]; then
    local have
    have="$(git -C "$path" rev-parse HEAD)"
    if [[ "$have" != "$commit" ]]; then
      echo "warning: $path is at $have, pinned $commit (left unchanged)" >&2
    else
      echo "$name: $path at $commit"
    fi
    return
  fi
  mkdir -p "$(dirname "$path")"
  git clone --filter=blob:none "$url" "$path"
  git -C "$path" checkout --detach "$commit"
  echo "$name: cloned at $commit"
}

clone kimodo
clone ardy

if (( sync )); then
  export UV_PYTHON_INSTALL_DIR="$root/.uv-python"
  if ! command -v cmake >/dev/null; then
    PATH="$(uvx --from cmake python -c 'import cmake; print(cmake.CMAKE_BIN_DIR)'):$PATH"
    export PATH
  fi
  uv sync ${reinstall[@]+"${reinstall[@]}"}
  .venv/bin/python - <<'EOF'
import torch, warp
assert torch.__version__.startswith("2.9.0"), torch.__version__
assert warp.__version__ == "1.14.0", warp.__version__
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), warp {warp.__version__}")
EOF
  # hf download skips complete files and resumes the rest; env.sh's offline mode must not apply.
  soma="$(pin soma-x path)"
  env -u HF_HUB_OFFLINE .venv/bin/hf download "$(pin soma-x hf_repo)" --revision "$(pin soma-x revision)" \
    --local-dir "$soma"
fi

echo "Next: scripts/fetch_artifacts.sh (graph and checkpoints); see docs/install.md."
