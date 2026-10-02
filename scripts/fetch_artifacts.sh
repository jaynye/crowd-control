#!/usr/bin/env bash
# Copy large run outputs listed in artifacts.toml, and download model checkpoints.
#
#   scripts/fetch_artifacts.sh --list
#   scripts/fetch_artifacts.sh --from HOST:/path/to/crowd-control [NAME ...]
#   scripts/fetch_artifacts.sh --verify [NAME ...]
#   scripts/fetch_artifacts.sh --checkpoints
#
# Run outputs keep their relative path (runs/<id>/<file>) and are checked against the
# manifest's sha256. Without NAME, the artifacts marked `default = true` are fetched.
# --from defaults to $CROWD_ARTIFACT_SOURCE. --checkpoints downloads the ARDY and Kimodo
# weights from Hugging Face into $CHECKPOINTS_DIR and $CHECKPOINT_DIR (scripts/env.sh).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
manifest=artifacts.toml
src="${CROWD_ARTIFACT_SOURCE:-}"
mode=fetch
names=()
while (( $# )); do
  case "$1" in
    --from) src="$2"; shift 2 ;;
    --list) mode=list; shift ;;
    --verify) mode=verify; shift ;;
    --checkpoints) mode=checkpoints; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) names+=("$1"); shift ;;
  esac
done

# field <section> <key>: value of `key = ...` inside `[section]`, quotes stripped.
field() {
  awk -v section="[$1]" -v key="$2" '
    /^\[/ { inside = ($0 == section) }
    inside && $1 == key { sub(/^[^=]*=[ \t]*/, ""); gsub(/"/, ""); print; exit }
  ' "$manifest"
}
sections() {  # sections <prefix>: names of [prefix.NAME] tables
  sed -n "s/^\[$1\.\(.*\)\]$/\1/p" "$manifest"
}

if [[ "$mode" == checkpoints ]]; then
  source scripts/env.sh
  unset HF_HUB_OFFLINE
  for name in $(sections checkpoint); do
    dir_var="$(field "checkpoint.$name" dir_env)"
    target="${!dir_var}/$(field "checkpoint.$name" name)"
    mkdir -p "$target"
    # hf download skips files already complete in --local-dir and resumes the rest.
    .venv/bin/hf download "$(field "checkpoint.$name" hf_repo)" \
      --revision "$(field "checkpoint.$name" revision)" --local-dir "$target"
    echo "$name: $target"
  done
  exit 0
fi

if (( ${#names[@]} == 0 )); then
  for name in $(sections artifact); do
    if [[ "$mode" == list || "$mode" == verify || "$(field "artifact.$name" default)" == true ]]; then
      names+=("$name")
    fi
  done
fi

status=0
for name in "${names[@]}"; do
  path="$(field "artifact.$name" path)"
  sha="$(field "artifact.$name" sha256)"
  [[ -n "$path" ]] || { echo "unknown artifact: $name" >&2; exit 2; }
  case "$mode" in
    list)
      have=missing
      [[ -f "$path" ]] && have=present
      printf '%-22s %8s MB  %-8s %s\n' "$name" "$(( $(field "artifact.$name" bytes) / 1000000 ))" "$have" "$path"
      continue ;;
    fetch)
      # Fetch when missing or when a previous copy fails its checksum (rsync resumes it).
      if [[ ! -f "$path" ]] || ! echo "$sha  $path" | sha256sum --check --quiet 2>/dev/null; then
        [[ -n "$src" ]] || { echo "set --from HOST:/path/to/crowd-control or CROWD_ARTIFACT_SOURCE" >&2; exit 2; }
        mkdir -p "$(dirname "$path")"
        rsync -a --partial --inplace --info=progress2 "$src/$path" "$path"
        # Small provenance files next to the artifact, when present.
        rsync -a --ignore-missing-args "$src/$(dirname "$path")/"{run.json,graph.json} "$(dirname "$path")/" || true
      fi ;;
  esac
  if [[ -f "$path" ]] && echo "$sha  $path" | sha256sum --check --quiet; then
    echo "$name: ok ($path)"
  else
    echo "$name: MISSING OR CORRUPT ($path)" >&2
    status=1
  fi
done
exit $status
