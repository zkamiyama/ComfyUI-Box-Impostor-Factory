#!/usr/bin/env bash
# BI_NODE adapter for a WSL ComfyUI process using Windows Node.js and Chromium.
set -euo pipefail

if ! command -v wslpath >/dev/null || ! command -v node.exe >/dev/null; then
  echo 'Run in WSL with Windows Node.js (node.exe) on PATH.' >&2
  exit 127
fi

args=()
for arg in "$@"; do
  if [[ $arg == /* ]]; then arg=$(wslpath -w "$arg"); fi
  args+=("$arg")
done
if [[ ${BI_CHROMIUM:-} == /* ]]; then
  BI_CHROMIUM=$(wslpath -w "$BI_CHROMIUM")
  export BI_CHROMIUM
fi

# Paths above are already in Windows notation. Avoid a second /p translation
# and pass through the caller's unrelated WSLENV entries and direction flags.
forward='BI_CHROMIUM:BI_SOFTWARE_GPU'
IFS=: read -r -a entries <<< "${WSLENV:-}"
for entry in "${entries[@]}"; do
  case ${entry%%/*} in
    ''|BI_CHROMIUM|BI_SOFTWARE_GPU) ;;
    *) forward+=":$entry" ;;
  esac
done
export WSLENV="$forward"
exec node.exe "${args[@]}"
