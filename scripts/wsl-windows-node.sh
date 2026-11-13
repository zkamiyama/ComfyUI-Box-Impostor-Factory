#!/usr/bin/env bash
# BI_NODE adapter for a WSL ComfyUI process using Windows Node.js and Chromium.
set -euo pipefail

if ! command -v wslpath >/dev/null; then
  echo 'This Box worker requires WSL (wslpath is missing).' >&2
  exit 127
fi

# systemd --user services commonly omit the Windows PATH imported by an interactive WSL shell.
if [[ -n ${BI_WINDOWS_NODE:-} ]]; then
  win_node=$BI_WINDOWS_NODE
elif command -v node.exe >/dev/null; then
  win_node=$(command -v node.exe)
else
  win_node=''
  for candidate in /mnt/c/nvm4w/nodejs/node.exe '/mnt/c/Program Files/nodejs/node.exe' '/mnt/c/Program Files (x86)/nodejs/node.exe'; do
    if [[ -x $candidate ]]; then win_node=$candidate; break; fi
  done
fi
if [[ -z $win_node || ! -x $win_node ]]; then
  echo 'Windows Node.js was not found. Set PATH or install Node.js for Windows.' >&2
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
exec "$win_node" "${args[@]}"
