#!/usr/bin/env bash
# Verify ComfyUI Box Impostor Factory without CUDA or the hardware graphics adapter.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="${BI_COMFY_DIR:-$(cd "$ROOT/../.." && pwd)}"
if [[ ! -f "$COMFY_DIR/main.py" ]]; then
  printf 'ComfyUI main.py was not found in %s. Set BI_COMFY_DIR.\n' "$COMFY_DIR" >&2
  exit 2
fi
if [[ -n "${BI_PYTHON:-}" ]]; then
  PYTHON="$BI_PYTHON"
elif [[ -x "$COMFY_DIR/../.venv/bin/python" ]]; then
  PYTHON="$COMFY_DIR/../.venv/bin/python"
else
  PYTHON=python
fi

# Always force software rendering. A caller's GPU-related settings may not
# change this smoke test into a hardware test.
export BI_SOFTWARE_GPU=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1
export ROCR_VISIBLE_DEVICES=-1 LIBGL_ALWAYS_SOFTWARE=1

if [[ "$(uname -r)" == *[Mm]icrosoft* ]]; then
  if [[ -z "${BI_NODE:-}" ]]; then
    if ! command -v node.exe >/dev/null; then
      echo 'WSL CPU smoke requires Windows Node.js; install Node.js or set BI_NODE.' >&2
      exit 2
    fi
    export BI_NODE="$ROOT/scripts/wsl-windows-node.sh"
  fi
  if [[ -z "${BI_CHROMIUM:-}" ]]; then
    edge='C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    if command -v wslpath >/dev/null && [[ -f "$(wslpath -u "$edge")" ]]; then
      export BI_CHROMIUM="$edge"
    else
      echo 'Set BI_CHROMIUM to an installed Windows Edge/Chromium executable.' >&2
      exit 2
    fi
  fi
fi

OUTPUT="${BI_CPU_OUTPUT:-$ROOT/.test-output/cpu-only/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
cd "$ROOT"
printf '[CPU] Python unit tests: %s\n' "$PYTHON"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py'
printf '[CPU] JavaScript unit tests\n'
node --test --test-concurrency=2 tests/*.test.mjs
printf '[CPU] Real ComfyUI queue, software graphics only\n'
"$PYTHON" tests/comfy_smoke.py --comfy "$COMFY_DIR" --output "$OUTPUT"
printf '[CPU] All checks passed. Evidence: %s\n' "$OUTPUT"
