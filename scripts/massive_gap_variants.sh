#!/usr/bin/env bash
# Portable entrypoint for MASSIVE Gap detach/RMS training and final evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "Python is required. Activate the training environment or set PYTHON_BIN." >&2
  exit 127
fi

exec "${python_bin}" -u "${SCRIPT_DIR}/run_massive_gap_variants.py" "$@"
