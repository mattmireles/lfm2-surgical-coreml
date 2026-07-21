#!/usr/bin/env bash
set -euo pipefail

# Copy and hash-check one registered pair plus the common source oracle.
# The Python orchestrator owns the manifest schema and refuses unknown pairs.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/Users/mm/Documents/GitHub/kokoro-coreml/.venv/bin/python}"

exec "${python_bin}" "${repo_root}/scripts/run_a17_parity.py" stage-resources "$@"
