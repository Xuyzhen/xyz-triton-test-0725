#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python -m pytest -sv "${SCRIPT_DIR}/from_vllm_ascend" "$@"
python -m pytest -sv "${SCRIPT_DIR}/from_vllm" "$@"
