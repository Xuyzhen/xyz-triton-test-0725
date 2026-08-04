#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
status=0

python -m pytest -sv -ra "${SCRIPT_DIR}/from_vllm_ascend" "$@" || status=1
python -m pytest -sv -ra "${SCRIPT_DIR}/from_vllm" "$@" || status=1

exit "${status}"
