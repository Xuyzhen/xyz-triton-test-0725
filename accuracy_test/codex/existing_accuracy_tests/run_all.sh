#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
status=0

bash "${SCRIPT_DIR}/run_from_vllm_ascend.sh" "$@" || status=1
bash "${SCRIPT_DIR}/run_from_vllm.sh" "$@" || status=1

exit "${status}"
