#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
status=0

bash "${SCRIPT_DIR}/existing_accuracy_tests/run_all.sh" "$@" || status=1
bash "${SCRIPT_DIR}/missing_accuracy_tests/run_all.sh" "$@" || status=1

exit "${status}"
