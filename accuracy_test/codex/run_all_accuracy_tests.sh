#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/existing_accuracy_tests/run_all.sh" "$@"
bash "${SCRIPT_DIR}/missing_accuracy_tests/run_all.sh" "$@"
