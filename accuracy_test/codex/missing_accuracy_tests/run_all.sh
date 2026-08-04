#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
status=0

for test_file in "${SCRIPT_DIR}"/test_*.py; do
    python -m pytest -sv -ra --continue-on-collection-errors "${test_file}" "$@" || status=1
done

exit "${status}"
