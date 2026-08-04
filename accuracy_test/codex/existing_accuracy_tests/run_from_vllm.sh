#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -m pytest -sv -ra --continue-on-collection-errors "${SCRIPT_DIR}/from_vllm" "$@"
