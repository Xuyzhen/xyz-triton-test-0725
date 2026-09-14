#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_npu_isolated.py "$@"
