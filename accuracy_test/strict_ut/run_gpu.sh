#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -c pytest.ini gpu -m gpu -v --tb=short "$@"
