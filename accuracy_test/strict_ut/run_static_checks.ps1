$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python generate_suite.py
python validate_suite.py
python -m compileall -q .
