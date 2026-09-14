$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pytest -c pytest.ini gpu -m gpu -v --tb=short @args
