$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pytest -c pytest.ini npu -m npu -v --tb=short @args
