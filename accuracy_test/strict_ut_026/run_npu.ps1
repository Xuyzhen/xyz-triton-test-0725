$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python run_npu_isolated.py @args
