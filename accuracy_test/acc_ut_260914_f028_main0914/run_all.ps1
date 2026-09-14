<#
.SYNOPSIS
    一键运行 acc_ut_260914_f028_main0914 全部算子精度测试 (Windows PowerShell).
.DESCRIPTION
    自动检测 CUDA / 昇腾 NPU 并运行对应侧。
    NPU 侧通过 run_npu_isolated.py 逐文件隔离进程运行（昇腾向量核异常会
    污染进程设备上下文，隔离运行避免跨文件相互影响）。
.PARAMETER ExtraArgs
    透传给 pytest 的额外参数。
.EXAMPLE
    .\run_all.ps1
    .\run_all.ps1 -ExtraArgs "-k","fill_num"
#>
param(
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)
Set-Location $RootDir

$env:PYTHONPATH = "$RootDir;$env:PYTHONPATH"

Write-Host "=================================================="
Write-Host "  acc_ut_260914_f028_main0914 算子精度测试（自动检测后端）"
Write-Host "=================================================="

# --- Detect available backends ---
$env:ACC_UT_BACKENDS = python -c @"
import sys
backends = []
try:
    import torch
    if torch.cuda.is_available():
        backends.append('cuda')
    if hasattr(torch, 'npu') and torch.npu.is_available():
        backends.append('npu')
except Exception:
    pass
print(','.join(backends))
"@
$backends = $env:ACC_UT_BACKENDS -split "," | Where-Object { $_ -ne "" }

if ($backends.Count -eq 0) {
    Write-Host "[ERROR] 当前机器既无 CUDA GPU 也无昇腾 NPU，无法运行 acc_ut_260914_f028_main0914" -ForegroundColor Red
    exit 1
}

$ranAny = $false

# --- CUDA side ---
if ($backends -contains "cuda") {
    Write-Host ">> 检测到 CUDA，运行 GPU 侧" -ForegroundColor Green
    python -c "import torch; print(f'torch={torch.__version__}  CUDA 已就绪')"
    $pytestArgs = @("$ScriptDir/gpu", "-v", "--tb=short", "-ra") + $ExtraArgs
    python -m pytest @pytestArgs
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 5) { exit $LASTEXITCODE }
    $ranAny = $true
}

# --- Ascend NPU side (isolated per-file subprocesses) ---
if ($backends -contains "npu") {
    Write-Host ">> 检测到昇腾 NPU，运行 NPU 侧（逐文件隔离进程）" -ForegroundColor Green
    python -c "import torch; import torch_npu; print(f'torch={torch.__version__}  NPU 已就绪')"
    $isolatedArgs = @()
    if ($ExtraArgs.Count -gt 0) { $isolatedArgs = $ExtraArgs }
    python "$ScriptDir/run_npu_isolated.py" @isolatedArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $ranAny = $true
}

if (-not $ranAny) { exit 1 }
Write-Host "`n全部所选后端测试完成。" -ForegroundColor Green
