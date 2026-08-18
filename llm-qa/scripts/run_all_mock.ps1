# 全量 Mock 回归
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
python -m llmqa.cli run --no-color
exit $LASTEXITCODE
