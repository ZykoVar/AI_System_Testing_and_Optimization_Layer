# Mock 冒烟：全套件 smoke 标签
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
python -m llmqa.cli run --tag smoke --no-color
exit $LASTEXITCODE
