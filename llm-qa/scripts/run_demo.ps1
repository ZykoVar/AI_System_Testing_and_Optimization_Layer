# 运行内置演示套件（Mock，无需 API Key）
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
python -m llmqa.demo
exit $LASTEXITCODE
