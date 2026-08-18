# 真实模型回归（需先设置环境变量，例如 $env:OPENAI_API_KEY）
# 用法：./scripts/run_real.ps1 -Provider openai -Suite security -Concurrency 2
param(
  [string]$Provider = "openai",
  [string]$Suite = "",
  [int]$Concurrency = 2
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
$args = @("run", "--provider", $Provider, "--concurrency", $Concurrency, "--no-color")
if ($Suite) { $args += @("--suite", $Suite) }
python -m llmqa.cli @args
exit $LASTEXITCODE
