# 激活虚拟环境并设置环境变量
$venvPath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
. $venvPath

# 确保项目根目录在 PYTHONPATH 中
$env:PYTHONPATH = $PSScriptRoot

Write-Host "========================================" -ForegroundColor Green
Write-Host " Enterprise RAG CS - Virtual Environment" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host " Python: $(python --version)" -ForegroundColor Cyan
Write-Host " Project: $PSScriptRoot" -ForegroundColor Cyan
Write-Host ""
Write-Host " Commands:" -ForegroundColor Yellow
Write-Host "   pytest tests/              # Run all tests" -ForegroundColor White
Write-Host "   python -m pytest tests/ -v # Verbose tests" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green
