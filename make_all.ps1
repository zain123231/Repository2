# make_all.ps1
# End-to-end execution script to reproduce all tables and figures for the GeoCLIP paper.
# This script has been upgraded to use a memory-safe execution strategy to prevent freezing.

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "GeoCLIP Scientific Evaluation Pipeline (Robust Mode)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

Write-Host "`n[LOG] Ensuring psutil is installed for system monitoring..." -ForegroundColor Yellow
python -m pip install psutil --quiet

Write-Host "`n[LOG] Starting robust Python orchestrator..." -ForegroundColor Cyan
python run_pipeline.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "Pipeline Complete! All artifacts are saved in the results/ directory." -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "`n==========================================================" -ForegroundColor Red
    Write-Host "Pipeline Halted! Check the logs above for the error." -ForegroundColor Red
    Write-Host "==========================================================" -ForegroundColor Red
}
