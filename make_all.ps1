# make_all.ps1
# End-to-end execution script to reproduce all tables and figures for the GeoCLIP paper.
# This script has been upgraded to use a memory-safe execution strategy to prevent freezing.

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "GeoCLIP Scientific Evaluation Pipeline (Robust Mode)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

Write-Host "`n[LOG] Setting up clean virtual environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
.venv\Scripts\python.exe -m pip install psutil --quiet

Write-Host "`n[LOG] Starting robust Python orchestrator..." -ForegroundColor Cyan
.venv\Scripts\python.exe run_pipeline.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "Pipeline Complete! All artifacts are saved in the results/ directory." -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "`n==========================================================" -ForegroundColor Red
    Write-Host "Pipeline Halted! Check the logs above for the error." -ForegroundColor Red
    Write-Host "==========================================================" -ForegroundColor Red
}
