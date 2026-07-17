# make_all.ps1
# End-to-end execution script to reproduce all tables and figures for the GeoCLIP paper.

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "GeoCLIP Scientific Evaluation Pipeline" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# 1. Evaluate A1-A4 variants on the main test set
Write-Host "`n[STEP 1] Evaluating A1-A4 Variants (Main Benchmark)..." -ForegroundColor Cyan
python src/evaluate_all.py --csv data/im2gps3k_test.csv --img-dir data/im2gps3k/images --cities data/global_cities.csv

# 1.1 Ablation Study
Write-Host "`n[STEP 1.1] Running Ablation Matrix..." -ForegroundColor Cyan
python src/ablation.py

# 2. Evaluate the 5 gallery compositions
Write-Host "`n[STEP 2] Building and Evaluating Gallery Compositions..." -ForegroundColor Cyan
python src/build_galleries.py --cities data/global_cities.csv
python src/evaluate_galleries.py --csv data/im2gps3k_test.csv --img-dir data/im2gps3k/images

# 3. Confidence-Gated Refinement Sweep
Write-Host "`n[STEP 3] Running Confidence-Gated Refinement Sweep..." -ForegroundColor Cyan
python src/evaluate_gated_refinement.py --val-csv data/val.csv --img-dir data/im2gps3k/images --cities data/global_cities.csv

# 4. StreetCLIP Baseline
Write-Host "`n[STEP 4] Running StreetCLIP Zero-Shot Baseline..." -ForegroundColor Cyan
python src/evaluate_streetclip.py --csv data/im2gps3k_test.csv --data-dir data/im2gps3k/images --cities data/global_cities.csv

# 5. Image-to-Image Retrieval Baseline
Write-Host "`n[STEP 5] Running Image Retrieval Baseline..." -ForegroundColor Cyan
python src/evaluate_retrieval.py --query-csv data/im2gps3k_test.csv --query-dir data/im2gps3k/images --ref-csv data/mp16_train.csv --ref-dir data/mp16/images

# 6. Iraqi Dataset Processing
Write-Host "`n[STEP 6] Processing Iraqi Dataset..." -ForegroundColor Cyan
python src/research_wikidata.py

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "Pipeline Complete! All artifacts are saved in the results/ directory." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
