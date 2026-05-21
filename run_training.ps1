# run_training.ps1  --  Train all models from scratch.
#
# Runs (in order):
#   1. notebooks/run_preprocess.py     — build sliding-window arrays
#   2. notebooks/train_bilstm_v2.py    — train BiLSTM v2 classifier
#   3. notebooks/train_twin.py         — train LSTM digital twin
#   4. notebooks/run_shap.py           — compute SHAP feature importances
#
# Exits on the first failure.  Outputs land in `models/` and `data/processed/`.

$ErrorActionPreference = "Stop"
$root      = Split-Path -Parent $MyInvocation.MyCommand.Definition
$notebooks = Join-Path $root "notebooks"

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Run-Step {
    param([string]$Label, [string]$Script)
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host " $Label" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Cyan
    & $py (Join-Path $notebooks $Script)
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Label' failed with exit code $LASTEXITCODE."
    }
}

Run-Step "1/4  Preprocess raw CSVs"        "run_preprocess.py"
Run-Step "2/4  Train BiLSTM v2 classifier" "train_bilstm_v2.py"
Run-Step "3/4  Train LSTM digital twin"    "train_twin.py"
Run-Step "4/4  Compute SHAP on v2"         "run_shap.py"

Write-Host ""
Write-Host "All training steps succeeded." -ForegroundColor Green
Write-Host "Models are in $(Join-Path $root 'models')."
