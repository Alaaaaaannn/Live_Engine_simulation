#!/usr/bin/env bash
# run_training.sh  --  Train all models from scratch.
#
# Runs (in order):
#   1. notebooks/run_preprocess.py     -- build sliding-window arrays
#   2. notebooks/train_bilstm_v2.py    -- train BiLSTM v2 classifier
#   3. notebooks/train_twin.py         -- train LSTM digital twin
#   4. notebooks/run_shap.py           -- compute SHAP feature importances
#
# Exits on the first failure.  Outputs land in `models/` and `data/processed/`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB="$ROOT/notebooks"
PY="${PYTHON:-python3}"

run_step() {
    local label="$1"
    local script="$2"
    echo ""
    echo "=============================================================="
    echo " $label"
    echo "=============================================================="
    "$PY" "$NB/$script"
}

run_step "1/4  Preprocess raw CSVs"        "run_preprocess.py"
run_step "2/4  Train BiLSTM v2 classifier" "train_bilstm_v2.py"
run_step "3/4  Train LSTM digital twin"    "train_twin.py"
run_step "4/4  Compute SHAP on v2"         "run_shap.py"

echo ""
echo "All training steps succeeded."
echo "Models are in $ROOT/models."
