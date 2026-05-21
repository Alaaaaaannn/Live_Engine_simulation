---
title: Engine Simulation Backend
emoji: ⚙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: FastAPI backend for the digital-twin engine fault simulator
---

# AI Digital Twin — Engine Fault Simulator

FastAPI + BiLSTM/LSTM digital-twin backend with a React/Three.js
front-end.  Detects three synthetic fuel/ignition fault classes on the
Bosch Engine Dataset, proposes corrective control actions, and validates
them through a learned next-state predictor.

## Layout

```
backend/        FastAPI service (BiLSTM v2 + LSTM twin + supervisory controller)
frontend/       Vite + React + R3F UI
notebooks/      Training & analysis scripts (run_preprocess, train_bilstm_v2, …)
models/         Trained model weights (populated by run_training)
data/raw/       Raw Bosch CSVs (mounted into the container, junctioned in dev)
data/processed/ Numpy sliding-window arrays + dataset_meta.json
```

## Quick start with Docker

Pre-requisites: trained models must exist in `models/` and the Bosch
CSVs in `data/raw/{gengine1,gengine2,pengines}/`.

```bash
# 1.  Train models (~2-4 h on CPU)
./run_training.sh                # macOS/Linux
./run_training.ps1               # Windows PowerShell

# 2.  Bring up the stack
docker compose up --build

# 3.  Open the UI
open http://localhost
```

The `frontend` container serves the SPA on port 80 and proxies `/api/*`
to the `backend` container at port 8000.

## Quick start without Docker

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (in another shell)
cd frontend
npm install
npm run dev     # Vite proxies /api/* to localhost:8000
```

## Training

`run_training.{sh,ps1}` runs four scripts in order:

1. `notebooks/run_preprocess.py` — builds sliding-window numpy arrays
   with **variable-magnitude fault augmentation** (offset scaled by
   Uniform(0.7, 1.5)).
2. `notebooks/train_bilstm_v2.py` — BiLSTM + self-attention classifier,
   F1-maximising threshold calibration, label-smoothing.
3. `notebooks/train_twin.py` — LSTM delta predictor for action
   validation.
4. `notebooks/run_shap.py` — SHAP feature importances on the v2 model.

Outputs land in `models/`:

| File                          | Used by                |
|-------------------------------|------------------------|
| `bilstm_v2_classifier.h5`     | backend classifier     |
| `bilstm_v2_thresholds.json`   | per-class gating       |
| `lstm_digital_twin.h5`        | backend twin           |
| `twin_meta.json`              | twin scaling constants |
| `shap_cache.json`             | per-fault explanations |

## Runtime tweakables

The UI's **TWEAKABLES** panel hits `POST /config/runtime` to mutate live
parameters without a restart:

| Tab          | What it changes                                                |
|--------------|----------------------------------------------------------------|
| THRESHOLDS   | Per-class confidence gate.  Predictions below threshold are forced to Normal. |
| FAULTS       | Injected fault magnitudes (standardised units).                |
| CONTROLLER   | Per-cycle fuel-trim and spark-advance step sizes.              |

`POST /config/runtime  { reset: true }` restores the bundled defaults.

## API summary

| Endpoint              | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `POST /simulate`      | Run one closed-loop cycle for a session.             |
| `POST /classify`      | Classify a raw (30 × n) sensor window.               |
| `POST /engine/select` | Switch active engine dataset.                        |
| `POST /fault/inject`  | Latch a fault into an active session.                |
| `GET  /status`        | Model load status + key metrics.                     |
| `GET  /config/runtime`| Tweakables snapshot.                                 |
| `POST /config/runtime`| Patch (or reset) tweakables.                         |
