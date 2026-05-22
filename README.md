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

NOTE: Currently non-functional due to S3 storage being disabled to save credits.

# AI Digital Twin Engine Fault Simulator

A full-stack, closed-loop **digital twin** of an internal-combustion engine.
The system streams sensor windows through a **BiLSTM + self-attention**
classifier, detects three synthetic fuel/ignition fault classes on the
**Bosch Engine Dataset**, proposes corrective control actions, and then
validates each proposed action against a **learned LSTM next-state predictor
(the "twin")** before applying it. The whole feedback loop is visualised in
a 3D React/Three.js dashboard with live SHAP explanations.

**Live demo:** https://engine-simulation-smoky.vercel.app
**Backend API:** https://alanfernandes-engine-simulation.hf.space

> Heads-up: the backend is hosted on Hugging Face Spaces' free tier and
> sleeps when idle. The first request after a cold period takes ~30–60 s
> to wake the container, subsequent calls are fast.

---

## Why this project exists

Classical engine fault detection pipelines treat detection and control as
two disconnected problems: a classifier flags a fault, a human (or a
hand-tuned PID) reacts. That gap is where most field failures hide,
the classifier may be right, but the action taken in response is unsafe
or sub-optimal for the current operating point.

This project bridges that gap end-to-end:

1. **Detect** the fault with a sequence model that respects temporal
   context (BiLSTM) and learns which sensor channels matter for which
   fault (self-attention).
2. **Propose** a corrective action from a small supervisory controller
   (per-cycle fuel-trim and spark-advance adjustments).
3. **Validate** the proposed action by asking a separately-trained
   **digital twin** "if I apply this, what will the next sensor window
   look like?" and only apply actions that the twin predicts will
   move the engine toward its healthy operating envelope.
4. **Explain** the decision with cached SHAP feature attributions, so
   the operator sees _which_ sensors drove the call.

The whole loop runs at interactive rates in the browser, against a real
deployed backend, with per-user persistence, i.e., it's a working
system, not a notebook demo.

---

## Novelty

- **Variable-magnitude fault augmentation.** During preprocessing, each
  injected fault's offset is scaled by `Uniform(0.7, 1.5)` instead of a
  fixed delta. This stops the classifier from memorising a single
  magnitude signature and forces it to learn the _shape_ of each fault.
- **BiLSTM + self-attention with F1-maximising threshold calibration.**
  Per-class confidence gates are calibrated post-hoc to maximise F1;
  predictions below threshold are forced to `Normal`, which keeps the
  controller from acting on weak evidence.
- **Twin as an action validator, not just a forecaster.** The LSTM
  delta-predictor is trained to forecast the _next_ sensor window
  given the current window + proposed action. The supervisory
  controller uses it as a safety filter, proposals that the twin
  predicts to worsen the state are rejected before they hit the engine
  model.
- **Live tweakability without restarts.** `POST /config/runtime` lets
  the operator mutate per-class thresholds, injected fault magnitudes,
  and controller step sizes from the UI. Useful for live demos and
  for stress-testing the gating logic.
- **Cached SHAP explanations per fault class.** Computed once during
  training (`run_shap.py`) and served as static JSON, so explanations
  render instantly in the UI without re-running SHAP at request time.

---

## Architecture & hosting

| Layer          | Service                          | Notes                                                                                       |
| -------------- | -------------------------------- | ------------------------------------------------------------------------------------------- |
| Frontend       | **Vercel**                       | Vite + React SPA. Deployed via the `vercel` CLI from `frontend/`.                           |
| Backend API    | **Hugging Face Spaces (Docker)** | FastAPI + TensorFlow/Keras. Free tier — cold-start ~30–60 s. Container runs as UID 1000.    |
| Database       | **Supabase Postgres**            | User auth + per-user `simulation_runs` / `simulation_cycles` tables. Session pooler (IPv4). |
| Object storage | **AWS S3**                       | Bucket `engine-simulation-miniproject` (`ap-southeast-2`). Holds model weights + raw CSVs.  |
| IAM            | **AWS IAM user** `miniproject`   | Read-only access keys consumed by `backend/storage.py` at container startup.                |
| Code           | **GitHub**                       | `master` is canonical. HF Space mirrors `master` → `main`.                                  |

### Request flow

```
Browser (Vercel SPA)
   │  fetch(VITE_API_URL + "/simulate", …)
   ▼
HF Space (FastAPI :8000)
   ├─ on startup: pulls model weights + Bosch CSVs from S3 → /app/models, /app/data
   ├─ classifier   (BiLSTM v2 + attention)
   ├─ controller   (supervisory step generator)
   ├─ twin         (LSTM delta predictor — action validator)
   └─ persistence  → Supabase (asyncpg, Session pooler)
```

### Deployment-specific gotchas (already handled in code)

- `backend/storage.py` fetches model files from S3 at container startup.
  `S3_ENDPOINT_URL` is left empty for native AWS - set it only for
  R2 / MinIO.
- `backend/db.py` rewrites `postgres://` URLs to `postgresql+asyncpg://`
  automatically, so a vanilla Supabase connection string works.
- The backend runs **degraded** if S3 or the DB fail at startup
  (caught in `lifespan`, error surfaced via `GET /status.message`).
  The container stays alive so HF Spaces doesn't roll the logs.
- HF Spaces forces UID 1000; the Dockerfile creates that user and
  `chown`s `/app` so `storage.py` can write `/app/models/`.
- Use Supabase's **Session pooler** (port 5432, IPv4). The Direct
  connection is IPv6-only and unreachable from HF. The Transaction
  pooler breaks asyncpg's prepared statements - do not use it.
- Run upserts are race-safe via Postgres `ON CONFLICT` against the
  `uq_runs_user_session` unique constraint + an in-process
  `asyncio.Lock` per `session_id`.
- Frontend history lives in `localStorage`, namespaced by user ID
  (`dt:simulation_history:v1:<user-id>`). Postgres is the
  authoritative server-side store.

---

## Tech stack

**Backend**: Python 3.11, FastAPI, Uvicorn, TensorFlow / Keras,
NumPy, SciPy, scikit-learn, SHAP, SQLAlchemy (async) + asyncpg, boto3,
python-jose (JWT), passlib + bcrypt, pydantic.

**Frontend**: Vite, React 18, React Three Fiber + Three.js, Zustand,
Tailwind, fetch-based API client.

**Infra / Ops**: Docker, docker-compose (local), Hugging Face Spaces
(prod backend), Vercel (prod frontend), Supabase (managed Postgres),
AWS S3 + IAM (model + data storage).

---

## Project layout

```
backend/        FastAPI service (BiLSTM v2 + LSTM twin + supervisory controller)
frontend/       Vite + React + R3F UI
notebooks/      Training & analysis scripts (run_preprocess, train_bilstm_v2, …)
models/         Trained model weights (populated by run_training, or pulled from S3 in prod)
data/raw/       Raw Bosch CSVs (mounted in the container, junctioned in dev)
data/processed/ Numpy sliding-window arrays + dataset_meta.json
scripts/        Misc ops scripts
figures/        Generated training / evaluation plots
Dockerfile               # root Dockerfile used by HF Spaces
docker-compose.yml       # local two-service stack (backend + frontend)
run_training.{sh,ps1}    # end-to-end training pipeline
HANDOFF.md               # detailed session notes / deployment notes
BILSTM_DEVELOPMENT.md    # classifier design write-up
```

---

## Clone & run it yourself

```bash
git clone https://github.com/Alaaaaaannn/Live_Engine_simulation.git
cd Live_Engine_simulation
```

You have three reasonable paths:

### A. Local with Docker (closest to production)

Pre-requisite: trained models in `models/` and Bosch CSVs in
`data/raw/{gengine1,gengine2,pengines}/`. If you don't have models yet,
run the training pipeline first (see below).

```bash
docker compose up --build
# UI:  http://localhost
# API: http://localhost:8000  (proxied as /api/* from the frontend)
```

The `frontend` container serves the SPA on port 80 and proxies
`/api/*` to the `backend` container at port 8000.

### B. Local without Docker

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (separate shell)
cd frontend
npm install
npm run dev    # Vite proxies /api/* to localhost:8000
```

### C. Run the training pipeline (only needed if you want fresh weights)

`run_training.{sh,ps1}` runs four scripts in order:

1. `notebooks/run_preprocess.py`: builds sliding-window numpy arrays
   with variable-magnitude fault augmentation (`Uniform(0.7, 1.5)`).
2. `notebooks/train_bilstm_v2.py`: BiLSTM + self-attention classifier,
   F1-maximising threshold calibration, label-smoothing.
3. `notebooks/train_twin.py`: LSTM delta predictor for action
   validation.
4. `notebooks/run_shap.py`: SHAP feature importances on the v2 model.

```bash
./run_training.sh        # macOS / Linux
./run_training.ps1       # Windows PowerShell
```

Outputs land in `models/`:

| File                        | Used by                |
| --------------------------- | ---------------------- |
| `bilstm_v2_classifier.h5`   | backend classifier     |
| `bilstm_v2_thresholds.json` | per-class gating       |
| `lstm_digital_twin.h5`      | backend twin           |
| `twin_meta.json`            | twin scaling constants |
| `shap_cache.json`           | per-fault explanations |

Training takes roughly 2–4 hours on a modern CPU.

---

## Environment variables

You only need these if you want to wire up auth + cloud storage like
the deployed stack. Pure local development with `docker compose` works
without any of them; the backend falls back to local files and a no-op
persistence path.

**Backend** (set in HF Space → _Settings → Variables and secrets_, or
in a local `backend/.env`):

| Variable                | Purpose                                                 |
| ----------------------- | ------------------------------------------------------- |
| `DATABASE_URL`          | Supabase Session-pooler connection string.              |
| `JWT_SECRET`            | Server-side JWT signing key.                            |
| `AWS_ACCESS_KEY_ID`     | IAM user `miniproject` access key.                      |
| `AWS_SECRET_ACCESS_KEY` | IAM user `miniproject` secret.                          |
| `AWS_REGION`            | `ap-southeast-2`.                                       |
| `S3_BUCKET`             | `engine-simulation-miniproject`.                        |
| `S3_ENDPOINT_URL`       | Leave empty for native AWS; set only for R2 / MinIO.    |
| `CORS_ORIGINS`          | Comma-separated allow-list (include the Vercel origin). |

**Frontend** (Vercel → _Project → Environment Variables_):

| Variable       | Purpose                                                           |
| -------------- | ----------------------------------------------------------------- |
| `VITE_API_URL` | Base URL of the backend (e.g. the HF Space URL, no trailing `/`). |

`backend/.env.example` is a placeholder template**do not put real
values in it**, it is committed and will leak via git.

---

## Runtime tweakables

The UI's **TWEAKABLES** panel hits `POST /config/runtime` to mutate
live parameters without a restart:

| Tab        | What it changes                                                                    |
| ---------- | ---------------------------------------------------------------------------------- |
| THRESHOLDS | Per-class confidence gate. Predictions below the threshold are forced to `Normal`. |
| FAULTS     | Injected fault magnitudes (standardised units).                                    |
| CONTROLLER | Per-cycle fuel-trim and spark-advance step sizes.                                  |

`POST /config/runtime  { "reset": true }` restores the bundled defaults.

---

## API summary

| Endpoint               | Purpose                                               |
| ---------------------- | ----------------------------------------------------- |
| `POST /simulate`       | Run one closed-loop cycle for a session.              |
| `POST /classify`       | Classify a raw (30 × n) sensor window.                |
| `POST /engine/select`  | Switch active engine dataset.                         |
| `POST /fault/inject`   | Latch a fault into an active session.                 |
| `GET  /status`         | Model load status + key metrics + degraded-mode info. |
| `GET  /config/runtime` | Tweakables snapshot.                                  |
| `POST /config/runtime` | Patch (or reset) tweakables.                          |

Auth endpoints (`/auth/register`, `/auth/login`, `/auth/logout`) issue
short-lived JWTs and back per-user simulation history.

---

## Further reading

- **`BILSTM_DEVELOPMENT.md`**: design notes for the v2 classifier
  (attention head, label-smoothing, threshold calibration).
- **`HANDOFF.md`**: operational notes, deployment-specific gotchas,
  diagnostic SQL, and the rationale behind each hosting choice
  (HF over Fly.io, S3 over R2, Session pooler over Direct, etc.).
