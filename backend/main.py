"""
main.py — FastAPI application entry point.
Run with: uvicorn main:app --reload --port 8000
"""
import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

import config
from models_loader    import store
from simulation_engine import run_cycle, get_or_create_session, reset_session, active_session_count
from classifier        import classify_window
from fault_injector    import inject_fault
from config            import FAULT_NAMES, WINDOW_SIZE
from schemas import (
    SimulateRequest, SimulateResponse,
    ClassifyRequest, ClassifyResponse,
    EngineSelectRequest, EngineSelectResponse,
    FaultInjectRequest, FaultInjectResponse,
    StatusResponse,
    RuntimeConfigRequest, RuntimeConfigResponse,
)
from db import (init_db, persist_cycle, User,
                 list_runs_for_user, list_cycles_for_user)
from auth import router as auth_router, get_current_user


# ── Lifespan: load models once at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    await init_db()
    yield

app = FastAPI(
    title="AI Digital Twin — Engine Fault Simulator",
    description="BiLSTM fault classifier + LSTM digital twin + supervisory controller",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow React dev server and deployed frontend) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


# ── POST /simulate ────────────────────────────────────────────────────────────

@app.post("/simulate", response_model=SimulateResponse, tags=["Simulation"])
async def simulate(req: SimulateRequest, user: User = Depends(get_current_user)):
    """
    Run one closed-loop simulation cycle for the authenticated user.
    """
    if not store.is_loaded:
        raise HTTPException(503, "Models not loaded yet. Retry in a moment.")
    try:
        resp = run_cycle(req)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Simulation error: {str(e)}")

    await persist_cycle(
        user_id       = user.id,
        session_id    = req.session_id,
        engine_id     = req.engine_id,
        cycle_idx     = resp.cycle_number,
        request_body  = req.model_dump(),
        response_body = resp.model_dump(),
    )
    return resp


# ── POST /classify ────────────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassifyResponse, tags=["Simulation"])
async def classify(req: ClassifyRequest):
    """
    Classify a raw sensor window (30 x n_features) into fault classes.
    Returns class label, confidence, and per-class probabilities.
    """
    if not store.is_loaded:
        raise HTTPException(503, "Models not loaded yet.")

    window = np.array(req.sensor_window, dtype=np.float32)
    if window.shape[0] != WINDOW_SIZE:
        raise HTTPException(400, f"Window must have exactly {WINDOW_SIZE} rows.")

    fault_class, confidence, prob_dict = classify_window(window)
    return ClassifyResponse(
        fault_class   = fault_class,
        fault_name    = FAULT_NAMES[fault_class],
        confidence    = round(confidence, 4),
        probabilities = {k: round(v, 4) for k, v in prob_dict.items()},
    )


# ── POST /engine/select ───────────────────────────────────────────────────────

@app.post("/engine/select", response_model=EngineSelectResponse, tags=["Engine"])
async def select_engine(req: EngineSelectRequest):
    """
    Switch the active engine dataset and reset simulation state.
    Loads corresponding pre-trained model weights (same models, different feature subsets).
    """
    if not store.is_loaded:
        raise HTTPException(503, "Models not loaded yet.")

    valid_engines = ["gengine1", "gengine2", "pengines"]
    if req.engine_id not in valid_engines:
        raise HTTPException(400, f"Unknown engine_id. Valid: {valid_engines}")

    meta = store.dataset_meta.get(req.engine_id)
    if meta is None:
        raise HTTPException(404, f"Engine '{req.engine_id}' metadata not found.")

    feature_cols = meta["feature_cols"]
    train_shape  = meta.get("train_shape", [0])
    sample_count = train_shape[0] if train_shape else 0

    return EngineSelectResponse(
        engine_id    = req.engine_id,
        n_features   = meta["n_features"],
        feature_cols = feature_cols,
        sample_count = sample_count,
        message      = f"Engine switched to {req.engine_id}. Simulation state reset.",
    )


# ── POST /fault/inject ────────────────────────────────────────────────────────

@app.post("/fault/inject", response_model=FaultInjectResponse, tags=["Simulation"])
async def inject_fault_endpoint(req: FaultInjectRequest):
    """
    Inject a one-shot synthetic fault into the current session's state window.
    """
    if not store.is_loaded:
        raise HTTPException(503, "Models not loaded yet.")

    valid = ["fault1", "fault2", "fault3"]
    if req.fault_type not in valid:
        raise HTTPException(400, f"Unknown fault_type. Valid: {valid}")

    # We need the session to inject into — use gengine1 as default engine
    from simulation_engine import _sessions
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found. "
                                  "Call /simulate first to initialise a session.")

    feature_cols = store.dataset_meta[sess.engine_id]["feature_cols"]
    try:
        sess.state_window = inject_fault(sess.state_window, req.fault_type, feature_cols)
        return FaultInjectResponse(
            session_id = req.session_id,
            fault_type = req.fault_type,
            applied    = True,
            message    = f"{req.fault_type} applied to session {req.session_id}.",
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── GET /status ───────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse, tags=["Health"])
async def status():
    """System health check. Returns model load status and key metrics."""
    return StatusResponse(
        models_loaded   = store.is_loaded,
        current_engine  = "gengine1",
        bilstm_f1       = store.get_bilstm_f1()  if store.is_loaded else 0.0,
        twin_rmse       = store.get_twin_rmse()   if store.is_loaded else 0.0,
        active_sessions = active_session_count(),
        message         = "All systems operational." if store.is_loaded else "Loading...",
    )


# ── /config/runtime ───────────────────────────────────────────────────────────

def _runtime_snapshot() -> RuntimeConfigResponse:
    return RuntimeConfigResponse(
        thresholds      = {k: float(v) for k, v in config.RUNTIME["thresholds"].items()},
        fault_offsets   = config.RUNTIME["fault_offsets"],
        ctrl_step_fuel  = float(config.RUNTIME["ctrl_step_fuel"]),
        ctrl_step_spark = float(config.RUNTIME["ctrl_step_spark"]),
    )


@app.get("/config/runtime", response_model=RuntimeConfigResponse, tags=["Config"])
async def get_runtime_config():
    """Return the live tweakables snapshot."""
    return _runtime_snapshot()


@app.post("/config/runtime", response_model=RuntimeConfigResponse, tags=["Config"])
async def set_runtime_config(req: RuntimeConfigRequest):
    """Shallow-merge the request body into RUNTIME (or reset to defaults)."""
    if req.reset:
        defaults = config.runtime_defaults()
        config.RUNTIME["thresholds"]      = defaults["thresholds"]
        config.RUNTIME["fault_offsets"]   = defaults["fault_offsets"]
        config.RUNTIME["ctrl_step_fuel"]  = defaults["ctrl_step_fuel"]
        config.RUNTIME["ctrl_step_spark"] = defaults["ctrl_step_spark"]
        return _runtime_snapshot()

    if req.thresholds is not None:
        for k, v in req.thresholds.items():
            if str(k) in {"0", "1", "2", "3"}:
                config.RUNTIME["thresholds"][str(k)] = float(v)

    if req.fault_offsets is not None:
        # Shallow-merge each fault entry so partial updates are allowed
        for fkey, fval in req.fault_offsets.items():
            if fkey not in config.RUNTIME["fault_offsets"]:
                continue
            target = config.RUNTIME["fault_offsets"][fkey]
            if isinstance(fval, dict):
                if "delta" in fval:
                    target["delta"] = float(fval["delta"])
                if "emissions" in fval and isinstance(fval["emissions"], dict):
                    for ek, ev in fval["emissions"].items():
                        target.setdefault("emissions", {})[ek] = float(ev)

    if req.ctrl_step_fuel is not None:
        config.RUNTIME["ctrl_step_fuel"] = float(req.ctrl_step_fuel)
    if req.ctrl_step_spark is not None:
        config.RUNTIME["ctrl_step_spark"] = float(req.ctrl_step_spark)

    return _runtime_snapshot()


# ── /history (user-scoped) ────────────────────────────────────────────────────

@app.get("/history/runs", tags=["History"])
async def history_runs(limit: int = 50, user: User = Depends(get_current_user)):
    """List recent simulation runs for the authenticated user."""
    return {"runs": await list_runs_for_user(user.id, limit=limit)}


@app.get("/history/runs/{session_id}/cycles", tags=["History"])
async def history_cycles(session_id: str, limit: int = 500,
                         user: User = Depends(get_current_user)):
    """Return persisted cycles for the user's session."""
    return {
        "session_id": session_id,
        "cycles": await list_cycles_for_user(user.id, session_id, limit=limit),
    }


# ── GET / ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "name"   : "AI Digital Twin Engine Simulator",
        "version": "1.0.0",
        "docs"   : "/docs",
        "status" : "/status",
    }
