"""
config.py — All constants and paths for the FastAPI backend.

`RUNTIME` is a process-wide, mutable dict.  Anything that should be
tweakable from the UI lives here.  Modules read from `config.RUNTIME[...]`
each call so changes take effect without restart.
"""
import os
import copy

# ── Root paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")

# ── Model file paths (BiLSTM v2 preferred) ─────────────────────────────────────
BILSTM_V2_PATH    = os.path.join(MODEL_DIR, "bilstm_v2_classifier.h5")
BILSTM_V1_PATH    = os.path.join(MODEL_DIR, "bilstm_classifier.h5")
BILSTM_PATH       = BILSTM_V2_PATH if os.path.exists(BILSTM_V2_PATH) else BILSTM_V1_PATH

THR_V2_PATH       = os.path.join(MODEL_DIR, "bilstm_v2_thresholds.json")
THR_V1_PATH       = os.path.join(MODEL_DIR, "bilstm_thresholds.json")
THRESHOLDS_PATH   = THR_V2_PATH if os.path.exists(THR_V2_PATH) else THR_V1_PATH

TWIN_PATH         = os.path.join(MODEL_DIR, "lstm_digital_twin.h5")
TWIN_META_PATH    = os.path.join(MODEL_DIR, "twin_meta.json")
SHAP_CACHE_PATH   = os.path.join(MODEL_DIR, "shap_cache.json")
DATASET_META_PATH = os.path.join(PROC_DIR, "dataset_meta.json")

# ── Simulation parameters ──────────────────────────────────────────────────────
WINDOW_SIZE   = 30
LAMBDA_TARGET = 0.0
LAMBDA_BAND   = 0.05
MAX_CYCLES    = 200

# Stability window — number of recent classifications used to compute
# the majority label / agreement fraction (Next-Step #7).
STABILITY_WINDOW = 10

# ── Default controller step sizes ──────────────────────────────────────────────
DEFAULT_CTRL_STEP_FUEL  = 0.03
DEFAULT_CTRL_STEP_SPARK = 0.05

# ── Default fault injection offsets (standardised units) ───────────────────────
DEFAULT_FAULT_OFFSETS = {
    "fault1": {"col": "Lambda", "delta": -1.5,
               "emissions": {"CO": +1.2, "HC": +0.7, "NOx": -0.6}},
    "fault2": {"col": "Lambda", "delta": +1.5,
               "emissions": {"CO": -0.5, "HC": +0.3, "NOx": +1.1}},
    "fault3": {"col": "IgnitionAngle", "delta": +2.0,
               "emissions": {"CO": +0.4, "HC": +1.0, "NOx": +0.5}},
}

# Default confidence thresholds.  Replaced from disk at startup, then
# mutable via /config/runtime.  Any classification below threshold is
# overridden to Normal (gating, Next-Step #2).
DEFAULT_THRESHOLDS = {"0": 0.50, "1": 0.50, "2": 0.50, "3": 0.50}

# ── RUNTIME — single mutable source of truth ───────────────────────────────────
# All modules that need to read live values do `from config import RUNTIME`
# and look up the keys here.  /config/runtime mutates this dict.
RUNTIME: dict = {
    "thresholds":      copy.deepcopy(DEFAULT_THRESHOLDS),
    "fault_offsets":   copy.deepcopy(DEFAULT_FAULT_OFFSETS),
    "ctrl_step_fuel":  DEFAULT_CTRL_STEP_FUEL,
    "ctrl_step_spark": DEFAULT_CTRL_STEP_SPARK,
}


def runtime_defaults() -> dict:
    """Return a deep copy of the initial defaults for /config/runtime resets."""
    return {
        "thresholds":      copy.deepcopy(DEFAULT_THRESHOLDS),
        "fault_offsets":   copy.deepcopy(DEFAULT_FAULT_OFFSETS),
        "ctrl_step_fuel":  DEFAULT_CTRL_STEP_FUEL,
        "ctrl_step_spark": DEFAULT_CTRL_STEP_SPARK,
    }


# Back-compat aliases — older modules might still import these names.
# They read the RUNTIME dict lazily via __getattr__ further down.
def __getattr__(name: str):
    if name == "FAULT_OFFSETS":
        return RUNTIME["fault_offsets"]
    if name == "CTRL_STEP_FUEL":
        return RUNTIME["ctrl_step_fuel"]
    if name == "CTRL_STEP_SPARK":
        return RUNTIME["ctrl_step_spark"]
    raise AttributeError(f"module 'config' has no attribute {name!r}")


# ── Engine dataset paths ───────────────────────────────────────────────────────
ENGINE_DIRS = {
    "gengine1": os.path.join(DATA_DIR, "gengine1"),
    "gengine2": os.path.join(DATA_DIR, "gengine2"),
    "pengines": os.path.join(DATA_DIR, "pengines"),
}

GENGINE1_COLS = [
    "Speed", "Load", "Lambda", "IgnitionAngle", "FuelCutoff",
    "ParticleNumbers", "CO", "CO2", "HC", "NOx", "O2", "TempExhaust", "TempCatalyst",
]
GENGINE2_COLS = [
    "Speed", "Load", "Lambda", "IgnitionAngle",
    "ParticleNumbers", "HC", "NOx", "TempExhaust", "TempCatalyst",
]

FAULT_NAMES = {0: "Normal", 1: "Rich Mixture", 2: "Lean Mixture", 3: "Ignition Fault"}

# ── Session cleanup ────────────────────────────────────────────────────────────
SESSION_TTL_MINUTES = 30
