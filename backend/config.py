"""
config.py — All constants and paths for the FastAPI backend.
"""
import os

# ── Root paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")

# ── Model file paths ───────────────────────────────────────────────────────────
BILSTM_PATH    = os.path.join(MODEL_DIR, "bilstm_classifier.h5")
TWIN_PATH      = os.path.join(MODEL_DIR, "lstm_digital_twin.h5")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "bilstm_thresholds.json")
TWIN_META_PATH  = os.path.join(MODEL_DIR, "twin_meta.json")
SHAP_CACHE_PATH = os.path.join(MODEL_DIR, "shap_cache.json")
DATASET_META_PATH = os.path.join(PROC_DIR, "dataset_meta.json")

# ── Simulation parameters ──────────────────────────────────────────────────────
WINDOW_SIZE     = 30       # timesteps per inference window
LAMBDA_TARGET   = 0.0      # stoichiometric target (standardized mean)
LAMBDA_BAND     = 0.05     # convergence tolerance (±0.05σ)
MAX_CYCLES      = 200      # max simulation cycles per session

# ── Control action step sizes ──────────────────────────────────────────────────
CTRL_STEP_FUEL  = 0.03     # fuel trim delta per cycle — ~50 cycles to heal (~25s at 2Hz)
CTRL_STEP_SPARK = 0.05     # spark advance delta per cycle — ~40 cycles for ignition fault

# ── Fault injection offsets (standardized units) ───────────────────────────────
# "emissions" keys define correlated emission offsets applied alongside the primary fault.
# Rich (fault1): excess fuel → high CO, elevated HC, suppressed NOx
# Lean (fault2): oxygen excess → low CO, slightly elevated HC, high NOx
# Ignition (fault3): incomplete combustion → elevated HC and CO, moderate NOx rise
FAULT_OFFSETS = {
    "fault1": {"col": "Lambda", "delta": -1.5,
               "emissions": {"CO": +1.2, "HC": +0.7, "NOx": -0.6}},
    "fault2": {"col": "Lambda", "delta": +1.5,
               "emissions": {"CO": -0.5, "HC": +0.3, "NOx": +1.1}},
    "fault3": {"col": "IgnitionAngle", "delta": +2.0,
               "emissions": {"CO": +0.4, "HC": +1.0, "NOx": +0.5}},
}

# ── Engine dataset paths ───────────────────────────────────────────────────────
ENGINE_DIRS = {
    "gengine1": os.path.join(DATA_DIR, "gengine1"),
    "gengine2": os.path.join(DATA_DIR, "gengine2"),
    "pengines": os.path.join(DATA_DIR, "pengines"),
}

GENGINE1_COLS = [
    "Speed", "Load", "Lambda", "IgnitionAngle", "FuelCutoff",
    "ParticleNumbers", "CO", "CO2", "HC", "NOx", "O2", "TempExhaust", "TempCatalyst"
]
GENGINE2_COLS = [
    "Speed", "Load", "Lambda", "IgnitionAngle",
    "ParticleNumbers", "HC", "NOx", "TempExhaust", "TempCatalyst"
]

FAULT_NAMES = {0: "Normal", 1: "Rich Mixture", 2: "Lean Mixture", 3: "Ignition Fault"}

# ── Session cleanup ────────────────────────────────────────────────────────────
SESSION_TTL_MINUTES = 30
