# -*- coding: utf-8 -*-
import os, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(HERE)
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed") + os.sep
MODEL_DIR = os.path.join(BASE_DIR, "models") + os.sep
DATA_DIR  = os.path.join(BASE_DIR, "data", "raw") + os.sep

# Prefer v2 thresholds if available
_v2_thr = os.path.join(MODEL_DIR, "bilstm_v2_thresholds.json")
_v1_thr = os.path.join(MODEL_DIR, "bilstm_thresholds.json")
_thr_path = _v2_thr if os.path.exists(_v2_thr) else _v1_thr

# Prefer v2 classifier if available
_v2_clf = os.path.join(MODEL_DIR, "bilstm_v2_classifier.h5")
_v1_clf = os.path.join(MODEL_DIR, "bilstm_classifier.h5")
_clf_path = _v2_clf if os.path.exists(_v2_clf) else _v1_clf

with open(PROC_DIR + "dataset_meta.json") as f:
    META = json.load(f)
with open(_thr_path) as f:
    BILSTM_META = json.load(f)
with open(MODEL_DIR + "twin_meta.json") as f:
    TWIN_META = json.load(f)
with open(MODEL_DIR + "shap_cache.json") as f:
    SHAP_CACHE = json.load(f)

FEATURE_COLS = META["gengine1"]["feature_cols"]
N_FEATURES   = META["gengine1"]["n_features"]
WINDOW_SIZE  = META["gengine1"]["window_size"]
LAM_IDX      = META["gengine1"]["lambda_col_idx"]
IGN_IDX      = META["gengine1"]["ignition_col_idx"]
FAULT_NAMES  = {int(k): v for k, v in META["fault_names"].items()}
FAULT_OFFSETS = META["gengine1"]["fault_offsets"]

print("All configs loaded.")

# ================

import tensorflow as tf
tf.random.set_seed(42)

classifier = tf.keras.models.load_model(_clf_path, compile=False)
twin       = tf.keras.models.load_model(MODEL_DIR + "lstm_digital_twin.h5", compile=False)

print(f"BiLSTM classifier : {classifier.name}")
print(f"LSTM digital twin : {twin.name}")

# ================

# ── Fault Classifier wrapper ──────────────────────────────────────────────────
def classify_window(window: np.ndarray):
    """
    Args: window (30, 13)
    Returns: (fault_class: int, confidence: float, probabilities: list)
    """
    x = window[np.newaxis, ...]                             # (1, 30, 13)
    probs = classifier.predict(x, verbose=0)[0]             # (4,)
    fault_class = int(np.argmax(probs))
    confidence  = float(probs[fault_class])
    return fault_class, confidence, probs.tolist()


# ── Supervisory Controller ─────────────────────────────────────────────────────
CTRL_STEP = 0.10     # per-cycle fuel trim step size (in standardized units)
SPARK_STEP = 0.50    # per-cycle spark advance step size

def compute_control_action(fault_class: int, lambda_current: float):
    """
    Rule-based controller. Returns (fuel_trim_delta, spark_advance_delta).
    """
    if fault_class == 1:   # Rich: too much fuel → reduce fuel
        return -CTRL_STEP, 0.0
    elif fault_class == 2: # Lean: too little fuel → increase fuel
        return +CTRL_STEP, 0.0
    elif fault_class == 3: # Ignition fault → advance spark timing
        return 0.0, -SPARK_STEP
    else:
        return 0.0, 0.0    # Normal: no action


# ── Digital Twin Validator ─────────────────────────────────────────────────────
FUEL_TO_LAMBDA = TWIN_META["fuel_to_lambda_scale"]
SPARK_TO_IGN   = TWIN_META["spark_to_ign_scale"]
LAMBDA_TARGET  = 0.0   # standardized mean ≈ stoichiometric

def validate_action(window: np.ndarray, fuel_trim: float, spark_adv: float,
                    lambda_current: float):
    """
    Twin predicts next state under the proposed action (delta prediction).
    next_state = current_last_row + predicted_delta
    Approves if predicted Lambda is closer to target than current Lambda.
    Returns (approved: bool, predicted_next_state: np.ndarray, predicted_lambda: float)
    """
    ctrl_cols = np.full((WINDOW_SIZE, 2), [fuel_trim, spark_adv], dtype=np.float32)
    x_twin    = np.concatenate([window, ctrl_cols], axis=1)[np.newaxis, ...]  # (1, 30, 15)

    current_last_row = window[-1].copy()  # (13,)
    if TWIN_META.get("prediction_type") == "delta":
        predicted_delta = twin.predict(x_twin, verbose=0)[0]  # (13,)
        next_state = current_last_row + predicted_delta
    else:
        next_state = twin.predict(x_twin, verbose=0)[0]

    pred_lambda    = float(next_state[LAM_IDX])
    current_dist   = abs(lambda_current - LAMBDA_TARGET)
    predicted_dist = abs(pred_lambda - LAMBDA_TARGET)

    approved = predicted_dist < current_dist   # Approve if Twin says Lambda improves
    return approved, next_state, pred_lambda


# ── Fault Injector ─────────────────────────────────────────────────────────────
def inject_fault(window: np.ndarray, fault_type: int):
    w = window.copy()
    if fault_type == 1:
        w[:, LAM_IDX] += FAULT_OFFSETS["rich"]
    elif fault_type == 2:
        w[:, LAM_IDX] += FAULT_OFFSETS["lean"]
    elif fault_type == 3:
        w[:, IGN_IDX] += FAULT_OFFSETS["ignition"]
    return w

print("Simulation components defined.")

# ================

# Load test trajectory (file 53)
test_file = DATA_DIR + "gengine1/raw_000053_measurement_000000.csv"
df_test   = pd.read_csv(test_file, header=0, names=FEATURE_COLS)
arr       = df_test[FEATURE_COLS].values.astype(np.float32)

# Simulation parameters
N_CYCLES         = 100
FAULT_INJECT_AT  = 20      # inject fault at cycle 20
FAULT_TYPE       = 1       # Rich mixture
AUTO_CORRECTION  = True

# Initialize state window with first 30 timesteps
state_window = arr[:WINDOW_SIZE].copy()
current_pos  = WINDOW_SIZE

# Storage
log = {
    "cycle": [], "fault_class": [], "fault_name": [], "confidence": [],
    "lambda_current": [], "lambda_predicted": [],
    "fuel_trim": [], "spark_adv": [],
    "twin_approved": [], "control_applied": [],
    "co": [], "hc": [], "nox": []
}

print(f"Starting simulation: {N_CYCLES} cycles, fault at cycle {FAULT_INJECT_AT}")
print(f"{'Cycle':>6} {'FaultClass':>14} {'Conf':>6} {'Lambda':>8} {'Action':>20} {'TwinDecision':>14}")
print("-" * 75)

for cycle in range(N_CYCLES):
    # ── Optional fault injection ──────────────────────────────────────────
    if cycle == FAULT_INJECT_AT:
        state_window = inject_fault(state_window, FAULT_TYPE)

    # ── Classify current window ───────────────────────────────────────────
    fault_class, confidence, probs = classify_window(state_window)
    lambda_current = float(state_window[-1, LAM_IDX])
    co_current     = float(state_window[-1, FEATURE_COLS.index("CO")])
    hc_current     = float(state_window[-1, FEATURE_COLS.index("HC")])
    nox_current    = float(state_window[-1, FEATURE_COLS.index("NOx")])

    # ── Supervisory control ───────────────────────────────────────────────
    fuel_trim, spark_adv = 0.0, 0.0
    twin_approved  = False
    lambda_predicted = lambda_current

    if AUTO_CORRECTION and fault_class != 0:
        fuel_trim, spark_adv = compute_control_action(fault_class, lambda_current)
        approved, next_state, lambda_predicted = validate_action(
            state_window, fuel_trim, spark_adv, lambda_current)
        twin_approved = approved

        if approved:
            # Apply correction: slide window forward using twin-predicted next state
            new_row = next_state.reshape(1, N_FEATURES)
            state_window = np.concatenate([state_window[1:], new_row], axis=0)
        else:
            # Twin rejected — slide forward with raw next timestep if available
            if current_pos < len(arr):
                new_row = arr[current_pos:current_pos+1]
                state_window = np.concatenate([state_window[1:], new_row], axis=0)
                current_pos += 1
    else:
        # No correction — slide forward with raw data
        if current_pos < len(arr):
            new_row = arr[current_pos:current_pos+1]
            state_window = np.concatenate([state_window[1:], new_row], axis=0)
            current_pos += 1

    # ── Log ───────────────────────────────────────────────────────────────
    action_str  = f"F{fuel_trim:+.2f} S{spark_adv:+.2f}" if fault_class != 0 else "none"
    twin_str    = "APPROVED" if twin_approved else ("REJECTED" if fault_class != 0 else "N/A")

    log["cycle"].append(cycle)
    log["fault_class"].append(fault_class)
    log["fault_name"].append(FAULT_NAMES[fault_class])
    log["confidence"].append(round(confidence * 100, 1))
    log["lambda_current"].append(round(lambda_current, 4))
    log["lambda_predicted"].append(round(lambda_predicted, 4))
    log["fuel_trim"].append(fuel_trim)
    log["spark_adv"].append(spark_adv)
    log["twin_approved"].append(twin_approved)
    log["control_applied"].append(AUTO_CORRECTION and twin_approved)
    log["co"].append(round(co_current, 4))
    log["hc"].append(round(hc_current, 4))
    log["nox"].append(round(nox_current, 4))

    if cycle % 10 == 0 or fault_class != 0:
        print(f"{cycle:>6} {FAULT_NAMES[fault_class]:>14} {confidence*100:>5.1f}% "
              f"{lambda_current:>8.4f} {action_str:>20} {twin_str:>14}")

log_df = pd.DataFrame(log)
print("-" * 75)
print(f"Simulation complete. Total cycles: {N_CYCLES}")
print(f"Fault detected at: cycles {log_df[log_df.fault_class != 0].cycle.tolist()[:5]}...")

# ================

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Lambda Convergence ────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(log_df["cycle"], log_df["lambda_current"],   color="#00d4ff", lw=1.8, label="Lambda (actual)")
ax1.plot(log_df["cycle"], log_df["lambda_predicted"], color="#ff3355", lw=1.5, ls="--", label="Lambda (twin predicted)")
ax1.axhspan(-0.05, 0.05, alpha=0.2, color="lime", label="Stoich. band (±0.05σ)")
ax1.axvline(FAULT_INJECT_AT, color="red", lw=1.5, ls=":", alpha=0.7, label=f"Fault injected (cycle {FAULT_INJECT_AT})")
ax1.set_title("Lambda Convergence", fontsize=11, fontweight="bold")
ax1.set_xlabel("Cycle"); ax1.set_ylabel("Standardized Lambda")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25, linestyle="--")
ax1.spines[["top","right"]].set_visible(False)

# ── CO ────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(log_df["cycle"], log_df["co"], color="#ff3355", lw=1.6)
ax2.axvline(FAULT_INJECT_AT, color="red", lw=1.2, ls=":", alpha=0.6)
ax2.set_title("CO Emission", fontsize=10, fontweight="bold")
ax2.set_xlabel("Cycle"); ax2.set_ylabel("Std. CO"); ax2.grid(True, alpha=0.25, linestyle="--")
ax2.spines[["top","right"]].set_visible(False)

# ── HC ────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(log_df["cycle"], log_df["hc"], color="#ffaa00", lw=1.6)
ax3.axvline(FAULT_INJECT_AT, color="red", lw=1.2, ls=":", alpha=0.6)
ax3.set_title("HC Emission", fontsize=10, fontweight="bold")
ax3.set_xlabel("Cycle"); ax3.set_ylabel("Std. HC"); ax3.grid(True, alpha=0.25, linestyle="--")
ax3.spines[["top","right"]].set_visible(False)

# ── NOx ───────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(log_df["cycle"], log_df["nox"], color="#7b68ee", lw=1.6)
ax4.axvline(FAULT_INJECT_AT, color="red", lw=1.2, ls=":", alpha=0.6)
ax4.set_title("NOx Emission", fontsize=10, fontweight="bold")
ax4.set_xlabel("Cycle"); ax4.set_ylabel("Std. NOx"); ax4.grid(True, alpha=0.25, linestyle="--")
ax4.spines[["top","right"]].set_visible(False)

# ── Fault Class Timeline ──────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
class_colors_map = {0: "#00ff88", 1: "#ff3355", 2: "#ffaa00", 3: "#7b68ee"}
for cls in range(4):
    mask = log_df["fault_class"] == cls
    ax5.scatter(log_df[mask]["cycle"], [cls] * mask.sum(),
                c=class_colors_map[cls], s=20, alpha=0.8, label=FAULT_NAMES[cls])
ax5.set_yticks(range(4)); ax5.set_yticklabels([FAULT_NAMES[c] for c in range(4)], fontsize=8)
ax5.set_title("Fault Class per Cycle", fontsize=10, fontweight="bold")
ax5.set_xlabel("Cycle"); ax5.legend(fontsize=7, loc="upper right")
ax5.grid(True, alpha=0.25, linestyle="--")
ax5.spines[["top","right"]].set_visible(False)

plt.suptitle(f"Closed-Loop Simulation — Rich Mixture Fault Injected at Cycle {FAULT_INJECT_AT}",
             fontsize=13, fontweight="bold")
plt.savefig(MODEL_DIR + "simulation_result.png", dpi=130, bbox_inches="tight")
# plt.show()
print("Simulation plot saved.")

# ================

# Count cycles from fault injection to convergence (Lambda back in stoich band)
fault_cycles  = log_df[log_df["cycle"] >= FAULT_INJECT_AT]
converge_mask = fault_cycles["lambda_current"].abs() < 0.05

if converge_mask.any():
    converge_cycle = int(fault_cycles[converge_mask].iloc[0]["cycle"])
    cycles_to_converge = converge_cycle - FAULT_INJECT_AT
    print(f" Lambda converged at cycle {converge_cycle} "
          f"({cycles_to_converge} cycles after fault injection)")
    print(f"  Target: < 50 cycles — {'PASS' if cycles_to_converge < 50 else 'FAIL'}")
else:
    print(" Lambda did not fully converge in this run.")

approved_count = log_df["twin_approved"].sum()
total_fault    = (log_df["fault_class"] != 0).sum()
print(f"\nTwin-approved corrections: {approved_count} / {total_fault} fault cycles")
print(f"Closed-loop complete. Proceed to build the FastAPI backend.")