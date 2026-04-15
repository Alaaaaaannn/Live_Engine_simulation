# -*- coding: utf-8 -*-
"""
Evaluation and twin_meta.json saving — run after training is complete.
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
os.chdir("D:/mini project/notebooks")

PROC_DIR  = "../data/processed/"
MODEL_DIR = "../models/"

with open(PROC_DIR + "dataset_meta.json", encoding="utf-8") as f:
    META = json.load(f)
with open(MODEL_DIR + "bilstm_thresholds.json", encoding="utf-8") as f:
    BILSTM_META = json.load(f)

FEATURE_COLS = META["gengine1"]["feature_cols"]
N_FEATURES   = META["gengine1"]["n_features"]
WINDOW_SIZE  = META["gengine1"]["window_size"]
LAM_IDX      = META["gengine1"]["lambda_col_idx"]
IGN_IDX      = META["gengine1"]["ignition_col_idx"]
FAULT_NAMES  = {int(k): v for k, v in META["fault_names"].items()}

FUEL_TO_LAMBDA = 0.3
SPARK_TO_IGN   = 0.10

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

print("Loading trained delta twin...")
twin = tf.keras.models.load_model(MODEL_DIR + "lstm_digital_twin.h5", compile=False)
print("Model loaded:", twin.name)

print("Loading test data...")
# Rebuild test data (fast, augment_frac=0.0)
DATA_DIR = "../data/raw/"
import pandas as pd

GENGINE1_TEST_IDS = list(range(53, 66))

def load_engine_files(directory, col_names, file_ids):
    dfs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"): continue
        fid = int(fname.split("_")[1])
        if fid not in file_ids: continue
        df = pd.read_csv(os.path.join(directory, fname), header=0, names=col_names)
        dfs.append(df)
    return dfs

g1_test_dfs = load_engine_files(DATA_DIR + "gengine1/", FEATURE_COLS, GENGINE1_TEST_IDS)

def build_delta_twin_data(df_list, feature_cols, window_size, stride=1, augment_frac=0.0, seed=42):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    n_feat = len(feature_cols)
    for df in df_list:
        arr = df[feature_cols].values.astype(np.float32)
        n_rows = len(arr)
        for start in range(0, n_rows - window_size - 1, stride):
            window      = arr[start : start + window_size]
            current_row = arr[start + window_size - 1]
            next_row    = arr[start + window_size]
            delta       = next_row - current_row
            ctrl_cols   = np.full((window_size, 2), [0.0, 0.0], dtype=np.float32)
            X_aug       = np.concatenate([window, ctrl_cols], axis=1)
            X_list.append(X_aug)
            y_list.append(delta)
    return np.stack(X_list).astype(np.float32), np.stack(y_list).astype(np.float32)

print("Building test set (stride=1)...")
X_twin_test, y_twin_test = build_delta_twin_data(g1_test_dfs, FEATURE_COLS, WINDOW_SIZE, stride=1)
print(f"Test: X={X_twin_test.shape}, y={y_twin_test.shape}")

print("Predicting...")
y_pred_delta = twin.predict(X_twin_test, batch_size=512, verbose=0)

print("\n=== Per-Channel DELTA RMSE ===")
print(f"{'Channel':<22}  {'RMSE (delta)':>14}  {'Baseline (zero-delta)':>22}  {'Better?':>8}")
print("-" * 72)
for i, col in enumerate(FEATURE_COLS):
    actual_delta = y_twin_test[:, i]
    pred_delta   = y_pred_delta[:, i]
    rmse_model   = float(np.sqrt(mean_squared_error(actual_delta, pred_delta)))
    rmse_zero    = float(np.sqrt(mean_squared_error(actual_delta, np.zeros_like(actual_delta))))
    better = "BEAT" if rmse_model < rmse_zero else "check"
    print(f"{col:<22}  {rmse_model:>14.5f}  {rmse_zero:>22.5f}  {better:>8}")

overall_rmse = float(np.sqrt(mean_squared_error(y_twin_test, y_pred_delta)))
print("-" * 72)
print(f"OVERALL RMSE: {overall_rmse:.5f}")

# Reconstruct absolute next state
current_states = X_twin_test[:, -1, :N_FEATURES]
actual_next    = current_states + y_twin_test
predicted_next = current_states + y_pred_delta

print("\n=== Reconstructed Next-State RMSE ===")
per_ch_rmse = {}
for i, col in enumerate(FEATURE_COLS):
    rmse = float(np.sqrt(mean_squared_error(actual_next[:, i], predicted_next[:, i])))
    per_ch_rmse[col] = rmse
    ok = "PASS" if rmse < 0.10 else ("OK" if rmse < 0.20 else "FAIL")
    print(f"{col:<22}  {rmse:>8.4f}  {ok}")

# Plot
key_channels = ["Lambda", "CO", "HC", "NOx"]
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
n_show = 500
for ax, col in zip(axes.flatten(), key_channels):
    i = FEATURE_COLS.index(col)
    ax.plot(actual_next[:n_show, i],    color="#00d4ff", lw=1.2, label="Actual next")
    ax.plot(predicted_next[:n_show, i], color="#ff3355", lw=1.2, ls="--", label="Twin predicted")
    rmse = np.sqrt(mean_squared_error(actual_next[:n_show, i], predicted_next[:n_show, i]))
    ax.set_title(f"{col}  RMSE={rmse:.4f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
plt.suptitle("LSTM Twin — Delta Prediction (current + predicted delta)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR + "twin_predictions.png", dpi=130, bbox_inches="tight")
plt.close()
print("Plot saved.")

# Directional accuracy
lam_actual_delta = y_twin_test[:, LAM_IDX]
lam_pred_delta   = y_pred_delta[:, LAM_IDX]
sign_match       = np.sign(lam_pred_delta) == np.sign(lam_actual_delta)
directional_acc  = sign_match.mean()
print(f"\nLambda directional accuracy: {directional_acc*100:.1f}%")

# Save twin_meta.json
overall_next_rmse = float(np.sqrt(mean_squared_error(actual_next, predicted_next)))
twin_meta = {
    "model_path"           : "lstm_digital_twin.h5",
    "prediction_type"      : "delta",
    "window_size"          : WINDOW_SIZE,
    "n_input_features"     : N_FEATURES + 2,
    "n_output_features"    : N_FEATURES,
    "feature_cols"         : FEATURE_COLS,
    "control_dims"         : ["fuel_trim_delta", "spark_advance_delta"],
    "fuel_to_lambda_scale" : FUEL_TO_LAMBDA,
    "spark_to_ign_scale"   : SPARK_TO_IGN,
    "lambda_col_idx"       : LAM_IDX,
    "ignition_col_idx"     : IGN_IDX,
    "overall_rmse"         : overall_next_rmse,
    "per_channel_rmse"     : per_ch_rmse,
    "directional_accuracy" : float(directional_acc),
}
with open(MODEL_DIR + "twin_meta.json", "w", encoding="utf-8") as f:
    json.dump(twin_meta, f, indent=2)

print(f"\ntwin_meta.json saved.")
print(f"prediction_type: delta")
print(f"overall_next_state_rmse: {overall_next_rmse:.5f}")
print("Done.")
