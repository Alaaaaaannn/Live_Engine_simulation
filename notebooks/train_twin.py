# Auto-extracted from 04_lstm_digital_twin.ipynb
import os; os.chdir("D:/mini project/notebooks")

import os, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
np.random.seed(42)

PROC_DIR  = "../data/processed/"
MODEL_DIR = "../models/"
DATA_DIR  = "../data/raw/"

with open(PROC_DIR + "dataset_meta.json") as f:
    META = json.load(f)
with open(MODEL_DIR + "bilstm_thresholds.json") as f:
    BILSTM_META = json.load(f)

FEATURE_COLS = META["gengine1"]["feature_cols"]
N_FEATURES   = META["gengine1"]["n_features"]
WINDOW_SIZE  = META["gengine1"]["window_size"]
LAM_IDX      = META["gengine1"]["lambda_col_idx"]
IGN_IDX      = META["gengine1"]["ignition_col_idx"]
FAULT_NAMES  = {int(k): v for k, v in META["fault_names"].items()}
N_CTRL       = 2
N_FEAT_TWIN  = N_FEATURES + N_CTRL  # 15

FUEL_TO_LAMBDA = 0.3   # realistic: 0.1 fuel trim → 0.03 lambda change
SPARK_TO_IGN   = 0.10

print(f"Features: {N_FEATURES}, Twin input dim: {N_FEAT_TWIN}")

# ================

GENGINE1_TRAIN_IDS = list(range(10, 20)) + list(range(30, 40))
GENGINE1_TEST_IDS  = list(range(53, 66))

def load_engine_files(directory, col_names, file_ids):
    dfs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"): continue
        fid = int(fname.split("_")[1])
        if fid not in file_ids: continue
        df = pd.read_csv(os.path.join(directory, fname), header=0, names=col_names)
        dfs.append(df)
    return dfs

g1_train_dfs = load_engine_files(DATA_DIR + "gengine1/", FEATURE_COLS, GENGINE1_TRAIN_IDS)
g1_test_dfs  = load_engine_files(DATA_DIR + "gengine1/", FEATURE_COLS, GENGINE1_TEST_IDS)

def build_delta_twin_data(df_list, feature_cols, window_size,
                          stride=1, augment_frac=0.4, seed=42):
    """
    Build (X, y) where:
      X: (N, window_size, n_features + 2)  — window + control action
      y: (N, n_features)                   — DELTA to next timestep (not absolute)

    Delta approach: model learns the change, not the next absolute value.
    At inference: next_state = current_state + twin_predicted_delta
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    n_feat = len(feature_cols)

    for df in df_list:
        arr = df[feature_cols].values.astype(np.float32)
        n_rows = len(arr)

        for start in range(0, n_rows - window_size - 1, stride):
            window      = arr[start : start + window_size]       # (30, n_feat)
            current_row = arr[start + window_size - 1]           # last row of window
            next_row    = arr[start + window_size]               # true next step
            delta       = next_row - current_row                 # (n_feat,) CHANGE

            if rng.random() < augment_frac:
                fuel_trim = rng.uniform(-0.12, 0.12)
                spark_adv = rng.uniform(-1.5, 1.5)
                # Modify delta to reflect the effect of control action
                delta_mod = delta.copy()
                delta_mod[LAM_IDX] += fuel_trim * FUEL_TO_LAMBDA
                delta_mod[IGN_IDX] += spark_adv * SPARK_TO_IGN
            else:
                fuel_trim, spark_adv = 0.0, 0.0
                delta_mod = delta.copy()

            ctrl_cols = np.full((window_size, 2), [fuel_trim, spark_adv], dtype=np.float32)
            X_aug     = np.concatenate([window, ctrl_cols], axis=1)  # (30, 15)

            X_list.append(X_aug)
            y_list.append(delta_mod)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    return X, y

print("Building DELTA twin training data (stride=1)...")
X_twin_train, y_twin_train = build_delta_twin_data(
    g1_train_dfs, FEATURE_COLS, WINDOW_SIZE, stride=1, augment_frac=0.35)
print("Building DELTA twin test data...")
X_twin_test, y_twin_test   = build_delta_twin_data(
    g1_test_dfs,  FEATURE_COLS, WINDOW_SIZE, stride=1, augment_frac=0.0)

print(f"\nTwin TRAIN: X={X_twin_train.shape}, y={y_twin_train.shape}")
print(f"Twin TEST : X={X_twin_test.shape},  y={y_twin_test.shape}")
print(f"\nDelta stats (what we're predicting):")
for i, col in enumerate(FEATURE_COLS):
    d = y_twin_train[:, i]
    print(f"  {col:<22}: mean={d.mean():.4f} std={d.std():.4f}")

# ================

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def build_lstm_twin(window_size, n_input_features, n_output_features):
    inputs = Input(shape=(window_size, n_input_features), name="window_with_ctrl")
    x = LSTM(128, return_sequences=True, name="lstm_1")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.10)(x)
    x = Dense(64, activation="relu", name="dense_1")(x)
    outputs = Dense(n_output_features, activation="linear", name="next_delta")(x)
    return Model(inputs, outputs, name="LSTM_DigitalTwin_Delta")

twin = build_lstm_twin(WINDOW_SIZE, N_FEAT_TWIN, N_FEATURES)
twin.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
twin.summary()

# ================

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_DIR + "lstm_digital_twin.h5",
                    monitor="val_loss", save_best_only=True, verbose=1)
]
print("Training LSTM Delta Twin...")
history = twin.fit(
    X_twin_train, y_twin_train,
    epochs=50, batch_size=512, validation_split=0.12,
    callbacks=callbacks, verbose=1
)
print("\nTraining complete.")

# ================

y_pred_delta = twin.predict(X_twin_test, batch_size=512, verbose=0)

print("=== Per-Channel DELTA RMSE ===")
print(f"{'Channel':<22}  {'RMSE (delta)':>14}  {'Baseline (zero-delta)':>22}  {'Better?':>8}")
print("-" * 72)
for i, col in enumerate(FEATURE_COLS):
    actual_delta = y_twin_test[:, i]
    pred_delta   = y_pred_delta[:, i]
    rmse_model   = float(np.sqrt(mean_squared_error(actual_delta, pred_delta)))
    rmse_zero    = float(np.sqrt(mean_squared_error(actual_delta, np.zeros_like(actual_delta))))
    better = "✓ BEAT" if rmse_model < rmse_zero else "check"
    print(f"{col:<22}  {rmse_model:>14.5f}  {rmse_zero:>22.5f}  {better:>8}")

overall_rmse = float(np.sqrt(mean_squared_error(y_twin_test, y_pred_delta)))
print("-" * 72)
print(f"{'OVERALL'::<22}  {overall_rmse:>14.5f}")
print(f"\nNote: Delta RMSE target (<0.05) means we predict the CHANGE accurately.")

# ================

# Reconstruct absolute next state from deltas: next = current + delta
# Use the last row of the test window as "current"
current_states   = X_twin_test[:, -1, :N_FEATURES]  # (N, 13) last row (features only)
actual_next      = current_states + y_twin_test       # true next state
predicted_next   = current_states + y_pred_delta      # model next state

print("=== Reconstructed Next-State RMSE (what the controller sees) ===")
print(f"{'Channel':<22}  {'RMSE':>8}  {'Target <0.10':>14}")
print("-" * 48)
key_channels = ["Lambda", "CO", "HC", "NOx"]
for i, col in enumerate(FEATURE_COLS):
    rmse = float(np.sqrt(mean_squared_error(actual_next[:, i], predicted_next[:, i])))
    ok   = "✓ PASS" if rmse < 0.10 else ("~ OK" if rmse < 0.20 else "✗")
    if col in key_channels or True:
        print(f"{col:<22}  {rmse:>8.4f}  {ok:>14}")

# Plot key channels
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
n_show = 500
for ax, col in zip(axes.flatten(), key_channels):
    i = FEATURE_COLS.index(col)
    ax.plot(actual_next[:n_show, i],    color="#00d4ff", lw=1.2, label="Actual next")
    ax.plot(predicted_next[:n_show, i], color="#ff3355", lw=1.2, ls="--", label="Twin predicted")
    rmse = np.sqrt(mean_squared_error(actual_next[:n_show, i], predicted_next[:n_show, i]))
    ax.set_title(f"{col}  RMSE={rmse:.4f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("LSTM Twin — Reconstructed Next State (current + predicted delta)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR + "twin_predictions.png", dpi=130, bbox_inches="tight")
plt.show()

# ================

# The twin's job is: will lambda get CLOSER to zero?
# Check directional accuracy of lambda prediction
lam_actual_delta = y_twin_test[:, LAM_IDX]
lam_pred_delta   = y_pred_delta[:, LAM_IDX]

# For each sample: does the sign of predicted delta match actual delta?
sign_match = np.sign(lam_pred_delta) == np.sign(lam_actual_delta)
directional_acc = sign_match.mean()
print(f"Lambda delta directional accuracy: {directional_acc*100:.1f}%")
print(f"(Does the twin correctly predict the direction of lambda change?)")

# Approval simulation: would the twin correctly approve/reject correction cycles?
# Simulate 1000 rich-fault scenarios: fuel_trim=-0.1 should reduce Lambda
correct_approvals = 0
total = 1000
rng = np.random.default_rng(42)
for _ in range(total):
    idx = rng.integers(0, len(X_twin_test))
    window = X_twin_test[idx, :, :N_FEATURES]
    current_lambda = float(window[-1, LAM_IDX])

    # Simulate a rich fault: lambda is negative (below zero), apply positive fuel trim
    # to push lambda back toward 0
    if current_lambda > 0:
        current_lambda = -abs(current_lambda)  # force to negative for test

    # Build input with positive fuel trim
    ctrl = np.full((WINDOW_SIZE, 2), [0.1, 0.0], dtype=np.float32)
    x_in = np.concatenate([window, ctrl], axis=1)[np.newaxis]
    pred_delta = twin.predict(x_in, verbose=0)[0]

    predicted_next_lambda = current_lambda + pred_delta[LAM_IDX]
    approved = abs(predicted_next_lambda) < abs(current_lambda)
    if approved:
        correct_approvals += 1

print(f"\nControl approval simulation (1000 rich-fault scenarios):")
print(f"  Correctly approved fuel-trim corrections: {correct_approvals/total*100:.1f}%")

# ================

import os
size_mb = os.path.getsize(MODEL_DIR + "lstm_digital_twin.h5") / 1e6
print(f"Saved: lstm_digital_twin.h5  ({size_mb:.1f} MB)")

# Compute per-channel reconstructed RMSE
per_ch_rmse = {}
for i, col in enumerate(FEATURE_COLS):
    per_ch_rmse[col] = float(np.sqrt(mean_squared_error(actual_next[:, i], predicted_next[:, i])))

twin_meta = {
    "model_path"           : "lstm_digital_twin.h5",
    "prediction_type"      : "delta",   # KEY: we predict change, not absolute
    "window_size"          : WINDOW_SIZE,
    "n_input_features"     : N_FEAT_TWIN,
    "n_output_features"    : N_FEATURES,
    "feature_cols"         : FEATURE_COLS,
    "control_dims"         : ["fuel_trim_delta", "spark_advance_delta"],
    "fuel_to_lambda_scale" : FUEL_TO_LAMBDA,
    "spark_to_ign_scale"   : SPARK_TO_IGN,
    "lambda_col_idx"       : LAM_IDX,
    "ignition_col_idx"     : IGN_IDX,
    "overall_rmse"         : float(np.sqrt(mean_squared_error(actual_next, predicted_next))),
    "per_channel_rmse"     : per_ch_rmse,
    "directional_accuracy" : float(directional_acc),
}

with open(MODEL_DIR + "twin_meta.json", "w") as f:
    json.dump(twin_meta, f, indent=2)

print("twin_meta.json saved. Proceed to Notebook 05: SHAP.")