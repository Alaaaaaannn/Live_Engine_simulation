import json

def md(src, uid):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src, 'id': f'md{uid}'}
def code(src, uid):
    return {'cell_type': 'code', 'metadata': {}, 'source': src, 'outputs': [], 'execution_count': None, 'id': f'cd{uid}'}

cells = []

cells.append(md(
    '# Notebook 04: LSTM Digital Twin\n\n'
    'Trains a next-state regression model used to **validate every proposed control action**\n'
    'before it is applied. No correction is ever committed without twin approval.\n\n'
    '**Architecture:** Input(30, n_features+2) → LSTM(64) → LSTM(32) → Dense(n_features)\n\n'
    '**Target:** RMSE < 0.05 (normalized) on CO, HC, NOx, Lambda next-state predictions.',
    '0001'))

cells.append(code(
    '''import os, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
np.random.seed(42)

PROC_DIR  = "../data/processed/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)

with open(PROC_DIR + "dataset_meta.json") as f:
    META = json.load(f)

FEATURE_COLS = META["gengine1"]["feature_cols"]
N_FEATURES   = META["gengine1"]["n_features"]
WINDOW_SIZE  = META["gengine1"]["window_size"]
LAM_IDX      = META["gengine1"]["lambda_col_idx"]
IGN_IDX      = META["gengine1"]["ignition_col_idx"]
FAULT_NAMES  = {int(k): v for k, v in META["fault_names"].items()}

# Control action column indices appended to the window
# We append 2 extra channels: [fuel_trim_delta, spark_advance_delta]
N_CTRL       = 2
N_FEAT_TWIN  = N_FEATURES + N_CTRL   # 13 + 2 = 15 input features for the twin

print(f"Features      : {N_FEATURES}")
print(f"Control dims  : {N_CTRL}")
print(f"Twin input dim: {N_FEAT_TWIN}")
print(f"Window size   : {WINDOW_SIZE}")''',
    '0002'))

cells.append(md('## 4.1 Prepare Twin Training Data\n\nFor each 30-step window, the **target** is the sensor values at timestep 31.\nWe augment with synthetic control actions to teach the twin how corrections shift the next state.', '0003'))

cells.append(code(
    '''def load_engine_files(directory, col_names, file_ids):
    import pandas as pd
    dfs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        fid = int(fname.split("_")[1])
        if fid not in file_ids:
            continue
        df = pd.read_csv(os.path.join(directory, fname), header=0, names=col_names)
        dfs.append(df)
    return dfs

GENGINE1_COLS = FEATURE_COLS  # 13 cols
DATA_DIR = "../data/raw/"
GENGINE1_TRAIN_IDS = list(range(10, 20)) + list(range(30, 40))
GENGINE1_TEST_IDS  = list(range(53, 66))

g1_train_dfs = load_engine_files(DATA_DIR + "gengine1/", GENGINE1_COLS, GENGINE1_TRAIN_IDS)
g1_test_dfs  = load_engine_files(DATA_DIR + "gengine1/", GENGINE1_COLS, GENGINE1_TEST_IDS)

# Control-action effect approximation:
# fuel_trim_delta=+0.1  →  Lambda increases by ~0.2 (lean correction)
# fuel_trim_delta=-0.1  →  Lambda decreases by ~0.2 (rich correction)
# spark_advance_delta=+1.0 → IgnitionAngle shifts by ~0.15 σ
FUEL_TO_LAMBDA = 2.0
SPARK_TO_IGN   = 0.15

def build_twin_data(df_list, feature_cols, window_size, stride=5,
                    augment_frac=0.4, seed=42):
    """
    Build (X, y) pairs for next-state prediction.
    X: (N, window_size, n_features + 2)  — window + control action broadcast
    y: (N, n_features)                   — next timestep sensor values
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    n_feat = len(feature_cols)

    for df in df_list:
        arr = df[feature_cols].values.astype(np.float32)  # (T, n_feat)
        n_rows = len(arr)

        for start in range(0, n_rows - window_size, stride):
            window = arr[start : start + window_size].copy()   # (30, n_feat)
            target = arr[start + window_size].copy()           # (n_feat,)  — true next state

            if rng.random() < augment_frac:
                # Inject a random control action
                fuel_trim    = rng.uniform(-0.15, 0.15)
                spark_adv    = rng.uniform(-2.0,  2.0)
                # Simulate effect on next state
                target_mod   = target.copy()
                target_mod[LAM_IDX] += fuel_trim * FUEL_TO_LAMBDA
                target_mod[IGN_IDX] += spark_adv * SPARK_TO_IGN
                ctrl_cols = np.full((window_size, 2),
                                    [fuel_trim, spark_adv], dtype=np.float32)
            else:
                fuel_trim, spark_adv = 0.0, 0.0
                target_mod = target.copy()
                ctrl_cols = np.zeros((window_size, 2), dtype=np.float32)

            # Append control dims to every timestep in the window
            X_aug = np.concatenate([window, ctrl_cols], axis=1)  # (30, 15)

            X_list.append(X_aug)
            y_list.append(target_mod)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    return X, y

print("Building twin training data...")
X_twin_train, y_twin_train = build_twin_data(g1_train_dfs, GENGINE1_COLS,
                                              WINDOW_SIZE, stride=5, augment_frac=0.4)
print("Building twin test data...")
X_twin_test, y_twin_test   = build_twin_data(g1_test_dfs,  GENGINE1_COLS,
                                              WINDOW_SIZE, stride=5, augment_frac=0.0)

print(f"\\nTwin TRAIN: X={X_twin_train.shape}, y={y_twin_train.shape}")
print(f"Twin TEST : X={X_twin_test.shape},  y={y_twin_test.shape}")''',
    '0004'))

cells.append(md('## 4.2 Build the LSTM Digital Twin Model', '0005'))

cells.append(code(
    '''import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow: {tf.__version__}")

def build_lstm_twin(window_size, n_input_features, n_output_features):
    """
    LSTM that predicts the next sensor state given a window + control action.
    """
    inputs = Input(shape=(window_size, n_input_features), name="window_with_ctrl")
    x = LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.20)(x)
    x = LSTM(32, return_sequences=False, name="lstm_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation="relu", name="dense_1")(x)
    x = Dropout(0.10)(x)
    outputs = Dense(n_output_features, activation="linear", name="next_state")(x)
    model = Model(inputs, outputs, name="LSTM_DigitalTwin")
    return model

twin = build_lstm_twin(WINDOW_SIZE, N_FEAT_TWIN, N_FEATURES)
twin.summary()''',
    '0006'))

cells.append(md('## 4.3 Train', '0007'))

cells.append(code(
    '''twin.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=7,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(
        MODEL_DIR + "lstm_digital_twin.h5",
        monitor="val_loss", save_best_only=True, verbose=1
    )
]

print("Training LSTM Digital Twin...")
history = twin.fit(
    X_twin_train, y_twin_train,
    epochs=50,
    batch_size=256,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)
print("\\nTraining complete.")''',
    '0008'))

cells.append(md('## 4.4 Training Curves', '0009'))

cells.append(code(
    '''fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(history.history["loss"],     label="Train MSE", color="#00d4ff", lw=1.8)
ax1.plot(history.history["val_loss"], label="Val MSE",   color="#ff3355", lw=1.8, ls="--")
ax1.set_title("Loss (MSE)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE")
ax1.legend(); ax1.grid(True, alpha=0.25, linestyle="--")
ax1.spines[["top","right"]].set_visible(False)

ax2.plot(history.history["mae"],     label="Train MAE", color="#00d4ff", lw=1.8)
ax2.plot(history.history["val_mae"], label="Val MAE",   color="#00ff88", lw=1.8, ls="--")
ax2.set_title("MAE", fontsize=11, fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE")
ax2.legend(); ax2.grid(True, alpha=0.25, linestyle="--")
ax2.spines[["top","right"]].set_visible(False)

plt.suptitle("LSTM Digital Twin — Training History", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR + "twin_training_history.png", dpi=130, bbox_inches="tight")
plt.show()''',
    '0010'))

cells.append(md('## 4.5 Per-Channel RMSE Evaluation', '0011'))

cells.append(code(
    '''y_pred_test = twin.predict(X_twin_test, batch_size=512, verbose=0)

print("=== Per-Channel RMSE on Test Set ===")
print(f"{'Channel':<22}  {'RMSE':>8}  {'MAE':>8}  {'Target < 0.05':>14}")
print("-" * 58)
rmse_all = []
for i, col in enumerate(FEATURE_COLS):
    rmse = float(np.sqrt(mean_squared_error(y_twin_test[:, i], y_pred_test[:, i])))
    mae  = float(np.mean(np.abs(y_twin_test[:, i] - y_pred_test[:, i])))
    ok   = "✓ PASS" if rmse < 0.05 else "✗ FAIL"
    print(f"{col:<22}  {rmse:>8.4f}  {mae:>8.4f}  {ok:>14}")
    rmse_all.append(rmse)

overall_rmse = float(np.sqrt(mean_squared_error(y_twin_test, y_pred_test)))
print("-" * 58)
print(f"{'OVERALL RMSE':<22}  {overall_rmse:>8.4f}")
print(f"\\nTarget: RMSE < 0.05 on emission-relevant channels (CO, HC, NOx, Lambda)")''',
    '0012'))

cells.append(md('## 4.6 Visualize Next-State Predictions', '0013'))

cells.append(code(
    '''# Plot actual vs predicted for Lambda, CO, HC, NOx over first 300 test samples
plot_cols_idx = [FEATURE_COLS.index(c) for c in ["Lambda", "CO", "HC", "NOx"]]
plot_cols_name = ["Lambda", "CO", "HC", "NOx"]
n_show = 300

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
colors_actual = "#00d4ff"
colors_pred   = "#ff3355"

for ax, (c_idx, c_name) in zip(axes.flatten(), zip(plot_cols_idx, plot_cols_name)):
    actual = y_twin_test[:n_show, c_idx]
    pred   = y_pred_test[:n_show, c_idx]
    t      = np.arange(n_show)
    ax.plot(t, actual, color=colors_actual, lw=1.2, alpha=0.9, label="Actual")
    ax.plot(t, pred,   color=colors_pred,   lw=1.2, alpha=0.9, ls="--", label="Predicted")
    rmse = np.sqrt(mean_squared_error(actual, pred))
    ax.set_title(f"{c_name}  (RMSE={rmse:.4f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Test Sample"); ax.set_ylabel("Standardized Value")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("LSTM Digital Twin — Actual vs Predicted Next State",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR + "twin_predictions.png", dpi=130, bbox_inches="tight")
plt.show()''',
    '0014'))

cells.append(md('## 4.7 Save Model Metadata', '0015'))

cells.append(code(
    '''import os
saved_path = MODEL_DIR + "lstm_digital_twin.h5"
size_mb    = os.path.getsize(saved_path) / 1e6
print(f"Saved: {saved_path}  ({size_mb:.1f} MB)")

twin_meta = {
    "model_path"          : "lstm_digital_twin.h5",
    "window_size"         : WINDOW_SIZE,
    "n_input_features"    : N_FEAT_TWIN,
    "n_output_features"   : N_FEATURES,
    "feature_cols"        : FEATURE_COLS,
    "control_dims"        : ["fuel_trim_delta", "spark_advance_delta"],
    "fuel_to_lambda_scale": FUEL_TO_LAMBDA,
    "spark_to_ign_scale"  : SPARK_TO_IGN,
    "lambda_col_idx"      : LAM_IDX,
    "ignition_col_idx"    : IGN_IDX,
    "overall_rmse"        : overall_rmse,
    "per_channel_rmse"    : {col: float(np.sqrt(mean_squared_error(
                                y_twin_test[:, i], y_pred_test[:, i])))
                              for i, col in enumerate(FEATURE_COLS)}
}

with open(MODEL_DIR + "twin_meta.json", "w") as f:
    json.dump(twin_meta, f, indent=2)

print("twin_meta.json saved.")
print("\\nProceed to Notebook 05: SHAP Explainability.")''',
    '0016'))

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': cells
}

out = 'D:/mini project/notebooks/04_lstm_digital_twin.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'Written: {out}')
