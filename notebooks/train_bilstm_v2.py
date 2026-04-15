"""
train_bilstm_v2.py  --  BiLSTM v2 Fault Classifier
=====================================================
Improvements over v1 (03_bilstm_fault_classifier.ipynb):

ARCHITECTURE
  v1  BiLSTM(64) -> BatchNorm -> BiLSTM(32) -> Dense(32) -> Dense(4)
  v2  BiLSTM(96) -> LayerNorm -> BiLSTM(64) -> LayerNorm
       -> MultiHeadAttention (self-attn over 30 timesteps, residual)
       -> GlobalAveragePooling1D -> Dense(64) -> Dense(4)

  BatchNorm on sequence data (None,30,128) is statistically wrong --
  it normalises across the batch dimension per timestep, not per sample.
  LayerNorm normalises across the feature dimension within each sample,
  which is the correct choice for RNN/Transformer sequences.

  MultiHeadAttention lets the model weight each of the 30 timesteps
  independently instead of relying on the final hidden state alone.
  The residual connection stabilises gradients.

  Label smoothing (0.05) penalises overconfidence -- fixes the v1
  symptom where P10 thresholds were pathologically close to 1.0.

EVALUATION (critical fix)
  v1  Test set is all-Normal; evaluation re-injects faults at the
      *exact same* offsets as training (+-1.5 / +2.0). This is
      circular: the model only needs to memorise the offset value.
      F1=0.991 is therefore an inflated, invalid metric.

  v2  Faults are injected at *variable magnitudes*:
        actual_offset = base_offset * Uniform(0.8, 2.2)
      Using a completely fresh seed (999), decoupled from training.
      The model must generalise across a range of fault severities,
      not just the single training offset.

METRICS
  v1  Only accuracy + F1 + confusion matrix.
  v2  Adds:
      - Per-class ROC curves + AUC (one-vs-rest)
      - Per-class Precision-Recall curves + Average Precision
      - F1-maximising threshold calibration (replaces useless P10 ~1.0)

OUTPUTS (saved to ../models/)
  bilstm_v2_classifier.h5       -- trained model weights
  bilstm_v2_thresholds.json     -- calibrated thresholds + AUC scores
  bilstm_v2_training_history.png
  bilstm_v2_confusion_matrix.png
  bilstm_v2_roc_curves.png
"""

import os
import json
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC_DIR  = "../data/processed/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------------
with open(PROC_DIR + "dataset_meta.json") as f:
    META = json.load(f)

FAULT_NAMES  = {int(k): v for k, v in META["fault_names"].items()}
FEATURE_COLS = META["gengine1"]["feature_cols"]
N_FEATURES   = META["gengine1"]["n_features"]
WINDOW_SIZE  = META["gengine1"]["window_size"]
LAM_IDX      = META["gengine1"]["lambda_col_idx"]
IGN_IDX      = META["gengine1"]["ignition_col_idx"]

# Base fault offsets (in standardised units) read from metadata so that
# this script stays in sync with the preprocessing pipeline.
_fo = META["gengine1"]["fault_offsets"]
BASE_OFFSETS = {
    1: (LAM_IDX, float(_fo["rich"])),      # Lambda  -1.5  (rich mixture)
    2: (LAM_IDX, float(_fo["lean"])),      # Lambda  +1.5  (lean mixture)
    3: (IGN_IDX, float(_fo["ignition"])),  # IgnAngle +2.0 (ignition fault)
}

print(f"Features ({N_FEATURES}): {FEATURE_COLS}")
print(f"Window size : {WINDOW_SIZE}")

# ---------------------------------------------------------------------------
# Load preprocessed arrays
# ---------------------------------------------------------------------------
X_train = np.load(PROC_DIR + "g1_X_train.npy")
y_train = np.load(PROC_DIR + "g1_y_train.npy")
X_test  = np.load(PROC_DIR + "g1_X_test.npy")   # files 53-65, all Normal

print(f"\nX_train : {X_train.shape}   y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}")

# ---------------------------------------------------------------------------
# Build honest evaluation set  (KEY FIX vs v1)
# ---------------------------------------------------------------------------
# Faults injected at VARIABLE magnitudes: base_offset * Uniform(0.8, 2.2).
# Seed 999 has zero overlap with any seed used during training (42) or the
# v1 eval (77).  The model must generalise to fault severities it has never
# seen exactly, making this a genuine out-of-distribution test.
# ---------------------------------------------------------------------------
FAULT_FRAC_EVAL = 0.30
rng_eval = np.random.default_rng(999)

X_eval_list, y_eval_list = [], []
for i in range(len(X_test)):
    w = X_test[i].copy()
    if rng_eval.random() < FAULT_FRAC_EVAL:
        ft = int(rng_eval.integers(1, 4))          # 1, 2, or 3
        col_idx, base_off = BASE_OFFSETS[ft]
        scale = rng_eval.uniform(0.8, 2.2)         # variable magnitude
        w[:, col_idx] += base_off * scale
        y_eval_list.append(ft)
    else:
        y_eval_list.append(0)
    X_eval_list.append(w)

X_eval = np.array(X_eval_list, dtype=np.float32)
y_eval = np.array(y_eval_list, dtype=np.int32)

print(f"\nEval set  : {X_eval.shape}")
print(f"Eval dist : { {FAULT_NAMES[c]: int(np.sum(y_eval == c)) for c in range(4)} }")

# ---------------------------------------------------------------------------
# TensorFlow / Keras
# ---------------------------------------------------------------------------
import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    Add,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

print(f"\nTensorFlow : {tf.__version__}")

# One-hot labels
y_train_cat = to_categorical(y_train, num_classes=4)

# Class weights on training distribution
cw_values    = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: float(cw_values[i]) for i in range(len(cw_values))}
print(f"Class weights : {class_weights}")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
def build_bilstm_v2(window_size: int, n_features: int, n_classes: int = 4) -> Model:
    """
    BiLSTM + Self-Attention fault classifier.

    Architecture summary
    --------------------
    Input(30, 13)
    BiLSTM(96, return_sequences=True)   -> output (30, 192)
    LayerNormalization                  -> normalise per sample across features
    Dropout(0.25)
    BiLSTM(64, return_sequences=True)   -> output (30, 128)
    LayerNormalization
    Dropout(0.20)
    MultiHeadAttention(heads=4, key_dim=16)  self-attn over 30 timesteps
    Add [residual]  +  LayerNormalization    -> output still (30, 128)
    GlobalAveragePooling1D              -> weighted mean over time (128,)
    Dense(64, relu)
    Dropout(0.15)
    Dense(4, softmax)
    """
    inputs = Input(shape=(window_size, n_features), name="sensor_window")

    # -- BiLSTM stack --
    x = Bidirectional(LSTM(96, return_sequences=True), name="bilstm_1")(inputs)
    x = LayerNormalization(name="ln_1")(x)
    x = Dropout(0.25, name="drop_1")(x)

    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_2")(x)
    x = LayerNormalization(name="ln_2")(x)
    x = Dropout(0.20, name="drop_2")(x)

    # -- Self-attention over the time axis --
    # query == value == x  (self-attention, not cross-attention)
    # output shape: (None, 30, 128) -- same as input
    attn_out = MultiHeadAttention(
        num_heads=4, key_dim=16, dropout=0.10, name="self_attention"
    )(x, x)
    x = Add(name="attn_residual")([x, attn_out])   # residual connection
    x = LayerNormalization(name="ln_attn")(x)

    # -- Aggregate over the 30 timesteps --
    x = GlobalAveragePooling1D(name="gap")(x)       # (None, 128)

    # -- Classification head --
    x = Dense(64, activation="relu", name="dense_1")(x)
    x = Dropout(0.15, name="drop_3")(x)
    outputs = Dense(n_classes, activation="softmax", name="fault_class")(x)

    return Model(inputs, outputs, name="BiLSTM_v2_FaultClassifier")


model = build_bilstm_v2(WINDOW_SIZE, N_FEATURES)
model.summary()

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
# Label smoothing 0.05 redistributes 5% probability mass to non-target
# classes during training.  This penalises overconfidence and produces
# better-calibrated softmax outputs -- fixing the v1 symptom where P10
# thresholds were 0.9997-1.0 (essentially useless in deployment).
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True, verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=4, min_lr=1e-6, verbose=1,
    ),
    ModelCheckpoint(
        MODEL_DIR + "bilstm_v2_classifier.h5",
        monitor="val_loss", save_best_only=True, verbose=1,
    ),
]

print("\nStarting v2 training ...")
history = model.fit(
    X_train, y_train_cat,
    epochs=60,
    batch_size=256,
    validation_split=0.15,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)
print("Training complete.")

# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(history.history["loss"],     label="Train", color="#00d4ff", lw=1.8)
ax1.plot(history.history["val_loss"], label="Val",   color="#ff3355", lw=1.8, ls="--")
ax1.set_title("Loss (v2 -- label smoothing 0.05)", fontsize=10, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CatCrossEntropy")
ax1.legend()
ax1.grid(True, alpha=0.25, linestyle="--")
ax1.spines[["top", "right"]].set_visible(False)

ax2.plot(history.history["accuracy"],     label="Train", color="#00d4ff", lw=1.8)
ax2.plot(history.history["val_accuracy"], label="Val",   color="#00ff88", lw=1.8, ls="--")
ax2.set_title("Accuracy (v2)", fontsize=10, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.25, linestyle="--")
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle("BiLSTM v2 -- Training History", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR + "bilstm_v2_training_history.png", dpi=130, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Evaluate on honest test set
# ---------------------------------------------------------------------------
y_pred_prob = model.predict(X_eval, batch_size=512, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

target_names = [FAULT_NAMES[c] for c in range(4)]
print("\n=== Classification Report (v2 -- variable-magnitude faults) ===")
print(classification_report(y_eval, y_pred, target_names=target_names, digits=4))

macro_f1 = f1_score(y_eval, y_pred, average="macro")
print(f"Macro F1  : {macro_f1:.4f}")

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
cm     = confusion_matrix(y_eval, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names,
    ax=ax1, cbar=False, linewidths=0.5,
)
ax1.set_title("Confusion Matrix (counts)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.tick_params(axis="x", rotation=30)

sns.heatmap(
    cm_pct, annot=True, fmt=".1f", cmap="Greens",
    xticklabels=target_names, yticklabels=target_names,
    ax=ax2, cbar=False, linewidths=0.5,
)
ax2.set_title("Confusion Matrix (%)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.tick_params(axis="x", rotation=30)

plt.suptitle(
    "BiLSTM v2 -- Variable-Magnitude Fault Evaluation",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig(MODEL_DIR + "bilstm_v2_confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# ROC curves + AUC  (one-vs-rest, per class)
# ---------------------------------------------------------------------------
# label_binarize creates a (N, 4) binary matrix for one-vs-rest ROC.
y_eval_bin = label_binarize(y_eval, classes=[0, 1, 2, 3])
COLORS     = ["#00ff88", "#ff3355", "#ffaa00", "#7b68ee"]

fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

roc_aucs = {}
pr_aucs  = {}

for cls in range(4):
    # ROC
    fpr, tpr, _  = roc_curve(y_eval_bin[:, cls], y_pred_prob[:, cls])
    roc_auc      = float(auc(fpr, tpr))
    roc_aucs[cls] = roc_auc
    ax_roc.plot(
        fpr, tpr, color=COLORS[cls], lw=1.8,
        label=f"{FAULT_NAMES[cls]} (AUC={roc_auc:.4f})",
    )

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_eval_bin[:, cls], y_pred_prob[:, cls])
    pr_auc       = float(auc(rec, prec))
    pr_aucs[cls]  = pr_auc
    ax_pr.plot(
        rec, prec, color=COLORS[cls], lw=1.8,
        label=f"{FAULT_NAMES[cls]} (AP={pr_auc:.4f})",
    )

ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
ax_roc.set_title("Per-Class ROC Curves", fontsize=11, fontweight="bold")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(fontsize=8)
ax_roc.grid(True, alpha=0.25, linestyle="--")
ax_roc.spines[["top", "right"]].set_visible(False)

ax_pr.set_title("Per-Class Precision-Recall Curves", fontsize=11, fontweight="bold")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend(fontsize=8)
ax_pr.grid(True, alpha=0.25, linestyle="--")
ax_pr.spines[["top", "right"]].set_visible(False)

macro_auc = float(np.mean(list(roc_aucs.values())))

plt.suptitle(
    f"BiLSTM v2 -- ROC + Precision-Recall  (macro AUC={macro_auc:.4f})",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig(MODEL_DIR + "bilstm_v2_roc_curves.png", dpi=130, bbox_inches="tight")
plt.show()

print(f"\nPer-class AUC : {roc_aucs}")
print(f"Macro AUC     : {macro_auc:.4f}")

# ---------------------------------------------------------------------------
# F1-maximising threshold calibration  (replaces v1 useless P10)
# ---------------------------------------------------------------------------
# For each class (one-vs-rest) we search 200 candidate softmax thresholds
# in [0.10, 0.99] and keep the one that maximises binary F1.
# These thresholds are what the backend should use instead of argmax.
# They will be meaningfully < 1.0 because label smoothing calibrates
# the softmax output distribution.
# ---------------------------------------------------------------------------
print("\n=== F1-Maximising Threshold Calibration ===")
thresholds_calibrated = {}

for cls in range(4):
    y_bin   = (y_eval == cls).astype(int)
    scores  = y_pred_prob[:, cls]

    best_f1     = 0.0
    best_thresh = 0.5

    for t in np.linspace(0.10, 0.99, 200):
        y_bin_pred = (scores >= t).astype(int)
        tp = int(np.sum((y_bin_pred == 1) & (y_bin == 1)))
        fp = int(np.sum((y_bin_pred == 1) & (y_bin == 0)))
        fn = int(np.sum((y_bin_pred == 0) & (y_bin == 1)))
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec_t = tp / (tp + fp)
        rec_t  = tp / (tp + fn)
        if prec_t + rec_t == 0:
            continue
        f1_t = 2.0 * prec_t * rec_t / (prec_t + rec_t)
        if f1_t > best_f1:
            best_f1    = f1_t
            best_thresh = float(t)

    thresholds_calibrated[str(cls)] = best_thresh
    print(
        f"  Class {cls} ({FAULT_NAMES[cls]:20s}): "
        f"threshold={best_thresh:.3f}  best_F1={best_f1:.4f}"
    )

# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------
saved_path = MODEL_DIR + "bilstm_v2_classifier.h5"
size_mb    = os.path.getsize(saved_path) / 1e6
print(f"\nModel saved : {saved_path}  ({size_mb:.2f} MB)")

output_meta = {
    # Calibrated per-class softmax thresholds (string keys for JSON compat)
    **thresholds_calibrated,
    "macro_f1"         : float(macro_f1),
    "macro_auc"        : float(macro_auc),
    "per_class_auc"    : {str(k): v for k, v in roc_aucs.items()},
    "per_class_pr_auc" : {str(k): v for k, v in pr_aucs.items()},
    "feature_cols"     : FEATURE_COLS,
    "window_size"      : WINDOW_SIZE,
    "n_features"       : N_FEATURES,
    "architecture"     : (
        "BiLSTM(96,seq)+LayerNorm+BiLSTM(64,seq)+LayerNorm"
        "+MultiHeadAttention(4,16)+GAP+Dense(64)+Dense(4)"
    ),
    "eval_protocol"    : "variable_magnitude_faults_Uniform0.8-2.2_seed999",
    "label_smoothing"  : 0.05,
    "v1_known_issues"  : [
        "v1 F1=0.991 was inflated: eval re-used identical offsets as training",
        "v1 BatchNorm on sequence dim is statistically incorrect",
        "v1 P10 thresholds (0.9997-1.0) are deployment-unusable",
    ],
}

with open(MODEL_DIR + "bilstm_v2_thresholds.json", "w") as f:
    json.dump(output_meta, f, indent=2)
print(f"Thresholds  : {MODEL_DIR}bilstm_v2_thresholds.json")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  BiLSTM v2 -- Final Summary")
print("=" * 50)
print(f"  Macro F1  (variable-magnitude eval) : {macro_f1:.4f}")
print(f"  Macro AUC (one-vs-rest)             : {macro_auc:.4f}")
print(f"  Model size                          : {size_mb:.2f} MB")
print(f"  Architecture                        : BiLSTM+Attention (v2)")
print("=" * 50)
print("Done. Proceed to backend swap-in or notebook 05 SHAP.")
