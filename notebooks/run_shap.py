# -*- coding: utf-8 -*-
"""
run_shap.py  --  SHAP feature attribution for BiLSTM v2
=========================================================

Paths derived from __file__ so this script runs from any working dir.
Loads bilstm_v2_classifier.h5 (not v1) and resamples its background from
the v2 training data (Next-Steps #5 and #8).
"""
import os
import json
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

np.random.seed(42)

HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(HERE)
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(PROC_DIR, "dataset_meta.json")) as f:
    META = json.load(f)

FEATURE_COLS = META["gengine1"]["feature_cols"]
N_FEATURES   = META["gengine1"]["n_features"]
WINDOW_SIZE  = META["gengine1"]["window_size"]
FAULT_NAMES  = {int(k): v for k, v in META["fault_names"].items()}

print(f"SHAP version : {shap.__version__}")
print(f"Features     : {FEATURE_COLS}")

# ---------------------------------------------------------------------------
# Load v2 classifier
# ---------------------------------------------------------------------------
import tensorflow as tf
tf.random.set_seed(42)

V2_PATH = os.path.join(MODEL_DIR, "bilstm_v2_classifier.h5")
if not os.path.exists(V2_PATH):
    raise FileNotFoundError(
        f"{V2_PATH} not found.  Run train_bilstm_v2.py first.")

model = tf.keras.models.load_model(V2_PATH, compile=False)
print("Model loaded:", model.name)

X_train = np.load(os.path.join(PROC_DIR, "g1_X_train.npy"))
X_test  = np.load(os.path.join(PROC_DIR, "g1_X_test.npy"))

# Background: 200 random v2-training samples (re-sampled vs v1)
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X_train), size=200, replace=False)
X_background = X_train[bg_idx].astype(np.float32)

# Explanation samples: 50 per fault class drawn from the v2-test set with
# fresh injected faults (variable severity matches training distribution)
LAM_IDX = FEATURE_COLS.index("Lambda")
IGN_IDX = FEATURE_COLS.index("IgnitionAngle")

def make_eval_set(X_base, n_per_class=50, seed=77):
    rng2 = np.random.default_rng(seed)
    sets = {0: [], 1: [], 2: [], 3: []}
    for i in rng2.choice(len(X_base), size=len(X_base), replace=False):
        w = X_base[i].copy()
        r = rng2.random()
        scale = float(rng2.uniform(0.8, 1.4))
        if r < 0.25 and len(sets[1]) < n_per_class:
            w[:, LAM_IDX] += -1.5 * scale; sets[1].append(w)
        elif r < 0.50 and len(sets[2]) < n_per_class:
            w[:, LAM_IDX] +=  1.5 * scale; sets[2].append(w)
        elif r < 0.75 and len(sets[3]) < n_per_class:
            w[:, IGN_IDX] +=  2.0 * scale; sets[3].append(w)
        elif len(sets[0]) < n_per_class:
            sets[0].append(w)
        if all(len(v) >= n_per_class for v in sets.values()):
            break
    return {k: np.array(v, dtype=np.float32) for k, v in sets.items()}

eval_sets = make_eval_set(X_test, n_per_class=50)
print("Eval set sizes:",
      {FAULT_NAMES[k]: v.shape[0] for k, v in eval_sets.items()})

np.save(os.path.join(MODEL_DIR, "shap_background.npy"), X_background)
print("Background saved:", X_background.shape)

# ---------------------------------------------------------------------------
# GradientExplainer on v2
# ---------------------------------------------------------------------------
explainer = shap.GradientExplainer(model, X_background)
print("Explainer created.")

shap_cache = {}

for cls in range(4):
    X_cls = eval_sets[cls]
    print(f"Computing SHAP for class {cls} ({FAULT_NAMES[cls]})... ",
          end="", flush=True)
    shap_vals = explainer.shap_values(X_cls)
    sv      = np.array(shap_vals[cls])
    sv_mean = np.mean(np.abs(sv), axis=(0, 1))
    sv_norm = sv_mean / (sv_mean.sum() + 1e-9)

    shap_cache[str(cls)] = {
        "fault_name"      : FAULT_NAMES[cls],
        "feature_names"   : FEATURE_COLS,
        "mean_abs_shap"   : sv_mean.tolist(),
        "normalized_shap": sv_norm.tolist(),
        "top_features"    : [
            {"feature": FEATURE_COLS[i], "importance": float(sv_norm[i])}
            for i in np.argsort(sv_norm)[::-1]
        ],
    }
    print(f"done.  top: {FEATURE_COLS[int(np.argmax(sv_norm))]}")

with open(os.path.join(MODEL_DIR, "shap_cache.json"), "w") as f:
    json.dump(shap_cache, f, indent=2)
print("SHAP cache saved.")

# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
colors_by_class = {0: "#00ff88", 1: "#ff3355", 2: "#ffaa00", 3: "#7b68ee"}
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for cls in range(4):
    ax  = axes[cls // 2, cls % 2]
    top = shap_cache[str(cls)]["top_features"]
    features    = [t["feature"]    for t in top]
    importances = [t["importance"] for t in top]
    ax.barh(range(len(features)), importances,
            color=colors_by_class[cls], edgecolor="none", alpha=0.85)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Normalised SHAP Importance", fontsize=9)
    ax.set_title(f"Fault {cls}: {FAULT_NAMES[cls]}",
                 fontsize=11, fontweight="bold",
                 color=colors_by_class[cls])
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

plt.suptitle("SHAP Feature Importance per Fault Class (BiLSTM v2)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "shap_importance.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)
print("SHAP chart saved.")
print("Done.")
