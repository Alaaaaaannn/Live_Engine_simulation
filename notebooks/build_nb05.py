import json

def md(src, uid):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src, 'id': f'md{uid}'}
def code(src, uid):
    return {'cell_type': 'code', 'metadata': {}, 'source': src, 'outputs': [], 'execution_count': None, 'id': f'cd{uid}'}

cells = []

cells.append(md(
    '# Notebook 05: SHAP Explainability\n\n'
    'Computes SHAP feature importance for the BiLSTM fault classifier.\n\n'
    '**Strategy:** Pre-compute SHAP explanations per fault class and cache them as JSON.\n'
    'The backend returns cached values at runtime — no live SHAP computation per API call.',
    '0001'))

cells.append(code(
    '''import os, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

PROC_DIR  = "../data/processed/"
MODEL_DIR = "../models/"

with open(PROC_DIR + "dataset_meta.json") as f:
    META = json.load(f)

FEATURE_COLS = META["gengine1"]["feature_cols"]
N_FEATURES   = META["gengine1"]["n_features"]
WINDOW_SIZE  = META["gengine1"]["window_size"]
FAULT_NAMES  = {int(k): v for k, v in META["fault_names"].items()}

print(f"SHAP version : {shap.__version__}")
print(f"Features     : {FEATURE_COLS}")''',
    '0002'))

cells.append(md('## 5.1 Load BiLSTM Model & Sample Data', '0003'))

cells.append(code(
    '''import tensorflow as tf
tf.random.set_seed(42)

model = tf.keras.models.load_model(MODEL_DIR + "bilstm_classifier.h5")
print("Model loaded:", model.name)

X_train = np.load(PROC_DIR + "g1_X_train.npy")
y_train = np.load(PROC_DIR + "g1_y_train.npy")
X_test  = np.load(PROC_DIR + "g1_X_test.npy")

# Background: 200 random training samples (used by GradientExplainer)
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X_train), size=200, replace=False)
X_background = X_train[bg_idx].astype(np.float32)

# Explanation samples: 50 per fault class (from eval set with injected faults)
LAM_IDX = FEATURE_COLS.index("Lambda")
IGN_IDX = FEATURE_COLS.index("IgnitionAngle")

def make_eval_set(X_base, n_per_class=50, seed=77):
    rng2 = np.random.default_rng(seed)
    sets = {0: [], 1: [], 2: [], 3: []}
    for i in rng2.choice(len(X_base), size=len(X_base), replace=False):
        w = X_base[i].copy()
        r = rng2.random()
        if r < 0.25 and len(sets[1]) < n_per_class:
            w[:, LAM_IDX] += -1.5; sets[1].append(w)
        elif r < 0.50 and len(sets[2]) < n_per_class:
            w[:, LAM_IDX] +=  1.5; sets[2].append(w)
        elif r < 0.75 and len(sets[3]) < n_per_class:
            w[:, IGN_IDX] +=  2.0; sets[3].append(w)
        elif len(sets[0]) < n_per_class:
            sets[0].append(w)
        if all(len(v) >= n_per_class for v in sets.values()):
            break
    return {k: np.array(v, dtype=np.float32) for k, v in sets.items()}

eval_sets = make_eval_set(X_test, n_per_class=50)
print("Eval set sizes:", {FAULT_NAMES[k]: v.shape[0] for k, v in eval_sets.items()})

# Save background for backend
np.save(MODEL_DIR + "shap_background.npy", X_background)
print("Background saved:", X_background.shape)''',
    '0004'))

cells.append(md('## 5.2 Compute SHAP Values (GradientExplainer)\n\nGradientExplainer works on Keras models and is much faster than KernelExplainer.\nWe average SHAP values across the time dimension (30 steps → 1 value per feature).', '0005'))

cells.append(code(
    '''# GradientExplainer on the Keras model
# Input: (batch, 30, 13) → Output: (batch, 4)
explainer = shap.GradientExplainer(model, X_background)
print("Explainer created.")

shap_cache = {}   # fault_class_int -> dict of feature_name -> mean_abs_shap

for cls in range(4):
    X_cls = eval_sets[cls]
    print(f"Computing SHAP for class {cls} ({FAULT_NAMES[cls]})... ", end="", flush=True)

    # shap_vals: list of arrays, one per output class
    # Each array: (n_samples, window_size, n_features)
    shap_vals = explainer.shap_values(X_cls, check_additivity=False)

    # shap_vals[cls]: SHAP values for output neuron `cls`
    sv = np.array(shap_vals[cls])              # (50, 30, 13)
    sv_mean = np.mean(np.abs(sv), axis=(0, 1)) # (13,) — mean |SHAP| per feature

    # Normalise to [0,1] for display
    sv_norm = sv_mean / (sv_mean.sum() + 1e-9)

    shap_cache[str(cls)] = {
        "fault_name"     : FAULT_NAMES[cls],
        "feature_names"  : FEATURE_COLS,
        "mean_abs_shap"  : sv_mean.tolist(),
        "normalized_shap": sv_norm.tolist(),
        "top_features"   : [
            {"feature": FEATURE_COLS[i], "importance": float(sv_norm[i])}
            for i in np.argsort(sv_norm)[::-1]
        ]
    }
    print(f"done. Top feature: {FEATURE_COLS[np.argmax(sv_norm)]}")

with open(MODEL_DIR + "shap_cache.json", "w") as f:
    json.dump(shap_cache, f, indent=2)
print("\\nSHAP cache saved:", MODEL_DIR + "shap_cache.json")''',
    '0006'))

cells.append(md('## 5.3 Visualize SHAP Bar Charts per Fault Class', '0007'))

cells.append(code(
    '''colors_by_class = {0: "#00ff88", 1: "#ff3355", 2: "#ffaa00", 3: "#7b68ee"}
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for cls in range(4):
    ax   = axes[cls // 2, cls % 2]
    data = shap_cache[str(cls)]
    top  = data["top_features"]

    features   = [t["feature"]    for t in top]
    importances = [t["importance"] for t in top]

    bars = ax.barh(range(len(features)), importances,
                   color=colors_by_class[cls], edgecolor="none", alpha=0.85)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Normalized SHAP Importance", fontsize=9)
    ax.set_title(f"Fault {cls}: {FAULT_NAMES[cls]}", fontsize=11, fontweight="bold",
                 color=colors_by_class[cls])
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    for bar, val in zip(bars, importances):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

plt.suptitle("SHAP Feature Importance per Fault Class (BiLSTM Classifier)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR + "shap_importance.png", dpi=130, bbox_inches="tight")
plt.show()
print("SHAP chart saved.")''',
    '0008'))

cells.append(md('## 5.4 Physics Validation\n\nVerify SHAP top features align with known combustion physics.', '0009'))

cells.append(code(
    '''print("=== Physics Validation of SHAP Top Features ===\\n")
expected_top = {
    0: ["Lambda", "CO", "NOx"],           # Normal: all sensors near nominal
    1: ["Lambda", "CO", "FuelCutoff"],    # Rich: Lambda low, CO high
    2: ["Lambda", "NOx", "O2"],           # Lean: Lambda high, NOx high
    3: ["IgnitionAngle", "TempExhaust"],  # Ignition: timing, exhaust temp
}

for cls in range(4):
    top3 = [t["feature"] for t in shap_cache[str(cls)]["top_features"][:3]]
    exp  = expected_top[cls]
    hits = sum(1 for f in exp if f in top3)
    print(f"Class {cls} ({FAULT_NAMES[cls]:20s}): top3={top3}  expected≥1_of={exp}  "
          f"{'✓ PASS' if hits >= 1 else '✗ CHECK'}")''',
    '0010'))

cells.append(code(
    '''print("\\nSHAP notebook complete.")
print("Cached explanations will be served by the FastAPI backend.")
print("Proceed to Notebook 06: Closed-Loop Simulation.")''',
    '0011'))

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': cells
}

out = 'D:/mini project/notebooks/05_shap_explainability.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'Written: {out}')
