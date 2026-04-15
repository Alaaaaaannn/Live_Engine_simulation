import json

def md(src, uid):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src, 'id': f'md{uid}'}
def code(src, uid):
    return {'cell_type': 'code', 'metadata': {}, 'source': src, 'outputs': [], 'execution_count': None, 'id': f'cd{uid}'}

cells = []

cells.append(md(
    '# Notebook 02: Preprocessing\n\n'
    'Produces the numpy arrays and metadata consumed by all subsequent notebooks and the backend.\n\n'
    '**Pipeline:**\n'
    '1. Load gengine1 train/test files with correct column names\n'
    '2. Inject synthetic fault labels (class 0-3) into trajectories\n'
    '3. Build sliding windows of 30 timesteps\n'
    '4. Save: `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, `feature_names.json`\n'
    '5. Repeat for gengine2 (shared features only)\n'
    '6. Prepare pengines (static tabular — separate treatment)',
    '0001'))

cells.append(code(
    '''import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "../data/raw/"
PROC_DIR  = "../data/processed/"
os.makedirs(PROC_DIR, exist_ok=True)

# ── Column names (from README) ─────────────────────────────────────────────────
GENGINE1_INPUTS  = ["Speed", "Load", "Lambda", "IgnitionAngle", "FuelCutoff"]
GENGINE1_OUTPUTS = ["ParticleNumbers", "CO", "CO2", "HC", "NOx", "O2", "TempExhaust", "TempCatalyst"]
GENGINE1_COLS    = GENGINE1_INPUTS + GENGINE1_OUTPUTS

GENGINE2_INPUTS  = ["Speed", "Load", "Lambda", "IgnitionAngle"]
GENGINE2_OUTPUTS = ["ParticleNumbers", "HC", "NoX", "TempExhaust", "TempCatalyst"]
GENGINE2_COLS    = GENGINE2_INPUTS + GENGINE2_OUTPUTS

# ── Shared feature subset (present in BOTH gengine1 and gengine2) ──────────────
# Used for the cross-engine model
SHARED_FEATURES = ["Speed", "Load", "Lambda", "IgnitionAngle", "HC", "NOx", "TempExhaust", "TempCatalyst"]

# ── Window / fault config ──────────────────────────────────────────────────────
WINDOW_SIZE       = 30       # timesteps per sequence
STRIDE            = 5        # step between windows (avoids near-duplicate windows)
FAULT_FRAC        = 0.25     # 25% of windows per training file get a fault injected
RICH_OFFSET       = -1.5     # Fault 1: Lambda shift  (rich = too much fuel)
LEAN_OFFSET       = +1.5     # Fault 2: Lambda shift  (lean = too little fuel)
IGN_OFFSET        = +2.0     # Fault 3: IgnitionAngle shift

# Fault class labels
LABEL_NORMAL  = 0
LABEL_RICH    = 1
LABEL_LEAN    = 2
LABEL_IGN     = 3

FAULT_NAMES = {0: "Normal", 1: "Rich Mixture", 2: "Lean Mixture", 3: "Ignition Fault"}

GENGINE1_TRAIN_IDS = list(range(10, 20)) + list(range(30, 40))
GENGINE1_TEST_IDS  = list(range(53, 66))
GENGINE2_TRAIN_IDS = list(range(0, 16))
GENGINE2_TEST_IDS  = list(range(16, 22))

print("Config loaded.")
print(f"  Window size  : {WINDOW_SIZE} timesteps")
print(f"  Stride       : {STRIDE}")
print(f"  Fault fraction per file: {FAULT_FRAC*100:.0f}%")
print(f"  Offsets      : Rich={RICH_OFFSET:+.1f}σ, Lean={LEAN_OFFSET:+.1f}σ, Ign={IGN_OFFSET:+.1f}σ")''',
    '0002'))

cells.append(md('## 2.1 Data Loading Helper', '0003'))

cells.append(code(
    '''def load_engine_files(directory, col_names, file_ids):
    """Load Bosch CSV files. Returns list of DataFrames."""
    dfs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        fid = int(fname.split("_")[1])
        if fid not in file_ids:
            continue
        df = pd.read_csv(os.path.join(directory, fname), header=0, names=col_names)
        df["file_id"] = fid
        dfs.append(df)
    return dfs

g1_train_dfs = load_engine_files(DATA_DIR + "gengine1/", GENGINE1_COLS, GENGINE1_TRAIN_IDS)
g1_test_dfs  = load_engine_files(DATA_DIR + "gengine1/", GENGINE1_COLS, GENGINE1_TEST_IDS)
g2_train_dfs = load_engine_files(DATA_DIR + "gengine2/", GENGINE2_COLS, GENGINE2_TRAIN_IDS)
g2_test_dfs  = load_engine_files(DATA_DIR + "gengine2/", GENGINE2_COLS, GENGINE2_TEST_IDS)

print(f"gengine1 train: {len(g1_train_dfs)} files | test: {len(g1_test_dfs)} files")
print(f"gengine2 train: {len(g2_train_dfs)} files | test: {len(g2_test_dfs)} files")''',
    '0004'))

cells.append(md(
    '## 2.2 Synthetic Fault Injection\n\n'
    'Since the Bosch dataset has **no fault labels**, we inject faults by perturbing\n'
    'sensor channels on a copy of the window. Each trajectory contributes:\n'
    '- 75% normal windows (stride-sampled)\n'
    '- 25% fault windows (randomly drawn from 3 fault types)\n\n'
    'Fault injection is applied to individual windows **after** striding, not to the raw file,\n'
    'so the original signal is never corrupted.',
    '0005'))

cells.append(code(
    '''def inject_fault(window: np.ndarray, fault_type: int, col_names: list) -> np.ndarray:
    """
    Apply synthetic fault perturbation to a (WINDOW_SIZE, n_features) array.
    Returns a modified copy.
    """
    w = window.copy()
    lam_idx = col_names.index("Lambda")
    ign_idx = col_names.index("IgnitionAngle")

    if fault_type == LABEL_RICH:
        w[:, lam_idx] += RICH_OFFSET
    elif fault_type == LABEL_LEAN:
        w[:, lam_idx] += LEAN_OFFSET
    elif fault_type == LABEL_IGN:
        w[:, ign_idx] += IGN_OFFSET
    return w


def build_windows(df_list, col_names, feature_cols, fault_fraction=0.0,
                  window_size=WINDOW_SIZE, stride=STRIDE, seed=42):
    """
    Build sliding windows from a list of DataFrames.
    fault_fraction: fraction of windows to replace with fault-injected versions (0 = test set).
    Returns X (N, window_size, n_features), y (N,)
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    n_feat = len(feature_cols)

    for df in df_list:
        arr = df[feature_cols].values.astype(np.float32)  # (T, n_feat)
        n_rows = len(arr)
        starts = list(range(0, n_rows - window_size + 1, stride))

        for start in starts:
            window = arr[start : start + window_size]           # (30, n_feat)

            if fault_fraction > 0 and rng.random() < fault_fraction:
                fault_type = rng.integers(1, 4)                 # 1, 2, or 3
                window = inject_fault(window, fault_type, feature_cols)
                label  = fault_type
            else:
                label = LABEL_NORMAL

            X_list.append(window)
            y_list.append(label)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


print("Functions defined.")''',
    '0006'))

cells.append(md('## 2.3 Build gengine1 Windows', '0007'))

cells.append(code(
    '''# gengine1 uses ALL 13 features
G1_FEATURE_COLS = GENGINE1_COLS  # 13 features

print("Building gengine1 TRAIN windows...")
X_g1_train, y_g1_train = build_windows(
    g1_train_dfs, GENGINE1_COLS, G1_FEATURE_COLS,
    fault_fraction=FAULT_FRAC, seed=42)

print("Building gengine1 TEST windows (no fault injection — raw signal)...")
X_g1_test, y_g1_test = build_windows(
    g1_test_dfs, GENGINE1_COLS, G1_FEATURE_COLS,
    fault_fraction=0.0, seed=0)

print(f"\\ngengine1 TRAIN: X={X_g1_train.shape}, y={y_g1_train.shape}")
print(f"gengine1 TEST : X={X_g1_test.shape},  y={y_g1_test.shape}")
print(f"\\nTRAIN class distribution:")
for cls, cnt in sorted(Counter(y_g1_train).items()):
    print(f"  {FAULT_NAMES[cls]:20s}: {cnt:7,}  ({cnt/len(y_g1_train)*100:.1f}%)")
print(f"\\nTEST class distribution (all Normal — raw test signal):")
for cls, cnt in sorted(Counter(y_g1_test).items()):
    print(f"  {FAULT_NAMES[cls]:20s}: {cnt:7,}  ({cnt/len(y_g1_test)*100:.1f}%)")''',
    '0008'))

cells.append(md(
    '## 2.4 Build gengine2 Windows (Shared-Feature Subset)\n\n'
    'gengine2 has only 9 columns (no CO, CO2, O2, FuelCutoff, ParticleNumbers from g1).\n'
    'We train a cross-engine model on the **8 shared features** present in both engines.',
    '0009'))

cells.append(code(
    '''# For gengine2, use the 8 features that overlap with gengine1
# gengine2 has: Speed, Load, Lambda, IgnitionAngle, ParticleNumbers, HC, NoX, TempExhaust, TempCatalyst
# Map gengine2 "NoX" -> equivalent to gengine1 "NOx"
# Rename for consistency
for df in g2_train_dfs + g2_test_dfs:
    df.rename(columns={"NoX": "NOx"}, inplace=True)

GENGINE2_COLS_RENAMED = [c.replace("NoX", "NOx") for c in GENGINE2_COLS]
G2_FEATURE_COLS = ["Speed", "Load", "Lambda", "IgnitionAngle",
                   "HC", "NOx", "TempExhaust", "TempCatalyst"]  # 8 shared features

print("Building gengine2 TRAIN windows...")
X_g2_train, y_g2_train = build_windows(
    g2_train_dfs, GENGINE2_COLS_RENAMED, G2_FEATURE_COLS,
    fault_fraction=FAULT_FRAC, seed=43)

print("Building gengine2 TEST windows...")
X_g2_test, y_g2_test = build_windows(
    g2_test_dfs, GENGINE2_COLS_RENAMED, G2_FEATURE_COLS,
    fault_fraction=0.0, seed=1)

print(f"\\ngengine2 TRAIN: X={X_g2_train.shape}, y={y_g2_train.shape}")
print(f"gengine2 TEST : X={X_g2_test.shape},  y={y_g2_test.shape}")''',
    '0010'))

cells.append(md('## 2.5 Build pengines Dataset (Static Tabular)', '0011'))

cells.append(code(
    '''# pengines is static (not time-series). We treat it as a transfer learning target.
# We use engine1 as train, engine2 as test (per the paper convention)
PENGINES_INPUTS  = ["engine_speed", "engine_load", "intake_valve_opening", "air_fuel_ratio"]
PENGINES_OUTPUTS = ["specific_fuel_consumption", "temperature_exhaust_manifold",
                    "temperature_in_catalyst", "engine_roughness_v",
                    "engine_roughness_s", "HC", "NOx"]
PENGINES_COLS    = PENGINES_INPUTS + PENGINES_OUTPUTS

pe1 = pd.read_excel(DATA_DIR + "pengines/engine1_normalized.xlsx", sheet_name="data")
pe2 = pd.read_excel(DATA_DIR + "pengines/engine2_normalized.xlsx", sheet_name="data")
pe1 = pe1.drop(columns=["Unnamed: 0"]).dropna().reset_index(drop=True)
pe2 = pe2.drop(columns=["Unnamed: 0"]).dropna().reset_index(drop=True)

# pengines has no time-series structure → no sliding windows
# We use 80% of engine1 as train, 20% as val, engine2 as test
split_idx = int(len(pe1) * 0.8)
pe_train  = pe1.iloc[:split_idx].copy()
pe_val    = pe1.iloc[split_idx:].copy()
pe_test   = pe2.copy()

# Inject faults into pengines train (same offsets on air_fuel_ratio ≈ Lambda)
# pengines uses air_fuel_ratio instead of Lambda
def inject_fault_pengines(arr, fault_type, col_names):
    w = arr.copy()
    afr_idx = col_names.index("air_fuel_ratio")
    ivo_idx = col_names.index("intake_valve_opening")
    if fault_type == LABEL_RICH:
        w[afr_idx] += RICH_OFFSET
    elif fault_type == LABEL_LEAN:
        w[afr_idx] += LEAN_OFFSET
    elif fault_type == LABEL_IGN:
        w[ivo_idx] += IGN_OFFSET
    return w

rng = np.random.default_rng(44)
pe_X, pe_y = [], []
for _, row in pe_train[PENGINES_COLS].iterrows():
    arr = row.values.astype(np.float32)
    if rng.random() < FAULT_FRAC:
        fault_type = rng.integers(1, 4)
        arr = inject_fault_pengines(arr, fault_type, PENGINES_COLS)
        pe_X.append(arr); pe_y.append(fault_type)
    else:
        pe_X.append(arr); pe_y.append(LABEL_NORMAL)

pe_X_train = np.array(pe_X, dtype=np.float32)
pe_y_train = np.array(pe_y, dtype=np.int32)
pe_X_test  = pe_test[PENGINES_COLS].values.astype(np.float32)
pe_y_test  = np.zeros(len(pe_X_test), dtype=np.int32)  # all normal (raw signal)

print(f"pengines TRAIN: X={pe_X_train.shape}, y={pe_y_train.shape}")
print(f"pengines TEST : X={pe_X_test.shape},  y={pe_y_test.shape}")''',
    '0012'))

cells.append(md('## 2.6 Class Balance Visualization', '0013'))

cells.append(code(
    '''fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (y, title) in zip(axes, [
        (y_g1_train, "gengine1 TRAIN"),
        (y_g2_train, "gengine2 TRAIN")]):
    counts  = [np.sum(y == c) for c in range(4)]
    colors  = ["#00ff88", "#ff3355", "#ffaa00", "#7b68ee"]
    bars    = ax.bar([FAULT_NAMES[c] for c in range(4)], counts,
                     color=colors, edgecolor="none", width=0.55)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{cnt:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(f"{title} Class Distribution", fontsize=11, fontweight="bold")
    ax.set_ylabel("Window Count", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

plt.tight_layout()
plt.savefig(PROC_DIR + "class_distribution.png", dpi=130, bbox_inches="tight")
plt.show()''',
    '0014'))

cells.append(md('## 2.7 Save All Processed Arrays', '0015'))

cells.append(code(
    '''# ── gengine1 ──────────────────────────────────────────────────────────────────
np.save(PROC_DIR + "g1_X_train.npy", X_g1_train)
np.save(PROC_DIR + "g1_y_train.npy", y_g1_train)
np.save(PROC_DIR + "g1_X_test.npy",  X_g1_test)
np.save(PROC_DIR + "g1_y_test.npy",  y_g1_test)

# ── gengine2 ──────────────────────────────────────────────────────────────────
np.save(PROC_DIR + "g2_X_train.npy", X_g2_train)
np.save(PROC_DIR + "g2_y_train.npy", y_g2_train)
np.save(PROC_DIR + "g2_X_test.npy",  X_g2_test)
np.save(PROC_DIR + "g2_y_test.npy",  y_g2_test)

# ── pengines ──────────────────────────────────────────────────────────────────
np.save(PROC_DIR + "pe_X_train.npy", pe_X_train)
np.save(PROC_DIR + "pe_y_train.npy", pe_y_train)
np.save(PROC_DIR + "pe_X_test.npy",  pe_X_test)
np.save(PROC_DIR + "pe_y_test.npy",  pe_y_test)

# ── Feature metadata ──────────────────────────────────────────────────────────
meta = {
    "gengine1": {
        "feature_cols": G1_FEATURE_COLS,
        "n_features": len(G1_FEATURE_COLS),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "lambda_col_idx": G1_FEATURE_COLS.index("Lambda"),
        "ignition_col_idx": G1_FEATURE_COLS.index("IgnitionAngle"),
        "train_shape": list(X_g1_train.shape),
        "test_shape":  list(X_g1_test.shape),
        "fault_offsets": {"rich": RICH_OFFSET, "lean": LEAN_OFFSET, "ignition": IGN_OFFSET}
    },
    "gengine2": {
        "feature_cols": G2_FEATURE_COLS,
        "n_features": len(G2_FEATURE_COLS),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "lambda_col_idx": G2_FEATURE_COLS.index("Lambda"),
        "ignition_col_idx": G2_FEATURE_COLS.index("IgnitionAngle"),
        "train_shape": list(X_g2_train.shape),
        "test_shape":  list(X_g2_test.shape),
        "fault_offsets": {"rich": RICH_OFFSET, "lean": LEAN_OFFSET, "ignition": IGN_OFFSET}
    },
    "pengines": {
        "feature_cols": PENGINES_COLS,
        "n_features": len(PENGINES_COLS),
        "lambda_equivalent_col": "air_fuel_ratio",
        "lambda_col_idx": PENGINES_COLS.index("air_fuel_ratio"),
        "train_shape": list(pe_X_train.shape),
        "test_shape":  list(pe_X_test.shape),
        "fault_offsets": {"rich": RICH_OFFSET, "lean": LEAN_OFFSET, "ignition": IGN_OFFSET}
    },
    "fault_names": FAULT_NAMES,
    "label_map": {"Normal": 0, "Rich Mixture": 1, "Lean Mixture": 2, "Ignition Fault": 3}
}

with open(PROC_DIR + "dataset_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("All arrays saved to ../data/processed/")
print("\\nSummary:")
for key in ["g1_X_train","g1_y_train","g1_X_test","g1_y_test",
            "g2_X_train","g2_y_train","g2_X_test","g2_y_test",
            "pe_X_train","pe_y_train","pe_X_test","pe_y_test"]:
    arr = np.load(PROC_DIR + key + ".npy")
    print(f"  {key+\'.npy\':<22} shape={str(arr.shape):<25} dtype={arr.dtype}")''',
    '0016'))

cells.append(md('## 2.8 Sanity Check: Visualize Fault Windows vs Normal Windows', '0017'))

cells.append(code(
    '''# Compare a normal window vs a fault-injected window for Lambda
rng_check = np.random.default_rng(99)

# Find one window of each class in g1_train
normal_idxs = np.where(y_g1_train == 0)[0]
rich_idxs   = np.where(y_g1_train == 1)[0]
lean_idxs   = np.where(y_g1_train == 2)[0]
ign_idxs    = np.where(y_g1_train == 3)[0]

lam_idx = G1_FEATURE_COLS.index("Lambda")
ign_col = G1_FEATURE_COLS.index("IgnitionAngle")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
pairs = [
    (normal_idxs[0], "Normal", "#00ff88", lam_idx, "Lambda"),
    (rich_idxs[0],   "Rich Mixture",  "#ff3355", lam_idx, "Lambda"),
    (lean_idxs[0],   "Lean Mixture",  "#ffaa00", lam_idx, "Lambda"),
    (ign_idxs[0],    "Ignition Fault","#7b68ee", ign_col, "IgnitionAngle"),
]

for col, (idx, label, color, feat_idx, feat_name) in enumerate(pairs):
    w = X_g1_train[idx]   # (30, 13)
    t = np.arange(WINDOW_SIZE)

    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    ax_top.plot(t, w[:, lam_idx], color=color, lw=1.8)
    ax_top.axhspan(-0.1, 0.1, alpha=0.15, color="lime")
    ax_top.set_title(f"{label}\\nLambda channel", fontsize=9, fontweight="bold")
    ax_top.set_xlabel("Timestep"); ax_top.set_ylabel("Std. Value")
    ax_top.grid(True, alpha=0.25, linestyle="--")
    ax_top.spines[["top","right"]].set_visible(False)

    ax_bot.plot(t, w[:, feat_idx], color=color, lw=1.8)
    ax_bot.set_title(f"{label}\\n{feat_name} channel", fontsize=9, fontweight="bold")
    ax_bot.set_xlabel("Timestep"); ax_bot.set_ylabel("Std. Value")
    ax_bot.grid(True, alpha=0.25, linestyle="--")
    ax_bot.spines[["top","right"]].set_visible(False)

plt.suptitle("Window Examples per Fault Class (gengine1 training set)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(PROC_DIR + "window_examples.png", dpi=130, bbox_inches="tight")
plt.show()
print("Preprocessing complete. Proceed to Notebook 03: BiLSTM Classifier.")''',
    '0018'))

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': cells
}

out = 'D:/mini project/notebooks/02_preprocessing.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'Written: {out}')
