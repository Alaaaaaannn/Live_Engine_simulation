import json, os

def md(src, uid):
    return {'cell_type':'markdown','metadata':{},'source':src,'id':f'md{uid}'}
def code(src, uid):
    return {'cell_type':'code','metadata':{},'source':src,'outputs':[],'execution_count':None,'id':f'cd{uid}'}

cells = []

cells.append(md(
'# Notebook 01: Data Exploration (EDA)\n\n'
'Explore the raw Bosch Engine Dataset before any modelling.\n'
'All files are **already standardized** (zero-mean, unit-variance on training set).',
'0001'))

cells.append(code(
'''import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

# ── Column Definitions (from README) ──────────────────────────────────────────
GENGINE1_INPUTS  = ["Speed", "Load", "Lambda", "IgnitionAngle", "FuelCutoff"]
GENGINE1_OUTPUTS = ["ParticleNumbers", "CO", "CO2", "HC", "NOx", "O2", "TempExhaust", "TempCatalyst"]
GENGINE1_COLS    = GENGINE1_INPUTS + GENGINE1_OUTPUTS

GENGINE2_INPUTS  = ["Speed", "Load", "Lambda", "IgnitionAngle"]
GENGINE2_OUTPUTS = ["ParticleNumbers", "HC", "NoX", "TempExhaust", "TempCatalyst"]
GENGINE2_COLS    = GENGINE2_INPUTS + GENGINE2_OUTPUTS

PENGINES_INPUTS  = ["engine_speed", "engine_load", "intake_valve_opening", "air_fuel_ratio"]
PENGINES_OUTPUTS = ["specific_fuel_consumption", "temperature_exhaust_manifold",
                    "temperature_in_catalyst", "engine_roughness_v",
                    "engine_roughness_s", "HC", "NOx"]
PENGINES_COLS = PENGINES_INPUTS + PENGINES_OUTPUTS

DATA_DIR = "../data/raw/"
os.makedirs("../data/processed", exist_ok=True)

print("Column setup complete.")
print(f"  gengine1: {len(GENGINE1_COLS)} columns ({len(GENGINE1_INPUTS)} inputs, {len(GENGINE1_OUTPUTS)} outputs)")
print(f"  gengine2: {len(GENGINE2_COLS)} columns ({len(GENGINE2_INPUTS)} inputs, {len(GENGINE2_OUTPUTS)} outputs)")
print(f"  pengines: {len(PENGINES_COLS)} columns ({len(PENGINES_INPUTS)} inputs, {len(PENGINES_OUTPUTS)} outputs)")''',
'0002'))

cells.append(md('## 1.1 Load All gengine1 Files', '0003'))

cells.append(code(
'''GENGINE1_TRAIN_IDS = list(range(10, 20)) + list(range(30, 40))
GENGINE1_TEST_IDS  = list(range(53, 66))
GENGINE2_TRAIN_IDS = list(range(0, 16))
GENGINE2_TEST_IDS  = list(range(16, 22))

def load_engine_files(directory, col_names, file_ids=None):
    """Load Bosch CSV files. Returns list of DataFrames with file_id and timestep columns."""
    files = sorted(os.listdir(directory))
    dfs = []
    for fname in files:
        if not fname.endswith(".csv"):
            continue
        fid = int(fname.split("_")[1])
        if file_ids is not None and fid not in file_ids:
            continue
        df = pd.read_csv(os.path.join(directory, fname), header=0, names=col_names)
        df["file_id"]  = fid
        df["timestep"] = np.arange(len(df))
        dfs.append(df)
    return dfs

g1_train_dfs = load_engine_files(DATA_DIR + "gengine1/", GENGINE1_COLS, GENGINE1_TRAIN_IDS)
g1_test_dfs  = load_engine_files(DATA_DIR + "gengine1/", GENGINE1_COLS, GENGINE1_TEST_IDS)
g1_train = pd.concat(g1_train_dfs, ignore_index=True)
g1_test  = pd.concat(g1_test_dfs,  ignore_index=True)

print(f"gengine1 TRAIN : {len(g1_train_dfs)} files, {len(g1_train):,} rows")
print(f"gengine1 TEST  : {len(g1_test_dfs)} files, {len(g1_test):,} rows")
print(f"Null count train: {g1_train[GENGINE1_COLS].isnull().sum().sum()}")''',
'0004'))

cells.append(md('## 1.2 Descriptive Statistics', '0005'))

cells.append(code(
'''print("=== GENGINE1 TRAINING SET STATISTICS ===")
print(g1_train[GENGINE1_COLS].describe().round(4).to_string())''',
'0006'))

cells.append(md('## 1.3 Distribution Plots for All Features', '0007'))

cells.append(code(
'''fig, axes = plt.subplots(3, 5, figsize=(22, 11))
axes = axes.flatten()

for i, col in enumerate(GENGINE1_COLS):
    ax = axes[i]
    color = "#00d4ff" if i < len(GENGINE1_INPUTS) else "#00ff88"
    ax.hist(g1_train[col], bins=80, color=color, edgecolor="none", alpha=0.82)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("Standardized Value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

for j in range(len(GENGINE1_COLS), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("gengine1 Training Set — Feature Distributions (already standardized)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../data/processed/eda_distributions.png", dpi=130, bbox_inches="tight")
plt.show()''',
'0008'))

cells.append(md('## 1.4 Correlation Heatmap', '0009'))

cells.append(code(
'''plt.figure(figsize=(12, 9))
corr = g1_train[GENGINE1_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8})
plt.title("gengine1 Feature Correlation Matrix", fontsize=12, fontweight="bold", pad=10)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../data/processed/eda_correlation.png", dpi=130, bbox_inches="tight")
plt.show()''',
'0010'))

cells.append(md('## 1.5 Sample Trajectory Plots', '0011'))

cells.append(code(
'''fig, axes = plt.subplots(5, 3, figsize=(20, 14))
plot_features = ["Lambda", "Speed", "Load", "IgnitionAngle", "CO"]
colors = ["#00d4ff", "#00ff88", "#ffaa00"]
files_to_plot = g1_train_dfs[:3]

for col_idx, df_sample in enumerate(files_to_plot):
    window = df_sample.iloc[:500]
    fid = df_sample["file_id"].iloc[0]
    for row_idx, feature in enumerate(plot_features):
        ax = axes[row_idx, col_idx]
        ax.plot(window["timestep"], window[feature],
                color=colors[col_idx], linewidth=0.9, alpha=0.9)
        if feature == "Lambda":
            ax.axhspan(-0.1, 0.1, alpha=0.15, color="green", label="Stoich. band")
            ax.legend(fontsize=7)
        ax.set_title(f"File {fid}: {feature}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Timestep (10 Hz)", fontsize=8)
        ax.set_ylabel("Std. Value", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("gengine1 — Sample Trajectories (first 500 timesteps)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../data/processed/eda_trajectories.png", dpi=130, bbox_inches="tight")
plt.show()''',
'0012'))

cells.append(md(
'## 1.6 Lambda Distribution & Fault Threshold Analysis\n\n'
'The data has NO fault labels. We synthesize faults by perturbing Lambda and IgnitionAngle.\n'
'We choose offsets in standardized units that are meaningful relative to the data range.',
'0013'))

cells.append(code(
'''lambda_train = g1_train["Lambda"]
ign_train    = g1_train["IgnitionAngle"]

print(f"Lambda       — mean: {lambda_train.mean():.4f}, std: {lambda_train.std():.4f}, "
      f"min: {lambda_train.min():.4f}, max: {lambda_train.max():.4f}")
print(f"IgnitionAngle — mean: {ign_train.mean():.4f}, std: {ign_train.std():.4f}, "
      f"min: {ign_train.min():.4f}, max: {ign_train.max():.4f}")

# Fault injection offsets (in standardized units)
RICH_OFFSET  = -1.5   # Fault 1: Lambda -= 1.5σ  (rich mixture, too much fuel)
LEAN_OFFSET  = +1.5   # Fault 2: Lambda += 1.5σ  (lean mixture, too little fuel)
IGN_OFFSET   = +2.0   # Fault 3: IgnitionAngle += 2.0σ (advanced timing fault)

pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("\\nLambda percentiles:")
for p, q in zip(pcts, np.percentile(lambda_train, pcts)):
    print(f"  P{p:2d}: {q:+.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(lambda_train, bins=120, color="#00d4ff", alpha=0.75, edgecolor="none", density=True)
ax1.axvline(RICH_OFFSET,  color="#ff3355", lw=2, ls="--", label=f"Rich fault offset ({RICH_OFFSET:+.1f}σ)")
ax1.axvline(LEAN_OFFSET,  color="#ffaa00", lw=2, ls="--", label=f"Lean fault offset ({LEAN_OFFSET:+.1f}σ)")
ax1.axvspan(-0.1, 0.1, alpha=0.2, color="lime", label="Normal band")
ax1.set_xlabel("Standardized Lambda", fontsize=11)
ax1.set_ylabel("Density", fontsize=11)
ax1.set_title("Lambda Distribution & Fault Thresholds", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25, linestyle="--")
ax1.spines[["top", "right"]].set_visible(False)

ax2.hist(ign_train, bins=120, color="#00ff88", alpha=0.75, edgecolor="none", density=True)
ax2.axvline(IGN_OFFSET, color="#ff3355", lw=2, ls="--", label=f"Ignition fault offset (+{IGN_OFFSET}σ)")
ax2.set_xlabel("Standardized IgnitionAngle", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("IgnitionAngle Distribution & Fault Threshold", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25, linestyle="--")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("../data/processed/eda_fault_thresholds.png", dpi=130, bbox_inches="tight")
plt.show()

print(f"\\nFault injection offsets chosen:")
print(f"  Fault 1 (Rich Mixture)  : Lambda += {RICH_OFFSET:+.1f}σ per timestep")
print(f"  Fault 2 (Lean Mixture)  : Lambda += {LEAN_OFFSET:+.1f}σ per timestep")
print(f"  Fault 3 (Ignition Fault): IgnitionAngle += {IGN_OFFSET:+.1f}σ per timestep")''',
'0014'))

cells.append(md('## 1.7 gengine2 & pengines Quick Checks', '0015'))

cells.append(code(
'''g2_train_dfs = load_engine_files(DATA_DIR + "gengine2/", GENGINE2_COLS, GENGINE2_TRAIN_IDS)
g2_train = pd.concat(g2_train_dfs, ignore_index=True)
print(f"gengine2 TRAIN: {len(g2_train_dfs)} files, {len(g2_train):,} rows")
print(g2_train[GENGINE2_COLS].describe().round(4).to_string())

print("\\n--- pengines ---")
pe1 = pd.read_excel(DATA_DIR + "pengines/engine1_normalized.xlsx", sheet_name="data")
pe2 = pd.read_excel(DATA_DIR + "pengines/engine2_normalized.xlsx", sheet_name="data")
pe1 = pe1.drop(columns=["Unnamed: 0"]).dropna().reset_index(drop=True)
pe2 = pe2.drop(columns=["Unnamed: 0"]).dropna().reset_index(drop=True)
print(f"pengines engine1: {pe1.shape}  engine2: {pe2.shape}")
print(pe1[PENGINES_COLS].describe().round(4).to_string())''',
'0016'))

cells.append(md(
'## 1.8 Summary\n\n'
'| Dataset | Train Rows | Test Rows | Features | Key Inputs |\n'
'|---------|-----------|----------|---------|--------|\n'
'| gengine1 | 247,824 | 185,384 | 13 | Speed, Load, Lambda, IgnitionAngle, FuelCutoff |\n'
'| gengine2 | 403,816 | 165,790 | 9 | Speed, Load, Lambda, IgnitionAngle |\n'
'| pengines | 795 (static) | — | 11 | engine_speed, engine_load, air_fuel_ratio |\n\n'
'**Findings:**\n'
'- All data pre-standardized, zero nulls\n'
'- **No fault labels** — synthesize by perturbing Lambda (±1.5σ) and IgnitionAngle (+2.0σ)\n'
'- gengine1 is the primary training dataset (richest features)\n'
'- Lambda column is the central control-relevant variable\n'
'- Proceed to `02_preprocessing.ipynb`',
'0017'))

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': cells
}

out_path = 'D:/mini project/notebooks/01_data_exploration.ipynb'
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'Written: {out_path}')
