# -*- coding: utf-8 -*-
"""
run_preprocess.py  --  Standalone preprocessing script
======================================================

Extracted from build_nb02.py.  Produces the numpy arrays and metadata
consumed by all downstream training scripts and the FastAPI backend.

Key differences vs build_nb02:
  * Adds **variable-magnitude fault augmentation** during training-window
    generation (Next-Step #4).  Each injected fault scales the base
    offset by Uniform(0.7, 1.5) so the model sees a spectrum of fault
    severities rather than a single fixed magnitude.
  * Self-contained: derives paths from __file__ so it can be launched
    from any working directory.
"""
import os
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths (derived from this file's location)
# ---------------------------------------------------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(HERE)              # mini_project/Mini-Project
DATA_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
GENGINE1_INPUTS  = ["Speed", "Load", "Lambda", "IgnitionAngle", "FuelCutoff"]
GENGINE1_OUTPUTS = ["ParticleNumbers", "CO", "CO2", "HC", "NOx",
                    "O2", "TempExhaust", "TempCatalyst"]
GENGINE1_COLS    = GENGINE1_INPUTS + GENGINE1_OUTPUTS

GENGINE2_INPUTS  = ["Speed", "Load", "Lambda", "IgnitionAngle"]
GENGINE2_OUTPUTS = ["ParticleNumbers", "HC", "NoX", "TempExhaust", "TempCatalyst"]
GENGINE2_COLS    = GENGINE2_INPUTS + GENGINE2_OUTPUTS

# ---------------------------------------------------------------------------
# Window / fault config
# ---------------------------------------------------------------------------
WINDOW_SIZE      = 30
STRIDE           = 5
FAULT_FRAC       = 0.25
RICH_OFFSET      = -1.5
LEAN_OFFSET      = +1.5
IGN_OFFSET       = +2.0

# Variable-magnitude augmentation (Next-Step #4).  The classifier sees
# faults scaled by Uniform(SEVERITY_LOW, SEVERITY_HIGH), closing the
# 1.0% miss-rate gap on low-severity ignition faults observed in v1.
SEVERITY_LOW     = 0.7
SEVERITY_HIGH    = 1.5

LABEL_NORMAL = 0
LABEL_RICH   = 1
LABEL_LEAN   = 2
LABEL_IGN    = 3
FAULT_NAMES  = {0: "Normal", 1: "Rich Mixture",
                2: "Lean Mixture", 3: "Ignition Fault"}

GENGINE1_TRAIN_IDS = list(range(10, 20)) + list(range(30, 40))
GENGINE1_TEST_IDS  = list(range(53, 66))
GENGINE2_TRAIN_IDS = list(range(0, 16))
GENGINE2_TEST_IDS  = list(range(16, 22))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_engine_files(directory, col_names, file_ids):
    dfs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        fid = int(fname.split("_")[1])
        if fid not in file_ids:
            continue
        df = pd.read_csv(os.path.join(directory, fname),
                         header=0, names=col_names)
        df["file_id"] = fid
        dfs.append(df)
    return dfs


def inject_fault(window, fault_type, col_names, severity_scale=1.0):
    w = window.copy()
    lam_idx = col_names.index("Lambda")
    ign_idx = col_names.index("IgnitionAngle")
    if fault_type == LABEL_RICH:
        w[:, lam_idx] += RICH_OFFSET * severity_scale
    elif fault_type == LABEL_LEAN:
        w[:, lam_idx] += LEAN_OFFSET * severity_scale
    elif fault_type == LABEL_IGN:
        w[:, ign_idx] += IGN_OFFSET * severity_scale
    return w


def build_windows(df_list, col_names, feature_cols, fault_fraction=0.0,
                  window_size=WINDOW_SIZE, stride=STRIDE, seed=42,
                  variable_severity=True):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for df in df_list:
        arr = df[feature_cols].values.astype(np.float32)
        n_rows = len(arr)
        starts = list(range(0, n_rows - window_size + 1, stride))

        for start in starts:
            window = arr[start:start + window_size]

            if fault_fraction > 0 and rng.random() < fault_fraction:
                fault_type = int(rng.integers(1, 4))
                if variable_severity:
                    scale = float(rng.uniform(SEVERITY_LOW, SEVERITY_HIGH))
                else:
                    scale = 1.0
                window = inject_fault(window, fault_type, feature_cols, scale)
                label  = fault_type
            else:
                label = LABEL_NORMAL

            X_list.append(window)
            y_list.append(label)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading gengine1 files...")
    g1_train_dfs = load_engine_files(os.path.join(DATA_DIR, "gengine1"),
                                      GENGINE1_COLS, GENGINE1_TRAIN_IDS)
    g1_test_dfs  = load_engine_files(os.path.join(DATA_DIR, "gengine1"),
                                      GENGINE1_COLS, GENGINE1_TEST_IDS)
    print(f"  train files: {len(g1_train_dfs)}  test files: {len(g1_test_dfs)}")

    print("Loading gengine2 files...")
    g2_train_dfs = load_engine_files(os.path.join(DATA_DIR, "gengine2"),
                                      GENGINE2_COLS, GENGINE2_TRAIN_IDS)
    g2_test_dfs  = load_engine_files(os.path.join(DATA_DIR, "gengine2"),
                                      GENGINE2_COLS, GENGINE2_TEST_IDS)
    print(f"  train files: {len(g2_train_dfs)}  test files: {len(g2_test_dfs)}")

    # rename NoX -> NOx for gengine2
    for df in g2_train_dfs + g2_test_dfs:
        df.rename(columns={"NoX": "NOx"}, inplace=True)
    GENGINE2_COLS_RENAMED = [c.replace("NoX", "NOx") for c in GENGINE2_COLS]

    # ---------------------------------------------------------------------
    # gengine1 windows (full 13-feature set)
    # ---------------------------------------------------------------------
    G1_FEATURE_COLS = GENGINE1_COLS
    print("\nBuilding gengine1 TRAIN windows (variable-magnitude faults)...")
    X_g1_train, y_g1_train = build_windows(
        g1_train_dfs, GENGINE1_COLS, G1_FEATURE_COLS,
        fault_fraction=FAULT_FRAC, seed=42, variable_severity=True)

    print("Building gengine1 TEST windows (no fault injection)...")
    X_g1_test, y_g1_test = build_windows(
        g1_test_dfs, GENGINE1_COLS, G1_FEATURE_COLS,
        fault_fraction=0.0, seed=0)

    print(f"  TRAIN  X={X_g1_train.shape}  y={y_g1_train.shape}")
    print(f"  TEST   X={X_g1_test.shape}   y={y_g1_test.shape}")
    print("  TRAIN class distribution:")
    for cls, cnt in sorted(Counter(y_g1_train).items()):
        print(f"    {FAULT_NAMES[cls]:20s}: {cnt:7,}  ({cnt/len(y_g1_train)*100:.1f}%)")

    # ---------------------------------------------------------------------
    # gengine2 windows (shared-feature subset)
    # ---------------------------------------------------------------------
    G2_FEATURE_COLS = ["Speed", "Load", "Lambda", "IgnitionAngle",
                       "HC", "NOx", "TempExhaust", "TempCatalyst"]

    print("\nBuilding gengine2 TRAIN windows (variable-magnitude faults)...")
    X_g2_train, y_g2_train = build_windows(
        g2_train_dfs, GENGINE2_COLS_RENAMED, G2_FEATURE_COLS,
        fault_fraction=FAULT_FRAC, seed=43, variable_severity=True)

    print("Building gengine2 TEST windows...")
    X_g2_test, y_g2_test = build_windows(
        g2_test_dfs, GENGINE2_COLS_RENAMED, G2_FEATURE_COLS,
        fault_fraction=0.0, seed=1)

    print(f"  TRAIN  X={X_g2_train.shape}  y={y_g2_train.shape}")
    print(f"  TEST   X={X_g2_test.shape}   y={y_g2_test.shape}")

    # ---------------------------------------------------------------------
    # pengines (static tabular)
    # ---------------------------------------------------------------------
    PENGINES_INPUTS  = ["engine_speed", "engine_load",
                        "intake_valve_opening", "air_fuel_ratio"]
    PENGINES_OUTPUTS = ["specific_fuel_consumption",
                        "temperature_exhaust_manifold",
                        "temperature_in_catalyst",
                        "engine_roughness_v", "engine_roughness_s",
                        "HC", "NOx"]
    PENGINES_COLS = PENGINES_INPUTS + PENGINES_OUTPUTS

    print("\nBuilding pengines arrays...")
    pe1 = pd.read_excel(os.path.join(DATA_DIR, "pengines",
                                      "engine1_normalized.xlsx"),
                         sheet_name="data")
    pe2 = pd.read_excel(os.path.join(DATA_DIR, "pengines",
                                      "engine2_normalized.xlsx"),
                         sheet_name="data")
    pe1 = pe1.drop(columns=["Unnamed: 0"]).dropna().reset_index(drop=True)
    pe2 = pe2.drop(columns=["Unnamed: 0"]).dropna().reset_index(drop=True)

    split_idx = int(len(pe1) * 0.8)
    pe_train  = pe1.iloc[:split_idx].copy()
    pe_test   = pe2.copy()

    def inject_fault_pengines(arr, ftype, col_names, scale=1.0):
        w = arr.copy()
        afr = col_names.index("air_fuel_ratio")
        ivo = col_names.index("intake_valve_opening")
        if ftype == LABEL_RICH:   w[afr] += RICH_OFFSET * scale
        elif ftype == LABEL_LEAN: w[afr] += LEAN_OFFSET * scale
        elif ftype == LABEL_IGN:  w[ivo] += IGN_OFFSET  * scale
        return w

    rng_pe = np.random.default_rng(44)
    pe_X, pe_y = [], []
    for _, row in pe_train[PENGINES_COLS].iterrows():
        arr = row.values.astype(np.float32)
        if rng_pe.random() < FAULT_FRAC:
            ftype = int(rng_pe.integers(1, 4))
            scale = float(rng_pe.uniform(SEVERITY_LOW, SEVERITY_HIGH))
            arr = inject_fault_pengines(arr, ftype, PENGINES_COLS, scale)
            pe_X.append(arr); pe_y.append(ftype)
        else:
            pe_X.append(arr); pe_y.append(LABEL_NORMAL)
    pe_X_train = np.array(pe_X, dtype=np.float32)
    pe_y_train = np.array(pe_y, dtype=np.int32)
    pe_X_test  = pe_test[PENGINES_COLS].values.astype(np.float32)
    pe_y_test  = np.zeros(len(pe_X_test), dtype=np.int32)

    print(f"  pengines TRAIN  X={pe_X_train.shape}  y={pe_y_train.shape}")
    print(f"  pengines TEST   X={pe_X_test.shape}   y={pe_y_test.shape}")

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    print("\nSaving arrays to", PROC_DIR)
    np.save(os.path.join(PROC_DIR, "g1_X_train.npy"), X_g1_train)
    np.save(os.path.join(PROC_DIR, "g1_y_train.npy"), y_g1_train)
    np.save(os.path.join(PROC_DIR, "g1_X_test.npy"),  X_g1_test)
    np.save(os.path.join(PROC_DIR, "g1_y_test.npy"),  y_g1_test)

    np.save(os.path.join(PROC_DIR, "g2_X_train.npy"), X_g2_train)
    np.save(os.path.join(PROC_DIR, "g2_y_train.npy"), y_g2_train)
    np.save(os.path.join(PROC_DIR, "g2_X_test.npy"),  X_g2_test)
    np.save(os.path.join(PROC_DIR, "g2_y_test.npy"),  y_g2_test)

    np.save(os.path.join(PROC_DIR, "pe_X_train.npy"), pe_X_train)
    np.save(os.path.join(PROC_DIR, "pe_y_train.npy"), pe_y_train)
    np.save(os.path.join(PROC_DIR, "pe_X_test.npy"),  pe_X_test)
    np.save(os.path.join(PROC_DIR, "pe_y_test.npy"),  pe_y_test)

    meta = {
        "gengine1": {
            "feature_cols":     G1_FEATURE_COLS,
            "n_features":       len(G1_FEATURE_COLS),
            "window_size":      WINDOW_SIZE,
            "stride":           STRIDE,
            "lambda_col_idx":   G1_FEATURE_COLS.index("Lambda"),
            "ignition_col_idx": G1_FEATURE_COLS.index("IgnitionAngle"),
            "train_shape":      list(X_g1_train.shape),
            "test_shape":       list(X_g1_test.shape),
            "fault_offsets":    {"rich": RICH_OFFSET,
                                 "lean": LEAN_OFFSET,
                                 "ignition": IGN_OFFSET},
            "severity_range":   [SEVERITY_LOW, SEVERITY_HIGH],
        },
        "gengine2": {
            "feature_cols":     G2_FEATURE_COLS,
            "n_features":       len(G2_FEATURE_COLS),
            "window_size":      WINDOW_SIZE,
            "stride":           STRIDE,
            "lambda_col_idx":   G2_FEATURE_COLS.index("Lambda"),
            "ignition_col_idx": G2_FEATURE_COLS.index("IgnitionAngle"),
            "train_shape":      list(X_g2_train.shape),
            "test_shape":       list(X_g2_test.shape),
            "fault_offsets":    {"rich": RICH_OFFSET,
                                 "lean": LEAN_OFFSET,
                                 "ignition": IGN_OFFSET},
            "severity_range":   [SEVERITY_LOW, SEVERITY_HIGH],
        },
        "pengines": {
            "feature_cols":            PENGINES_COLS,
            "n_features":              len(PENGINES_COLS),
            "lambda_equivalent_col":   "air_fuel_ratio",
            "lambda_col_idx":          PENGINES_COLS.index("air_fuel_ratio"),
            "ignition_col_idx":        PENGINES_COLS.index("intake_valve_opening"),
            "train_shape":             list(pe_X_train.shape),
            "test_shape":              list(pe_X_test.shape),
            "fault_offsets":           {"rich": RICH_OFFSET,
                                        "lean": LEAN_OFFSET,
                                        "ignition": IGN_OFFSET},
        },
        "fault_names": FAULT_NAMES,
        "label_map":   {"Normal": 0, "Rich Mixture": 1,
                        "Lean Mixture": 2, "Ignition Fault": 3},
    }

    with open(os.path.join(PROC_DIR, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("dataset_meta.json written.")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
