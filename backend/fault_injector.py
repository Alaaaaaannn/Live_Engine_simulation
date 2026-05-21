"""
fault_injector.py — Synthetic fault injection into sensor windows.
Operates on copies of the window; never mutates in place.
"""
import numpy as np
import config


def inject_fault(window: np.ndarray, fault_type: str, feature_cols: list) -> np.ndarray:
    """
    Apply a synthetic fault perturbation to a (window_size, n_features) array.

    Reads offsets from `config.RUNTIME["fault_offsets"]` on every call so
    the UI tweakables panel can override magnitudes live.

    Args:
        window      : numpy array (WINDOW_SIZE, n_features)
        fault_type  : "fault1" | "fault2" | "fault3"
        feature_cols: list of feature names matching columns in window

    Returns:
        Modified copy of window with fault applied.

    Raises:
        ValueError: if fault_type is unrecognized or column not found
    """
    offsets = config.RUNTIME["fault_offsets"]
    if fault_type not in offsets:
        raise ValueError(
            f"Unknown fault_type '{fault_type}'. Valid: {list(offsets.keys())}")

    fault_cfg = offsets[fault_type]
    col_name  = fault_cfg["col"]
    delta     = float(fault_cfg["delta"])

    if col_name not in feature_cols:
        raise ValueError(f"Column '{col_name}' not in feature_cols for this engine.")

    col_idx = feature_cols.index(col_name)
    w = window.copy()
    w[:, col_idx] += delta

    # Apply correlated emission offsets so the charts show realistic co-movement
    em_offsets = fault_cfg.get("emissions", {})
    for em_col, em_delta in em_offsets.items():
        if em_col in feature_cols:
            em_idx = feature_cols.index(em_col)
            w[:, em_idx] += float(em_delta)

    return w
