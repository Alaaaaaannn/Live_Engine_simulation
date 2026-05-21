"""
classifier.py — BiLSTM fault classification wrapper.

Adds:
  * Name-aligned schema mapping so the v2 13-feature model can score
    gengine2 (8-feature) and pengines (11-feature, renamed) windows
    without misaligning columns.
  * Confidence gating — predictions below the per-class threshold from
    RUNTIME["thresholds"] are overridden to Normal.
  * Temporal-stability tracker — a per-session deque of recent labels;
    exposes the majority label and agreement fraction.
"""
from __future__ import annotations

from collections import deque

import numpy as np

import config
from config import FAULT_NAMES, WINDOW_SIZE, STABILITY_WINDOW
from models_loader import store


# ── Temporal stability ─────────────────────────────────────────────────────────
# Per-session rolling label buffer.  Sessions are referenced by string id;
# entries are pruned when sessions die (simulation_engine handles eviction).
_stability_buffers: dict[str, deque] = {}


# ── Name → name synonyms across engine datasets ───────────────────────────────
# The classifier and twin were trained on gengine1's 13-feature schema.
# Other engines use partially overlapping column names (pengines renames
# everything).  Map each non-gengine1 column name to its gengine1 equivalent
# so name-aligned padding lands real data in the right model slot instead
# of leaving it as a zero-padded "missing" channel.
_COLUMN_SYNONYMS: dict[str, str] = {
    # pengines → gengine1
    "engine_speed":                 "Speed",
    "engine_load":                  "Load",
    "air_fuel_ratio":               "Lambda",
    "intake_valve_opening":         "IgnitionAngle",
    "temperature_exhaust_manifold": "TempExhaust",
    "temperature_in_catalyst":      "TempCatalyst",
}


def _classifier_input_features() -> int:
    """Return the number of features the loaded classifier expects.

    The Keras model's input is (None, WINDOW_SIZE, n_features) — we read
    n_features from the input shape so backend stays correct across
    v1/v2 retrainings and feature-count changes.
    """
    return int(store.classifier.input_shape[-1])


def _model_feature_cols() -> list[str]:
    """The feature schema the trained model expects (gengine1's order)."""
    return store.dataset_meta["gengine1"]["feature_cols"]


def align_to_model_schema(
    window: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    """Re-project a (W, n_engine_features) window into the model's feature
    schema by NAME.  Columns the engine doesn't have are left at 0
    (their standardised mean), which means "no signal here" rather than
    "extreme rich/lean".

    If the engine's feature_cols already match the model's schema, the
    input is returned untouched (modulo dtype).
    """
    expected   = _classifier_input_features()
    model_cols = _model_feature_cols()

    if list(feature_cols) == model_cols[:len(feature_cols)] and len(feature_cols) == expected:
        return window.astype(np.float32)

    W   = window.shape[0]
    out = np.zeros((W, expected), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        target = col if col in model_cols else _COLUMN_SYNONYMS.get(col)
        if target is None or target not in model_cols:
            continue
        j = model_cols.index(target)
        if j < expected:
            out[:, j] = window[:, i]
    return out


def project_to_engine_schema(
    aligned_row: np.ndarray,
    feature_cols: list[str],
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse of `align_to_model_schema` for a single row vector.

    Maps a model-schema row (length `expected`) back to the engine's
    feature_cols order.  Columns not represented in the model output
    fall back to the corresponding entry of `fallback` (or 0 if None).
    """
    model_cols = _model_feature_cols()
    n          = len(feature_cols)
    out        = np.zeros(n, dtype=np.float32) if fallback is None else fallback.astype(np.float32).copy()
    for i, col in enumerate(feature_cols):
        target = col if col in model_cols else _COLUMN_SYNONYMS.get(col)
        if target is None or target not in model_cols:
            continue
        j = model_cols.index(target)
        out[i] = float(aligned_row[j])
    return out


def classify_window(
    window: np.ndarray,
    feature_cols: list[str] | None = None,
) -> tuple[int, float, dict]:
    """Classify a (WINDOW_SIZE, n_features) sensor window.

    If `feature_cols` is provided and differs from the model's training
    schema, the window is name-aligned (columns placed in the position
    the model expects, missing columns zero-filled).  Without
    `feature_cols`, the window is assumed to already be in the model's
    schema.

    Returns (raw_class, confidence, prob_dict).  Confidence gating is
    handled by the higher-level wrapper `classify_with_gate`.
    """
    if window.shape[0] != WINDOW_SIZE:
        raise ValueError(
            f"Expected window of {WINDOW_SIZE} steps, got {window.shape[0]}")

    expected = _classifier_input_features()

    if feature_cols is not None:
        x = align_to_model_schema(window, feature_cols)
    else:
        n_have = window.shape[1]
        if n_have < expected:
            pad = np.zeros((WINDOW_SIZE, expected - n_have), dtype=np.float32)
            x   = np.concatenate([window.astype(np.float32), pad], axis=1)
        elif n_have > expected:
            x = window[:, :expected].astype(np.float32)
        else:
            x = window.astype(np.float32)

    probs = store.classifier.predict(x[np.newaxis, ...], verbose=0)[0]
    cls   = int(np.argmax(probs))
    conf  = float(probs[cls])
    pdict = {FAULT_NAMES[i]: float(probs[i]) for i in range(4)}
    return cls, conf, pdict


def _engine_schema_fully_supported(feature_cols: list[str] | None) -> bool:
    """The classifier was trained on gengine1's full 13-column schema.
    Engines that omit any of those columns (or rename them) suffer from
    a training-distribution mismatch the model never saw: zero-padded
    channels in positions like CO/CO2/O2 look anomalous to the model.

    Returns True only when every column the model was trained on is
    present (by exact name) in the caller's feature_cols.
    """
    if feature_cols is None:
        return True   # legacy path — caller asserts schema match
    model_cols = _model_feature_cols()
    fc_set     = set(feature_cols)
    return all(col in fc_set for col in model_cols)


def classify_with_gate(
    window: np.ndarray,
    feature_cols: list[str] | None = None,
) -> tuple[int, float, dict, int]:
    """Run inference and apply confidence gating.

    Returns:
        raw_class, confidence, prob_dict, gated_class

    `gated_class` equals `raw_class` when confidence ≥ threshold; otherwise
    it is forced to 0 (Normal).  Thresholds come from `config.RUNTIME`.

    For engines whose feature schema does not fully cover the model's
    training schema (e.g. gengine2 lacks CO/CO2/O2/FuelCutoff), the
    gated class is unconditionally forced to Normal — the model's
    predictions on padded zero-channels are not trustworthy.  The raw
    class is still returned for diagnostics, and the simulation
    engine's persistent-fault path still drives controller actions.
    """
    raw_class, confidence, prob_dict = classify_window(window, feature_cols)
    thresholds = config.RUNTIME.get("thresholds") or {}
    t = float(thresholds.get(str(raw_class), 0.0))
    gated_class = raw_class if confidence >= t else 0
    if not _engine_schema_fully_supported(feature_cols):
        gated_class = 0
    return raw_class, confidence, prob_dict, gated_class


def record_label(session_id: str, label: int) -> None:
    """Append a label to the per-session stability buffer."""
    buf = _stability_buffers.get(session_id)
    if buf is None:
        buf = deque(maxlen=STABILITY_WINDOW)
        _stability_buffers[session_id] = buf
    buf.append(int(label))


def temporal_stability(session_id: str) -> tuple[int, float]:
    """Return (majority_label, agreement_fraction) over the session's buffer.

    Agreement is `count_of_majority / buffer_length`.  Returns (0, 1.0)
    for sessions with no recorded labels yet.
    """
    buf = _stability_buffers.get(session_id)
    if not buf:
        return 0, 1.0
    counts = np.bincount(list(buf), minlength=4)
    majority = int(np.argmax(counts))
    agreement = float(counts[majority]) / float(len(buf))
    return majority, agreement


def forget_session(session_id: str) -> None:
    """Drop the stability buffer for a session (called on session expiry)."""
    _stability_buffers.pop(session_id, None)
