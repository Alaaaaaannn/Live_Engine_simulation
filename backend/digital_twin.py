"""
digital_twin.py — LSTM digital twin prediction and approval logic.

The twin uses delta prediction: it predicts the CHANGE in sensor values,
not the absolute next state.  At inference:
    next_state = current_last_row + predicted_delta

The twin was trained on gengine1's 13-feature schema.  When called with
a different engine's window we name-align to the model's schema (and
project the output back to the caller's schema) using the same helpers
as the classifier.
"""
import numpy as np
from classifier import align_to_model_schema, project_to_engine_schema
from models_loader import store
from config import WINDOW_SIZE, LAMBDA_TARGET, LAMBDA_BAND


def _twin_input_features() -> int:
    """Number of channels the twin expects (engine features + 2 actions)."""
    return int(store.twin.input_shape[-1])


def _twin_state_features() -> int:
    """Number of feature channels the twin operates on (input - 2 actions)."""
    return _twin_input_features() - 2


def _run_twin(
    window: np.ndarray,
    fuel_trim: float,
    spark_advance: float,
    feature_cols: list[str] | None,
) -> np.ndarray:
    """Project the window into the twin's schema, run inference, return
    the predicted next state mapped back into the caller's schema."""
    n_steps    = window.shape[0]
    model_n    = _twin_state_features()
    have_n     = window.shape[1]

    if feature_cols is not None and have_n != model_n:
        aligned = align_to_model_schema(window, feature_cols)
    else:
        aligned = window.astype(np.float32)

    ctrl   = np.full((n_steps, 2), [fuel_trim, spark_advance], dtype=np.float32)
    x_twin = np.concatenate([aligned, ctrl], axis=1)[np.newaxis, ...]

    meta            = store.twin_meta
    aligned_last    = aligned[-1].copy()                          # (model_n,)
    predicted_raw   = store.twin.predict(x_twin, verbose=0)[0]    # (model_n,)
    aligned_next    = (aligned_last + predicted_raw) if meta.get("prediction_type") == "delta" else predicted_raw

    if feature_cols is not None and have_n != model_n:
        # Map back to engine schema, falling back to the current row for
        # columns the model doesn't represent.
        return project_to_engine_schema(aligned_next, feature_cols, fallback=window[-1])
    return aligned_next


def validate_action(
    window: np.ndarray,
    fuel_trim: float,
    spark_advance: float,
    lambda_current: float,
    lambda_col_idx: int,
    ignition_col_idx: int,
    feature_cols: list[str] | None = None,
) -> tuple[bool, np.ndarray, float]:
    """
    Run the digital twin on the proposed control action.
    Approves the action if twin predicts Lambda convergence toward LAMBDA_TARGET.

    Args:
        window           : (WINDOW_SIZE, n_features) current state window
        fuel_trim        : proposed fuel trim delta
        spark_advance    : proposed spark advance delta
        lambda_current   : current Lambda value (standardized)
        lambda_col_idx   : index of Lambda in feature array
        ignition_col_idx : index of IgnitionAngle in feature array
        feature_cols     : engine feature names (enables name-aligned padding
                           for non-gengine1 engines)

    Returns:
        approved         (bool)
        next_state       (np.ndarray, shape n_features in caller's schema)
        predicted_lambda (float)
    """
    next_state       = _run_twin(window, fuel_trim, spark_advance, feature_cols)
    predicted_lambda = float(next_state[lambda_col_idx])
    current_dist     = abs(lambda_current - LAMBDA_TARGET)
    predicted_dist   = abs(predicted_lambda - LAMBDA_TARGET)

    # Approve if twin predicts Lambda gets meaningfully closer to target
    approved = predicted_dist < current_dist

    return approved, next_state, predicted_lambda


def predict_next_state(
    window: np.ndarray,
    fuel_trim: float = 0.0,
    spark_advance: float = 0.0,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Raw next-state prediction without approval logic.
    Returns predicted sensor array (n_features,) in the caller's schema.
    """
    return _run_twin(window, fuel_trim, spark_advance, feature_cols)
