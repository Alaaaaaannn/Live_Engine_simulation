"""
digital_twin.py — LSTM digital twin prediction and approval logic.

The twin uses delta prediction: it predicts the CHANGE in sensor values,
not the absolute next state.  At inference:
    next_state = current_last_row + predicted_delta
"""
import numpy as np
from models_loader import store
from config import WINDOW_SIZE, LAMBDA_TARGET, LAMBDA_BAND


def validate_action(
    window: np.ndarray,
    fuel_trim: float,
    spark_advance: float,
    lambda_current: float,
    lambda_col_idx: int,
    ignition_col_idx: int,
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

    Returns:
        approved         (bool)
        next_state       (np.ndarray, shape n_features)
        predicted_lambda (float)
    """
    n_steps = window.shape[0]
    ctrl    = np.full((n_steps, 2), [fuel_trim, spark_advance], dtype=np.float32)
    x_twin  = np.concatenate([window, ctrl], axis=1)[np.newaxis, ...]  # (1, 30, n+2)

    # Current last row (the "present" state)
    current_last_row = window[-1].copy()  # (n_features,)

    meta = store.twin_meta
    if meta.get("prediction_type") == "delta":
        # Model predicts the CHANGE; reconstruct absolute next state
        predicted_delta = store.twin.predict(x_twin, verbose=0)[0]  # (n_features,)
        next_state      = current_last_row + predicted_delta
    else:
        # Legacy: model predicts absolute next state directly
        next_state = store.twin.predict(x_twin, verbose=0)[0]

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
) -> np.ndarray:
    """
    Raw next-state prediction without approval logic.
    Returns predicted sensor array (n_features,).
    """
    n_steps = window.shape[0]
    ctrl    = np.full((n_steps, 2), [fuel_trim, spark_advance], dtype=np.float32)
    x_twin  = np.concatenate([window, ctrl], axis=1)[np.newaxis, ...]

    current_last_row = window[-1].copy()

    meta = store.twin_meta
    if meta.get("prediction_type") == "delta":
        predicted_delta = store.twin.predict(x_twin, verbose=0)[0]
        return current_last_row + predicted_delta
    else:
        return store.twin.predict(x_twin, verbose=0)[0]
