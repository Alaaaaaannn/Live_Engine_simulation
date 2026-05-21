"""
simulation_engine.py — Core closed-loop simulation logic.

Holds per-session state and runs one cycle per /simulate call.
"""
import time
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Optional

import config
from config import (
    WINDOW_SIZE, LAMBDA_TARGET, LAMBDA_BAND, MAX_CYCLES,
    GENGINE1_COLS, GENGINE2_COLS, DATA_DIR, FAULT_NAMES,
)
from classifier    import (classify_with_gate, record_label,
                            temporal_stability, forget_session)
from controller    import compute_control_action
from digital_twin  import validate_action
from fault_injector import inject_fault
from models_loader  import store
from schemas        import (SimulateRequest, SimulateResponse,
                             ControlAction, TwinPrediction, ShapFeature,
                             ParameterState)


# ── Parameter-state evaluator ─────────────────────────────────────────────────
# The BiLSTM looks at a 30-step temporal window. A slider override only
# touches the last row, so even extreme values get averaged out by the
# remaining 29 trajectory rows and the model reports "Normal".
# This is correct behaviour for the classifier — it's measuring whether
# a real fault PATTERN exists — but the UI also needs to tell the user
# what their slider values actually MEAN in isolation. Hence this
# deterministic readout, derived purely from the current request fields.

_PARAM_LABELS = {
    # (negative-direction label, positive-direction label)
    "lambda":   ("Rich Mixture",   "Lean Mixture"),
    "rpm":      ("Engine Lugging", "Over-Rev"),
    "load":     ("Idle / No Load", "Heavy Load"),
    "ignition": ("Retarded Spark", "Knock Risk"),
}

# Standardised-deviation thresholds.  Below NOMINAL_BAND the parameter
# is considered normal; above SEVERE_LIMIT the severity saturates.
_NOMINAL_BAND = 0.6
_SEVERE_LIMIT = 3.0


def _evaluate_parameter_state(req: SimulateRequest) -> ParameterState:
    """Inspect the slider values themselves and report what they mean.

    Independent of the BiLSTM — answers "what is the user asking the
    engine to do RIGHT NOW", not "what fault pattern has built up over
    the last 30 cycles".
    """
    signed = {
        "lambda":   req.lambda_val,
        "rpm":      req.rpm,
        "load":     req.load,
        "ignition": req.ignition_angle,
    }
    deviations = {k: abs(v) for k, v in signed.items()}
    dominant   = max(deviations, key=deviations.get)
    max_dev    = deviations[dominant]

    if max_dev < _NOMINAL_BAND:
        return ParameterState(label="Nominal", severity=0.0, param=None)

    severity = min(
        1.0,
        (max_dev - _NOMINAL_BAND) / (_SEVERE_LIMIT - _NOMINAL_BAND),
    )
    sign = signed[dominant]
    label = _PARAM_LABELS[dominant][0 if sign < 0 else 1]
    return ParameterState(
        label    = label,
        severity = round(float(severity), 3),
        param    = dominant,
    )


# ── Session state ──────────────────────────────────────────────────────────────

class SimulationSession:
    """Holds the rolling state window and cycle counter for one session."""

    def __init__(self, engine_id: str):
        self.engine_id    = engine_id
        self.cycle        = 0
        self.state_window : Optional[np.ndarray] = None   # (30, n_features)
        self.raw_traj     : Optional[np.ndarray] = None   # full trajectory
        self.traj_pos     = WINDOW_SIZE
        self.created_at   = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        # Persistent fault state — set on injection, cleared when healed
        self.active_fault_type: Optional[str]   = None
        self.fault_lam_offset: float            = 0.0   # residual lambda offset still in state
        self._init_trajectory()

    def _init_trajectory(self):
        """Load the first test trajectory for this engine."""
        meta         = store.dataset_meta[self.engine_id]
        feature_cols = meta["feature_cols"]

        if self.engine_id == "gengine1":
            fpath = os.path.join(DATA_DIR, "gengine1",
                                 "raw_000053_measurement_000000.csv")
            df = pd.read_csv(fpath, header=0, names=GENGINE1_COLS)
        elif self.engine_id == "gengine2":
            fpath = os.path.join(DATA_DIR, "gengine2",
                                 "raw_000016_measurement_000000.csv")
            df = pd.read_csv(fpath, header=0, names=GENGINE2_COLS)
            df.rename(columns={"NoX": "NOx"}, inplace=True)
        else:
            # pengines — use a synthetic trajectory from engine1 xlsx
            from config import ENGINE_DIRS
            import openpyxl
            fpath = os.path.join(ENGINE_DIRS["pengines"], "engine1_normalized.xlsx")
            df = pd.read_excel(fpath, sheet_name="data")
            df = df.drop(columns=["Unnamed: 0"]).dropna().reset_index(drop=True)

        self.raw_traj    = df[feature_cols].values.astype(np.float32)
        self.state_window = self.raw_traj[:WINDOW_SIZE].copy()
        self.traj_pos     = WINDOW_SIZE

    def advance(self, next_row: Optional[np.ndarray] = None):
        """Slide the state window forward by one timestep."""
        if next_row is not None:
            new = next_row.reshape(1, -1)
        elif self.traj_pos < len(self.raw_traj):
            new = self.raw_traj[self.traj_pos:self.traj_pos + 1]
            self.traj_pos += 1
        else:
            new = self.state_window[-1:].copy()   # repeat last row at end of trajectory

        self.state_window = np.concatenate([self.state_window[1:], new], axis=0)
        self.cycle += 1
        self.last_accessed = datetime.utcnow()

    @property
    def is_expired(self) -> bool:
        from config import SESSION_TTL_MINUTES
        return datetime.utcnow() - self.last_accessed > timedelta(minutes=SESSION_TTL_MINUTES)


# ── Session registry ───────────────────────────────────────────────────────────

_sessions: dict[str, SimulationSession] = {}

def get_or_create_session(session_id: str, engine_id: str) -> SimulationSession:
    _prune_expired()
    if session_id not in _sessions:
        _sessions[session_id] = SimulationSession(engine_id)
    sess = _sessions[session_id]
    sess.last_accessed = datetime.utcnow()
    return sess

def reset_session(session_id: str, engine_id: str) -> SimulationSession:
    _sessions[session_id] = SimulationSession(engine_id)
    return _sessions[session_id]

def _prune_expired():
    expired = [sid for sid, s in _sessions.items() if s.is_expired]
    for sid in expired:
        del _sessions[sid]
        forget_session(sid)

def active_session_count() -> int:
    _prune_expired()
    return len(_sessions)


# ── Main simulation step ───────────────────────────────────────────────────────

def run_cycle(req: SimulateRequest) -> SimulateResponse:
    """
    Run one simulation cycle. Called by POST /simulate.
    Thread-safe per session_id (GIL protects dict access for single-worker servers).
    """
    sess         = get_or_create_session(req.session_id, req.engine_id)
    feature_cols = store.dataset_meta[req.engine_id]["feature_cols"]
    meta         = store.dataset_meta[req.engine_id]
    lam_idx      = meta["lambda_col_idx"]
    ign_idx      = meta["ignition_col_idx"]

    # ── 1. Apply user slider overrides to the last row of the window ──────────
    window = sess.state_window.copy()

    # Map slider indices safely (only override columns present in this engine).
    # When auto-correction is active, skip the lambda override so that
    # corrections can actually heal the state window toward stoichiometric.
    lambda_override = req.lambda_val if not req.auto_correction else None

    slider_map = {
        "Lambda":        lambda_override,
        "Speed":         req.rpm,
        "Load":          req.load,
        "IgnitionAngle": req.ignition_angle,
    }
    for col, val in slider_map.items():
        if val is not None and col in feature_cols:
            idx = feature_cols.index(col)
            window[-1, idx] = val   # override only the last timestep

    # ── 2. Fault injection ────────────────────────────────────────────────────
    _FO = config.RUNTIME["fault_offsets"]
    # New fault requested — latch the delta into the session
    if req.fault_inject and req.fault_inject in ("fault1", "fault2", "fault3"):
        sess.active_fault_type = req.fault_inject
        sess.fault_lam_offset  = _FO[req.fault_inject]["delta"]

    # Apply the session-local residual offset to the WORKING COPY of the window
    # every cycle while a fault is active.  The session state_window itself
    # stays on the clean trajectory — only the working copy used for
    # classification and control is shifted.  This guarantees the BiLSTM
    # consistently sees all 30 rows perturbed across the full healing sequence.
    if sess.active_fault_type is not None and abs(sess.fault_lam_offset) > 0.01:
        fault_type  = sess.active_fault_type
        orig_delta  = _FO[fault_type]["delta"]
        col_name    = _FO[fault_type]["col"]
        residual    = sess.fault_lam_offset

        if col_name in feature_cols:
            col_idx = feature_cols.index(col_name)
            window[:, col_idx] += residual

        # Scale emission offsets by heal fraction (1.0 at fault start → 0.0 healed)
        heal_frac = abs(residual / orig_delta) if orig_delta != 0 else 1.0
        for em_col, em_delta in _FO[fault_type].get("emissions", {}).items():
            if em_col in feature_cols:
                em_idx = feature_cols.index(em_col)
                window[:, em_idx] += em_delta * heal_frac

    # ── 3. Classify (with confidence gating + stability tracking) ────────────
    raw_class, confidence, _, fault_class = classify_with_gate(window, feature_cols)
    record_label(req.session_id, fault_class)
    stability_label, stability_agreement = temporal_stability(req.session_id)
    lambda_current = float(window[-1, lam_idx])

    # ── 4. Supervisory control ────────────────────────────────────────────────
    fuel_trim, spark_adv = 0.0, 0.0
    twin_approved        = False
    lambda_predicted     = lambda_current
    next_state           = None

    CTRL_STEP_FUEL  = float(config.RUNTIME["ctrl_step_fuel"])
    CTRL_STEP_SPARK = float(config.RUNTIME["ctrl_step_spark"])

    # Healing is driven by the RESIDUAL OFFSET, not by the classifier output.
    # This ensures the parameter converges fully over ~25 cycles even if the
    # BiLSTM drops out of fault detection before the offset reaches zero.
    if req.auto_correction and sess.active_fault_type in ("fault1", "fault2") \
            and abs(sess.fault_lam_offset) > 0.01:
        # Direction: rich offset is negative (push up), lean is positive (push down)
        if sess.fault_lam_offset < 0:
            fuel_trim = CTRL_STEP_FUEL
            sess.fault_lam_offset = min(0.0, sess.fault_lam_offset + CTRL_STEP_FUEL)
        else:
            fuel_trim = -CTRL_STEP_FUEL
            sess.fault_lam_offset = max(0.0, sess.fault_lam_offset - CTRL_STEP_FUEL)

        traj_lam = float(sess.state_window[-1, lam_idx])
        lambda_predicted = float(np.clip(traj_lam + sess.fault_lam_offset, -4.0, 4.0))
        twin_approved = True

    elif req.auto_correction and sess.active_fault_type == "fault3" \
            and abs(sess.fault_lam_offset) > 0.01:
        # Ignition fault: step the angle offset back toward 0 each cycle.
        # fault_lam_offset holds the ignition angle residual (+2.0 → 0).
        spark_adv = -CTRL_STEP_SPARK
        sess.fault_lam_offset = max(0.0, sess.fault_lam_offset - CTRL_STEP_SPARK)
        lambda_predicted = lambda_current   # lambda unaffected by ignition correction
        twin_approved = True

    elif req.auto_correction and fault_class != 0 and sess.active_fault_type is None:
        # Fallback: classifier detected fault but no persistent session fault.
        fuel_trim, spark_adv = compute_control_action(fault_class, lambda_current)
        approved, next_state, lambda_predicted = validate_action(
            window, fuel_trim, spark_adv, lambda_current, lam_idx, ign_idx,
            feature_cols=feature_cols,
        )
        twin_approved = approved

    # Clear the persistent fault once the residual offset is within convergence band
    if sess.active_fault_type and abs(sess.fault_lam_offset) <= 0.15:
        sess.active_fault_type = None
        sess.fault_lam_offset  = 0.0

    # ── 5. Advance state window ───────────────────────────────────────────────
    # While a fault is active, always advance from raw trajectory so the
    # session window stays clean (fault offset is applied fresh each cycle).
    if sess.active_fault_type is not None:
        sess.advance(None)
    elif twin_approved and next_state is not None:
        sess.advance(next_state)
    else:
        sess.advance(None)

    # ── 6. Read current emissions ─────────────────────────────────────────────
    co_current  = float(window[-1, feature_cols.index("CO")])  if "CO"  in feature_cols else 0.0
    hc_current  = float(window[-1, feature_cols.index("HC")])  if "HC"  in feature_cols else 0.0
    nox_current = float(window[-1, feature_cols.index("NOx")]) if "NOx" in feature_cols else 0.0

    # Predict emissions under the applied action for display
    co_pred  = co_current
    hc_pred  = hc_current
    nox_pred = nox_current
    if next_state is not None:
        co_pred  = float(next_state[feature_cols.index("CO")])  if "CO"  in feature_cols else 0.0
        hc_pred  = float(next_state[feature_cols.index("HC")])  if "HC"  in feature_cols else 0.0
        nox_pred = float(next_state[feature_cols.index("NOx")]) if "NOx" in feature_cols else 0.0

    # ── 7. Convergence check ──────────────────────────────────────────────────
    converged = abs(lambda_current - LAMBDA_TARGET) < LAMBDA_BAND

    # ── 8. SHAP (only on fault events, return cached values) ──────────────────
    shap_features = None
    if fault_class != 0:
        try:
            top = store.get_shap_for_class(fault_class)[:7]
            shap_features = [
                ShapFeature(feature=t["feature"], importance=t["importance"])
                for t in top
            ]
        except KeyError:
            shap_features = None

    return SimulateResponse(
        cycle_number    = sess.cycle,
        fault_class     = fault_class,
        fault_name      = FAULT_NAMES[fault_class],
        fault_confidence = round(confidence, 4),
        parameter_state = _evaluate_parameter_state(req),
        lambda_current  = round(lambda_current, 4),
        lambda_predicted = round(lambda_predicted, 4),
        co_current      = round(co_current, 4),
        hc_current      = round(hc_current, 4),
        nox_current     = round(nox_current, 4),
        control_action  = ControlAction(
            fuel_trim     = round(fuel_trim, 4),
            spark_advance = round(spark_adv, 4)
        ),
        twin = TwinPrediction(
            lambda_predicted = round(lambda_predicted, 4),
            co_predicted     = round(co_pred, 4),
            hc_predicted     = round(hc_pred, 4),
            nox_predicted    = round(nox_pred, 4),
            approved         = twin_approved
        ),
        converged         = converged,
        shap_features     = shap_features,
        stability_label   = int(stability_label),
        stability_agreement = round(float(stability_agreement), 4),
        raw_fault_class   = int(raw_class),
    )
