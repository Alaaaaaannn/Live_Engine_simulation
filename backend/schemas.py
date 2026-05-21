"""
schemas.py — Pydantic request/response models for all API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ── /simulate ─────────────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    session_id:      str   = Field(...,  description="UUID identifying the simulation session")
    engine_id:       str   = Field("gengine1", description="Active engine dataset")
    lambda_val:      float = Field(0.0,  ge=-4.0, le=4.0,   description="Lambda slider (standardized)")
    rpm:             float = Field(0.0,  ge=-2.0, le=3.0,   description="Speed/RPM (standardized)")
    load:            float = Field(0.0,  ge=-2.0, le=3.0,   description="Engine load (standardized)")
    ignition_angle:  float = Field(0.0,  ge=-4.0, le=4.0,   description="Ignition angle (standardized)")
    co_baseline:     float = Field(0.0,  ge=-2.0, le=3.0,   description="CO baseline (standardized)")
    hc_baseline:     float = Field(0.0,  ge=-2.0, le=3.0,   description="HC baseline (standardized)")
    fault_inject:    Optional[str] = Field(None, description="fault1 | fault2 | fault3 | null")
    auto_correction: bool  = Field(True, description="Enable supervisory controller")
    cycle_number:    int   = Field(0,    ge=0)

class ControlAction(BaseModel):
    fuel_trim:     float
    spark_advance: float

class TwinPrediction(BaseModel):
    lambda_predicted: float
    co_predicted:     float
    hc_predicted:     float
    nox_predicted:    float
    approved:         bool

class ShapFeature(BaseModel):
    feature:    str
    importance: float

class ParameterState(BaseModel):
    """Deterministic readout of what the current slider values represent.

    Independent of the BiLSTM — the classifier looks at a 30-step window
    so a single slider extreme gets averaged out. This object reflects
    the live operator intent directly.
    """
    label:    str
    severity: float = Field(..., ge=0.0, le=1.0)
    param:    Optional[str] = None  # 'lambda' | 'rpm' | 'load' | 'ignition' | None

class SimulateResponse(BaseModel):
    cycle_number:      int
    fault_class:       int
    fault_name:        str
    fault_confidence:  float
    lambda_current:    float
    lambda_predicted:  float
    co_current:        float
    hc_current:        float
    nox_current:       float
    control_action:    ControlAction
    twin:              TwinPrediction
    converged:         bool
    shap_features:     Optional[List[ShapFeature]] = None
    # Temporal stability (Next-Step #7).  majority label over the last
    # STABILITY_WINDOW cycles and the fraction of cycles agreeing with it.
    stability_label:     Optional[int]   = None
    stability_agreement: Optional[float] = None
    # Raw (un-gated) classifier output, useful for diagnostics
    raw_fault_class:     Optional[int]   = None
    # Deterministic interpretation of the live slider values
    parameter_state:     Optional[ParameterState] = None


# ── /classify ─────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    sensor_window: List[List[float]] = Field(
        ..., description="30 x n_features window of standardized sensor values"
    )

class ClassifyResponse(BaseModel):
    fault_class:      int
    fault_name:       str
    confidence:       float
    probabilities:    Dict[str, float]


# ── /engine/select ────────────────────────────────────────────────────────────

class EngineSelectRequest(BaseModel):
    engine_id: str = Field(..., description="gengine1 | gengine2 | pengines")

class EngineSelectResponse(BaseModel):
    engine_id:      str
    n_features:     int
    feature_cols:   List[str]
    sample_count:   int
    message:        str


# ── /fault/inject ─────────────────────────────────────────────────────────────

class FaultInjectRequest(BaseModel):
    session_id:  str
    fault_type:  str = Field(..., description="fault1 | fault2 | fault3")

class FaultInjectResponse(BaseModel):
    session_id:  str
    fault_type:  str
    applied:     bool
    message:     str


# ── /status ───────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    models_loaded:    bool
    current_engine:   str
    bilstm_f1:        float
    twin_rmse:        float
    active_sessions:  int
    message:          str


# ── /config/runtime ───────────────────────────────────────────────────────────

class RuntimeConfigResponse(BaseModel):
    """Mirror of `config.RUNTIME` exposed for the tweakables panel."""
    thresholds:      Dict[str, float]
    fault_offsets:   Dict[str, Any]
    ctrl_step_fuel:  float
    ctrl_step_spark: float


class RuntimeConfigRequest(BaseModel):
    """Partial update — only the keys that should change are sent."""
    thresholds:      Optional[Dict[str, float]] = None
    fault_offsets:   Optional[Dict[str, Any]]   = None
    ctrl_step_fuel:  Optional[float]            = None
    ctrl_step_spark: Optional[float]            = None
    reset:           Optional[bool]             = Field(
        False, description="If true, restore the bundled defaults.")
