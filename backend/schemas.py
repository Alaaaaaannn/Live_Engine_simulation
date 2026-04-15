"""
schemas.py — Pydantic request/response models for all API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict


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
