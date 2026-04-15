"""
controller.py — Rule-based supervisory controller.
No ML here: pure logic mapping fault class → corrective action.
"""
from config import CTRL_STEP_FUEL, CTRL_STEP_SPARK


def compute_control_action(fault_class: int, lambda_current: float) -> tuple[float, float]:
    """
    Map detected fault class to a (fuel_trim_delta, spark_advance_delta) action.

    Fault 1 (Rich, λ too low):  reduce fuel → increase λ toward 1.0
    Fault 2 (Lean, λ too high): increase fuel → decrease λ toward 1.0
    Fault 3 (Ignition fault):   retard spark timing
    Normal:                     no action

    Returns: (fuel_trim_delta, spark_advance_delta) in standardized units
    """
    if fault_class == 1:          # Rich mixture
        return -CTRL_STEP_FUEL, 0.0
    elif fault_class == 2:        # Lean mixture
        return +CTRL_STEP_FUEL, 0.0
    elif fault_class == 3:        # Ignition timing fault
        return 0.0, -CTRL_STEP_SPARK
    else:                         # Normal — no action
        return 0.0, 0.0
