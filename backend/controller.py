"""
controller.py — Rule-based supervisory controller.
No ML here: pure logic mapping fault class → corrective action.
"""
import config


def compute_control_action(fault_class: int, lambda_current: float) -> tuple[float, float]:
    """
    Map detected fault class to a (fuel_trim_delta, spark_advance_delta) action.

    Fault 1 (Rich, λ too low):  reduce fuel → increase λ toward 1.0
    Fault 2 (Lean, λ too high): increase fuel → decrease λ toward 1.0
    Fault 3 (Ignition fault):   retard spark timing
    Normal:                     no action

    Reads step sizes from `config.RUNTIME` on every call so the UI
    tweakables panel can adjust them live without a restart.
    """
    fuel_step  = float(config.RUNTIME["ctrl_step_fuel"])
    spark_step = float(config.RUNTIME["ctrl_step_spark"])

    if fault_class == 1:          # Rich mixture
        return -fuel_step, 0.0
    elif fault_class == 2:        # Lean mixture
        return +fuel_step, 0.0
    elif fault_class == 3:        # Ignition timing fault
        return 0.0, -spark_step
    else:                         # Normal — no action
        return 0.0, 0.0
