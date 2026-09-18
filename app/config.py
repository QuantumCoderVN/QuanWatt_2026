# ============================================================
# config.py
# Application configuration and settings
# ============================================================

from typing import Dict, Any


# Default solver settings
DEFAULT_PHASE_QUBITS = 8
DEFAULT_MAX_ITER = 8
DEFAULT_TOLERANCE = 1e-7
DEFAULT_PHASE_TARGET = 0.40
DEFAULT_C = 1.0

# Network size limits
MIN_BUSES = 2
MAX_BUSES = 14

# Phase qubit range
MIN_PHASE_QUBITS = 4
MAX_PHASE_QUBITS = 8

# Iteration range
MIN_ITERATIONS = 1
MAX_ITERATIONS = 12

# Timeout settings (seconds)
DEFAULT_TIMEOUT = 240  # 4 minutes application budget
FUNCTION_TIMEOUT = 300  # 5 minutes Vercel function limit


def get_default_solver_settings() -> Dict[str, Any]:
    """
    Return default solver configuration.
    """
    return {
        "phase_qubits": DEFAULT_PHASE_QUBITS,
        "max_iter": DEFAULT_MAX_ITER,
        "tolerance": DEFAULT_TOLERANCE,
        "phase_target": DEFAULT_PHASE_TARGET,
        "C": DEFAULT_C,
        "timeout_seconds": DEFAULT_TIMEOUT,
    }


def validate_solver_settings(settings: Dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate solver settings.

    Returns:
        (is_valid, error_message)
    """
    phase_qubits = settings.get("phase_qubits", DEFAULT_PHASE_QUBITS)
    if not (MIN_PHASE_QUBITS <= phase_qubits <= MAX_PHASE_QUBITS):
        return False, f"Phase qubits must be between {MIN_PHASE_QUBITS} and {MAX_PHASE_QUBITS}"

    max_iter = settings.get("max_iter", DEFAULT_MAX_ITER)
    if not (MIN_ITERATIONS <= max_iter <= MAX_ITERATIONS):
        return False, f"Max iterations must be between {MIN_ITERATIONS} and {MAX_ITERATIONS}"

    tolerance = settings.get("tolerance", DEFAULT_TOLERANCE)
    if tolerance <= 0:
        return False, "Tolerance must be positive"

    phase_target = settings.get("phase_target", DEFAULT_PHASE_TARGET)
    if not (0 < phase_target < 1):
        return False, "Phase target must be between 0 and 1"

    C = settings.get("C", DEFAULT_C)
    if C <= 0:
        return False, "C constant must be positive"

    return True, None


def get_case_limits() -> Dict[str, Any]:
    """
    Return network size limits and defaults.
    """
    return {
        "min_buses": MIN_BUSES,
        "max_buses": MAX_BUSES,
        "min_phase_qubits": MIN_PHASE_QUBITS,
        "max_phase_qubits": MAX_PHASE_QUBITS,
        "min_iterations": MIN_ITERATIONS,
        "max_iterations": MAX_ITERATIONS,
    }
