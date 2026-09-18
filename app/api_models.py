# ============================================================
# api_models.py
# API request/response models and validation
# ============================================================

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SolverSettings:
    """Solver configuration settings."""
    phase_qubits: int = 8
    max_iter: int = 8
    tolerance: float = 1e-7
    phase_target: float = 0.40
    C: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_qubits": self.phase_qubits,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "phase_target": self.phase_target,
            "C": self.C,
        }


@dataclass
class IterationResult:
    """Single FDLS iteration results."""
    iteration: int
    mismatch: float
    mismatch_p: List[float]
    mismatch_q: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "mismatch": self.mismatch,
            "mismatch_p": self.mismatch_p,
            "mismatch_q": self.mismatch_q,
        }


@dataclass
class BusResult:
    """Per-bus power flow results."""
    bus_id: int
    bus_type: int
    vm: float
    va_deg: float
    p_calc: float
    q_calc: float
    p_spec: float
    q_spec: float
    p_mismatch: float
    q_mismatch: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bus_id": self.bus_id,
            "bus_type": self.bus_type,
            "vm": self.vm,
            "va_deg": self.va_deg,
            "p_calc_mw": self.p_calc,
            "q_calc_mvar": self.q_calc,
            "p_spec_mw": self.p_spec,
            "q_spec_mvar": self.q_spec,
            "p_mismatch_pu": self.p_mismatch,
            "q_mismatch_pu": self.q_mismatch,
        }


@dataclass
class SolverResult:
    """Complete solver results for one method (HHL or Classical)."""
    name: str
    status: str  # "converged", "max_iterations", "timeout", "error"
    converged: bool
    iterations: int
    final_mismatch: float
    elapsed_time: float
    loss_history: List[float]
    iteration_results: List[IterationResult]
    bus_results: List[BusResult]
    first_delta_theta: Optional[List[float]]
    first_delta_v: Optional[List[float]]
    settings: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "status": self.status,
            "converged": self.converged,
            "iterations": self.iterations,
            "final_mismatch_pu": self.final_mismatch,
            "elapsed_time_seconds": self.elapsed_time,
            "loss_history": self.loss_history,
            "iteration_results": [ir.to_dict() for ir in self.iteration_results],
            "bus_results": [br.to_dict() for br in self.bus_results],
            "first_delta_theta": self.first_delta_theta,
            "first_delta_v": self.first_delta_v,
            "settings": self.settings,
        }

        if self.diagnostics:
            result["diagnostics"] = self.diagnostics

        if self.error_message:
            result["error_message"] = self.error_message

        return result


@dataclass
class ComparisonResult:
    """Comparison between HHL and Classical results."""
    hhl_result: SolverResult
    classical_result: SolverResult
    voltage_magnitude_errors: List[float]
    voltage_angle_errors: List[float]
    max_vm_error: float
    max_va_error: float
    first_theta_error: Optional[float]
    first_v_error: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hhl": self.hhl_result.to_dict(),
            "classical": self.classical_result.to_dict(),
            "comparison": {
                "voltage_magnitude_errors": self.voltage_magnitude_errors,
                "voltage_angle_errors_deg": self.voltage_angle_errors,
                "max_vm_error": self.max_vm_error,
                "max_va_error_deg": self.max_va_error,
                "first_theta_error": self.first_theta_error,
                "first_v_error": self.first_v_error,
            }
        }


def numpy_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_serializable(item) for item in obj]
    return obj


def build_solver_result(
    name: str,
    fdls_result: Dict[str, Any],
    settings: Dict[str, Any],
    base_mva: float,
    bus_ids: np.ndarray,
    bus_types: np.ndarray,
    diagnostics: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
) -> SolverResult:
    """
    Build SolverResult from FDLS output.
    """
    # Determine status
    if error_message:
        status = "error"
    elif fdls_result.get("timeout", False):
        status = "timeout"
    elif fdls_result["converged"]:
        status = "converged"
    else:
        status = "max_iterations"

    # Build iteration results
    iteration_results = []
    for i, loss in enumerate(fdls_result["loss_history"]):
        mismatch_p = fdls_result["mismatch_p_history"][i] if i < len(fdls_result["mismatch_p_history"]) else []
        mismatch_q = fdls_result["mismatch_q_history"][i] if i < len(fdls_result["mismatch_q_history"]) else []

        iteration_results.append(IterationResult(
            iteration=i,
            mismatch=float(loss),
            mismatch_p=[float(x) for x in mismatch_p],
            mismatch_q=[float(x) for x in mismatch_q],
        ))

    # Build bus results
    bus_results = []
    for i, bus_id in enumerate(bus_ids):
        bus_results.append(BusResult(
            bus_id=int(bus_id),
            bus_type=int(bus_types[i]),
            vm=float(fdls_result["vm"][i]),
            va_deg=float(fdls_result["va_deg"][i]),
            p_calc=float(fdls_result["P_calc"][i] * base_mva),
            q_calc=float(fdls_result["Q_calc"][i] * base_mva),
            p_spec=float(fdls_result["P_spec"][i] * base_mva),
            q_spec=float(fdls_result["Q_spec"][i] * base_mva),
            p_mismatch=float(fdls_result["bus_mismatch_p"][i]),
            q_mismatch=float(fdls_result["bus_mismatch_q"][i]),
        ))

    # Extract first corrections
    first_delta_theta = None
    if fdls_result["first_theta_solution"] is not None:
        first_delta_theta = [float(x) for x in fdls_result["first_theta_solution"]]

    first_delta_v = None
    if fdls_result["first_v_solution"] is not None:
        first_delta_v = [float(x) for x in fdls_result["first_v_solution"]]

    return SolverResult(
        name=name,
        status=status,
        converged=fdls_result["converged"],
        iterations=fdls_result["iterations"],
        final_mismatch=float(fdls_result["final_loss"]),
        elapsed_time=float(fdls_result["elapsed_time"]),
        loss_history=[float(x) for x in fdls_result["loss_history"]],
        iteration_results=iteration_results,
        bus_results=bus_results,
        first_delta_theta=first_delta_theta,
        first_delta_v=first_delta_v,
        settings=settings,
        diagnostics=diagnostics,
        error_message=error_message,
    )
