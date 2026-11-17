# ============================================================
# solver.py
# FDLS power-flow orchestration with Quapp HHL linear solves
# ============================================================

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, Callable

from .network import (
    validate_case,
    make_ybus,
    make_B_matrices,
    validate_B_matrices,
    prepare_case_indices,
)
from .fdls import fdls_power_flow
from .hhl_solver_quapp import hhl_solve_with_diagnostics
from .config import validate_solver_settings, get_default_solver_settings, DEFAULT_TIMEOUT
from .api_models import build_solver_result, SolverResult


def compute_adaptive_c(matrix: np.ndarray, default_c: float = 1.0) -> float:
    """
    Compute adaptive reciprocal-rotation constant C for custom networks.

    For custom networks: C = min(1, 0.5 × minimum absolute eigenvalue)
    For the default 3-bus case: use the fixed default C
    """
    try:
        if matrix.shape[0] <= 8:
            eigenvalues = np.linalg.eigvals(matrix)
            min_abs_eigenvalue = np.min(np.abs(eigenvalues))
            adaptive_c = min(1.0, 0.5 * min_abs_eigenvalue)
            return float(adaptive_c)
        return default_c
    except Exception:
        return default_c


def solve_power_flow(
    case: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
    use_adaptive_c: bool = False,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Optional[SolverResult], Optional[str]]:
    """
    Solve power flow with FDLS on the proxy; each Ax=b via Quapp HHL.

    Returns:
        (SolverResult, error_message) — error_message is None on success
    """
    def emit(event: Dict[str, Any]) -> None:
        if on_progress is None:
            return
        try:
            on_progress(event)
        except Exception:
            pass

    is_valid, error = validate_case(case)
    if not is_valid:
        return None, error

    prepare_case_indices(case)

    if settings is None:
        settings = get_default_solver_settings()

    is_valid, error = validate_solver_settings(settings)
    if not is_valid:
        return None, error

    try:
        emit({
            "type": "status",
            "phase": "build_network",
            "message": "Building network matrices (Ybus, B′, B″)…",
            "elapsed_s": 0,
        })

        Ybus = make_ybus(case)
        Bprime, Bdouble = make_B_matrices(case, Ybus)

        is_valid, error = validate_B_matrices(Bprime, Bdouble)
        if not is_valid:
            return None, error

        hhl_c = settings.get("C", 1.0)
        applied_scaling = {}

        if use_adaptive_c and len(Bprime) > 0:
            adaptive_c_prime = compute_adaptive_c(Bprime, hhl_c)
            applied_scaling["Bprime"] = {
                "original_c": hhl_c,
                "adaptive_c": adaptive_c_prime,
            }
        else:
            adaptive_c_prime = hhl_c

        if use_adaptive_c and len(Bdouble) > 0:
            adaptive_c_double = compute_adaptive_c(Bdouble, hhl_c)
            applied_scaling["Bdouble"] = {
                "original_c": hhl_c,
                "adaptive_c": adaptive_c_double,
            }
        else:
            adaptive_c_double = hhl_c

        phase_qubits = settings.get("phase_qubits", 8)
        max_iter = settings.get("max_iter", 8)
        tolerance = settings.get("tolerance", 1e-7)
        phase_target = settings.get("phase_target", 0.40)
        timeout_seconds = settings.get("timeout_seconds", DEFAULT_TIMEOUT)

        print("\n🔵 Running HHL Quantum Solver (via Quapp)...")
        print(f"   Timeout budget: {timeout_seconds:.0f}s")
        hhl_diagnostics = {
            "bprime_scaling": {},
            "bdouble_scaling": {},
            "quapp_jobs": [],
            "solve_log": [],
        }

        solve_count = [0]
        hhl_start = time.time()

        def hhl_solver_wrapper(A, b):
            is_bprime = A.shape == Bprime.shape and np.allclose(A, Bprime)
            matrix_name = "B' (P-θ)" if is_bprime else "B'' (Q-V)"
            solve_count[0] += 1
            print(f"   → Solving {matrix_name} system (solve #{solve_count[0]})...")

            emit({
                "type": "status",
                "phase": "quapp_invoke",
                "message": f"Invoking Quapp HHL for {matrix_name} (solve #{solve_count[0]})…",
                "elapsed_s": round(time.time() - hhl_start, 1),
                "max_iter": max_iter,
                "solve_index": solve_count[0],
                "matrix": "Bprime" if is_bprime else "Bdouble",
            })

            c_value = adaptive_c_prime if is_bprime else adaptive_c_double
            result = hhl_solve_with_diagnostics(
                A, b,
                phase_qubits=phase_qubits,
                phase_target=phase_target,
                C=c_value,
            )

            mode = result.get("execution_mode")
            fallback = bool(result.get("fallback_used"))
            print(f"      execution_mode={mode} fallback={fallback} residual={result['residual']:.3e}")

            entry = {
                "solve_index": solve_count[0],
                "matrix": "Bprime" if is_bprime else "Bdouble",
                "execution_mode": mode,
                "fallback_used": fallback,
                "fallback_reason": result.get("fallback_reason"),
                "residual": result["residual"],
                "jobs": result.get("jobs") or [],
            }
            hhl_diagnostics["solve_log"].append(entry)
            if result.get("jobs"):
                hhl_diagnostics["quapp_jobs"].extend(result["jobs"])

            matrix_key = "bprime_scaling" if is_bprime else "bdouble_scaling"
            if not hhl_diagnostics[matrix_key]:
                hhl_diagnostics[matrix_key] = {
                    "lambda_bound": result["lambda_bound"],
                    "t_eff": result["t_eff"],
                    "k_coefficient": result["k_coefficient"],
                    "residual": result["residual"],
                    "raw_amplitude_norm": result["raw_amplitude_norm"],
                    "applied_c": c_value,
                    "execution_mode": mode,
                    "fallback_used": fallback,
                }

            emit({
                "type": "status",
                "phase": "quapp_done",
                "message": (
                    f"Finished {matrix_name} via {mode}"
                    + (" (fallback)" if fallback else "")
                    + f" · residual {result['residual']:.3e}"
                ),
                "elapsed_s": round(time.time() - hhl_start, 1),
                "max_iter": max_iter,
                "solve_index": solve_count[0],
                "matrix": "Bprime" if is_bprime else "Bdouble",
                "execution_mode": mode,
                "fallback_used": fallback,
                "residual": float(result["residual"]),
            })

            return result["solution"]

        hhl_result = fdls_power_flow(
            case=case,
            Ybus=Ybus,
            Bprime=Bprime,
            Bdouble=Bdouble,
            solver=hhl_solver_wrapper,
            max_iter=max_iter,
            tolerance=tolerance,
            name="HHL",
            timeout_seconds=timeout_seconds,
            on_progress=on_progress,
        )

        if applied_scaling:
            hhl_diagnostics["applied_scaling"] = applied_scaling

        n_quapp = sum(
            1 for e in hhl_diagnostics["solve_log"]
            if e.get("execution_mode") == "quapp_statevector"
        )
        n_fallback = sum(1 for e in hhl_diagnostics["solve_log"] if e.get("fallback_used"))
        hhl_diagnostics["quapp_solve_count"] = n_quapp
        hhl_diagnostics["fallback_solve_count"] = n_fallback

        hhl_time = time.time() - hhl_start
        print(f"   ✓ HHL completed in {hhl_time:.2f}s")
        print(f"   ✓ Iterations: {hhl_result['iterations']}")
        print(f"   ✓ Final mismatch: {hhl_result['final_loss']:.2e} p.u.")
        print(f"   ✓ Converged: {hhl_result['converged']}")
        print(f"   ✓ Timeout reached: {hhl_result.get('timeout', False)}")
        print(f"   ✓ Total linear solves: {solve_count[0]} (Quapp={n_quapp}, fallback={n_fallback})")

        bus_ids = case["bus"][:, 0]
        bus_types = case["bus"][:, 1]
        base_mva = case["base_mva"]

        solver_result = build_solver_result(
            name="HHL",
            fdls_result=hhl_result,
            settings={
                "phase_qubits": phase_qubits,
                "phase_target": phase_target,
                "C": hhl_c,
                "adaptive_c_used": use_adaptive_c,
            },
            base_mva=base_mva,
            bus_ids=bus_ids,
            bus_types=bus_types,
            diagnostics=hhl_diagnostics,
        )

        return solver_result, None

    except Exception as e:
        return None, f"Solver error: {str(e)}"
