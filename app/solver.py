# ============================================================
# solver.py
# Main solver orchestration for HHL and Classical comparison
# ============================================================

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple

from network import (
    validate_case,
    make_ybus,
    make_B_matrices,
    validate_B_matrices,
    prepare_case_indices,
)
from fdls import fdls_power_flow, classical_solve
from hhl_solver import hhl_solve, hhl_solve_with_diagnostics
from config import validate_solver_settings, get_default_solver_settings
from api_models import (
    build_solver_result,
    ComparisonResult,
    SolverResult,
    numpy_to_serializable,
)


def compute_adaptive_c(matrix: np.ndarray, default_c: float = 1.0) -> float:
    """
    Compute adaptive reciprocal-rotation constant C for custom networks.

    For custom networks: C = min(1, 0.5 × minimum absolute eigenvalue)
    For the default 3-bus case: use the fixed default C

    Parameters:
        matrix: The matrix to analyze
        default_c: Default C value if no adaptation needed

    Returns:
        Adaptive C value
    """
    try:
        # For small matrices, compute eigenvalues
        if matrix.shape[0] <= 8:
            eigenvalues = np.linalg.eigvals(matrix)
            min_abs_eigenvalue = np.min(np.abs(eigenvalues))
            adaptive_c = min(1.0, 0.5 * min_abs_eigenvalue)
            return float(adaptive_c)
        else:
            # For larger matrices, use default
            return default_c
    except:
        return default_c


def solve_power_flow(
    case: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
    use_adaptive_c: bool = False,
) -> Tuple[Optional[ComparisonResult], Optional[str]]:
    """
    Solve power flow problem using both HHL and Classical methods.

    Parameters:
        case: Network case dictionary with bus, gen, branch data
        settings: Solver settings (phase_qubits, max_iter, tolerance, etc.)
        use_adaptive_c: Whether to use adaptive C for custom networks

    Returns:
        (ComparisonResult, error_message) - error_message is None on success
    """
    # Validate case
    is_valid, error = validate_case(case)
    if not is_valid:
        return None, error

    # Prepare case indices
    prepare_case_indices(case)

    # Get default settings if not provided
    if settings is None:
        settings = get_default_solver_settings()

    # Validate settings
    is_valid, error = validate_solver_settings(settings)
    if not is_valid:
        return None, error

    try:
        # Build network matrices
        Ybus = make_ybus(case)
        Bprime, Bdouble = make_B_matrices(case, Ybus)

        # Validate matrices
        is_valid, error = validate_B_matrices(Bprime, Bdouble)
        if not is_valid:
            return None, error

        # Compute adaptive C if requested
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

        # Extract settings
        phase_qubits = settings.get("phase_qubits", 8)
        max_iter = settings.get("max_iter", 8)
        tolerance = settings.get("tolerance", 1e-7)
        phase_target = settings.get("phase_target", 0.40)
        timeout_seconds = settings.get("timeout_seconds", 240)

        # Track overall timing
        start_time = time.time()

        # Classical solver
        print(f"\n🔷 Running Classical Solver...")

        def classical_solver_wrapper(A, b):
            return classical_solve(A, b)

        classical_result = fdls_power_flow(
            case=case,
            Ybus=Ybus,
            Bprime=Bprime,
            Bdouble=Bdouble,
            solver=classical_solver_wrapper,
            max_iter=max_iter,
            tolerance=tolerance,
            name="Classical",
            timeout_seconds=timeout_seconds,
        )

        classical_time = time.time() - start_time
        print(f"   ✓ Classical completed in {classical_time:.4f}s")
        print(f"   ✓ Iterations: {classical_result['iterations']}")
        print(f"   ✓ Final mismatch: {classical_result['final_loss']:.2e} p.u.")
        print(f"   ✓ Converged: {classical_result['converged']}")

        remaining_time = timeout_seconds - classical_time

        if remaining_time <= 0:
            return None, "Timeout reached during classical solve"

        # HHL solver with diagnostics tracking
        print(f"\n🔵 Running HHL Quantum Solver...")
        print(f"   (This will take ~30-40 seconds for 3-bus case)")
        print(f"   Remaining timeout: {remaining_time:.1f}s")
        hhl_diagnostics = {
            "bprime_scaling": {},
            "bdouble_scaling": {},
        }

        # Track first solve diagnostics
        first_bprime_solve = True
        first_bdouble_solve = True

        def hhl_solver_wrapper(A, b):
            nonlocal first_bprime_solve, first_bdouble_solve

            # Determine which matrix we're solving
            is_bprime = (A.shape == Bprime.shape and np.allclose(A, Bprime))
            matrix_name = "B' (P-θ)" if is_bprime else "B'' (Q-V)"

            progress_printer(matrix_name)

            # Use appropriate C value
            c_value = adaptive_c_prime if is_bprime else adaptive_c_double

            # Get diagnostics on first solve of each type
            if (is_bprime and first_bprime_solve) or (not is_bprime and first_bdouble_solve):
                result = hhl_solve_with_diagnostics(
                    A, b,
                    phase_qubits=phase_qubits,
                    phase_target=phase_target,
                    C=c_value,
                )

                matrix_key = "bprime_scaling" if is_bprime else "bdouble_scaling"
                hhl_diagnostics[matrix_key] = {
                    "lambda_bound": result["lambda_bound"],
                    "t_eff": result["t_eff"],
                    "k_coefficient": result["k_coefficient"],
                    "residual": result["residual"],
                    "raw_amplitude_norm": result["raw_amplitude_norm"],
                    "applied_c": c_value,
                }

                if is_bprime:
                    first_bprime_solve = False
                else:
                    first_bdouble_solve = False

                return result["solution"]
            else:
                # Regular solve without diagnostics
                return hhl_solve(
                    A, b,
                    phase_qubits=phase_qubits,
                    phase_target=phase_target,
                    C=c_value,
                )

        hhl_start = time.time()
        solve_count = [0]  # Track number of linear solves

        def progress_printer(matrix_name):
            solve_count[0] += 1
            print(f"   → Solving {matrix_name} system (solve #{solve_count[0]})...")

        hhl_result = fdls_power_flow(
            case=case,
            Ybus=Ybus,
            Bprime=Bprime,
            Bdouble=Bdouble,
            solver=hhl_solver_wrapper,
            max_iter=max_iter,
            tolerance=tolerance,
            name="HHL",
            timeout_seconds=remaining_time,
        )

        # Add scaling info to diagnostics
        if applied_scaling:
            hhl_diagnostics["applied_scaling"] = applied_scaling

        hhl_time = time.time() - hhl_start
        print(f"   ✓ HHL completed in {hhl_time:.2f}s")
        print(f"   ✓ Iterations: {hhl_result['iterations']}")
        print(f"   ✓ Final mismatch: {hhl_result['final_loss']:.2e} p.u.")
        print(f"   ✓ Converged: {hhl_result['converged']}")
        print(f"   ✓ Total linear solves: {solve_count[0]}")

        # Build result objects
        bus_ids = case["bus"][:, 0]
        bus_types = case["bus"][:, 1]
        base_mva = case["base_mva"]

        classical_solver_result = build_solver_result(
            name="Classical",
            fdls_result=classical_result,
            settings={"method": "numpy.linalg.solve"},
            base_mva=base_mva,
            bus_ids=bus_ids,
            bus_types=bus_types,
        )

        hhl_solver_result = build_solver_result(
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

        # Compute comparison metrics
        vm_errors = [
            abs(hhl_result["vm"][i] - classical_result["vm"][i])
            for i in range(len(bus_ids))
        ]

        va_errors = [
            abs(hhl_result["va_deg"][i] - classical_result["va_deg"][i])
            for i in range(len(bus_ids))
        ]

        max_vm_error = max(vm_errors) if vm_errors else 0.0
        max_va_error = max(va_errors) if va_errors else 0.0

        # Compare first corrections
        first_theta_error = None
        if (hhl_result["first_theta_solution"] is not None and
            classical_result["first_theta_solution"] is not None):
            first_theta_error = float(np.linalg.norm(
                hhl_result["first_theta_solution"] - classical_result["first_theta_solution"]
            ))

        first_v_error = None
        if (hhl_result["first_v_solution"] is not None and
            classical_result["first_v_solution"] is not None):
            first_v_error = float(np.linalg.norm(
                hhl_result["first_v_solution"] - classical_result["first_v_solution"]
            ))

        comparison = ComparisonResult(
            hhl_result=hhl_solver_result,
            classical_result=classical_solver_result,
            voltage_magnitude_errors=vm_errors,
            voltage_angle_errors=va_errors,
            max_vm_error=max_vm_error,
            max_va_error=max_va_error,
            first_theta_error=first_theta_error,
            first_v_error=first_v_error,
        )

        return comparison, None

    except Exception as e:
        return None, f"Solver error: {str(e)}"
