# ============================================================
# fdls.py
# Fast Decoupled Load Flow solver
# ============================================================

import numpy as np
from typing import Dict, Callable, Optional
import time

from network import (
    make_specified_power,
    initial_voltage,
    calculated_power,
    VM, VA
)


def fdls_power_flow(
    case: Dict,
    Ybus: np.ndarray,
    Bprime: np.ndarray,
    Bdouble: np.ndarray,
    solver: Callable,
    max_iter: int = 8,
    tolerance: float = 1e-7,
    name: str = "FDLS",
    timeout_seconds: Optional[float] = None
) -> Dict:
    """
    Fast Decoupled Load Flow using specified linear solver.

    FDLS equations:
        P-theta step: B' Δθ = ΔP / |V|
        Q-V step:     B'' ΔV = ΔQ / |V|

    Parameters:
        case: Network case dictionary
        Ybus: Bus admittance matrix
        Bprime: Reduced -imag(Ybus) for non-slack buses
        Bdouble: Reduced -imag(Ybus) for PQ buses
        solver: Linear solver function(A, b) -> x
        max_iter: Maximum iterations
        tolerance: Convergence tolerance for mismatch norm
        name: Solver name for labeling
        timeout_seconds: Optional timeout budget

    Returns:
        Dictionary with convergence info, voltages, powers, and first corrections
    """
    start_time = time.time()

    non_slack = case["non_slack"]
    pq = case["pq"]
    slack = case["slack"]
    pv = case["pv"]

    P_spec, Q_spec = make_specified_power(case)

    vm, va = initial_voltage(case)

    loss_history = []
    mismatch_p_history = []
    mismatch_q_history = []

    first_theta_solution = None
    first_v_solution = None

    converged = False
    timeout_reached = False

    for iteration in range(max_iter):
        # Check timeout
        if timeout_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                timeout_reached = True
                break

        P_calc, Q_calc = calculated_power(Ybus, vm, va)

        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc

        mismatch_p = dP[non_slack]
        mismatch_q = dQ[pq]

        loss = np.sqrt(
            np.sum(mismatch_p ** 2)
            +
            np.sum(mismatch_q ** 2)
        )

        loss_history.append(float(loss))
        mismatch_p_history.append(mismatch_p.copy())
        mismatch_q_history.append(mismatch_q.copy())

        if loss < tolerance:
            converged = True
            break

        # P-theta step
        if len(non_slack) > 0:
            rhs_p = mismatch_p / vm[non_slack]

            try:
                dtheta = solver(Bprime, rhs_p)
                dtheta = np.real(dtheta)

                if first_theta_solution is None:
                    first_theta_solution = dtheta.copy()

                va[non_slack] += dtheta
            except Exception as e:
                print(f"[ERROR] {name} P-theta solve failed at iteration {iteration}: {e}")
                break

        # Q-V step
        if len(pq) > 0:
            rhs_q = mismatch_q / vm[pq]

            try:
                dV = solver(Bdouble, rhs_q)
                dV = np.real(dV)

                if first_v_solution is None:
                    first_v_solution = dV.copy()

                vm[pq] += dV
            except Exception as e:
                print(f"[ERROR] {name} Q-V solve failed at iteration {iteration}: {e}")
                break

        # Enforce constraints
        va[slack] = np.deg2rad(case["bus"][slack, VA])
        vm[slack] = case["bus"][slack, VM]

        for idx in pv:
            vm[idx] = case["bus"][idx, VM]

    # Final power calculation
    P_final, Q_final = calculated_power(Ybus, vm, va)

    final_dP = P_spec - P_final
    final_dQ = Q_spec - Q_final

    final_loss = np.sqrt(
        np.sum(final_dP[non_slack] ** 2)
        +
        np.sum(final_dQ[pq] ** 2)
    )

    elapsed_time = time.time() - start_time

    # Compute bus-wise mismatches
    bus_mismatch_p = final_dP
    bus_mismatch_q = final_dQ

    return {
        "name": name,
        "vm": vm,
        "va": va,
        "va_deg": np.rad2deg(va),
        "loss_history": loss_history,
        "mismatch_p_history": mismatch_p_history,
        "mismatch_q_history": mismatch_q_history,
        "final_loss": float(final_loss),
        "converged": converged,
        "timeout": timeout_reached,
        "iterations": len(loss_history),
        "P_calc": P_final,
        "Q_calc": Q_final,
        "P_spec": P_spec,
        "Q_spec": Q_spec,
        "bus_mismatch_p": bus_mismatch_p,
        "bus_mismatch_q": bus_mismatch_q,
        "first_theta_solution": first_theta_solution,
        "first_v_solution": first_v_solution,
        "elapsed_time": elapsed_time,
    }


def classical_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Classical linear solver using NumPy.
    """
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    return np.real(x)
