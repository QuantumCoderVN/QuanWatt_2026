# ============================================================
# fdls.py
# ============================================================

import numpy as np

from config import FDLS_MAX_ITER, FDLS_TOL, PRINT_EVERY_ITER
from ieee_case import (
    VM,
    VA,
    make_specified_power,
    initial_voltage,
    calculated_power,
)
from quantum_solvers import clean_real_vector


def fdls_power_flow(case, Ybus, Bprime, Bdouble, solver, name):
    """
    FDLS loop.

    Cả hai bước đều dùng cùng solver:
        Classical / HHL / VQLS

    P-theta:
        B' Δθ = ΔP / |V|

    Q-V:
        B'' ΔV = ΔQ / |V|
    """

    non_slack = case["non_slack"]
    pq = case["pq"]
    slack = case["slack"]
    pv = case["pv"]

    P_spec, Q_spec = make_specified_power(case)

    vm, va = initial_voltage(case)

    loss_history = []

    first_theta_solution = None
    first_v_solution = None

    for iteration in range(FDLS_MAX_ITER):
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

        if PRINT_EVERY_ITER:
            print("\n" + "-" * 80)
            print(f"{name} | FDLS iteration {iteration}")
            print("-" * 80)
            print("P_calc:", P_calc)
            print("Q_calc:", Q_calc)
            print("dP:", dP)
            print("dQ:", dQ)
            print("mismatch_p:", mismatch_p)
            print("mismatch_q:", mismatch_q)
            print("loss:", loss)

        if loss < FDLS_TOL:
            break

        # ========================================================
        # P-theta step
        # ========================================================
        rhs_p = mismatch_p / vm[non_slack]

        if PRINT_EVERY_ITER:
            print("\nP-theta system:")
            print("Bprime:")
            print(Bprime)
            print("rhs_p = mismatch_p / Vm:")
            print(rhs_p)

        dtheta = solver(Bprime, rhs_p, label=f"{name} theta iter {iteration}")
        dtheta = clean_real_vector(dtheta)

        if first_theta_solution is None:
            first_theta_solution = dtheta.copy()

        if PRINT_EVERY_ITER:
            print(f"{name} solution dtheta:")
            print(dtheta)

        va[non_slack] += dtheta

        # ========================================================
        # Q-V step
        # ========================================================
        rhs_q = mismatch_q / vm[pq]

        if PRINT_EVERY_ITER:
            print("\nQ-V system:")
            print("Bdouble:")
            print(Bdouble)
            print("rhs_q = mismatch_q / Vm:")
            print(rhs_q)

        dV = solver(Bdouble, rhs_q, label=f"{name} V iter {iteration}")
        dV = clean_real_vector(dV)

        if first_v_solution is None:
            first_v_solution = dV.copy()

        if PRINT_EVERY_ITER:
            print(f"{name} solution dV:")
            print(dV)

        vm[pq] += dV

        # Keep slack fixed
        va[slack] = np.deg2rad(case["bus"][slack, VA])
        vm[slack] = case["bus"][slack, VM]

        # Keep PV voltage magnitude fixed
        for idx in pv:
            vm[idx] = case["bus"][idx, VM]

    P_final, Q_final = calculated_power(Ybus, vm, va)

    final_dP = P_spec - P_final
    final_dQ = Q_spec - Q_final

    final_loss = np.sqrt(
        np.sum(final_dP[non_slack] ** 2)
        +
        np.sum(final_dQ[pq] ** 2)
    )

    return {
        "name": name,
        "vm": vm,
        "va": va,
        "va_deg": np.rad2deg(va),
        "loss_history": loss_history,
        "final_loss": float(final_loss),
        "converged": final_loss < FDLS_TOL,
        "iterations": len(loss_history),
        "P_calc": P_final,
        "Q_calc": Q_final,
        "first_theta_solution": first_theta_solution,
        "first_v_solution": first_v_solution,
    }