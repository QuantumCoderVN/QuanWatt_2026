# ============================================================
# sc_pdipm_solver.py
# SC-PDIPM solver for bounded DC-OPF
#
# Main idea:
#   OPF is solved by SC-PDIPM.
#   At each iteration, SC-PDIPM builds a KKT linear system:
#
#       KKT * direction = rhs
#
#   direction = [dX, dZ, dlambda, dmu]
#
#   The KKT system can be solved by:
#       - Classical linear solver
#       - VQLS linear solver
#
#   Therefore VQLS is NOT an OPF solver directly.
#   VQLS is only the linear solver inside SC-PDIPM.
# ============================================================

import numpy as np

from config_qopf3 import (
    PDIPM_MAX_ITERS,
    PDIPM_TOL_FEAS,
    PDIPM_TOL_GRAD,
    PDIPM_TOL_COMP,
    PDIPM_XI,
    PDIPM_SIGMA,
    PDIPM_GAMMA0,
    PDIPM_MU0,
    PDIPM_USE_STEP_CONTROL,
    PDIPM_STEP_KAPPA,
    PDIPM_MAX_LINESEARCH,
    PRINT_PDIPM_ITER,
    PDIPM_STOP_IF_VQLS_BAD,
    PDIPM_VQLS_MAX_REL_RESIDUAL,
)

from case3_dc_opf import objective_value, parse_solution
from kkt_builder import build_dc_opf_qp
from vqls_opf_solver import solve_linear_classical, solve_linear_vqls


# ============================================================
# Build inequalities
# ============================================================

def build_all_inequalities(qp):
    """
    Convert all inequalities to:

        G(X) = A_G X - b_G <= 0

    Includes:
        1. Branch flow inequalities from Aineq X <= bineq
        2. Upper bounds:  X_i <= ub_i
        3. Lower bounds:  lb_i <= X_i  ->  -X_i <= -lb_i

    Returns:
        A_G, b_G, names
    """

    nvar = qp["nvar"]

    A_rows = []
    b_vals = []
    names = []

    # --------------------------------------------------------
    # Branch inequalities
    # --------------------------------------------------------
    Aineq = qp["Aineq"]
    bineq = qp["bineq"]

    if Aineq.shape[0] > 0:
        for i in range(Aineq.shape[0]):
            A_rows.append(Aineq[i, :].copy())
            b_vals.append(float(bineq[i]))

            if "ineq_names" in qp and i < len(qp["ineq_names"]):
                names.append(qp["ineq_names"][i])
            else:
                names.append(f"branch_ineq_{i}")

    # --------------------------------------------------------
    # Variable bounds
    # --------------------------------------------------------
    lb = qp["lb"]
    ub = qp["ub"]
    var_names = qp["variable_names"]

    for i in range(nvar):
        # Upper bound: X_i <= ub_i
        if np.isfinite(ub[i]):
            row = np.zeros(nvar)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(float(ub[i]))
            names.append(f"{var_names[i]}_upper")

        # Lower bound: lb_i <= X_i  ->  -X_i <= -lb_i
        if np.isfinite(lb[i]):
            row = np.zeros(nvar)
            row[i] = -1.0
            A_rows.append(row)
            b_vals.append(float(-lb[i]))
            names.append(f"{var_names[i]}_lower")

    if len(A_rows) == 0:
        A_G = np.zeros((0, nvar))
        b_G = np.zeros(0)
    else:
        A_G = np.vstack(A_rows)
        b_G = np.asarray(b_vals, dtype=float)

    return A_G, b_G, names


# ============================================================
# Objective and initialization
# ============================================================

def qp_objective_value(X, qp):
    """
    QP objective:

        0.5 X^T H X + f^T X
    """

    H = qp["H"]
    f = qp["f"]

    return float(0.5 * X @ H @ X + f @ X)


def make_initial_X(qp):
    """
    Build an initial X.

    X = [Pg variables, theta variables]

    Steps:
        1. Distribute total load among generators inside bounds.
        2. Solve theta approximately from Aeq X = beq.
        3. Keep variables inside finite bounds.

    This initial point does not need to be perfectly optimal.
    SC-PDIPM will correct it.
    """

    nvar = qp["nvar"]
    ng = qp["ng"]

    Aeq = qp["Aeq"]
    beq = qp["beq"]

    lb = qp["lb"]
    ub = qp["ub"]

    X0 = np.zeros(nvar)

    total_load = float(np.sum(qp["Pd"]))

    pg_lb = lb[:ng].copy()
    pg_ub = ub[:ng].copy()

    pg_lb[~np.isfinite(pg_lb)] = 0.0
    pg_ub[~np.isfinite(pg_ub)] = total_load

    if total_load < np.sum(pg_lb) - 1e-12:
        raise ValueError("Infeasible start: total load < sum(Pmin).")

    if total_load > np.sum(pg_ub) + 1e-12:
        raise ValueError("Infeasible start: total load > sum(Pmax).")

    Pg = pg_lb.copy()
    remaining = total_load - np.sum(Pg)

    capacity = pg_ub - pg_lb
    cap_sum = np.sum(capacity)

    if cap_sum > 1e-14:
        Pg += remaining * capacity / cap_sum

    # Keep generators strictly inside finite bounds if possible.
    eps = 1e-4

    for i in range(ng):
        if np.isfinite(lb[i]) and np.isfinite(ub[i]):
            Pg[i] = min(max(Pg[i], lb[i] + eps), ub[i] - eps)
        elif np.isfinite(lb[i]):
            Pg[i] = max(Pg[i], lb[i] + eps)
        elif np.isfinite(ub[i]):
            Pg[i] = min(Pg[i], ub[i] - eps)

    X0[:ng] = Pg

    # Estimate theta from Aeq X = beq.
    Atheta = Aeq[:, ng:]
    rhs_theta = beq - Aeq[:, :ng] @ Pg

    theta0 = np.linalg.lstsq(Atheta, rhs_theta, rcond=None)[0]
    X0[ng:] = theta0

    # Keep all variables strictly inside finite bounds if possible.
    for i in range(nvar):
        if np.isfinite(lb[i]) and np.isfinite(ub[i]):
            X0[i] = min(max(X0[i], lb[i] + eps), ub[i] - eps)
        elif np.isfinite(lb[i]):
            X0[i] = max(X0[i], lb[i] + eps)
        elif np.isfinite(ub[i]):
            X0[i] = min(X0[i], ub[i] - eps)

    return X0


# ============================================================
# Residuals and conditions
# ============================================================

def compute_residuals(X, Z, lam, mu, gamma, qp, A_G, b_G):
    """
    Residuals of the SC-PDIPM KKT conditions.

    Lagrangian:

        L = 0.5 X^T H X + f^T X
            + lambda^T (Aeq X - beq)
            + mu^T (A_G X - b_G + Z)
            - gamma * sum(log(Z))

    KKT residuals:
        r_dual  = H X + f + Aeq^T lambda + A_G^T mu
        r_eq    = Aeq X - beq
        r_ineq  = A_G X - b_G + Z
        r_cent  = mu - gamma / Z
    """

    H = qp["H"]
    f = qp["f"]
    Aeq = qp["Aeq"]

    r_dual = H @ X + f + Aeq.T @ lam + A_G.T @ mu
    r_eq = Aeq @ X - qp["beq"]
    r_ineq = A_G @ X - b_G + Z

    # Paper-style centrality residual:
    r_cent_paper = mu - gamma / Z

    # Complementarity:
    complementarity = mu * Z

    return {
        "r_dual": r_dual,
        "r_eq": r_eq,
        "r_ineq": r_ineq,
        "r_cent_paper": r_cent_paper,
        "complementarity": complementarity,
    }


def conditions(X, Z, lam, mu, gamma, qp, A_G, b_G):
    """
    Compute stopping conditions:
        feascond: equality and inequality feasibility
        gradcond: stationarity
        compcond: average complementarity
    """

    r = compute_residuals(X, Z, lam, mu, gamma, qp, A_G, b_G)

    feascond = max(
        np.linalg.norm(r["r_eq"], ord=np.inf),
        np.linalg.norm(r["r_ineq"], ord=np.inf),
    )

    gradcond = np.linalg.norm(r["r_dual"], ord=np.inf) / (
        1.0 + max(np.linalg.norm(lam, ord=np.inf), np.linalg.norm(mu, ord=np.inf))
    )

    compcond = float(np.dot(mu, Z) / len(Z))

    return feascond, gradcond, compcond, r


# ============================================================
# KKT system
# ============================================================

def build_pdipm_kkt(X, Z, lam, mu, gamma, qp, A_G, b_G):
    """
    Build the SC-PDIPM KKT linear system.

    Unknown direction:

        direction = [dX, dZ, dlambda, dmu]

    System:

        [ H      0          Aeq.T     A_G.T ] [dX ]   -[r_dual]
        [ 0   diag(mu/Z)     0         I   ] [dZ ] = -[r_cent]
        [ Aeq    0           0         0   ] [dl ]   -[r_eq]
        [ A_G    I           0         0   ] [dmu]   -[r_ineq]

    This is the matrix that later goes into VQLS.
    """

    H = qp["H"]
    Aeq = qp["Aeq"]

    n = qp["nvar"]
    ni = A_G.shape[0]
    meq = Aeq.shape[0]

    r = compute_residuals(X, Z, lam, mu, gamma, qp, A_G, b_G)

    KKT = np.block([
        [
            H,
            np.zeros((n, ni)),
            Aeq.T,
            A_G.T,
        ],
        [
            np.zeros((ni, n)),
            np.diag(mu / Z),
            np.zeros((ni, meq)),
            np.eye(ni),
        ],
        [
            Aeq,
            np.zeros((meq, ni)),
            np.zeros((meq, meq)),
            np.zeros((meq, ni)),
        ],
        [
            A_G,
            np.eye(ni),
            np.zeros((ni, meq)),
            np.zeros((ni, ni)),
        ],
    ])

    rhs = -np.concatenate([
        r["r_dual"],
        r["r_cent_paper"],
        r["r_eq"],
        r["r_ineq"],
    ])

    return KKT, rhs, r


def split_direction(direction, qp, ni):
    """
    Split direction vector:

        direction = [dX, dZ, dlambda, dmu]
    """

    n = qp["nvar"]
    meq = qp["Aeq"].shape[0]

    dX = direction[:n]
    dZ = direction[n:n + ni]
    dlam = direction[n + ni:n + ni + meq]
    dmu = direction[n + ni + meq:]

    return dX, dZ, dlam, dmu


# ============================================================
# Step size
# ============================================================

def fraction_to_boundary(v, dv, xi=0.995):
    """
    Choose maximum step alpha such that:

        v + alpha * dv > 0

    for variables that must remain positive, such as Z and mu.
    """

    idx = dv < 0.0

    if not np.any(idx):
        return 1.0

    alpha = np.min(-v[idx] / dv[idx])

    return min(1.0, xi * alpha)


def merit_value(X, Z, lam, mu, gamma, qp, A_G, b_G):
    """
    Simple merit value for step control.
    """

    feascond, gradcond, compcond, _ = conditions(
        X, Z, lam, mu, gamma, qp, A_G, b_G
    )

    return max(feascond, gradcond, compcond)

class VQLSResidualError(RuntimeError):
    def __init__(self, message, info):
        super().__init__(message)
        self.info = info

# ============================================================
# Linear solver selection
# ============================================================

def solve_kkt_direction(KKT, rhs, it, linear_solver="classical"):
    """
    Solve one SC-PDIPM KKT system.

    linear_solver:
        - "classical"
        - "vqls"

    If linear_solver="vqls", the returned direction is really from VQLS.
    This means the SC-PDIPM update is controlled by VQLS.
    """

    linear_solver = linear_solver.lower()

    if linear_solver == "classical":
        direction, info = solve_linear_classical(KKT, rhs)
        return direction, info

    if linear_solver == "vqls":
        direction, info = solve_linear_vqls(
            KKT,
            rhs,
            label=f"VQLS KKT solve at SC-PDIPM iteration {it}",
        )

        if (
            PDIPM_STOP_IF_VQLS_BAD
            and info["relative_residual"] > PDIPM_VQLS_MAX_REL_RESIDUAL
        ):
            raise VQLSResidualError(
                "VQLS residual too large for safe SC-PDIPM update. "
                f"relative_residual={info['relative_residual']:.6e}, "
                f"threshold={PDIPM_VQLS_MAX_REL_RESIDUAL:.6e}",
                info,
            )

        return direction, info

    raise ValueError(
        f"Unknown linear_solver={linear_solver}. "
        "Use 'classical' or 'vqls'."
    )


# ============================================================
# Main SC-PDIPM solver
# ============================================================

def solve_opf_scpdipm(case, linear_solver="classical"):
    """
    Solve bounded DC-OPF using SC-PDIPM.

    Parameters
    ----------
    case:
        IEEE14-subcase-3bus case data.

    linear_solver:
        "classical" or "vqls".

    Returns
    -------
    result dict containing:
        X, Z, lambda, mu, cost, residuals, history, vqls_loss_history.
    """

    linear_solver = linear_solver.lower()

    if linear_solver not in ("classical", "vqls"):
        raise ValueError("linear_solver must be 'classical' or 'vqls'.")

    qp = build_dc_opf_qp(case)

    A_G, b_G, ineq_names = build_all_inequalities(qp)

    n = qp["nvar"]
    ni = A_G.shape[0]
    meq = qp["Aeq"].shape[0]

    if ni == 0:
        raise ValueError("SC-PDIPM needs inequalities. No inequality found.")

    X = make_initial_X(qp)

    # Start with positive slack:
    # G(X) = A_G X - b_G
    # Need G(X) + Z = 0 at optimum.
    slack_from_X = b_G - A_G @ X
    Z = np.maximum(slack_from_X, 1.0)

    lam = np.zeros(meq)
    mu = PDIPM_MU0 * np.ones(ni)

    gamma = PDIPM_GAMMA0

    history = []
    total_vqls_loss_history = []

    if PRINT_PDIPM_ITER:
        print("=" * 120)
        print(f"SC-PDIPM iterations using linear_solver = {linear_solver}")
        print("=" * 120)
        print(
            f"{'it':>3s} "
            f"{'obj_qp':>14s} "
            f"{'feas':>12s} "
            f"{'grad':>12s} "
            f"{'gap':>12s} "
            f"{'gamma':>12s} "
            f"{'lin_res':>12s} "
            f"{'lin_rel':>12s} "
            f"{'cond(KKT)':>12s}"
        )

    converged = False

    for it in range(PDIPM_MAX_ITERS + 1):
        feascond, gradcond, compcond, residuals = conditions(
            X, Z, lam, mu, gamma, qp, A_G, b_G
        )

        obj_qp = qp_objective_value(X, qp)

        KKT, rhs, _ = build_pdipm_kkt(X, Z, lam, mu, gamma, qp, A_G, b_G)

        try:
            kkt_cond = np.linalg.cond(KKT)
        except Exception:
            kkt_cond = np.inf

        hist_item = {
            "it": it,
            "X": X.copy(),
            "Z": Z.copy(),
            "lambda": lam.copy(),
            "mu": mu.copy(),
            "obj_qp": obj_qp,
            "feascond": feascond,
            "gradcond": gradcond,
            "compcond": compcond,
            "gamma": gamma,
            "kkt_cond": kkt_cond,
            "linear_solver": None,
            "linear_residual": None,
            "linear_relative_residual": None,
        }

        history.append(hist_item)

        if (
            feascond < PDIPM_TOL_FEAS
            and gradcond < PDIPM_TOL_GRAD
            and compcond < PDIPM_TOL_COMP
        ):
            converged = True

            if PRINT_PDIPM_ITER:
                print(
                    f"{it:3d} "
                    f"{obj_qp:14.6e} "
                    f"{feascond:12.3e} "
                    f"{gradcond:12.3e} "
                    f"{compcond:12.3e} "
                    f"{gamma:12.3e} "
                    f"{'-':>12s} "
                    f"{'-':>12s} "
                    f"{kkt_cond:12.3e}"
                )

            break

        direction, linear_info = solve_kkt_direction(
            KKT,
            rhs,
            it,
            linear_solver=linear_solver,
        )

        history[-1]["linear_solver"] = linear_info["solver"]
        history[-1]["linear_residual"] = linear_info["residual"]
        history[-1]["linear_relative_residual"] = linear_info["relative_residual"]

        if linear_info["solver"] == "vqls":
            if "loss_history" in linear_info:
                total_vqls_loss_history.extend(linear_info["loss_history"])

        if PRINT_PDIPM_ITER:
            print(
                f"{it:3d} "
                f"{obj_qp:14.6e} "
                f"{feascond:12.3e} "
                f"{gradcond:12.3e} "
                f"{compcond:12.3e} "
                f"{gamma:12.3e} "
                f"{linear_info['residual']:12.3e} "
                f"{linear_info['relative_residual']:12.3e} "
                f"{kkt_cond:12.3e}"
            )

        dX, dZ, dlam, dmu = split_direction(direction, qp, ni)

        alpha_p = fraction_to_boundary(Z, dZ, xi=PDIPM_XI)
        alpha_d = fraction_to_boundary(mu, dmu, xi=PDIPM_XI)

        if PDIPM_USE_STEP_CONTROL:
            old_merit = merit_value(X, Z, lam, mu, gamma, qp, A_G, b_G)

            accepted = False

            for _ in range(PDIPM_MAX_LINESEARCH + 1):
                X_trial = X + alpha_p * dX
                Z_trial = Z + alpha_p * dZ
                lam_trial = lam + alpha_d * dlam
                mu_trial = mu + alpha_d * dmu

                if np.min(Z_trial) <= 0.0 or np.min(mu_trial) <= 0.0:
                    alpha_p *= PDIPM_STEP_KAPPA
                    alpha_d *= PDIPM_STEP_KAPPA
                    continue

                new_merit = merit_value(
                    X_trial,
                    Z_trial,
                    lam_trial,
                    mu_trial,
                    gamma,
                    qp,
                    A_G,
                    b_G,
                )

                if new_merit <= old_merit * 1.05:
                    accepted = True
                    break

                alpha_p *= PDIPM_STEP_KAPPA
                alpha_d *= PDIPM_STEP_KAPPA

            if not accepted:
                X_trial = X + alpha_p * dX
                Z_trial = Z + alpha_p * dZ
                lam_trial = lam + alpha_d * dlam
                mu_trial = mu + alpha_d * dmu

        else:
            X_trial = X + alpha_p * dX
            Z_trial = Z + alpha_p * dZ
            lam_trial = lam + alpha_d * dlam
            mu_trial = mu + alpha_d * dmu

        X = X_trial
        Z = np.maximum(Z_trial, 1e-14)
        lam = lam_trial
        mu = np.maximum(mu_trial, 1e-14)

        gamma = PDIPM_SIGMA * float(np.dot(mu, Z) / ni)

    final_feas, final_grad, final_comp, final_residuals = conditions(
        X, Z, lam, mu, gamma, qp, A_G, b_G
    )

    G_value = A_G @ X - b_G

    active = []

    for i, s in enumerate(Z):
        if s < 1e-6:
            active.append(ineq_names[i])

    result = {
        "method": f"SC-PDIPM ({linear_solver})",
        "linear_solver": linear_solver,
        "converged": converged,
        "iterations": history[-1]["it"],
        "X": X,
        "Z": Z,
        "lambda": lam,
        "mu": mu,
        "gamma": gamma,
        "qp": qp,
        "A_G": A_G,
        "b_G": b_G,
        "ineq_names": ineq_names,
        "G_value": G_value,
        "active_inequalities": active,
        "objective_qp": qp_objective_value(X, qp),
        "cost": objective_value(X, case),
        "parsed": parse_solution(X, case),
        "feascond": final_feas,
        "gradcond": final_grad,
        "compcond": final_comp,
        "residuals": final_residuals,
        "history": history,
        "vqls_loss_history": total_vqls_loss_history,
    }

    return result


# ============================================================
# Backward-compatible wrappers
# ============================================================

def solve_opf_scpdipm_classical(case):
    """
    Backward-compatible wrapper.
    """

    return solve_opf_scpdipm(case, linear_solver="classical")


def solve_opf_scpdipm_vqls(case):
    """
    VQLS-based SC-PDIPM wrapper.
    """

    return solve_opf_scpdipm(case, linear_solver="vqls")