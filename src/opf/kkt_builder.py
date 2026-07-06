# ============================================================
# kkt_builder.py
# Build bounded DC-OPF QP matrices from IEEE14-subcase-3bus
# ============================================================

import numpy as np

from pypower.ext2int import ext2int
from pypower.makeBdc import makeBdc

from pypower.idx_bus import BUS_TYPE, REF, PD, VA
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PMAX, PMIN
from pypower.idx_brch import RATE_A

from config_qopf3 import (
    ENFORCE_GEN_BOUNDS,
    ENFORCE_THETA_BOUNDS,
    THETA_MIN_DEG,
    THETA_MAX_DEG,
    ENFORCE_BRANCH_LIMITS,
    ACTIVE_TOL,
)

from case3_dc_opf import (
    extract_quadratic_cost_pu,
    make_variable_names,
)


def _to_dense(A):
    if hasattr(A, "toarray"):
        return A.toarray()
    return np.asarray(A, dtype=float)


def _copy_ppc(ppc):
    out = {}
    for k, v in ppc.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def build_dc_opf_qp(case):
    """
    Build bounded DC-OPF QP:

        minimize 0.5 z^T H z + f^T z
        subject to Aeq z = beq
                   lb <= z <= ub
                   Aineq z <= bineq

    Variables:

        z = [Pg_online_pu, theta_non_slack]

    DC balance:

        Bbus theta + Pbusinj = Cg Pg - Pd

    Rearranged:

        Cg Pg - Bbus[:, non_slack] theta_non_slack
        =
        Pd + Pbusinj + Bbus[:, slack] theta_slack
    """

    ppc = _copy_ppc(case["ppc"])
    ppc_int = ext2int(ppc)

    baseMVA = float(ppc_int["baseMVA"])
    bus = ppc_int["bus"]
    gen = ppc_int["gen"]
    branch = ppc_int["branch"]
    gencost = ppc_int["gencost"]

    online = gen[:, GEN_STATUS] > 0
    gen = gen[online, :]
    gencost = gencost[online, :]

    nb = bus.shape[0]
    ng = gen.shape[0]

    Bbus, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)

    Bbus = _to_dense(Bbus)
    Bf = _to_dense(Bf)
    Pbusinj = np.asarray(Pbusinj, dtype=float).reshape(-1)
    Pfinj = np.asarray(Pfinj, dtype=float).reshape(-1)

    slack_candidates = np.where(bus[:, BUS_TYPE] == REF)[0]

    if len(slack_candidates) == 0:
        raise ValueError("Không tìm thấy slack bus.")

    slack = int(slack_candidates[0])
    non_slack = [i for i in range(nb) if i != slack]
    ntheta = len(non_slack)

    Cg = np.zeros((nb, ng), dtype=float)

    for j in range(ng):
        gen_bus = int(gen[j, GEN_BUS])
        Cg[gen_bus, j] = 1.0

    Pd = bus[:, PD] / baseMVA
    theta_slack = np.deg2rad(bus[slack, VA])

    A_pg = Cg
    A_theta = -Bbus[:, non_slack]

    Aeq = np.hstack([A_pg, A_theta])
    beq = Pd + Pbusinj + Bbus[:, slack] * theta_slack

    case_int_for_cost = {
        "baseMVA": baseMVA,
        "gencost": gencost,
        "gen": gen,
        "bus": bus,
    }

    c2_pu, c1_pu, c0 = extract_quadratic_cost_pu(case_int_for_cost)

    nvar = ng + ntheta

    H = np.zeros((nvar, nvar), dtype=float)
    f = np.zeros(nvar, dtype=float)

    for i in range(ng):
        H[i, i] = 2.0 * c2_pu[i]
        f[i] = c1_pu[i]

    # --------------------------------------------------------
    # Bounds: lb <= z <= ub
    # --------------------------------------------------------
    lb = -np.inf * np.ones(nvar)
    ub = np.inf * np.ones(nvar)

    if ENFORCE_GEN_BOUNDS:
        for i in range(ng):
            lb[i] = gen[i, PMIN] / baseMVA
            ub[i] = gen[i, PMAX] / baseMVA

    if ENFORCE_THETA_BOUNDS:
        theta_min = np.deg2rad(THETA_MIN_DEG)
        theta_max = np.deg2rad(THETA_MAX_DEG)

        for k in range(ntheta):
            idx = ng + k
            lb[idx] = theta_min
            ub[idx] = theta_max

    # --------------------------------------------------------
    # Branch flow constraints:
    #
    # flow = Bf theta + Pfinj
    #
    # With theta_slack fixed:
    #
    # flow = Bf[:, non_slack] theta_non_slack
    #        + Bf[:, slack] theta_slack
    #        + Pfinj
    #
    # We write:
    #
    # Aineq z <= bineq
    # --------------------------------------------------------
    Aineq_rows = []
    bineq_vals = []
    ineq_names = []

    Fmat = np.zeros((branch.shape[0], nvar), dtype=float)
    Fmat[:, ng:] = Bf[:, non_slack]

    f_const = Bf[:, slack] * theta_slack + Pfinj

    if ENFORCE_BRANCH_LIMITS:
        for ell in range(branch.shape[0]):
            rate_mva = branch[ell, RATE_A]

            if rate_mva <= 0:
                continue

            rate_pu = rate_mva / baseMVA

            # flow <= rate
            Aineq_rows.append(Fmat[ell, :])
            bineq_vals.append(rate_pu - f_const[ell])
            ineq_names.append(f"branch{ell+1}_flow_upper")

            # -flow <= rate
            Aineq_rows.append(-Fmat[ell, :])
            bineq_vals.append(rate_pu + f_const[ell])
            ineq_names.append(f"branch{ell+1}_flow_lower")

    if len(Aineq_rows) > 0:
        Aineq = np.vstack(Aineq_rows)
        bineq = np.array(bineq_vals, dtype=float)
    else:
        Aineq = np.zeros((0, nvar), dtype=float)
        bineq = np.zeros(0, dtype=float)

    return {
        "H": H,
        "f": f,
        "Aeq": Aeq,
        "beq": beq,
        "Aineq": Aineq,
        "bineq": bineq,
        "ineq_names": ineq_names,
        "lb": lb,
        "ub": ub,
        "Bbus": Bbus,
        "Bf": Bf,
        "Pbusinj": Pbusinj,
        "Pfinj": Pfinj,
        "Fmat": Fmat,
        "f_const": f_const,
        "Pd": Pd,
        "Cg": Cg,
        "baseMVA": baseMVA,
        "nb": nb,
        "ng": ng,
        "ntheta": ntheta,
        "nvar": nvar,
        "slack": slack,
        "non_slack": non_slack,
        "variable_names": make_variable_names(case),
        "ppc_int": ppc_int,
        "gen_int": gen,
        "bus_int": bus,
        "branch_int": branch,
    }


def build_equality_only_kkt(qp):
    """
    Đây là KKT cũ, chỉ dùng để in và so sánh.

    Nó chưa có Pg bounds, nên có thể cho Pg âm.
    """

    H = qp["H"]
    f = qp["f"]
    Aeq = qp["Aeq"]
    beq = qp["beq"]

    KKT = np.block([
        [H, Aeq.T],
        [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]
    ])

    rhs = np.concatenate([-f, beq])

    return KKT, rhs


def build_active_set_kkt(qp, z, tol=ACTIVE_TOL):
    """
    Sau khi solve bounded QP, ta biết constraint nào đang active.

    Khi đó có thể build lại ma trận KKT active-set:

        [ H      Aactive.T ] [ z  ] = [ -f      ]
        [ Aactive   0      ] [ nu ]   [ bactive ]

    Aactive gồm:
        - Aeq
        - các bound đang active
        - các branch inequality đang active

    Đây là ma trận KKT có ý nghĩa vật lý hơn equality-only KKT.
    """

    H = qp["H"]
    f = qp["f"]
    Aeq = qp["Aeq"]
    beq = qp["beq"]
    Aineq = qp["Aineq"]
    bineq = qp["bineq"]
    lb = qp["lb"]
    ub = qp["ub"]
    names = qp["variable_names"]

    nvar = qp["nvar"]

    A_rows = []
    b_vals = []
    active_names = []

    # Equality constraints
    for i in range(Aeq.shape[0]):
        A_rows.append(Aeq[i, :])
        b_vals.append(beq[i])
        active_names.append(f"power_balance_bus_internal_{i}")

    # Active lower/upper bounds
    for i in range(nvar):
        if np.isfinite(lb[i]) and abs(z[i] - lb[i]) <= tol:
            row = np.zeros(nvar)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(lb[i])
            active_names.append(f"{names[i]}_lower_bound_active")

        if np.isfinite(ub[i]) and abs(z[i] - ub[i]) <= tol:
            row = np.zeros(nvar)
            row[i] = 1.0
            A_rows.append(row)
            b_vals.append(ub[i])
            active_names.append(f"{names[i]}_upper_bound_active")

    # Active branch inequalities
    if Aineq.shape[0] > 0:
        violation_margin = bineq - Aineq @ z

        for i, margin in enumerate(violation_margin):
            if abs(margin) <= tol:
                A_rows.append(Aineq[i, :])
                b_vals.append(bineq[i])
                active_names.append(qp["ineq_names"][i])

    Aactive = np.vstack(A_rows)
    bactive = np.array(b_vals, dtype=float)

    KKT = np.block([
        [H, Aactive.T],
        [Aactive, np.zeros((Aactive.shape[0], Aactive.shape[0]))]
    ])

    rhs = np.concatenate([-f, bactive])

    x_kkt = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    z_from_active_kkt = x_kkt[:nvar]
    multiplier = x_kkt[nvar:]

    residual = np.linalg.norm(KKT @ x_kkt - rhs)

    return {
        "KKT_active": KKT,
        "rhs_active": rhs,
        "Aactive": Aactive,
        "bactive": bactive,
        "active_names": active_names,
        "z_from_active_kkt": z_from_active_kkt,
        "multipliers": multiplier,
        "active_kkt_residual": residual,
    }


def diagnose_matrix(A, name="A"):
    A = np.asarray(A, dtype=float)

    print("=" * 80)
    print(f"Matrix diagnostic: {name}")
    print("=" * 80)
    print("shape:", A.shape)
    print("rank :", np.linalg.matrix_rank(A))

    try:
        print("cond :", np.linalg.cond(A))
    except Exception as e:
        print("cond failed:", e)

    symmetric = np.allclose(A, A.T, atol=1e-10)
    print("symmetric:", symmetric)

    if symmetric:
        eigvals = np.linalg.eigvalsh(A)
        print("min eigenvalue:", eigvals.min())
        print("max eigenvalue:", eigvals.max())

        num_pos = np.sum(eigvals > 1e-9)
        num_neg = np.sum(eigvals < -1e-9)
        num_zero = len(eigvals) - num_pos - num_neg

        print("positive eigenvalues:", num_pos)
        print("negative eigenvalues:", num_neg)
        print("near-zero eigenvalues:", num_zero)

        if num_pos > 0 and num_neg > 0:
            print("conclusion: symmetric indefinite KKT matrix")
        elif num_neg == 0 and num_zero == 0:
            print("conclusion: positive definite")
        elif num_neg == 0:
            print("conclusion: positive semidefinite")
        else:
            print("conclusion: not positive definite")

    print()