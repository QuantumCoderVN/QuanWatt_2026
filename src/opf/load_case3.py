"""Tao ma tran SC-PDIPM KKT 3-bus va xu ly dung pipeline cua paper.

Khac voi he KKT equality-only truoc day, ma tran moi dung dung cau truc block
cua phuong trinh (2) trong paper:

    [ L_xx      0       J_H^T    J_G^T ] [dX     ]   [ grad_x L       ]
    [   0    diag(mu/Z)    0        I   ] [dZ     ] =-[ mu-gamma/Z    ]
    [  J_H      0          0        0   ] [dlambda]   [ H(X)          ]
    [  J_G      I          0        0   ] [dmu    ]   [ G(X)+Z        ]

Ma tran co kich thuoc 16 x 16, tuong ung 4 qubit:
    n_X = 5, n_Z = 4, n_lambda = 3, n_mu = 4.

Pipeline theo paper:
    1. Lap KKT A_raw d = rhs_raw tai mot interior iterate.
    2. Left-ILU: A_pre = M^{-1} A_raw, b_pre = M^{-1} rhs_raw.
    3. Spectral scaling: A_vqls = A_pre/alpha, b_vqls = b_pre/alpha.
    4. Amplitude encoding: |b> = b_vqls/||b_vqls||.
    5. Phan ra A_vqls thanh Pauli/LCU trong code VQLS.

Left preconditioning khong doi an, nen search direction thu duoc tu VQLS chinh
la d = [dX, dZ, dlambda, dmu]; khong co buoc x = D y nhu symmetric scaling.

Yeu cau:
    pip install numpy scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import SuperLU, spilu


np.set_printoptions(
    precision=8,
    suppress=True,
    linewidth=240,
)


# =============================================================================
# CAU HINH LEFT-ILU
# =============================================================================

# fill_factor=1 giu ILU la incomplete, tranh bien thanh LU chinh xac.
ILU_DROP_TOL = 1e-4
ILU_FILL_FACTOR = 1.0
ILU_PERMUTATION = "NATURAL"
ILU_DIAG_PIVOT_THRESHOLD = 0.0


@dataclass(frozen=True)
class SCPDIPMProblem:
    labels: tuple[str, ...]
    A_raw: np.ndarray
    rhs_raw: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    lambda_eq: np.ndarray
    mu: np.ndarray
    gamma: float
    equality_residual: np.ndarray
    inequality_value: np.ndarray
    complementarity_residual: np.ndarray
    primal_inequality_residual: np.ndarray


@dataclass(frozen=True)
class PaperPreparedSystem:
    problem: SCPDIPMProblem
    ilu: SuperLU
    A_preconditioned: np.ndarray
    b_preconditioned: np.ndarray
    alpha: float
    A_vqls: np.ndarray
    b_vqls_raw: np.ndarray
    beta: float
    b_state: np.ndarray
    n_qubits: int


# =============================================================================
# 1. DU LIEU CASE3SC VA CAC JACOBIAN DC-OPF
# =============================================================================


def build_case3sc_derivatives() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Tra ve Hessian, q, J_H, J_G va vector tai Pd.

    Bien X:
        [theta_2, theta_3, Pg_1, Pg_2, Pg_3]

    Equality H(X)=0:
        B_reduced theta - Pg + Pd = 0, co 3 rang buoc.

    Bon inequality duoc chon de tao ma tran 16 x 16:
        G1 = -Pg1                 <= 0
        G2 = -Pg2                 <= 0
        G3 = Pg3 - Pg3_max        <= 0
        G4 = -Pg3 + Pg3_min       <= 0

    Day la tap rang buoc dai dien de ma tran co du cac block SC-PDIPM va co
    kich thuoc 2^n. Khi trien khai OPF day du, them cac bound/line constraint
    con lai; pipeline left-ILU va VQLS khong thay doi.
    """

    base_mva = 100.0
    loads_mw = np.array([110.0, 110.0, 95.0])

    branches = (
        (1, 3, 0.62),
        (3, 2, 0.75),
        (1, 2, 0.90),
    )

    Bbus = np.zeros((3, 3), dtype=float)

    for from_bus, to_bus, reactance in branches:
        i = from_bus - 1
        j = to_bus - 1
        bij = base_mva / reactance
        Bbus[i, i] += bij
        Bbus[j, j] += bij
        Bbus[i, j] -= bij
        Bbus[j, i] -= bij

    B_reduced = Bbus[:, 1:]
    generator_incidence = np.eye(3)
    J_H = np.hstack((B_reduced, -generator_incidence))

    J_G = np.zeros((4, 5), dtype=float)
    J_G[0, 2] = -1.0
    J_G[1, 3] = -1.0
    J_G[2, 4] = 1.0
    J_G[3, 4] = -1.0

    # f(Pg)=0.11 Pg1^2+5 Pg1+0.085 Pg2^2+1.2 Pg2.
    hessian = np.diag([0.0, 0.0, 0.22, 0.17, 0.0])
    linear_cost = np.array([0.0, 0.0, 5.0, 1.2, 0.0])

    return hessian, linear_cost, J_H, J_G, loads_mw


def inequality_values(X: np.ndarray) -> np.ndarray:
    Pg1, Pg2, Pg3 = X[2], X[3], X[4]
    Pg3_max = 1e-4
    Pg3_min = -1e-4

    return np.array(
        [
            -Pg1,
            -Pg2,
            Pg3 - Pg3_max,
            -Pg3 + Pg3_min,
        ],
        dtype=float,
    )


# =============================================================================
# 2. LAP DUNG MA TRAN KKT PHUONG TRINH (2)
# =============================================================================


def build_new_scpdipm_kkt() -> SCPDIPMProblem:
    """Lap ma tran KKT 16 x 16 tai mot strict-interior iterate."""

    hessian, linear_cost, J_H, J_G, loads_mw = build_case3sc_derivatives()

    # Chon dispatch co tong Pg bang tong Pd va giai goc DC de H(X)=0.
    Pg = np.array([160.0, 155.0, 0.0])
    required_injection = Pg - loads_mw
    theta = np.linalg.lstsq(J_H[:, :2], required_injection, rcond=None)[0]
    X = np.concatenate((theta, Pg))

    # Diem trong: Z>0, mu>0. Khong bat buoc moi residual da bang 0;
    # he Newton/KKT chinh la de tinh search direction giam cac residual nay.
    Z = np.ones(4)
    lambda_eq = np.zeros(3)
    mu = np.ones(4)
    gamma = 0.1

    H_value = J_H @ X + loads_mw
    G_value = inequality_values(X)

    grad_lagrangian = (
        hessian @ X
        + linear_cost
        + J_H.T @ lambda_eq
        + J_G.T @ mu
    )

    complementarity_residual = mu - gamma / Z
    primal_inequality_residual = G_value + Z

    n_X = 5
    n_i = 4
    n_e = 3

    zero_X_i = np.zeros((n_X, n_i))
    zero_i_X = np.zeros((n_i, n_X))
    zero_X_e = np.zeros((n_X, n_e))
    zero_i_e = np.zeros((n_i, n_e))
    zero_e_i = np.zeros((n_e, n_i))
    zero_e_e = np.zeros((n_e, n_e))
    zero_i_i = np.zeros((n_i, n_i))

    # Thu tu block dung paper: [dX, dZ, dlambda, dmu].
    A_raw = np.block(
        [
            [hessian, zero_X_i, J_H.T, J_G.T],
            [zero_i_X, np.diag(mu / Z), zero_i_e, np.eye(n_i)],
            [J_H, zero_e_i, zero_e_e, zero_e_i],
            [J_G, np.eye(n_i), zero_i_e, zero_i_i],
        ]
    )

    rhs_raw = -np.concatenate(
        (
            grad_lagrangian,
            complementarity_residual,
            H_value,
            primal_inequality_residual,
        )
    )

    labels = (
        "dtheta_2",
        "dtheta_3",
        "dPg_1",
        "dPg_2",
        "dPg_3",
        "dZ_1",
        "dZ_2",
        "dZ_3",
        "dZ_4",
        "dlambda_1",
        "dlambda_2",
        "dlambda_3",
        "dmu_1",
        "dmu_2",
        "dmu_3",
        "dmu_4",
    )

    if A_raw.shape != (16, 16):
        raise RuntimeError("KKT moi phai co kich thuoc 16 x 16.")

    if not np.allclose(A_raw, A_raw.T, atol=1e-12):
        raise RuntimeError("KKT truoc left preconditioning phai symmetric.")

    if np.linalg.matrix_rank(A_raw) < 16:
        raise RuntimeError("KKT moi bi suy bien.")

    return SCPDIPMProblem(
        labels=labels,
        A_raw=A_raw.astype(complex),
        rhs_raw=rhs_raw.astype(complex),
        X=X,
        Z=Z,
        lambda_eq=lambda_eq,
        mu=mu,
        gamma=gamma,
        equality_residual=H_value,
        inequality_value=G_value,
        complementarity_residual=complementarity_residual,
        primal_inequality_residual=primal_inequality_residual,
    )


# =============================================================================
# 3. LEFT-ILU PRECONDITIONING DUNG NHU PAPER
# =============================================================================


def left_ilu_prepare_for_vqls(
    problem: SCPDIPMProblem,
) -> PaperPreparedSystem:
    """Ap dung A_pre=M^{-1}A va b_pre=M^{-1}b, sau do chuan hoa."""

    A_raw = problem.A_raw
    rhs_raw = problem.rhs_raw

    ilu = spilu(
        csc_matrix(A_raw),
        drop_tol=ILU_DROP_TOL,
        fill_factor=ILU_FILL_FACTOR,
        permc_spec=ILU_PERMUTATION,
        diag_pivot_thresh=ILU_DIAG_PIVOT_THRESHOLD,
    )

    # Khong lap nghich dao M^{-1} truc tiep. Dung triangular solves cua ILU.
    A_preconditioned = ilu.solve(A_raw)
    b_preconditioned = ilu.solve(rhs_raw)

    condition_raw = np.linalg.cond(A_raw)
    condition_preconditioned = np.linalg.cond(A_preconditioned)

    if not np.isfinite(condition_preconditioned):
        raise RuntimeError("Left-ILU tao ma tran co condition number vo han.")

    if condition_preconditioned >= condition_raw:
        raise RuntimeError(
            "Left-ILU khong giam condition number; can dieu chinh tham so ILU."
        )

    alpha = float(np.linalg.norm(A_preconditioned, ord=2))
    A_vqls = A_preconditioned / alpha
    b_vqls_raw = b_preconditioned / alpha

    beta = float(np.linalg.norm(b_vqls_raw))
    b_state = b_vqls_raw / beta

    n_qubits = int(np.log2(A_vqls.shape[0]))

    if 2**n_qubits != A_vqls.shape[0]:
        raise RuntimeError("Kich thuoc sau preconditioning khong phai 2^n.")

    return PaperPreparedSystem(
        problem=problem,
        ilu=ilu,
        A_preconditioned=A_preconditioned,
        b_preconditioned=b_preconditioned,
        alpha=alpha,
        A_vqls=A_vqls,
        b_vqls_raw=b_vqls_raw,
        beta=beta,
        b_state=b_state,
        n_qubits=n_qubits,
    )


# =============================================================================
# 4. CHAN DOAN PAULI/LCU KHONG CAN QISKIT
# =============================================================================


def pauli_term_count(matrix: np.ndarray, atol: float = 1e-10) -> int:
    """Dem so he so Pauli khac 0 cua ma tran kich thuoc 2^n."""

    single_pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    dimension = matrix.shape[0]
    n_qubits = int(np.log2(dimension))
    count = 0

    for symbols in product("IXYZ", repeat=n_qubits):
        pauli = single_pauli[symbols[0]]

        for symbol in symbols[1:]:
            pauli = np.kron(pauli, single_pauli[symbol])

        coefficient = np.trace(pauli.conj().T @ matrix) / dimension

        if abs(coefficient) > atol:
            count += 1

    return count


# =============================================================================
# 5. KIEM CHUNG VA XUAT DU LIEU CHO CODE VQLS
# =============================================================================


def verify_solution_equivalence(system: PaperPreparedSystem) -> dict[str, float]:
    """Kiem tra left preconditioning va spectral scaling khong doi nghiem."""

    raw_solution = np.linalg.solve(
        system.problem.A_raw,
        system.problem.rhs_raw,
    )

    prepared_solution = np.linalg.solve(
        system.A_vqls,
        system.b_vqls_raw,
    )

    relative_solution_difference = float(
        np.linalg.norm(prepared_solution - raw_solution)
        / np.linalg.norm(raw_solution)
    )

    raw_relative_residual = float(
        np.linalg.norm(
            system.problem.A_raw @ prepared_solution
            - system.problem.rhs_raw
        )
        / np.linalg.norm(system.problem.rhs_raw)
    )

    return {
        "relative_solution_difference": relative_solution_difference,
        "raw_relative_residual": raw_relative_residual,
    }


def print_summary(system: PaperPreparedSystem) -> None:
    problem = system.problem
    verification = verify_solution_equivalence(system)

    print("=" * 100)
    print("NEW SC-PDIPM KKT MATRIX - PAPER EQUATION (2)")
    print("=" * 100)
    print("Unknown order:", list(problem.labels))
    print(f"Matrix dimension                         = {problem.A_raw.shape}")
    print(f"Number of VQLS state qubits              = {system.n_qubits}")
    print(f"Raw KKT symmetric/Hermitian              = {np.allclose(problem.A_raw, problem.A_raw.conj().T)}")
    print(f"Left-ILU matrix Hermitian                 = {np.allclose(system.A_preconditioned, system.A_preconditioned.conj().T)}")
    print(f"cond_2(A_raw)                             = {np.linalg.cond(problem.A_raw):.10f}")
    print(f"cond_2(M^-1 A_raw)                        = {np.linalg.cond(system.A_preconditioned):.10f}")
    print(f"||A_vqls||_2                              = {np.linalg.norm(system.A_vqls, 2):.10f}")
    print(f"ILU drop tolerance                        = {ILU_DROP_TOL}")
    print(f"ILU fill factor                           = {ILU_FILL_FACTOR}")
    print(f"||M^-1 A-I||_F                            = {np.linalg.norm(system.A_preconditioned-np.eye(16)):.10f}")
    print(f"Raw KKT Pauli terms                       = {pauli_term_count(problem.A_raw)}")
    print(f"Left-ILU/scaled Pauli terms               = {pauli_term_count(system.A_vqls)}")
    print(f"Solution equivalence error                = {verification['relative_solution_difference']:.12e}")
    print(f"Residual in original KKT                  = {verification['raw_relative_residual']:.12e}")

    print("\nInterior iterate:")
    print("X      =", problem.X)
    print("Z      =", problem.Z)
    print("lambda =", problem.lambda_eq)
    print("mu     =", problem.mu)
    print("gamma  =", problem.gamma)

    print("\nA_RAW (full SC-PDIPM KKT) =")
    print(np.real_if_close(problem.A_raw))

    print("\nRHS_RAW =")
    print(np.real_if_close(problem.rhs_raw))

    print("\nA_PRECONDITIONED = M^-1 A_RAW =")
    print(np.real_if_close(system.A_preconditioned))

    print("\nB_PRECONDITIONED = M^-1 RHS_RAW =")
    print(np.real_if_close(system.b_preconditioned))

    print("\nA_MATRIX = A_VQLS =")
    print(np.real_if_close(system.A_vqls))

    print("\nB_VECTOR_RAW =")
    print(np.real_if_close(system.b_vqls_raw))

    print("\nB_VECTOR = |b> =")
    print(np.real_if_close(system.b_state))


def save_npz(system: PaperPreparedSystem, path: Path) -> Path:
    np.savez(
        path,
        A_RAW=system.problem.A_raw,
        B_RAW=system.problem.rhs_raw,
        A_PRECONDITIONED=system.A_preconditioned,
        B_PRECONDITIONED=system.b_preconditioned,
        A_MATRIX=system.A_vqls,
        B_VECTOR_RAW=system.b_vqls_raw,
        B_VECTOR=system.b_state,
        B_VECTOR_NORM=np.array(system.beta),
        SPECTRAL_SCALE=np.array(system.alpha),
        LABELS=np.asarray(system.problem.labels),
    )
    return path


# =============================================================================
# CAC BIEN SAN SANG IMPORT VAO CODE VQLS
# =============================================================================


PROBLEM = build_new_scpdipm_kkt()
PREPARED = left_ilu_prepare_for_vqls(PROBLEM)

A_RAW = PROBLEM.A_raw
B_RAW = PROBLEM.rhs_raw

A_PRECONDITIONED = PREPARED.A_preconditioned
B_PRECONDITIONED = PREPARED.b_preconditioned

A_MATRIX = PREPARED.A_vqls
B_VECTOR_RAW = PREPARED.b_vqls_raw
B_VECTOR_NORM = PREPARED.beta
B_VECTOR = PREPARED.b_state

N_QUBITS_INPUT = PREPARED.n_qubits
VARIABLE_LABELS = PROBLEM.labels


def main() -> None:
    print_summary(PREPARED)
    output_path = save_npz(
        PREPARED,
        Path(__file__).resolve().parent / "scpdipm_case3sc_paper_matrix.npz",
    )
    print(f"\nSaved VQLS matrix package to: {output_path}")


if __name__ == "__main__":
    main()