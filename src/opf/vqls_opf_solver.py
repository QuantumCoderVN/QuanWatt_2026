# ============================================================
# vqls_opf_solver.py
# Linear solvers for SC-PDIPM KKT systems:
#   - classical
#   - VQLS statevector simulation
# ============================================================

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from config_qopf3 import (
    VQLS_SEED,
    VQLS_LAYERS,
    VQLS_MAXITER,
    VQLS_RESTARTS,
    VQLS_INIT_SCALE,
    VQLS_OPT_METHOD,
    VQLS_SCALE_SYSTEM,
    VQLS_USE_RUIZ_SCALING,
    VQLS_RUIZ_ITERS,
    VQLS_MAX_PAD_DIM,
    VQLS_REAL_ANSATZ,
)


def next_power_of_two(n):
    dim = 1
    while dim < n:
        dim *= 2
    return dim


def solve_linear_classical(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    info = {
        "solver": "classical",
        "residual": np.linalg.norm(A @ x - b),
        "relative_residual": np.linalg.norm(A @ x - b) / (np.linalg.norm(b) + 1e-14),
        "cond": np.linalg.cond(A),
    }

    return x, info


def pad_to_power_of_two(A, b):
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if len(b) != n:
        raise ValueError("b dimension does not match A.")

    dim = next_power_of_two(n)

    if dim > VQLS_MAX_PAD_DIM:
        raise ValueError(
            f"VQLS padded dimension {dim} exceeds VQLS_MAX_PAD_DIM={VQLS_MAX_PAD_DIM}."
        )

    A_pad = np.eye(dim, dtype=complex)
    b_pad = np.zeros(dim, dtype=complex)

    A_pad[:n, :n] = A
    b_pad[:n] = b

    return A_pad, b_pad, n


def symmetric_ruiz_scaling(A, b, iters=10, eps=1e-12):
    """
    Symmetric scaling:

        A x = b

    Let:

        x = S y

    Then:

        S A S y = S b

    This keeps symmetry if A is symmetric.
    """

    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    n = A.shape[0]
    s = np.ones(n, dtype=float)
    A_scaled = A.copy()

    for _ in range(iters):
        row_norms = np.linalg.norm(A_scaled, axis=1)
        d = 1.0 / np.sqrt(np.maximum(row_norms, eps))

        s *= d
        A_scaled = d[:, None] * A_scaled * d[None, :]

    b_scaled = s * b

    return A_scaled, b_scaled, s


def num_vqls_params(n_qubits, layers):
    if VQLS_REAL_ANSATZ:
        return n_qubits * layers

    return 2 * n_qubits * layers


def recover_scaled_solution(A, b, x_prime):
    """
    VQLS returns a normalized direction x_prime.

    Recover scalar k by minimizing:

        || A (k x_prime) - b ||

    For real KKT systems, we force k to be real.
    """

    b_prime = A @ x_prime
    denom = np.vdot(b_prime, b_prime)

    if abs(denom) < 1e-14:
        k = 0.0
    else:
        k = np.vdot(b_prime, b) / denom

    # Important fix:
    # For real OPF/KKT systems, the recovered direction should be real.
    if VQLS_REAL_ANSATZ:
        k = float(np.real(k))

    return k * x_prime, k


def apply_ansatz(qc, params, qubits, layers):
    """
    Ansatz for VQLS.

    If VQLS_REAL_ANSATZ=True:
        use RY + CX only.
        This keeps the state mostly real and is better for real KKT systems.

    If VQLS_REAL_ANSATZ=False:
        use RY + RZ + CX.
    """

    n = len(qubits)

    if VQLS_REAL_ANSATZ:
        expected = n * layers

        if len(params) != expected:
            raise ValueError(f"Expected {expected} parameters, got {len(params)}.")

        k = 0

        for _ in range(layers):
            for q in qubits:
                qc.ry(params[k], q)
                k += 1

            if n >= 2:
                for q1, q2 in zip(qubits[:-1], qubits[1:]):
                    qc.cx(q1, q2)

                if n > 2:
                    qc.cx(qubits[-1], qubits[0])

        return

    expected = 2 * n * layers

    if len(params) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(params)}.")

    k = 0

    for _ in range(layers):
        for q in qubits:
            qc.ry(params[k], q)
            k += 1

        for q in qubits:
            qc.rz(params[k], q)
            k += 1

        if n >= 2:
            for q1, q2 in zip(qubits[:-1], qubits[1:]):
                qc.cx(q1, q2)

            if n > 2:
                qc.cx(qubits[-1], qubits[0])


def make_state(params, n_qubits, layers):
    qc = QuantumCircuit(n_qubits)
    apply_ansatz(qc, params, list(range(n_qubits)), layers)
    return Statevector.from_instruction(qc).data

def vqls_global_cost(params, A, b, n_qubits, layers):
    """
    Global VQLS cost:

        C = 1 - |<b|A|x>|^2 / ( ||A|x>||^2 ||b||^2 )

    C = 0 means A|x> is parallel to |b>.
    """

    x_state = make_state(params, n_qubits, layers)

    Ax = A @ x_state

    norm_Ax = np.vdot(Ax, Ax).real
    norm_b = np.vdot(b, b).real

    if norm_Ax < 1e-14 or norm_b < 1e-14:
        return 1.0

    overlap = np.vdot(b, Ax)

    cost = 1.0 - (abs(overlap) ** 2) / (norm_Ax * norm_b)

    return float(np.real(cost))


def solve_linear_vqls(A, b, label="VQLS"):
    """
    Solve A x = b using VQLS statevector simulation.

    This is intended for SC-PDIPM KKT systems.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    A_pad, b_pad, n_original = pad_to_power_of_two(A, b)

    original_cond = np.linalg.cond(A_pad)

    if VQLS_USE_RUIZ_SCALING:
        A_scaled, b_scaled, s_diag = symmetric_ruiz_scaling(
            A_pad,
            b_pad,
            iters=VQLS_RUIZ_ITERS,
        )
    else:
        A_scaled = A_pad.copy()
        b_scaled = b_pad.copy()
        s_diag = np.ones(A_pad.shape[0], dtype=float)

    scaled_cond = np.linalg.cond(A_scaled)

    norm_scale = 1.0

    if VQLS_SCALE_SYSTEM:
        norm_scale = np.linalg.norm(A_scaled, ord=2)

        if norm_scale < 1e-14:
            raise ValueError("A norm is near zero.")

        A_work = A_scaled / norm_scale
        b_work = b_scaled / norm_scale
    else:
        A_work = A_scaled
        b_work = b_scaled

    dim = A_work.shape[0]
    n_qubits = int(np.log2(dim))
    n_params = num_vqls_params(n_qubits, VQLS_LAYERS)

    rng = np.random.default_rng(VQLS_SEED)

    best = None

    print("=" * 80)
    print(f"{label}")
    print("=" * 80)
    print("original dimension :", n_original)
    print("padded dimension   :", dim)
    print("n_qubits           :", n_qubits)
    print("n_params           :", n_params)
    print("cond original      :", original_cond)
    print("cond after scaling :", scaled_cond)
    print("norm scale         :", norm_scale)
    print()

    for r in range(VQLS_RESTARTS):
        history = []

        w0 = VQLS_INIT_SCALE * rng.standard_normal(n_params)

        def objective(w):
            c = vqls_global_cost(w, A_work, b_work, n_qubits, VQLS_LAYERS)
            history.append(c)
            return c

        res = minimize(
            objective,
            w0,
            method=VQLS_OPT_METHOD,
            options={
                "maxiter": VQLS_MAXITER,
                "rhobeg": 0.5,
                "catol": 1e-10,
            },
        )

        if best is None or res.fun < best["fun"]:
            best = {
                "fun": float(res.fun),
                "res": res,
                "history": history,
            }

        print(f"[{label}] restart {r + 1}/{VQLS_RESTARTS}")
        print("success   :", res.success)
        print("message   :", res.message)
        print("final loss:", res.fun)
        print("nfev      :", res.nfev)
        print()

    y_prime = make_state(best["res"].x, n_qubits, VQLS_LAYERS)

    # Solve scaled system:
    #     A_work y = b_work
    y_rec_pad, k = recover_scaled_solution(A_work, b_work, y_prime)

    # Undo symmetric scaling:
    #     x = S y
    x_rec_pad = s_diag * y_rec_pad

    if VQLS_REAL_ANSATZ:
        x = np.real(x_rec_pad[:n_original])
    else:
        x = np.real_if_close(x_rec_pad[:n_original]).real

    residual = np.linalg.norm(A @ x - b)
    relative_residual = residual / (np.linalg.norm(b) + 1e-14)

    info = {
        "solver": "vqls",
        "loss": best["fun"],
        "loss_history": best["history"],
        "k": k,
        "residual": residual,
        "relative_residual": relative_residual,
        "original_cond": original_cond,
        "scaled_cond": scaled_cond,
        "padded_dim": dim,
        "n_qubits": n_qubits,
    }

    print("=" * 80)
    print(f"{label} final diagnostic")
    print("=" * 80)
    print("loss              :", info["loss"])
    print("residual          :", info["residual"])
    print("relative residual :", info["relative_residual"])
    print("recovery k        :", info["k"])
    print()

    return x, info