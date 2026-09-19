"""
VQLS 4x4: compare signed recovered solution values from
1) original SWAP-parity sign recovery
2) full quantum state tomography
against the classical solution.

Key features
------------
- A is a 4x4 matrix and is automatically decomposed into Pauli terms by Qiskit:
      SparsePauliOp.from_operator(Operator(A_MATRIX))
- The final plot compares signed solution values, not probabilities:
      classical x, SWAP-parity recovered x, tomography recovered x
  in one grouped bar chart.
- Results are saved after running into an output directory, including optimized weights,
  cost history, recovered solutions, recovery runtimes, figures, and metadata.
- Later, use --plot-only to redraw figures without re-running VQLS optimization
  or recovery.

Usage
-----
Run full VQLS + recovery + save + plot:
    python vqls_4x4_signed_solution_compare_save.py

Only redraw plots from saved results:
    python vqls_4x4_signed_solution_compare_save.py --plot-only

Optional:
    python vqls_4x4_signed_solution_compare_save.py --shots 200000 --steps 160
    python vqls_4x4_signed_solution_compare_save.py --output-dir output_run_01
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator


# ============================================================
# User settings
# ============================================================
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_RESULTS_FILE = "vqls_4x4_saved_results.npz"
DEFAULT_SUMMARY_FILE = "vqls_4x4_saved_summary.json"
DEFAULT_SOLUTION_FIG = "vqls_4x4_signed_solution_values.png"
DEFAULT_COST_FIG = "vqls_4x4_cost_convergence.png"
DEFAULT_RUNTIME_FIG = "vqls_4x4_recovery_runtime.png"

RNG_SEED = 0
PAULI_ATOL = 1e-10
PAULI_RTOL = 1e-10
N_LAYERS = 3
Q_DELTA = 0.001

# 4x4 system, so N_QUBITS = 2.
# This is intentionally not manually decomposed into Pauli strings.
A_MATRIX = np.array(
    [
        [4.0,  0.8,  0.3, -0.2],
        [0.8,  3.5, -0.4,  0.5],
        [0.3, -0.4,  2.8,  0.7],
        [-0.2, 0.5,  0.7,  3.2],
    ],
    dtype=complex,
)

B_VECTOR_RAW = np.array([1.30, -0.62, 0.70, -0.31], dtype=complex)
B_VECTOR_NORM = np.linalg.norm(B_VECTOR_RAW)
B_VECTOR = B_VECTOR_RAW / B_VECTOR_NORM


# ============================================================
# Utilities
# ============================================================
def infer_n_qubits_from_matrix(A):
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A_MATRIX must be square.")
    dim = A.shape[0]
    n = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError("A_MATRIX size must be 2^n x 2^n.")
    return n, A


def pauli_decompose_matrix(A_matrix, atol=1e-10, rtol=1e-10):
    """Automatically decompose A into Pauli basis using Qiskit."""
    n_qubits, A_matrix = infer_n_qubits_from_matrix(A_matrix)
    op = Operator(A_matrix, input_dims=(2,) * n_qubits, output_dims=(2,) * n_qubits)
    A_pauli = SparsePauliOp.from_operator(op, atol=atol, rtol=rtol)
    labels = [p.to_label() for p in A_pauli.paulis]
    coeffs = np.asarray(A_pauli.coeffs, dtype=complex)
    return n_qubits, A_matrix, A_pauli, labels, coeffs


def make_Ub_gate(b_vector):
    b_vector = np.asarray(b_vector, dtype=complex)
    b_vector = b_vector / np.linalg.norm(b_vector)
    return StatePreparation(b_vector)


def add_measure_all(qc):
    qc_m = qc.copy()
    qc_m.measure_all()
    return qc_m


def run_counts(qc, shots, backend=None, seed=0):
    if backend is None:
        backend = AerSimulator(seed_simulator=seed)
    qc_m = add_measure_all(qc)
    tqc = transpile(qc_m, backend)
    return backend.run(tqc, shots=int(shots)).result().get_counts()


def counts_to_probability_vector(counts, n_bits):
    probs = np.zeros(2**n_bits, dtype=float)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Empty counts.")
    for bitstring, count in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        probs[idx] += count / total
    return probs


def normalize_global_sign(x, reference):
    """Flip global sign/phase so x is closest to reference."""
    x = np.asarray(x, dtype=complex)
    reference = np.asarray(reference, dtype=complex)
    overlap = np.vdot(reference, x)
    if abs(overlap) < 1e-14:
        return x
    return x * np.exp(-1j * np.angle(overlap))


def best_global_sign_real(x, reference):
    x = np.real_if_close(np.asarray(x, dtype=complex)).real
    reference = np.real_if_close(np.asarray(reference, dtype=complex)).real
    if np.linalg.norm(x - reference) <= np.linalg.norm(-x - reference):
        return x
    return -x


def relative_error(x, x_ref):
    return np.linalg.norm(x - x_ref) / max(np.linalg.norm(x_ref), 1e-14)


def residual_norm(A, x, b):
    return np.linalg.norm(A @ x - b)


def scale_normalized_state_to_solution(normalized_state, A_matrix, b_vector):
    """
    Given normalized quantum state x_prime, find scalar k minimizing
        || A (k x_prime) - b ||_2.
    """
    x_prime = np.asarray(normalized_state, dtype=complex)
    b_prime = A_matrix @ x_prime
    denom = np.vdot(b_prime, b_prime)
    if abs(denom) < 1e-14:
        k = 0.0
    else:
        k = np.vdot(b_prime, b_vector) / denom
    return k * x_prime, k, b_prime


# ============================================================
# Build VQLS object from A
# ============================================================
N_QUBITS, A_MATRIX, A_PAULI, PAULI_LABELS, C = pauli_decompose_matrix(
    A_MATRIX, atol=PAULI_ATOL, rtol=PAULI_RTOL
)
TOT_QUBITS = N_QUBITS + 1
ANCILLA_IDX = N_QUBITS
NUM_PAULI = len(C)
N_PARAMS = N_QUBITS * N_LAYERS
U_B_GATE = make_Ub_gate(B_VECTOR)


# ============================================================
# VQLS circuit components
# ============================================================
def apply_U_b(qc, qubits, dagger=False):
    qc.append(U_B_GATE.inverse() if dagger else U_B_GATE, qubits)


def apply_controlled_pauli_string(qc, pauli_label, qubits, ancilla):
    n = len(qubits)
    if len(pauli_label) != n:
        raise ValueError("Pauli label length mismatch.")
    for str_idx, p in enumerate(pauli_label):
        # Qiskit label convention: rightmost char acts on qubit 0.
        q = qubits[n - 1 - str_idx]
        if p == "I":
            continue
        if p == "X":
            qc.cx(ancilla, q)
        elif p == "Y":
            qc.cy(ancilla, q)
        elif p == "Z":
            qc.cz(ancilla, q)
        else:
            raise ValueError(f"Invalid Pauli: {p}")


def apply_CA(qc, l, qubits, ancilla):
    apply_controlled_pauli_string(qc, PAULI_LABELS[l], qubits, ancilla)


def apply_variational(qc, params, qubits):
    """Real RY ansatz; suitable for signed real recovery."""
    params = np.asarray(params, dtype=float)
    expected = len(qubits) * N_LAYERS
    if len(params) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(params)}.")
    for q in qubits:
        qc.h(q)
    k = 0
    for layer in range(N_LAYERS):
        for q in qubits:
            qc.ry(params[k], q)
            k += 1
        if layer < N_LAYERS - 1:
            for q1, q2 in zip(qubits[:-1], qubits[1:]):
                qc.cx(q1, q2)


def build_ansatz_circuit(weights):
    qc = QuantumCircuit(N_QUBITS)
    apply_variational(qc, weights, list(range(N_QUBITS)))
    return qc


# ============================================================
# Hadamard test and VQLS cost
# ============================================================
def hadamard_test(weights, l, lp, j, part):
    qc = QuantumCircuit(TOT_QUBITS)
    main = list(range(N_QUBITS))

    qc.h(ANCILLA_IDX)
    if part == "Im":
        qc.p(-np.pi / 2, ANCILLA_IDX)

    apply_variational(qc, weights, main)
    apply_CA(qc, lp, main, ANCILLA_IDX)

    apply_U_b(qc, main, dagger=True)
    if j != -1:
        qc.cz(ANCILLA_IDX, main[j])
    apply_U_b(qc, main, dagger=False)

    apply_CA(qc, l, main, ANCILLA_IDX)
    qc.h(ANCILLA_IDX)
    return qc


def measure_z_ancilla(weights, l, lp, j, part):
    qc = hadamard_test(weights, l, lp, j, part)
    state = Statevector.from_instruction(qc)
    pauli_str = "Z" + "I" * N_QUBITS
    Z_obs = SparsePauliOp.from_list([(pauli_str, 1.0)])
    return state.expectation_value(Z_obs).real


def mu(weights, l, lp, j):
    return measure_z_ancilla(weights, l, lp, j, "Re") + 1j * measure_z_ancilla(weights, l, lp, j, "Im")


def psi_norm(weights):
    norm = 0.0 + 0.0j
    for l in range(NUM_PAULI):
        for lp in range(NUM_PAULI):
            norm += np.conj(C[l]) * C[lp] * mu(weights, l, lp, -1)
    return norm.real


def cost_local(weights):
    norm = psi_norm(weights)
    if abs(norm) < 1e-12:
        return 1e6
    mu_sum = 0.0 + 0.0j
    for l in range(NUM_PAULI):
        for lp in range(NUM_PAULI):
            for j in range(N_QUBITS):
                mu_sum += np.conj(C[l]) * C[lp] * mu(weights, l, lp, j)
    return 0.5 - 0.5 * mu_sum.real / (N_QUBITS * norm)


# ============================================================
# Recovery method 1: original SWAP-parity sign recovery
# ============================================================
def cyclic_left_shift_matrix_qiskit(dim):
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        U[i, (i + 1) % dim] = 1.0
    return U


def build_vqls_probability_circuit(weights):
    return build_ansatz_circuit(weights)


def build_swap_parity_circuit(weights, shifted=False):
    readout = N_QUBITS
    main = list(range(N_QUBITS))
    dim = 2**N_QUBITS
    qc = QuantumCircuit(N_QUBITS + 1)
    apply_variational(qc, weights, main)
    if shifted:
        qc.unitary(cyclic_left_shift_matrix_qiskit(dim), main, label="cyclic_left_shift")
    qc.swap(0, readout)
    qc.h(readout)
    return qc


def parity_interference_from_counts(counts, n_main):
    full_probs = counts_to_probability_vector(counts, n_main + 1)
    L = 2**n_main
    pc = np.zeros(L // 2, dtype=float)
    for k in range(L // 2):
        full_index = 2 * k  # readout=0, main_index=2k
        pc[k] = 2.0 * full_probs[full_index]
    return pc


def recover_signs_from_swap_parity_counts(prob_orig, counts_parity_unshifted, counts_parity_shifted, first_sign=1, zero_tol=1e-12):
    prob_orig = np.asarray(prob_orig, dtype=float)
    L = len(prob_orig)
    parity_0 = parity_interference_from_counts(counts_parity_unshifted, N_QUBITS)
    parity_1 = parity_interference_from_counts(counts_parity_shifted, N_QUBITS)

    parity_array_0 = []
    for i, pc in enumerate(parity_0):
        a = prob_orig[2 * i]
        b = prob_orig[2 * i + 1]
        if a < zero_tol or b < zero_tol:
            parity_array_0.append(1)
        elif pc < a or pc < b:
            parity_array_0.append(-1)
        else:
            parity_array_0.append(1)

    rotated_prob = np.roll(prob_orig, -1)
    parity_array_1 = []
    for i, pc in enumerate(parity_1):
        a = rotated_prob[2 * i]
        b = rotated_prob[2 * i + 1]
        if a < zero_tol or b < zero_tol:
            parity_array_1.append(1)
        elif pc < a or pc < b:
            parity_array_1.append(-1)
        else:
            parity_array_1.append(1)

    parity_intertwined = []
    for i in range(max(len(parity_array_0), len(parity_array_1))):
        if i < len(parity_array_0):
            parity_intertwined.append(parity_array_0[i])
        if i < len(parity_array_1):
            parity_intertwined.append(parity_array_1[i])

    if len(parity_intertwined) != L:
        raise ValueError(f"Expected {L} parities, got {len(parity_intertwined)}.")

    first = 1 if first_sign >= 0 else -1
    signs = []
    s = first
    for i in range(L):
        signs.append(s)
        s = s * parity_intertwined[i]

    signs = np.asarray(signs, dtype=int)
    signed_state = signs * np.sqrt(np.maximum(prob_orig, 0.0))

    return {
        "prob_orig": prob_orig,
        "sign_array": signs,
        "signed_normalized_state": signed_state,
        "parity_couple_0": parity_0,
        "parity_couple_1": parity_1,
        "parity_array_0": np.asarray(parity_array_0),
        "parity_array_1": np.asarray(parity_array_1),
        "parity_intertwined": np.asarray(parity_intertwined),
        "cycle_consistent": bool(s == first),
    }


def recover_by_swap_parity(weights, shots, first_sign, seed=0, backend=None):
    qc_prob = build_vqls_probability_circuit(weights)
    qc_p0 = build_swap_parity_circuit(weights, shifted=False)
    qc_p1 = build_swap_parity_circuit(weights, shifted=True)

    counts_prob = run_counts(qc_prob, shots=shots, backend=backend, seed=seed)
    counts_p0 = run_counts(qc_p0, shots=shots, backend=backend, seed=seed + 1)
    counts_p1 = run_counts(qc_p1, shots=shots, backend=backend, seed=seed + 2)

    prob_orig = counts_to_probability_vector(counts_prob, N_QUBITS)
    out = recover_signs_from_swap_parity_counts(prob_orig, counts_p0, counts_p1, first_sign=first_sign)
    out["counts_prob"] = counts_prob
    out["counts_parity_unshifted"] = counts_p0
    out["counts_parity_shifted"] = counts_p1
    return out


# ============================================================
# Recovery method 2: quantum tomography
# ============================================================
def unique_tomography_bases(n):
    import itertools
    return ["".join(b) for b in itertools.product("XYZ", repeat=n)]


def all_pauli_strings(n):
    import itertools
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)]


def pauli_to_measurement_basis(pauli_string):
    return "".join("Z" if p == "I" else p for p in pauli_string)


def add_basis_rotation(qc, basis):
    n = len(basis)
    for i, b in enumerate(basis):
        q = n - 1 - i
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
        elif b == "Z":
            pass
        else:
            raise ValueError(f"Unknown basis char: {b}")


def make_tomography_circuit(state_circuit, basis):
    n = state_circuit.num_qubits
    U = state_circuit.copy()
    U.remove_final_measurements(inplace=True)
    qc = QuantumCircuit(n, n)
    qc.compose(U, range(n), inplace=True)
    add_basis_rotation(qc, basis)
    qc.measure(range(n), range(n))
    return qc


def pauli_eigenvalue_from_bitstring(pauli_string, bitstring):
    eig = 1
    for p, bit in zip(pauli_string, bitstring):
        if p == "I":
            continue
        eig *= 1 if bit == "0" else -1
    return eig


def estimate_pauli_expectation(pauli_string, counts):
    shots = sum(counts.values())
    expval = 0.0
    for bitstring, count in counts.items():
        expval += pauli_eigenvalue_from_bitstring(pauli_string, bitstring) * count / shots
    return expval


def single_pauli_matrix(p):
    if p == "I":
        return np.array([[1, 0], [0, 1]], dtype=complex)
    if p == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if p == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if p == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError(f"Unknown Pauli: {p}")


def pauli_matrix(pauli_string):
    mats = [single_pauli_matrix(p) for p in pauli_string]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def project_to_physical_density_matrix(rho):
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals, 0.0, None)
    if np.sum(vals) <= 0:
        raise ValueError("All eigenvalues vanished after clipping.")
    vals = vals / np.sum(vals)
    return vecs @ np.diag(vals) @ vecs.conj().T


def quantum_state_tomography(state_circuit, shots, backend=None, seed=0, make_physical=True):
    n = state_circuit.num_qubits
    if backend is None:
        backend = AerSimulator(seed_simulator=seed)
    bases = unique_tomography_bases(n)
    circuits = [make_tomography_circuit(state_circuit, basis) for basis in bases]
    tqc = transpile(circuits, backend)
    result = backend.run(tqc, shots=int(shots)).result()

    counts_by_basis = {basis: result.get_counts(i) for i, basis in enumerate(bases)}
    expectations = {}
    for P in all_pauli_strings(n):
        if P == "I" * n:
            expectations[P] = 1.0
        else:
            expectations[P] = estimate_pauli_expectation(P, counts_by_basis[pauli_to_measurement_basis(P)])

    dim = 2**n
    rho = np.zeros((dim, dim), dtype=complex)
    for P, expval in expectations.items():
        rho += expval * pauli_matrix(P)
    rho /= dim
    rho = 0.5 * (rho + rho.conj().T)
    if make_physical:
        rho = project_to_physical_density_matrix(rho)
    return rho, expectations, counts_by_basis


def amplitudes_from_density_matrix_real_reference(rho, reference_sign=1, reference_index=0):
    rho = np.asarray(rho, dtype=complex)
    p_ref = np.real(rho[reference_index, reference_index])
    if p_ref <= 0:
        raise ValueError("Reference amplitude is zero or invalid.")
    a_ref = (1 if reference_sign >= 0 else -1) * np.sqrt(p_ref)
    # rho[ref, i] = a_ref * a_i for real pure states.
    amps = np.real(rho[reference_index, :]) / a_ref
    norm = np.linalg.norm(amps)
    if norm > 0:
        amps = amps / norm
    return amps


def recover_by_tomography(weights, shots, reference_sign, seed=0, backend=None):
    qc = build_ansatz_circuit(weights)
    rho, expectations, counts_by_basis = quantum_state_tomography(qc, shots=shots, backend=backend, seed=seed, make_physical=True)
    state = amplitudes_from_density_matrix_real_reference(rho, reference_sign=reference_sign, reference_index=0)
    purity = np.real(np.trace(rho @ rho))
    return {
        "rho": rho,
        "normalized_state": state,
        "purity": purity,
        "expectations": expectations,
        "counts_by_basis": counts_by_basis,
    }


# ============================================================
# Optimization, saving, and plotting
# ============================================================
def run_full_experiment(shots=200000, steps=160, timing_repeats=3, seed=0):
    np.random.seed(seed)
    backend = AerSimulator(seed_simulator=seed)

    print(f"Matrix size: {A_MATRIX.shape[0]} x {A_MATRIX.shape[1]} ({N_QUBITS} qubits)")
    print(f"Number of Pauli terms from automatic decomposition: {NUM_PAULI}")
    print(f"Condition number of A: {np.linalg.cond(A_MATRIX):.6e}")
    print("\nAutomatic Pauli decomposition:")
    print(A_PAULI)

    w_init = Q_DELTA * np.random.randn(N_PARAMS)
    cost_history = []

    def cost_with_log(w):
        c = cost_local(w)
        cost_history.append(c)
        return c

    t_opt0 = time.perf_counter()
    res = minimize(
        cost_with_log,
        w_init,
        method="COBYLA",
        options={"rhobeg": 0.5, "maxiter": int(steps), "catol": 1e-8},
    )
    t_opt1 = time.perf_counter()
    w_opt = res.x

    # Classical solution for normalized b and raw b.
    x_classical = np.linalg.solve(A_MATRIX, B_VECTOR)
    x_classical_raw = np.linalg.solve(A_MATRIX, B_VECTOR_RAW)
    x_classical_norm = x_classical / np.linalg.norm(x_classical)

    first_sign_anchor = 1 if np.real(x_classical_norm[0]) >= 0 else -1

    # Main recovery once, these are the values used for solution comparison.
    t0 = time.perf_counter()
    swap = recover_by_swap_parity(w_opt, shots=shots, first_sign=first_sign_anchor, seed=seed, backend=backend)
    x_swap_norm = best_global_sign_real(swap["signed_normalized_state"], x_classical_norm)
    x_swap, k_swap, _ = scale_normalized_state_to_solution(x_swap_norm, A_MATRIX, B_VECTOR)
    x_swap = best_global_sign_real(x_swap, x_classical)
    t1 = time.perf_counter()
    swap_main_time = t1 - t0

    t0 = time.perf_counter()
    tomo = recover_by_tomography(w_opt, shots=shots, reference_sign=first_sign_anchor, seed=seed + 100, backend=backend)
    x_tomo_norm = best_global_sign_real(tomo["normalized_state"], x_classical_norm)
    x_tomo, k_tomo, _ = scale_normalized_state_to_solution(x_tomo_norm, A_MATRIX, B_VECTOR)
    x_tomo = best_global_sign_real(x_tomo, x_classical)
    t1 = time.perf_counter()
    tomo_main_time = t1 - t0

    # Optional timing repeats: only for recovery stage, not optimization.
    swap_times = []
    tomo_times = []
    for r in range(int(timing_repeats)):
        t0 = time.perf_counter()
        _ = recover_by_swap_parity(w_opt, shots=shots, first_sign=first_sign_anchor, seed=seed + 10 * r, backend=backend)
        swap_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = recover_by_tomography(w_opt, shots=shots, reference_sign=first_sign_anchor, seed=seed + 1000 + 10 * r, backend=backend)
        tomo_times.append(time.perf_counter() - t0)

    metrics = {
        "swap_rel_error": relative_error(x_swap, x_classical),
        "tomo_rel_error": relative_error(x_tomo, x_classical),
        "swap_residual": residual_norm(A_MATRIX, x_swap, B_VECTOR),
        "tomo_residual": residual_norm(A_MATRIX, x_tomo, B_VECTOR),
        "classical_residual": residual_norm(A_MATRIX, x_classical, B_VECTOR),
        "swap_main_time": swap_main_time,
        "tomo_main_time": tomo_main_time,
        "swap_runtime_mean": float(np.mean(swap_times)) if len(swap_times) else swap_main_time,
        "swap_runtime_std": float(np.std(swap_times, ddof=1)) if len(swap_times) > 1 else 0.0,
        "tomo_runtime_mean": float(np.mean(tomo_times)) if len(tomo_times) else tomo_main_time,
        "tomo_runtime_std": float(np.std(tomo_times, ddof=1)) if len(tomo_times) > 1 else 0.0,
        "optimization_time": t_opt1 - t_opt0,
        "tomo_purity": float(tomo["purity"]),
        "final_cost": float(res.fun),
    }

    data = {
        "w_init": w_init,
        "w_opt": w_opt,
        "cost_history": np.asarray(cost_history),
        "A_MATRIX": A_MATRIX,
        "B_VECTOR": B_VECTOR,
        "B_VECTOR_RAW": B_VECTOR_RAW,
        "x_classical": x_classical,
        "x_classical_raw": x_classical_raw,
        "x_classical_norm": x_classical_norm,
        "x_swap_norm": x_swap_norm,
        "x_tomo_norm": x_tomo_norm,
        "x_swap": x_swap,
        "x_tomo": x_tomo,
        "k_swap": np.asarray(k_swap),
        "k_tomo": np.asarray(k_tomo),
        "swap_prob_orig": swap["prob_orig"],
        "swap_sign_array": swap["sign_array"],
        "tomo_rho": tomo["rho"],
        "swap_times": np.asarray(swap_times),
        "tomo_times": np.asarray(tomo_times),
        "shots": np.asarray(shots),
        "steps": np.asarray(steps),
        "seed": np.asarray(seed),
        "metrics_json": np.asarray(json.dumps(metrics)),
    }
    return data, metrics


def json_safe(obj):
    """Convert common scientific/Python path objects into JSON-safe values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_results(data, metrics, results_file=DEFAULT_RESULTS_FILE, summary_file=DEFAULT_SUMMARY_FILE):
    np.savez(results_file, **data)
    summary = {
        "matrix_size": "4x4",
        "n_qubits": N_QUBITS,
        "num_pauli_terms": NUM_PAULI,
        "pauli_labels": PAULI_LABELS,
        "shots": int(np.asarray(data["shots"]).item()),
        "steps": int(np.asarray(data["steps"]).item()),
        "seed": int(np.asarray(data["seed"]).item()),
        "metrics": metrics,
        "results_file": str(results_file),
        "summary_file": str(summary_file),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_safe)
    print(f"\nSaved numerical results to: {results_file}")
    print(f"Saved summary to:           {summary_file}")


def load_results(results_file=DEFAULT_RESULTS_FILE):
    path = Path(results_file)
    if not path.exists():
        raise FileNotFoundError(f"Saved results not found: {results_file}. Run without --plot-only first.")
    loaded = np.load(path, allow_pickle=True)
    return {k: loaded[k] for k in loaded.files}


def as_real_array(x):
    return np.real_if_close(np.asarray(x)).real


def resolve_output_path(path_like, output_dir):
    """
    Return an output path. Relative filenames are placed inside output_dir;
    absolute paths are kept unchanged.
    """
    path = Path(path_like)
    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / path


def plot_signed_solution_values(data, fig_file=DEFAULT_SOLUTION_FIG):
    """One grouped bar chart: signed values of classical, SWAP, tomography."""
    x_classical = as_real_array(data["x_classical"])
    x_swap = as_real_array(data["x_swap"])
    x_tomo = as_real_array(data["x_tomo"])
    dim = len(x_classical)
    idx = np.arange(dim)
    width = 0.25

    metrics = json.loads(str(np.asarray(data["metrics_json"]).item())) if "metrics_json" in data else {}

    plt.figure(figsize=(11, 5))
    plt.axhline(0.0, linewidth=1.0)
    plt.bar(idx - width, x_classical, width, label="Classical")
    plt.bar(idx, x_swap, width, label=f"SWAP-parity, rel.err={metrics.get('swap_rel_error', np.nan):.2e}")
    plt.bar(idx + width, x_tomo, width, label=f"Tomography, rel.err={metrics.get('tomo_rel_error', np.nan):.2e}")

    for offset, values in [(-width, x_classical), (0, x_swap), (width, x_tomo)]:
        for i, val in enumerate(values):
            va = "bottom" if val >= 0 else "top"
            dy = 0.01 if val >= 0 else -0.01
            plt.text(i + offset, val + dy, f"{val:.3f}", ha="center", va=va, fontsize=8, rotation=0)

    plt.xticks(idx, [str(i) for i in idx])
    plt.xlabel("Basis index")
    plt.ylabel("Signed solution value")
    plt.title("Signed solution values: Classical vs SWAP-parity vs Tomography")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_file, dpi=140, bbox_inches="tight")
    print(f"Saved signed solution comparison figure to: {fig_file}")


def plot_cost_convergence(data, fig_file=DEFAULT_COST_FIG):
    cost_history = as_real_array(data["cost_history"])
    plt.figure(figsize=(8, 4.5))
    plt.plot(cost_history, "o-", linewidth=1.8, markersize=3)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Cost C_L")
    plt.title("Cost convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=140, bbox_inches="tight")
    print(f"Saved cost convergence figure to: {fig_file}")



def plot_recovery_runtime(data, fig_file=DEFAULT_RUNTIME_FIG):
    """Plot recovery runtime of SWAP-parity and tomography from saved data."""
    metrics = json.loads(str(np.asarray(data["metrics_json"]).item())) if "metrics_json" in data else {}

    swap_times = as_real_array(data.get("swap_times", np.array([])))
    tomo_times = as_real_array(data.get("tomo_times", np.array([])))

    # Prefer repeat timing arrays when available; otherwise fall back to main single run.
    if len(swap_times) > 0:
        swap_mean = float(np.mean(swap_times))
        swap_std = float(np.std(swap_times, ddof=1)) if len(swap_times) > 1 else 0.0
    else:
        swap_mean = float(metrics.get("swap_main_time", np.nan))
        swap_std = 0.0

    if len(tomo_times) > 0:
        tomo_mean = float(np.mean(tomo_times))
        tomo_std = float(np.std(tomo_times, ddof=1)) if len(tomo_times) > 1 else 0.0
    else:
        tomo_mean = float(metrics.get("tomo_main_time", np.nan))
        tomo_std = 0.0

    labels = ["SWAP-parity", "Tomography"]
    means = [swap_mean, tomo_mean]
    stds = [swap_std, tomo_std]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, means, yerr=stds, capsize=8)
    plt.ylabel("Recovery runtime (seconds)")
    plt.title("Recovery runtime comparison")
    plt.grid(True, axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y, f"{mean:.3f}s", ha="center", va="bottom", fontsize=10)

    footer = (
        f"Measured only after VQLS optimization. "
        f"shots={int(np.asarray(data['shots']).item())}, "
        f"repeats={max(len(swap_times), len(tomo_times))}."
    )
    plt.figtext(0.5, -0.02, footer, ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=140, bbox_inches="tight")
    print(f"Saved recovery runtime figure to: {fig_file}")


def print_report(data):
    metrics = json.loads(str(np.asarray(data["metrics_json"]).item())) if "metrics_json" in data else {}
    print("\n" + "=" * 70)
    print("SIGNED SOLUTION COMPARISON")
    print("=" * 70)
    print("Classical x:")
    print(as_real_array(data["x_classical"]))
    print("\nSWAP-parity recovered x:")
    print(as_real_array(data["x_swap"]))
    print("\nTomography recovered x:")
    print(as_real_array(data["x_tomo"]))
    print("\nOptimized weights w_opt:")
    print(as_real_array(data["w_opt"]))
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Load saved .npz and only redraw figures.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory used to save/load output files.")
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--summary-file", default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--solution-fig", default=DEFAULT_SOLUTION_FIG)
    parser.add_argument("--cost-fig", default=DEFAULT_COST_FIG)
    parser.add_argument("--runtime-fig", default=DEFAULT_RUNTIME_FIG)
    parser.add_argument("--shots", type=int, default=200000)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = resolve_output_path(args.results_file, output_dir)
    summary_file = resolve_output_path(args.summary_file, output_dir)
    solution_fig = resolve_output_path(args.solution_fig, output_dir)
    cost_fig = resolve_output_path(args.cost_fig, output_dir)
    runtime_fig = resolve_output_path(args.runtime_fig, output_dir)

    if args.plot_only:
        data = load_results(results_file)
        print(f"Loaded saved results from: {results_file}")
    else:
        data, metrics = run_full_experiment(
            shots=args.shots,
            steps=args.steps,
            timing_repeats=args.timing_repeats,
            seed=args.seed,
        )
        save_results(data, metrics, results_file, summary_file)

    print_report(data)
    plot_signed_solution_values(data, solution_fig)
    plot_cost_convergence(data, cost_fig)
    plot_recovery_runtime(data, runtime_fig)

    print(f"\nAll output files are in: {output_dir.resolve()}")
    print("\nTo redraw figures later without re-running VQLS:")
    print(
        f"    python {Path(__file__).name} --plot-only "
        f"--output-dir {output_dir} --results-file {results_file.name}"
    )


if __name__ == "__main__":
    main()
