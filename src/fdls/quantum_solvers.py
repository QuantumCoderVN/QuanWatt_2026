# ============================================================
# quantum_solvers.py
# Classical, HHL, VQLS solvers
# ============================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, RYGate, StatePreparation
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator

from config import (
    HHL_PHASE_QUBITS,
    HHL_PHASE_TARGET,
    HHL_C,
    HHL_DEBUG_COMPARE_CLASSICAL,
    PRINT_SOLVER_DETAIL,
    VQLS_STEPS,
    VQLS_RHOBEG,
    VQLS_Q_DELTA,
    VQLS_RNG_SEED,
    PAULI_ATOL,
    PAULI_RTOL,
    MAX_PAULI_TERMS,
    USE_COMPLEX_ANSATZ,
    VQLS_LAYERS,
)


# ============================================================
# COMMON UTILS
# ============================================================

def clean_real_vector(x, tol=1e-8):
    x = np.asarray(x)

    if np.max(np.abs(np.imag(x))) > tol:
        print("[WARN] nghiệm có phần ảo đáng kể:")
        print(x)

    return np.real(x)


def pad_to_power_of_two(A, b):
    """
    Quantum state cần kích thước 2^n.

    Nếu A là 1x1, pad thành 2x2:
        [[a, 0],
         [0, 1]]

    Nếu A là 2x2, giữ nguyên.

    Sau khi giải, lấy lại n phần tử đầu.
    """

    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A phải là ma trận vuông.")

    if len(b) != n:
        raise ValueError("Kích thước b không khớp A.")

    dim = 1
    while dim < n:
        dim *= 2

    # HHL code của bạn dùng ít nhất 1 target qubit => dim tối thiểu 2
    if dim < 2:
        dim = 2

    A_pad = np.eye(dim, dtype=complex)
    b_pad = np.zeros(dim, dtype=complex)

    A_pad[:n, :n] = A
    b_pad[:n] = b

    return A_pad, b_pad, n


def recover_scaled_solution(A, b, x_prime):
    """
    Khôi phục scale giống đúng code VQLS của bạn:

        b_prime = A @ x_prime
        k = <b_prime, b> / <b_prime, b_prime>
        x = k * x_prime
    """

    b_prime = A @ x_prime
    denom = np.vdot(b_prime, b_prime)

    if abs(denom) < 1e-12:
        k = 0.0
    else:
        k = np.vdot(b_prime, b) / denom

    x_recovered = k * x_prime

    return x_recovered, k

def gershgorin_lambda_bound(A):
    """
    Cận trên trị riêng theo Gershgorin.

    Không tính eigenvalue thật.
    Phù hợp hơn với tinh thần HHL.
    """

    A = np.asarray(A, dtype=complex)

    diag_abs = np.abs(np.diag(A))
    row_sum_abs = np.sum(np.abs(A), axis=1)

    radii = row_sum_abs - diag_abs
    bound = np.max(diag_abs + radii)

    if bound <= 1e-12:
        raise ValueError("Gershgorin bound quá nhỏ, không thể scale HHL.")

    return float(bound)

def classical_solve(A, b, label="Classical"):
    """
    Classical solver:
        A x = b

    Thêm label để đồng bộ signature với HHL và VQLS:
        solver(A, b, label=...)
    """

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    print(f"\n[{label}] Classical solution:")
    print(np.real_if_close(x))

    print(f"[{label}] residual ||A x - b|| = {np.linalg.norm(A @ x - b):.6e}")

    return clean_real_vector(x)


# ============================================================
# HHL - theo code HHL của bạn
# ============================================================

def controlled_unitary_power(qc, U, power, control_qubit, target_qubits):
    """
    Giữ đúng cấu trúc code HHL của bạn:
        U_power = matrix_power(U, power)
        gate.unitary(U_power)
        gate.to_gate().control(1)
    """

    U_power = np.linalg.matrix_power(U, power)

    gate = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    gate.unitary(U_power, list(range(len(target_qubits))))

    controlled_gate = gate.to_gate().control(1)
    qc.append(controlled_gate, [control_qubit] + list(target_qubits))


def qpe_circuit(qc, phase_reg, target_reg, U_matrix):
    """
    Giữ đúng QPE của code bạn:
    - H trên phase register
    - controlled-U^{2^k}
    - inverse QFT
    """

    for w in phase_reg:
        qc.h(w)

    for i, ctrl in enumerate(reversed(phase_reg)):
        power = 2 ** i
        controlled_unitary_power(qc, U_matrix, power, ctrl, target_reg)

    iqft = QFT(num_qubits=len(phase_reg), inverse=True, do_swaps=False)
    qc.append(iqft, phase_reg)


def control_rotation_gate(qc, control_reg, target_qubit, t, C=1.0):
    """
    Giữ đúng controlled RY của code bạn:
        phi từ bitstring
        nếu phi > 0.5 thì phi -= 1
        lambda = 2*pi/t * phi
        amp = C/lambda
        theta = 2*arcsin(amp)
    """

    n = len(control_reg)

    for d in range(1, 2**n):
        bin_str = f"{d:0{n}b}"

        phi = sum(int(bit) * 2 ** (-(j + 1)) for j, bit in enumerate(bin_str))

        if phi > 0.5:
            phi = phi - 1.0

        lam = (2 * np.pi / t) * phi

        if abs(lam) < 1e-12:
            continue

        amp = C / lam

        if abs(amp) > 1:
            continue

        theta = 2 * np.arcsin(amp)

        cry = RYGate(theta).control(n, ctrl_state=bin_str)
        qc.append(cry, list(control_reg) + [target_qubit])


def hhl_solve(A, b, label="HHL"):
    """
    HHL thật theo code của bạn:
        |b>
        QPE
        controlled RY
        inverse QPE
        lấy amplitude ancilla = 1, phase = 0
        recover scale bằng k
    """

    A_pad, b_pad, n_original = pad_to_power_of_two(A, b)

    if np.linalg.norm(b_pad) < 1e-14:
        return np.zeros(n_original)

    dim = A_pad.shape[0]
    n_qubits = int(np.log2(dim))
    phase_qubits = HHL_PHASE_QUBITS
    ancilla_qubit = 1
    total_qubits = n_qubits + phase_qubits + ancilla_qubit

    b_norm = b_pad / np.linalg.norm(b_pad)

    lambda_bound = gershgorin_lambda_bound(A_pad)

    t_eff = HHL_PHASE_TARGET * 2 * np.pi / lambda_bound

    U_matrix = expm(1j * A_pad * t_eff)

    if PRINT_SOLVER_DETAIL:
        print(f"\n[{label}] HHL scaling diagnostic")
        print("Gershgorin lambda bound:")
        print(lambda_bound)
        print("Effective HHL t:")
        print(t_eff)
        print("Max phase upper bound:")
        print(lambda_bound * t_eff / (2 * np.pi))
        print("QPE resolution:")
        print(1 / (2 ** phase_qubits))

    assert np.allclose(
        U_matrix.conj().T @ U_matrix,
        np.eye(dim),
        atol=1e-8
    ), "U không unitary."

    phase_reg = QuantumRegister(phase_qubits, name="phase")
    target_reg = QuantumRegister(n_qubits, name="target")
    ancilla_reg = QuantumRegister(ancilla_qubit, name="ancilla")

    qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL")

    phase_indices = list(range(phase_qubits))
    target_indices = list(range(phase_qubits, phase_qubits + n_qubits))
    ancilla_index = phase_qubits + n_qubits

    # Step 1: initialize |b>
    qc_hhl.initialize(b_norm, target_indices)

    # Step 2: QPE
    qpe_circuit(qc_hhl, phase_indices, target_indices, U_matrix)

    # Step 3: controlled RY
    control_rotation_gate(
        qc_hhl,
        phase_indices,
        ancilla_index,
        t=t_eff,
        C=HHL_C
    )

    # Step 4: inverse QPE
    qc_qpe_only = QuantumCircuit(phase_qubits + n_qubits, name="QPE")
    qpe_circuit(
        qc_qpe_only,
        list(range(phase_qubits)),
        list(range(phase_qubits, phase_qubits + n_qubits)),
        U_matrix
    )

    inv_qpe = qc_qpe_only.inverse()
    inv_qpe.name = "QPE†"

    qc_hhl.append(inv_qpe, phase_indices + target_indices)

    # Statevector simulation
    sv = Statevector.from_instruction(qc_hhl)

    # Lấy amplitude tại:
    # phase = 0...0
    # ancilla = 1
    # target = từng basis state
    #
    # Qiskit little-endian:
    # index = phase_bits + target_bits << phase_qubits + ancilla << (...)
    raw = np.zeros(dim, dtype=complex)

    for target_state in range(dim):
        idx = 0
        idx |= target_state << phase_qubits
        idx |= 1 << (phase_qubits + n_qubits)

        raw[target_state] = sv.data[idx]

    if np.linalg.norm(raw) < 1e-14:
        print(f"[WARN] {label}: HHL hậu chọn ancilla=1 gần zero.")
        return np.zeros(n_original)

    # Recover scale theo cùng công thức k
    x_recovered_pad, k = recover_scaled_solution(A_pad, b_pad, raw)

    x = x_recovered_pad[:n_original]

    print(f"\n[{label}] HHL raw amplitude ancilla=1:")
    print(np.real_if_close(raw))

    print(f"[{label}] HHL recovery coefficient k:")
    print(k)

    print(f"[{label}] HHL recovered solution:")
    print(np.real_if_close(x))

    print(f"[{label}] residual ||A x - b|| = {np.linalg.norm(A @ x - b):.6e}")
    x_classical = np.linalg.solve(A, b)

    print(f"[{label}] Classical reference solution:")
    print(np.real_if_close(x_classical))

    print(f"[{label}] HHL error ||x_hhl - x_classical||:")
    print(np.linalg.norm(x - x_classical))

    print(f"[{label}] HHL relative error:")
    print(np.linalg.norm(x - x_classical) / np.linalg.norm(x_classical))
    return clean_real_vector(x)


# ============================================================
# VQLS - theo code VQLS của bạn
# ============================================================

def infer_n_qubits_from_matrix(A):
    A = np.asarray(A, dtype=complex)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A_MATRIX phải là ma trận vuông.")

    dim = A.shape[0]
    n_qubits = int(np.log2(dim))

    if 2**n_qubits != dim:
        raise ValueError("Kích thước A_MATRIX phải là 2^n x 2^n.")

    return n_qubits, A


def pauli_decompose_matrix(A_matrix, atol=1e-10, rtol=1e-10, max_terms=None):
    n_qubits, A_matrix = infer_n_qubits_from_matrix(A_matrix)

    op = Operator(
        A_matrix,
        input_dims=(2,) * n_qubits,
        output_dims=(2,) * n_qubits,
    )

    A_pauli = SparsePauliOp.from_operator(op, atol=atol, rtol=rtol)

    if max_terms is not None and len(A_pauli.coeffs) > max_terms:
        idx = np.argsort(np.abs(A_pauli.coeffs))[::-1][:max_terms]
        A_pauli = SparsePauliOp(A_pauli.paulis[idx], A_pauli.coeffs[idx])

    pauli_labels = [p.to_label() for p in A_pauli.paulis]
    coeffs = np.asarray(A_pauli.coeffs, dtype=complex)

    return n_qubits, A_matrix, A_pauli, pauli_labels, coeffs


def vqls_solve(A, b, label="VQLS"):
    """
    VQLS theo đúng cấu trúc code của bạn:
    - A bất kỳ kích thước 2^n x 2^n
    - SparsePauliOp.from_operator
    - StatePreparation(b)
    - controlled Pauli string
    - H/Ry/Rz/CX ansatz
    - Hadamard test
    - local cost
    - COBYLA
    - recover x = k x'
    """

    A_pad, b_pad, n_original = pad_to_power_of_two(A, b)

    if np.linalg.norm(b_pad) < 1e-14:
        return np.zeros(n_original)

    b_pad = b_pad.astype(complex)

    N_QUBITS, A_MATRIX, A_PAULI, PAULI_LABELS, C = pauli_decompose_matrix(
        A_pad,
        atol=PAULI_ATOL,
        rtol=PAULI_RTOL,
        max_terms=MAX_PAULI_TERMS,
    )

    B_VECTOR = b_pad / np.linalg.norm(b_pad)

    TOT_QUBITS = N_QUBITS + 1
    ANCILLA_IDX = N_QUBITS
    NUM_PAULI = len(C)

    if USE_COMPLEX_ANSATZ:
        N_PARAMS = 2 * N_QUBITS * VQLS_LAYERS
    else:
        N_PARAMS = N_QUBITS

    U_B_GATE = StatePreparation(B_VECTOR)

    def apply_U_b(qc, qubits, dagger=False):
        if dagger:
            qc.append(U_B_GATE.inverse(), qubits)
        else:
            qc.append(U_B_GATE, qubits)

    def apply_controlled_pauli_string(qc, pauli_label, qubits, ancilla):
        n = len(qubits)

        if len(pauli_label) != n:
            raise ValueError("Độ dài Pauli label không khớp số qubit.")

        for str_idx, p in enumerate(pauli_label):
            q = qubits[n - 1 - str_idx]

            if p == "I":
                continue
            elif p == "X":
                qc.cx(ancilla, q)
            elif p == "Y":
                qc.cy(ancilla, q)
            elif p == "Z":
                qc.cz(ancilla, q)
            else:
                raise ValueError(f"Pauli không hợp lệ: {p}")

    def apply_CA(qc, l, qubits, ancilla):
        apply_controlled_pauli_string(qc, PAULI_LABELS[l], qubits, ancilla)

    def apply_variational(qc, params, qubits):
        params = np.asarray(params, dtype=float)

        for q in qubits:
            qc.h(q)

        if not USE_COMPLEX_ANSATZ:
            if len(params) != len(qubits):
                raise ValueError(f"Ansatz cũ cần {len(qubits)} tham số.")

            for i, q in enumerate(qubits):
                qc.ry(params[i], q)

            return

        expected = 2 * len(qubits) * VQLS_LAYERS

        if len(params) != expected:
            raise ValueError(
                f"Ansatz phức cần {expected} tham số, nhưng nhận {len(params)}."
            )

        k = 0

        for layer in range(VQLS_LAYERS):
            for q in qubits:
                qc.ry(params[k], q)
                k += 1

            for q in qubits:
                qc.rz(params[k], q)
                k += 1

            if layer < VQLS_LAYERS - 1:
                for q1, q2 in zip(qubits[:-1], qubits[1:]):
                    qc.cx(q1, q2)

    def hadamard_test(weights, l, lp, j, part):
        qc = QuantumCircuit(TOT_QUBITS)
        main_qubits = list(range(N_QUBITS))

        qc.h(ANCILLA_IDX)

        if part == "Im":
            qc.p(-np.pi / 2, ANCILLA_IDX)

        apply_variational(qc, weights, main_qubits)

        # apply A_lp trước
        apply_CA(qc, lp, main_qubits, ANCILLA_IDX)

        # B_j = U_b Z_j U_b†
        apply_U_b(qc, main_qubits, dagger=True)

        if j != -1:
            qc.cz(ANCILLA_IDX, main_qubits[j])

        apply_U_b(qc, main_qubits, dagger=False)

        # apply A_l sau
        apply_CA(qc, l, main_qubits, ANCILLA_IDX)

        qc.h(ANCILLA_IDX)

        return qc

    def measure_z_ancilla(weights, l, lp, j, part):
        qc = hadamard_test(weights, l, lp, j, part)
        state = Statevector.from_instruction(qc)

        pauli_str = "Z" + "I" * N_QUBITS
        Z_obs = SparsePauliOp.from_list([(pauli_str, 1.0)])

        return state.expectation_value(Z_obs).real

    def mu(weights, l, lp, j):
        re = measure_z_ancilla(weights, l, lp, j, "Re")
        im = measure_z_ancilla(weights, l, lp, j, "Im")
        return re + 1j * im

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

    rng = np.random.default_rng(VQLS_RNG_SEED)
    w_init = VQLS_Q_DELTA * rng.standard_normal(N_PARAMS)

    cost_history = []

    def cost_with_log(w):
        c = cost_local(w)
        cost_history.append(c)
        return c

    print(f"\n[{label}] VQLS information")
    print(f"N_QUBITS:        {N_QUBITS}")
    print(f"NUM_PAULI:       {NUM_PAULI}")
    print(f"N_PARAMS:        {N_PARAMS}")
    print(f"cond(A):         {np.linalg.cond(A_MATRIX):.6e}")
    print(f"Pauli decomp:")
    print(A_PAULI)

    res = minimize(
        cost_with_log,
        w_init,
        method="COBYLA",
        options={
            "rhobeg": VQLS_RHOBEG,
            "maxiter": VQLS_STEPS,
            "catol": 1e-8,
        },
    )

    w = res.x

    qc_state = QuantumCircuit(N_QUBITS)
    apply_variational(qc_state, w, list(range(N_QUBITS)))
    state = Statevector.from_instruction(qc_state)

    x_prime_normalized = state.data

    x_recovered_pad, k_coeff = recover_scaled_solution(
        A_MATRIX,
        b_pad,
        x_prime_normalized
    )

    x = x_recovered_pad[:n_original]

    print(f"\n[{label}] Final VQLS cost:")
    print(res.fun)

    print(f"[{label}] Optimized parameters:")
    print(w)

    print(f"[{label}] x_prime_normalized:")
    print(np.real_if_close(x_prime_normalized))

    print(f"[{label}] Recovery coefficient k:")
    print(k_coeff)

    print(f"[{label}] Recovered VQLS solution:")
    print(np.real_if_close(x))

    print(f"[{label}] residual ||A x - b|| = {np.linalg.norm(A @ x - b):.6e}")

    return clean_real_vector(x)