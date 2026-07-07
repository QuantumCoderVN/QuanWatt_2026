"""
VQLS on Qiskit - Quantum Variational Linear Solver.

Phiên bản này:
1. Nhận ma trận A bất kỳ kích thước 2^n x 2^n.
2. Dùng Qiskit SparsePauliOp.from_operator để phân rã A sang Pauli basis.
3. Tự động dựng controlled-Pauli A_l trong Hadamard test.
4. Không còn hard-code C, A_PAULI, apply_CA cho từng Pauli term.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
import matplotlib.pyplot as plt
from qiskit.circuit.library import StatePreparation


# ============================================================
# Hyperparameters
# ============================================================
N_QUBITS_INPUT = 3          # số qubit của ma trận A: dim(A) = 2^N_QUBITS_INPUT
N_SHOTS = 10**6
STEPS = 20
ETA = 0.8
Q_DELTA = 0.001
RNG_SEED = 0

PAULI_ATOL = 1e-10
PAULI_RTOL = 1e-10

# Nếu A ngẫu nhiên dense thì số Pauli term có thể lên tới 4^n.
# Để None nếu muốn phân rã chính xác.
# Đặt ví dụ MAX_PAULI_TERMS = 12 để chạy thử nhanh nhưng khi đó A bị xấp xỉ.
MAX_PAULI_TERMS = None

# Ansatz:
# False: giữ ansatz cũ H + RY, phù hợp nghiệm gần thực.
# True : dùng RY + RZ + entanglement, phù hợp hơn với A bất kỳ/phức.
USE_COMPLEX_ANSATZ = True
N_LAYERS = 1

np.random.seed(RNG_SEED)


# ============================================================
# MODULE 0: Arbitrary matrix A and Qiskit Pauli decomposition
# ============================================================



def infer_n_qubits_from_matrix(A):
    """Kiểm tra A là ma trận vuông 2^n x 2^n và trả về n."""
    A = np.asarray(A, dtype=complex)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A_MATRIX phải là ma trận vuông.")

    dim = A.shape[0]
    n_qubits = int(np.log2(dim))

    if 2**n_qubits != dim:
        raise ValueError("Kích thước A_MATRIX phải là 2^n x 2^n.")

    return n_qubits, A


def pauli_decompose_matrix(A_matrix, atol=1e-10, rtol=1e-10, max_terms=None):
    """
    Phân rã A_matrix sang Pauli basis bằng Qiskit.

    A = sum_l C[l] * P_l

    Trả về:
        n_qubits
        A_matrix
        A_pauli
        pauli_labels
        coeffs
    """
    n_qubits, A_matrix = infer_n_qubits_from_matrix(A_matrix)

    op = Operator(
        A_matrix,
        input_dims=(2,) * n_qubits,
        output_dims=(2,) * n_qubits,
    )

    A_pauli = SparsePauliOp.from_operator(op, atol=atol, rtol=rtol)

    # Tùy chọn truncate theo hệ số lớn nhất để chạy thử nhanh.
    # Nếu max_terms=None thì giữ chính xác toàn bộ.
    if max_terms is not None and len(A_pauli.coeffs) > max_terms:
        idx = np.argsort(np.abs(A_pauli.coeffs))[::-1][:max_terms]
        A_pauli = SparsePauliOp(A_pauli.paulis[idx], A_pauli.coeffs[idx])

    pauli_labels = [p.to_label() for p in A_pauli.paulis]
    coeffs = np.asarray(A_pauli.coeffs, dtype=complex)

    return n_qubits, A_matrix, A_pauli, pauli_labels, coeffs



# # ============================================================
# # FDLS A_theta matrix: Bprime · Δθ = ΔP / V
# # ============================================================

# A_MATRIX = np.array([
#     [19.99664967, -4.78186315],
#     [-4.78186315,  4.75996315],
# ], dtype=complex)

# B_VECTOR_RAW = np.array([
#      0.21038092,
#     -0.89294760,
# ], dtype=complex)

# B_VECTOR_NORM = np.linalg.norm(B_VECTOR_RAW)
# B_VECTOR = B_VECTOR_RAW / B_VECTOR_NORM
# ============================================================
# OPF-like KKT/Newton linear system
# Form:
#     [ H   J^T ] [dx      ] = [ -grad_L ]
#     [ J    0  ] [dlambda ]   [ -g      ]
#
# Đây là ma trận dạng OPF/KKT, nhưng test này chỉ kiểm tra
# khả năng giải hệ tuyến tính A x = b của VQLS.
# ============================================================

# A_MATRIX = np.array([
#     [2.00, 0.30, 1.00, 0.25],
#     [0.30, 1.50, 0.35, 1.15],
#     [1.00, 0.35, 0.00, 0.00],
#     [0.25, 1.15, 0.00, 0.00],
# ], dtype=complex)

# B_VECTOR_RAW = np.array([
#     -0.12,
#      0.08,
#      0.03,
#     -0.04,
# ], dtype=complex)

# B_VECTOR_NORM = np.linalg.norm(B_VECTOR_RAW)
# B_VECTOR = B_VECTOR_RAW / B_VECTOR_NORM
# ============================================================
# OPF KKT linear system from MATPOWER case3sc
# A_OPF_KKT · x = b_OPF_rhs
# ============================================================

# A_MATRIX = np.array([
#     [   0.000000,    0.000000,   0.000000,  0.000000,  0.000000, -111.111111,  244.444444, -133.333333],
#     [   0.000000,    0.000000,   0.000000,  0.000000,  0.000000, -161.290323, -133.333333,  294.623656],
#     [   0.000000,    0.000000,   0.220000,  0.000000,  0.000000,   -1.000000,   -0.000000,   -0.000000],
#     [   0.000000,    0.000000,   0.000000,  0.170000,  0.000000,   -0.000000,   -1.000000,   -0.000000],
#     [   0.000000,    0.000000,   0.000000,  0.000000,  0.000000,   -0.000000,   -0.000000,   -1.000000],
#     [-111.111111, -161.290323,  -1.000000, -0.000000, -0.000000,    0.000000,    0.000000,    0.000000],
#     [ 244.444444, -133.333333,  -0.000000, -1.000000, -0.000000,    0.000000,    0.000000,    0.000000],
#     [-133.333333,  294.623656,  -0.000000, -0.000000, -1.000000,    0.000000,    0.000000,    0.000000],
# ], dtype=complex)

# B_VECTOR_RAW = np.array([
#     -0.000000,
#     -0.000000,
#     -5.000000,
#     -1.200000,
#     -0.000000,
#   -110.000000,
#   -110.000000,
#    -95.000000,
# ], dtype=complex)

# B_VECTOR_NORM = np.linalg.norm(B_VECTOR_RAW)
# B_VECTOR = B_VECTOR_RAW / B_VECTOR_NORM

# ============================================================
# 2x2 SPD test matrix
# Symmetric positive definite
# cond(A) ≈ 2.7836
# ============================================================

A_MATRIX = np.array([
    [4.0, 1.0],
    [1.0, 3.0],
], dtype=complex)

B_VECTOR_RAW = np.array([
    1.0,
    2.0,
], dtype=complex)

B_VECTOR_NORM = np.linalg.norm(B_VECTOR_RAW)
B_VECTOR = B_VECTOR_RAW / B_VECTOR_NORM
# Phân rã Pauli tự động bằng Qiskit
N_QUBITS, A_MATRIX, A_PAULI, PAULI_LABELS, C = pauli_decompose_matrix(
    A_MATRIX,
    atol=PAULI_ATOL,
    rtol=PAULI_RTOL,
    max_terms=MAX_PAULI_TERMS,
)

TOT_QUBITS = N_QUBITS + 1
ANCILLA_IDX = N_QUBITS
NUM_PAULI = len(C)

if USE_COMPLEX_ANSATZ:
    N_PARAMS = 2 * N_QUBITS * N_LAYERS
else:
    N_PARAMS = N_QUBITS


# ============================================================
# MODULE 1: Pauli decomposition and circuit components
# ============================================================
# def apply_U_b(qc, qubits):
#     """
#     Apply U_b = H⊗H⊗...⊗H.

#     Khi đó:
#         |b> = U_b |0> = uniform state.
#     """
#     for q in qubits:
#         qc.h(q)

def make_Ub_gate(b_vector):
    """
    Tạo gate U_b sao cho:
        U_b |0...0> = |b>
    """
    b_vector = np.asarray(b_vector, dtype=complex)
    b_vector = b_vector / np.linalg.norm(b_vector)

    return StatePreparation(b_vector)


U_B_GATE = make_Ub_gate(B_VECTOR)


def apply_U_b(qc, qubits, dagger=False):
    """
    Nếu dagger=False:
        apply U_b

    Nếu dagger=True:
        apply U_b†
    """
    if dagger:
        qc.append(U_B_GATE.inverse(), qubits)
    else:
        qc.append(U_B_GATE, qubits)

def apply_controlled_pauli_string(qc, pauli_label, qubits, ancilla):
    """
    Apply controlled-Pauli string theo convention của Qiskit.

    Qiskit convention:
        label bên phải ứng với qubit 0.
        label bên trái ứng với qubit n-1.

    Ví dụ:
        pauli_label = "IZX" với n=3
        X nằm trên qubit 0
        Z nằm trên qubit 1
        I nằm trên qubit 2
    """
    n = len(qubits)

    if len(pauli_label) != n:
        raise ValueError("Độ dài Pauli label không khớp số qubit.")

    for str_idx, p in enumerate(pauli_label):
        # Qiskit: character bên phải là qubit 0
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
    """
    Apply controlled-A_l, trong đó A_l là Pauli string thứ l
    lấy từ phân rã Pauli tự động của Qiskit.
    """
    apply_controlled_pauli_string(qc, PAULI_LABELS[l], qubits, ancilla)


# ============================================================
# MODULE 2: Variational ansatz
# ============================================================
def apply_variational(qc, params, qubits):
    """
    Ansatz.

    Nếu USE_COMPLEX_ANSATZ=False:
        Giữ ansatz cũ của bạn:
            H trên mỗi qubit, sau đó RY.

    Nếu USE_COMPLEX_ANSATZ=True:
        Dùng ansatz mạnh hơn:
            H -> nhiều layer RY/RZ + CX entanglement.
        Ansatz này biểu diễn được phase tốt hơn khi A là ma trận phức.
    """
    params = np.asarray(params, dtype=float)

    for q in qubits:
        qc.h(q)

    if not USE_COMPLEX_ANSATZ:
        if len(params) != len(qubits):
            raise ValueError(f"Ansatz cũ cần {len(qubits)} tham số.")
        for i, q in enumerate(qubits):
            qc.ry(params[i], q)
        return

    expected = 2 * len(qubits) * N_LAYERS
    if len(params) != expected:
        raise ValueError(f"Ansatz phức cần {expected} tham số, nhưng nhận {len(params)}.")

    k = 0
    for layer in range(N_LAYERS):
        for q in qubits:
            qc.ry(params[k], q)
            k += 1

        # for q in qubits:
        #     qc.rz(params[k], q)
        #     k += 1

        # Entanglement chain
        if layer < N_LAYERS - 1:
            for q1, q2 in zip(qubits[:-1], qubits[1:]):
                qc.cx(q1, q2)


# ============================================================
# MODULE 3: Hadamard test
# ============================================================
def hadamard_test(weights, l, lp, j, part):
    """
    Measure Re/Im of:

        <0| V† A_l† U_b Z_j U_b† A_lp V |0>

    Với Pauli string:
        A_l† = A_l

    Do gate tác động lên state theo thứ tự phải-sang-trái,
    muốn operator là A_l B_j A_lp thì trong circuit phải apply:
        A_lp -> B_j -> A_l
    """
    qc = QuantumCircuit(TOT_QUBITS)
    main_qubits = list(range(N_QUBITS))

    qc.h(ANCILLA_IDX)

    if part == "Im":
        qc.p(-np.pi / 2, ANCILLA_IDX)

    apply_variational(qc, weights, main_qubits)

    # Quan trọng: apply A_lp trước
    apply_CA(qc, lp, main_qubits, ANCILLA_IDX)

    # B_j = U_b Z_j U_b†
    # Ở đây U_b = H^n nên U_b† = U_b
    apply_U_b(qc, main_qubits, dagger=True)

    if j != -1:
        qc.cz(ANCILLA_IDX, main_qubits[j])

    apply_U_b(qc, main_qubits, dagger=False)

    # Quan trọng: apply A_l sau
    apply_CA(qc, l, main_qubits, ANCILLA_IDX)

    qc.h(ANCILLA_IDX)

    return qc


def measure_z_ancilla(weights, l, lp, j, part):
    """Measure expectation value của Z trên ancilla."""
    qc = hadamard_test(weights, l, lp, j, part)
    state = Statevector.from_instruction(qc)

    # Ancilla là qubit cuối cùng, nên trong Qiskit Pauli label,
    # Z nằm bên trái nhất.
    pauli_str = "Z" + "I" * N_QUBITS
    Z_obs = SparsePauliOp.from_list([(pauli_str, 1.0)])

    return state.expectation_value(Z_obs).real


def mu(weights, l, lp, j):
    """Compute μ = Re + i·Im."""
    re = measure_z_ancilla(weights, l, lp, j, "Re")
    im = measure_z_ancilla(weights, l, lp, j, "Im")
    return re + 1j * im


# ============================================================
# MODULE 4: Cost function
# ============================================================
def psi_norm(weights):
    """
    Compute:
        <ψ|ψ> = <x| A†A |x>
    """
    norm = 0.0 + 0.0j

    for l in range(NUM_PAULI):
        for lp in range(NUM_PAULI):
            norm += np.conj(C[l]) * C[lp] * mu(weights, l, lp, -1)

    return norm.real


def cost_local(weights):
    """Compute local cost function C_L."""
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
# MODULE 5: Optimization
# ============================================================
# def parameter_shift_gradient(w, cost_fn):
#     """Compute gradient bằng parameter-shift rule."""
#     grad = np.zeros_like(w)
#     shift = np.pi / 2

#     for i in range(len(w)):
#         wp = w.copy()
#         wp[i] += shift

#         wm = w.copy()
#         wm[i] -= shift

#         grad[i] = 0.5 * (cost_fn(wp) - cost_fn(wm))

#     return grad


def main():
    """Main optimization loop."""
    w_init = Q_DELTA * np.random.randn(N_PARAMS)

    print(f"Number of qubits inferred from A: {N_QUBITS}")
    print(f"Total qubits including ancilla:   {TOT_QUBITS}")
    print(f"Number of ansatz parameters:      {N_PARAMS}")
    print(f"Number of Pauli terms:            {NUM_PAULI}")
    print(f"Condition number of A:            {np.linalg.cond(A_MATRIX):.3e}")

    print(f"\nInitial parameters:\n{w_init}")

    print("\n" + "=" * 60)
    print("Pauli decomposition of A from Qiskit")
    print("=" * 60)
    print(A_PAULI)

    # Kiểm tra phân rã có khớp A không
    A_reconstructed = np.asarray(A_PAULI.to_matrix(), dtype=complex)
    decomp_err = np.linalg.norm(A_reconstructed - A_MATRIX)

    print(f"\nDecomposition check ||A_pauli - A_matrix|| = {decomp_err:.6e}")

    cost_history = []

    def cost_with_log(w):
        c = cost_local(w)
        cost_history.append(c)
        return c

    print("\n" + "=" * 60)
    print("Optimization using SciPy COBYLA")
    print("=" * 60)

    from scipy.optimize import minimize

    res = minimize(
        cost_with_log,
        w_init,
        method="COBYLA",
        options={
            "rhobeg": 0.5,
            "maxiter": STEPS,
            "catol": 1e-8,
        },
    )

    w = res.x

    print(f"\n[SciPy] Iterations: {len(cost_history)}, Final cost: {res.fun:.2e}")

    log_indices = [0, 5, 10, 20, len(cost_history) // 2, len(cost_history) - 1]
    seen = set()

    for i in log_indices:
        if 0 <= i < len(cost_history) and i not in seen:
            print(f"  Step {i:3d}: Cost = {cost_history[i]:.7f}")
            seen.add(i)

    print(f"\nOptimized parameters:\n{w}")
    print(f"Final cost: {res.fun:.2e}")

    # ============================================================
    # Classical comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("Comparison with classical solution")
    print("=" * 60)

    A_matrix = A_MATRIX
    dim = 2**N_QUBITS

    # Vì apply_U_b = H^n, vector b là uniform state.
    b_vector = B_VECTOR

    try:
        x_classical = np.linalg.solve(A_matrix, b_vector)
    except np.linalg.LinAlgError:
        print("A bị suy biến hoặc gần suy biến. Dùng least-squares thay cho solve.")
        x_classical = np.linalg.lstsq(A_matrix, b_vector, rcond=None)[0]

    x_norm = x_classical / np.linalg.norm(x_classical)
    c_probs = np.abs(x_norm) ** 2

    # State từ ansatz VQLS
    qc_state = QuantumCircuit(N_QUBITS)
    apply_variational(qc_state, w, list(range(N_QUBITS)))
    state = Statevector.from_instruction(qc_state)

    # Quan trọng:
    # Ở phiên bản này ta KHÔNG reverse qubit order sang PennyLane nữa.
    # A_MATRIX, A_PAULI và Statevector đều dùng cùng convention của Qiskit.
    x_prime_normalized = state.data

    # ============================================================
    # Solution Recovery via Scaling Factor k
    # ============================================================
    b_prime = A_matrix @ x_prime_normalized

    denom = np.vdot(b_prime, b_prime)
    if abs(denom) < 1e-12:
        k_coeff = 0.0
    else:
        k_coeff = np.vdot(b_prime, b_vector) / denom

    x_vqls_recovered = k_coeff * x_prime_normalized
    # ============================================================
    # Compare recovered VQLS solution with RAW FDLS classical solution
    # ============================================================

    # Vì B_VECTOR đã normalize, x_vqls_recovered hiện đang là nghiệm của:
    #     A · x = B_VECTOR_RAW / ||B_VECTOR_RAW||
    #
    # Muốn quay lại nghiệm gốc của FDLS:
    #     A · Δθ = B_VECTOR_RAW
    #
    # thì nhân lại với ||B_VECTOR_RAW||.
    x_vqls_recovered_raw = B_VECTOR_NORM * x_vqls_recovered

    # Nghiệm cổ điển của hệ FDLS raw
    x_classical_raw = np.linalg.solve(A_matrix, B_VECTOR_RAW)

    print(f"\nRecovered RAW FDLS solution from VQLS Δθ = ||b_raw|| · x_VQLS =")
    print(np.real_if_close(x_vqls_recovered_raw))

    print(f"\nClassical RAW FDLS solution Δθ =")
    print(np.real_if_close(x_classical_raw))

    raw_abs_err = np.linalg.norm(x_vqls_recovered_raw - x_classical_raw)
    raw_rel_err = raw_abs_err / np.linalg.norm(x_classical_raw)
    raw_residual = np.linalg.norm(A_matrix @ x_vqls_recovered_raw - B_VECTOR_RAW)

    print(f"\nRAW absolute error ||x_VQLS_raw - x_classical_raw|| = {raw_abs_err:.6e}")
    print(f"RAW relative error                              = {raw_rel_err:.6e}")
    print(f"RAW residual ||A·x_VQLS_raw - b_raw||          = {raw_residual:.6e}")
    print(f"\nb' = A · x'_VQLS     = {np.real_if_close(b_prime)}")
    print(f"b  input             = {np.real_if_close(b_vector)}")

    print(f"\nRecovery coefficient k = {k_coeff}")
    print(f"  |k|    = {np.abs(k_coeff):.8f}")
    print(f"  arg(k) = {np.angle(k_coeff):.8f} rad")

    print(f"\nRecovered VQLS solution x_VQLS = k·x' =")
    print(np.real_if_close(x_vqls_recovered))

    print(f"\nClassical solution x_classical =")
    print(np.real_if_close(x_classical))

    abs_err = np.linalg.norm(x_vqls_recovered - x_classical)
    rel_err = abs_err / np.linalg.norm(x_classical)

    print(f"\nAbsolute error ||x_VQLS - x_classical|| = {abs_err:.6e}")
    print(f"Relative error                         = {rel_err:.6e}")

    residual = np.linalg.norm(A_matrix @ x_vqls_recovered - b_vector)

    print(f"\nResidual check ||A·x_VQLS - b|| = {residual:.6e}")
    print(f"  A·x_VQLS = {np.real_if_close(A_matrix @ x_vqls_recovered)}")
    print(f"  b        = {np.real_if_close(b_vector)}")

    print("\n" + "=" * 70)
    print("SOLUTION RECOVERY COMPLETE")
    print("=" * 70)

    q_probs = np.abs(x_prime_normalized) ** 2

    print(f"\n{'Index':<8}{'Classical':<15}{'Quantum':<15}{'|Diff|':<12}")

    for i in range(dim):
        print(
            f"{i:<8}"
            f"{c_probs[i]:<15.6f}"
            f"{q_probs[i]:<15.6f}"
            f"{abs(c_probs[i] - q_probs[i]):<12.6f}"
        )

    fidelity = np.sum(np.sqrt(c_probs * q_probs)) ** 2
    print(f"\nFidelity: {fidelity:.6f}")
    

    state_fidelity = np.abs(np.vdot(x_norm, x_prime_normalized)) ** 2
    print(f"State fidelity |<x_classical_norm|x_vqls>|^2 = {state_fidelity:.8f}")

    def exact_global_cost(weights):
        qc_state = QuantumCircuit(N_QUBITS)
        apply_variational(qc_state, weights, list(range(N_QUBITS)))
        x = Statevector.from_instruction(qc_state).data

        b = B_VECTOR
        psi = A_MATRIX @ x

        numerator = np.abs(np.vdot(b, psi)) ** 2
        denominator = np.vdot(psi, psi).real

        return 1.0 - numerator / denominator
    print(f"Exact global cost = {exact_global_cost(w):.8e}")

    # ============================================================
    # Plotting
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(cost_history, "g-o", linewidth=2, markersize=4)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cost C_L")
    axes[0].set_yscale("log")
    axes[0].set_title("Cost convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(dim), c_probs, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Basis state")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Classical solution")

    axes[2].bar(range(dim), q_probs, color="seagreen", edgecolor="black")
    axes[2].set_xlabel("Basis state")
    axes[2].set_ylabel("Probability")
    axes[2].set_title("VQLS solution")

    plt.tight_layout()
    plt.savefig("vqls_results.png", dpi=120, bbox_inches="tight")
    print("\nPlot saved to: vqls_results.png")

    return w, cost_history, c_probs, q_probs


if __name__ == "__main__":
    main()