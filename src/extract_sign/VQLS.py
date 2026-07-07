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
STEPS = 150
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
N_LAYERS = 3

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
# 4x4 signed test matrix for SWAP-parity recovery
# Symmetric real matrix, dim = 2^2, cond(A) moderate
#
# Designed raw solution:
#     x_raw = [0.4, -0.3, 0.2, -0.1]
#
# Therefore this is a good test for sign extraction.
# ============================================================

A_MATRIX = np.array([
    [4.0,  0.8,  0.3, -0.2],
    [0.8,  3.5, -0.4,  0.5],
    [0.3, -0.4,  2.8,  0.7],
    [-0.2, 0.5,  0.7,  3.2],
], dtype=complex)

B_VECTOR_RAW = np.array([
    1.30,
   -0.62,
    0.70,
   -0.31,
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
    N_PARAMS = N_QUBITS * N_LAYERS
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
    Ansatz thực RY-only.

    Mạch:
        H trên mỗi qubit
        sau đó mỗi layer:
            RY trên từng qubit
            CX entanglement chain giữa các layer

    Số tham số:
        N_PARAMS = N_QUBITS * N_LAYERS

    Ansatz này phù hợp với phương pháp khôi phục dấu ± vì state tạo ra là real.
    """
    params = np.asarray(params, dtype=float)

    for q in qubits:
        qc.h(q)

    expected = len(qubits) * N_LAYERS

    if len(params) != expected:
        raise ValueError(
            f"Ansatz RY-only cần {expected} tham số, nhưng nhận {len(params)}."
        )

    k = 0

    for layer in range(N_LAYERS):
        for q in qubits:
            qc.ry(params[k], q)
            k += 1

        # Entanglement chain giữa các layer.
        # Với N_LAYERS = 3, CX sẽ được thêm sau layer 0 và layer 1.
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

# ============================================================
# MODULE 5B: Shot-based SWAP + Hadamard sign extraction
# ============================================================

def cyclic_left_shift_matrix_qiskit(dim):
    """
    Unitary U_shift sao cho nếu state có amplitude c[i],
    output có amplitude d[i] = c[(i + 1) mod dim].

    Đây là phép dịch vòng trái trên vector nghiệm theo thứ tự index
    của Qiskit Statevector/counts:
        index i tương ứng bitstring q[n-1]...q[0].
    """
    U_shift = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        U_shift[i, (i + 1) % dim] = 1.0

    return U_shift


def build_vqls_probability_circuit(weights):
    """
    Circuit đo p_i = |c_i|^2 của ansatz VQLS.

    Không đọc nghiệm bằng Statevector.data.
    Khi chạy thực tế, thêm measurement rồi lấy counts.
    """
    qc = QuantumCircuit(N_QUBITS)
    apply_variational(qc, weights, list(range(N_QUBITS)))
    return qc


def build_swap_parity_circuit(weights, shifted=False):
    """
    Circuit SWAP + Hadamard để lấy dấu tương đối.

    Qiskit convention:
        - index i của vector dùng bitstring q[n-1]...q[0]
        - c_0 và c_1 khác nhau ở qubit 0

    Vì vậy để tạo giao thoa cho các cặp:
        (c0,c1), (c2,c3), ...
    ta SWAP qubit 0 với readout qubit.

    Nếu shifted=True:
        áp dụng cyclic left shift trước, nên các cặp trở thành:
        (c1,c2), (c3,c4), ..., (c[L-1],c0)
    """
    readout = N_QUBITS
    main_qubits = list(range(N_QUBITS))
    dim = 2**N_QUBITS

    qc = QuantumCircuit(N_QUBITS + 1)

    # Chuẩn bị nghiệm VQLS |x(theta)>
    apply_variational(qc, weights, main_qubits)

    # Dịch vòng vector nếu cần
    if shifted:
        qc.unitary(
            cyclic_left_shift_matrix_qiskit(dim),
            main_qubits,
            label="cyclic_left_shift",
        )

    # Qiskit: c_{2k} và c_{2k+1} khác nhau ở q0
    qc.swap(0, readout)

    # Hadamard tạo giao thoa:
    # xác suất readout=0, q0=0 chứa |c_{2k} + c_{2k+1}|^2 / 2
    qc.h(readout)

    return qc


def add_measure_all(qc):
    """Trả về bản copy có measurement toàn bộ qubit."""
    qc_meas = qc.copy()
    qc_meas.measure_all()
    return qc_meas


def sample_counts_from_statevector_for_debug(qc, shots=N_SHOTS, seed=RNG_SEED):
    """
    Fallback mô phỏng shot nếu chưa truyền backend thật.

    Lưu ý:
        Hàm này chỉ dùng Statevector để SINH counts giả lập.
        Không dùng Statevector để đọc trực tiếp nghiệm.

    Khi chạy hardware hoặc Aer backend thật, truyền backend vào run_counts().
    """
    state = Statevector.from_instruction(qc)
    probs = np.abs(state.data) ** 2
    probs = probs / np.sum(probs)

    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=int(shots), p=probs)

    n_bits = qc.num_qubits
    counts = {}

    for idx in samples:
        bitstring = format(int(idx), f"0{n_bits}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


def run_counts(qc, shots=N_SHOTS, backend=None, seed=RNG_SEED):
    """
    Chạy circuit và trả về counts.

    Nếu backend=None:
        dùng AerSimulator để đo bằng shot.

    Nếu backend != None:
        chạy trên backend bạn truyền vào.
    """
    qc_meas = add_measure_all(qc)

    if backend is None:
        try:
            from qiskit_aer import AerSimulator
        except ImportError:
            raise ImportError(
                "Bạn cần cài qiskit-aer để chạy shot-based simulation:\n"
                "    pip install qiskit-aer\n"
                "hoặc truyền backend thật vào run_counts(..., backend=backend)."
            )

        backend = AerSimulator(seed_simulator=seed)

    job = backend.run(qc_meas, shots=int(shots))
    return job.result().get_counts()


def counts_to_probability_vector(counts, n_bits):
    """
    Đổi counts dạng {'q[n-1]...q[0]': count} sang vector xác suất.

    Với Qiskit counts, int(bitstring, 2) khớp trực tiếp với index vector
    nếu dùng cùng convention q[n-1]...q[0].
    """
    probs = np.zeros(2**n_bits, dtype=float)
    total = sum(counts.values())

    if total == 0:
        raise ValueError("Counts rỗng, không thể suy ra xác suất.")

    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        idx = int(clean, 2)
        probs[idx] += count / total

    return probs


def parity_interference_from_counts(counts, n_main):
    """
    Từ counts của SWAP-parity circuit, lấy:

        pc_k ≈ |c_{2k} + c_{2k+1}|^2

    Sau SWAP(q0, readout) + H(readout), xác suất của trạng thái
    readout=0 và main_index=2k là:

        1/2 * |c_{2k} + c_{2k+1}|^2

    Do đó nhân 2 để thu được pc_k.
    """
    full_probs = counts_to_probability_vector(counts, n_main + 1)
    L = 2**n_main

    pc = np.zeros(L // 2, dtype=float)

    for k in range(L // 2):
        # full_index = readout * 2^n + main_index
        # chọn readout=0 và main_index=2k
        full_index = 2 * k
        pc[k] = 2.0 * full_probs[full_index]

    return pc


def recover_signs_from_swap_parity_counts(
    prob_orig,
    counts_parity_unshifted,
    counts_parity_shifted,
    first_sign=1,
    zero_tol=1e-12,
):
    """
    Khôi phục dấu biên độ thực từ shot probabilities/counts.

    prob_orig:
        p_i = |c_i|^2, lấy từ counts của circuit đo ansatz gốc.

    counts_parity_unshifted:
        counts từ mạch SWAP-parity chưa dịch vòng,
        dùng cho các cặp (c0,c1), (c2,c3), ...

    counts_parity_shifted:
        counts từ mạch SWAP-parity sau cyclic left shift,
        dùng cho các cặp (c1,c2), (c3,c4), ...

    first_sign:
        dấu neo của phần tử đầu tiên, dùng để cố định global sign.
    """
    prob_orig = np.asarray(prob_orig, dtype=float)
    L = len(prob_orig)

    if L == 0 or (L & (L - 1)) != 0:
        raise ValueError("prob_orig phải có độ dài là lũy thừa của 2.")

    parity_couple_0 = parity_interference_from_counts(
        counts_parity_unshifted,
        N_QUBITS,
    )

    parity_couple_1 = parity_interference_from_counts(
        counts_parity_shifted,
        N_QUBITS,
    )

    # Mạch chưa shift: lấy dấu tương đối của (c0,c1), (c2,c3), ...
    parity_array_0 = []

    for i, pc in enumerate(parity_couple_0):
        a = prob_orig[2 * i]
        b = prob_orig[2 * i + 1]

        if a < zero_tol or b < zero_tol:
            # Nếu amplitude gần 0 thì dấu tương đối không ổn định.
            # Chọn +1 để tránh nhiễu shot làm sai dây chuyền.
            parity_array_0.append(1)
        elif pc < a or pc < b:
            parity_array_0.append(-1)
        else:
            parity_array_0.append(1)

    # Mạch shifted: prob gốc sau dịch trái là [p1,p2,...,p0]
    rotated_prob_orig = np.roll(prob_orig, -1)

    parity_array_1 = []

    for i, pc in enumerate(parity_couple_1):
        a = rotated_prob_orig[2 * i]
        b = rotated_prob_orig[2 * i + 1]

        if a < zero_tol or b < zero_tol:
            parity_array_1.append(1)
        elif pc < a or pc < b:
            parity_array_1.append(-1)
        else:
            parity_array_1.append(1)

    # Ghép xen kẽ:
    #   s0*s1, s1*s2, s2*s3, ...
    parity_intertwined = []

    for i in range(max(len(parity_array_0), len(parity_array_1))):
        if i < len(parity_array_0):
            parity_intertwined.append(parity_array_0[i])
        if i < len(parity_array_1):
            parity_intertwined.append(parity_array_1[i])

    if len(parity_intertwined) != L:
        raise ValueError(
            f"Sai số lượng parity: cần {L}, nhận {len(parity_intertwined)}."
        )

    first = 1 if first_sign >= 0 else -1

    signs = []
    s = first

    for i in range(L):
        signs.append(s)
        s = s * parity_intertwined[i]

    cycle_consistent = s == first

    if not cycle_consistent:
        print(
            "Warning: cyclic parity check failed; "
            "có thể do shot noise hoặc nghiệm có phần tử gần 0."
        )

    signs = np.asarray(signs, dtype=int)

    signed_normalized_state = signs * np.sqrt(np.maximum(prob_orig, 0.0))

    return {
        "parity_couple_0": parity_couple_0,
        "parity_couple_1": parity_couple_1,
        "parity_array_0": parity_array_0,
        "parity_array_1": parity_array_1,
        "parity_intertwined": parity_intertwined,
        "sign_array": signs,
        "signed_normalized_state": signed_normalized_state,
        "cycle_consistent": cycle_consistent,
    }


def extract_signed_vqls_solution_shot_based(
    weights,
    first_sign=1,
    shots=N_SHOTS,
    backend=None,
    seed=RNG_SEED,
):
    """
    Module tổng hợp:

        1. Đo p_i = |c_i|^2 bằng shot.
        2. Đo SWAP-parity chưa shift.
        3. Đo SWAP-parity sau cyclic shift.
        4. Khôi phục dấu.
        5. Trả về signed normalized VQLS state.

    Đây là phần có thể lắp thay cho cách đọc Statevector trực tiếp.
    """
    qc_prob = build_vqls_probability_circuit(weights)
    qc_parity_0 = build_swap_parity_circuit(weights, shifted=False)
    qc_parity_1 = build_swap_parity_circuit(weights, shifted=True)

    counts_prob = run_counts(
        qc_prob,
        shots=shots,
        backend=backend,
        seed=seed,
    )

    counts_parity_0 = run_counts(
        qc_parity_0,
        shots=shots,
        backend=backend,
        seed=seed + 1,
    )

    counts_parity_1 = run_counts(
        qc_parity_1,
        shots=shots,
        backend=backend,
        seed=seed + 2,
    )

    prob_orig = counts_to_probability_vector(counts_prob, N_QUBITS)

    sign_data = recover_signs_from_swap_parity_counts(
        prob_orig=prob_orig,
        counts_parity_unshifted=counts_parity_0,
        counts_parity_shifted=counts_parity_1,
        first_sign=first_sign,
    )

    return {
        "counts_prob": counts_prob,
        "counts_parity_unshifted": counts_parity_0,
        "counts_parity_shifted": counts_parity_1,
        "prob_orig": prob_orig,
        **sign_data,
    }


def scale_signed_vqls_state_to_solution(
    signed_normalized_state,
    A_matrix,
    b_vector,
):
    """
    Dùng cùng logic scale k của code cũ, nhưng đầu vào là state đã khôi phục dấu
    từ shot + SWAP-parity, không phải Statevector.data.
    """
    x_prime = np.asarray(signed_normalized_state, dtype=complex)
    b_prime = A_matrix @ x_prime

    denom = np.vdot(b_prime, b_prime)

    if abs(denom) < 1e-12:
        k_coeff = 0.0
    else:
        k_coeff = np.vdot(b_prime, b_vector) / denom

    x_recovered = k_coeff * x_prime

    return x_recovered, k_coeff, b_prime

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
    # Classical comparison + SHOT-BASED solution recovery only
    # ============================================================
    print("\n" + "=" * 60)
    print("Comparison with classical solution")
    print("=" * 60)

    A_matrix = A_MATRIX
    dim = 2**N_QUBITS
    b_vector = B_VECTOR

    try:
        x_classical = np.linalg.solve(A_matrix, b_vector)
    except np.linalg.LinAlgError:
        print("A bị suy biến hoặc gần suy biến. Dùng least-squares thay cho solve.")
        x_classical = np.linalg.lstsq(A_matrix, b_vector, rcond=None)[0]

    try:
        x_classical_raw = np.linalg.solve(A_matrix, B_VECTOR_RAW)
    except np.linalg.LinAlgError:
        print("A bị suy biến hoặc gần suy biến. Dùng least-squares cho raw system.")
        x_classical_raw = np.linalg.lstsq(A_matrix, B_VECTOR_RAW, rcond=None)[0]

    x_norm = x_classical / np.linalg.norm(x_classical)
    c_probs = np.abs(x_norm) ** 2

    # ============================================================
    # SHOT-BASED SWAP-PARITY SIGN RECOVERY
    # ============================================================
    # Chỉ dùng:
    #   counts -> p_i = |c_i|^2
    #   SWAP + Hadamard -> dấu tương đối
    #   scale k -> nghiệm VQLS
    #
    # Không đọc nghiệm bằng:
    #   Statevector.data
    #
    # first_sign_anchor dùng để cố định global sign.
    # Khi debug thì có thể neo theo nghiệm classical.
    # Khi chạy thực nghiệm, đặt thủ công theo hiểu biết bài toán:
    #   first_sign_anchor = 1
    # hoặc:
    #   first_sign_anchor = -1
    first_sign_anchor = 1 if np.real(x_classical[0]) >= 0 else -1

    signed_readout = extract_signed_vqls_solution_shot_based(
        w,
        first_sign=first_sign_anchor,
        shots=N_SHOTS,
        backend=None,   # Nếu có backend thật/Aer riêng thì thay None bằng backend đó.
        seed=RNG_SEED,
    )

    x_vqls_recovered, k_coeff, b_prime = scale_signed_vqls_state_to_solution(
        signed_readout["signed_normalized_state"],
        A_matrix,
        b_vector,
    )

    x_vqls_recovered_raw = B_VECTOR_NORM * x_vqls_recovered

    q_probs = signed_readout["prob_orig"]

    print("\n" + "=" * 70)
    print("SHOT-BASED SWAP-PARITY SIGN RECOVERY")
    print("=" * 70)

    print(f"First sign anchor s0 = {first_sign_anchor}")

    print("\nMeasured original probabilities p_i = |c_i|^2 from shots:")
    print(np.real_if_close(signed_readout["prob_orig"]))

    print("\nParity couple values for unshifted pairs:")
    print(np.real_if_close(signed_readout["parity_couple_0"]))

    print("\nParity array 0 for pairs (c0,c1), (c2,c3), ...:")
    print(signed_readout["parity_array_0"])

    print("\nParity couple values for shifted pairs:")
    print(np.real_if_close(signed_readout["parity_couple_1"]))

    print("\nParity array 1 for shifted pairs (c1,c2), (c3,c4), ...:")
    print(signed_readout["parity_array_1"])

    print("\nIntertwined relative parities:")
    print(signed_readout["parity_intertwined"])

    print("\nRecovered sign array:")
    print(signed_readout["sign_array"])

    print(f"\nCycle parity consistent: {signed_readout['cycle_consistent']}")

    print("\nRecovered signed normalized VQLS state from shots:")
    print(np.real_if_close(signed_readout["signed_normalized_state"]))

    print("\nRecovery coefficient k from shot-based state =")
    print(k_coeff)
    print(f"  |k|    = {np.abs(k_coeff):.8f}")
    print(f"  arg(k) = {np.angle(k_coeff):.8f} rad")

    print("\nRecovered VQLS solution x_VQLS from shot-based SWAP-parity =")
    print(np.real_if_close(x_vqls_recovered))

    print("\nClassical solution x_classical =")
    print(np.real_if_close(x_classical))

    abs_err = np.linalg.norm(x_vqls_recovered - x_classical)
    rel_err = abs_err / np.linalg.norm(x_classical)
    residual = np.linalg.norm(A_matrix @ x_vqls_recovered - b_vector)

    print(f"\nAbsolute error ||x_VQLS - x_classical|| = {abs_err:.6e}")
    print(f"Relative error                         = {rel_err:.6e}")
    print(f"Residual check ||A·x_VQLS - b||        = {residual:.6e}")

    print(f"\nA·x_VQLS = {np.real_if_close(A_matrix @ x_vqls_recovered)}")
    print(f"b input  = {np.real_if_close(b_vector)}")

    print("\nRecovered RAW FDLS solution from shot-based VQLS:")
    print(np.real_if_close(x_vqls_recovered_raw))

    print("\nClassical RAW FDLS solution:")
    print(np.real_if_close(x_classical_raw))

    raw_abs_err = np.linalg.norm(x_vqls_recovered_raw - x_classical_raw)
    raw_rel_err = raw_abs_err / np.linalg.norm(x_classical_raw)
    raw_residual = np.linalg.norm(A_matrix @ x_vqls_recovered_raw - B_VECTOR_RAW)

    print(f"\nRAW absolute error ||x_VQLS_raw - x_classical_raw|| = {raw_abs_err:.6e}")
    print(f"RAW relative error                              = {raw_rel_err:.6e}")
    print(f"RAW residual ||A·x_VQLS_raw - b_raw||          = {raw_residual:.6e}")

    print(f"\nb' = A · x'_VQLS_shot =")
    print(np.real_if_close(b_prime))

    print("\n" + "=" * 70)
    print("SHOT-BASED SOLUTION RECOVERY COMPLETE")
    print("=" * 70)

    print(f"\n{'Index':<8}{'Classical prob':<18}{'Shot prob':<18}{'|Diff|':<12}")

    for i in range(dim):
        print(
            f"{i:<8}"
            f"{c_probs[i]:<18.6f}"
            f"{q_probs[i]:<18.6f}"
            f"{abs(c_probs[i] - q_probs[i]):<12.6f}"
        )

    fidelity = np.sum(np.sqrt(c_probs * q_probs)) ** 2
    print(f"\nProbability fidelity from shots: {fidelity:.6f}")

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