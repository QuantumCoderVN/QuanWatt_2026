"""
VQLS on Qiskit - Quantum Variational Linear Solver.

Phiên bản này:
1. Nhận ma trận A bất kỳ kích thước 2^n x 2^n.
2. Dùng Qiskit SparsePauliOp.from_operator để phân rã A sang Pauli basis.
3. Tự động dựng controlled-Pauli A_l trong Hadamard test.
4. Không còn hard-code C, A_PAULI, apply_CA cho từng Pauli term.
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit, transpile
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
# MODULE 5B: Quantum Tomography solution recovery
# ============================================================

def all_pauli_strings(n):
    """
    Sinh toàn bộ Pauli string độ dài n trên {I, X, Y, Z}.

    Convention giống Qiskit label/counts:
        ký tự bên trái ứng với q[n-1], ký tự bên phải ứng với q[0].
    """
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)]


def unique_tomography_bases(n):
    """
    Full tomography measurement bases: toàn bộ chuỗi trong {X, Y, Z}^n.

    Số setting đo = 3^n.
    """
    return ["".join(b) for b in itertools.product("XYZ", repeat=n)]


def pauli_to_measurement_basis(pauli_string):
    """
    Chuyển Pauli string sang basis đo tương thích.

    Nếu gặp I thì chọn đo Z cho qubit đó vì I không cần basis riêng.
    """
    basis = []

    for p in pauli_string:
        if p == "I":
            basis.append("Z")
        else:
            basis.append(p)

    return "".join(basis)


def add_basis_rotation(qc, basis):
    """
    Thêm rotation để đo X/Y/Z bằng Z measurement.

    basis được viết theo thứ tự q[n-1] ... q[0].
    Qiskit qubit index là q[0], q[1], ..., q[n-1].
    Vì vậy basis[i] ứng với qubit n-1-i.
    """
    n = len(basis)

    for i, b in enumerate(basis):
        qubit = n - 1 - i

        if b == "X":
            qc.h(qubit)
        elif b == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif b == "Z":
            pass
        else:
            raise ValueError(f"Unknown tomography basis character: {b}")


def build_vqls_state_circuit(weights):
    """
    Circuit chuẩn bị trạng thái nghiệm VQLS |x(theta)>.

    Đây là circuit sẽ được đưa vào quantum tomography.
    """
    qc = QuantumCircuit(N_QUBITS)
    apply_variational(qc, weights, list(range(N_QUBITS)))
    return qc


def make_tomography_circuit(state_circuit, basis):
    """
    Tạo một tomography circuit:
        1. chuẩn bị |x(theta)>
        2. xoay sang basis X/Y/Z cần đo
        3. measurement sẽ được thêm trong run_counts hoặc batch runner
    """
    n = state_circuit.num_qubits

    U = state_circuit.copy()
    U.remove_final_measurements(inplace=True)

    qc = QuantumCircuit(n)
    qc.compose(U, range(n), inplace=True)
    add_basis_rotation(qc, basis)

    return qc


def add_measure_all(qc):
    """Trả về bản copy có measurement toàn bộ qubit."""
    qc_meas = qc.copy()
    qc_meas.measure_all()
    return qc_meas


def run_counts(qc, shots=N_SHOTS, backend=None, seed=RNG_SEED):
    """
    Chạy một circuit và trả về counts.

    Nếu backend=None thì dùng AerSimulator.
    Nếu truyền backend thật, hàm sẽ dùng backend đó.
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

    transpiled = transpile(qc_meas, backend)
    job = backend.run(transpiled, shots=int(shots))
    return job.result().get_counts()


def run_counts_batch(circuits, shots=N_SHOTS, backend=None, seed=RNG_SEED):
    """
    Chạy nhiều tomography circuits trong một batch để nhanh hơn.
    """
    measured_circuits = [add_measure_all(qc) for qc in circuits]

    if backend is None:
        try:
            from qiskit_aer import AerSimulator
        except ImportError:
            raise ImportError(
                "Bạn cần cài qiskit-aer để chạy tomography simulation:\n"
                "    pip install qiskit-aer\n"
                "hoặc truyền backend thật vào run_counts_batch(..., backend=backend)."
            )

        backend = AerSimulator(seed_simulator=seed)

    transpiled = transpile(measured_circuits, backend)
    job = backend.run(transpiled, shots=int(shots))
    result = job.result()

    return [result.get_counts(i) for i in range(len(measured_circuits))]


def counts_to_probability_vector(counts, n_bits):
    """
    Đổi counts dạng {'q[n-1]...q[0]': count} sang vector xác suất.
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


def pauli_eigenvalue_from_bitstring(pauli_string, bitstring):
    """
    Tính eigenvalue +/-1 của Pauli string từ bitstring đo được.

    Với I: bỏ qua qubit đó.
    Với X/Y/Z sau khi đã rotate basis:
        bit 0 -> +1
        bit 1 -> -1
    """
    clean = bitstring.replace(" ", "")
    eig = 1

    for p, bit in zip(pauli_string, clean):
        if p == "I":
            continue
        if bit == "0":
            eig *= +1
        elif bit == "1":
            eig *= -1
        else:
            raise ValueError(f"Invalid measured bit: {bit}")

    return eig


def estimate_pauli_expectation(pauli_string, counts):
    """Ước lượng <P> từ counts đo trong basis tương thích."""
    shots = sum(counts.values())

    if shots == 0:
        raise ValueError("Counts rỗng, không thể ước lượng Pauli expectation.")

    expval = 0.0

    for bitstring, count in counts.items():
        eig = pauli_eigenvalue_from_bitstring(pauli_string, bitstring)
        expval += eig * count / shots

    return expval


def single_pauli_matrix(p):
    """Trả về ma trận 2x2 của I, X, Y, Z."""
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
    """
    Build full Pauli matrix theo convention q[n-1] ... q[0].
    """
    mats = [single_pauli_matrix(p) for p in pauli_string]

    result = mats[0]
    for mat in mats[1:]:
        result = np.kron(result, mat)

    return result


def project_to_physical_density_matrix(rho):
    """
    Projection đơn giản về ma trận mật độ vật lý:
        1. Hermitize.
        2. Diagonalize.
        3. Clip eigenvalue âm về 0.
        4. Chuẩn hóa trace = 1.
    """
    rho = 0.5 * (rho + rho.conj().T)

    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)

    total = np.sum(eigvals)
    if total <= 0:
        raise ValueError("All eigenvalues vanished after clipping.")

    eigvals = eigvals / total
    rho_physical = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    return rho_physical


def quantum_state_tomography(
    state_circuit,
    shots=N_SHOTS,
    backend=None,
    seed=RNG_SEED,
    make_physical=True,
):
    """
    Khôi phục density matrix rho bằng full Pauli-basis tomography.

    Returns
    -------
    rho:
        Reconstructed density matrix.
    pauli_expectations:
        Dict Pauli string -> expectation value.
    counts_by_basis:
        Dict measurement basis -> raw counts.
    """
    n = state_circuit.num_qubits
    bases = unique_tomography_bases(n)

    tomography_circuits = [
        make_tomography_circuit(state_circuit, basis)
        for basis in bases
    ]

    counts_list = run_counts_batch(
        tomography_circuits,
        shots=shots,
        backend=backend,
        seed=seed,
    )

    counts_by_basis = {
        basis: counts_list[i]
        for i, basis in enumerate(bases)
    }

    pauli_expectations = {}

    for P in all_pauli_strings(n):
        if P == "I" * n:
            pauli_expectations[P] = 1.0
            continue

        basis = pauli_to_measurement_basis(P)
        counts = counts_by_basis[basis]
        pauli_expectations[P] = estimate_pauli_expectation(P, counts)

    dim = 2**n
    rho = np.zeros((dim, dim), dtype=complex)

    for P, expval in pauli_expectations.items():
        rho += expval * pauli_matrix(P)

    rho = rho / dim

    # Ép Hermitian để giảm sai số số học.
    rho = 0.5 * (rho + rho.conj().T)

    if make_physical:
        rho = project_to_physical_density_matrix(rho)

    return rho, pauli_expectations, counts_by_basis


def amplitudes_from_density_matrix_real_reference(
    rho,
    n,
    reference_index=0,
    reference_sign=+1,
):
    """
    Khôi phục biên độ thực có dấu từ density matrix.

    Giả sử:
        rho = |x><x|
        |x> có biên độ thực
        biên độ reference_index khác 0

    Với state thực:
        rho[ref, i] = a_ref * a_i
    nên:
        a_i = rho[ref, i] / a_ref
    """
    if reference_sign not in [+1, -1]:
        raise ValueError("reference_sign must be +1 or -1.")

    p_ref = np.real(rho[reference_index, reference_index])

    if p_ref <= 1e-14:
        raise ValueError(
            "Reference amplitude gần 0 hoặc không hợp lệ. "
            "Hãy chọn reference_index khác hoặc dùng extraction_mode='dominant_eigenvector'."
        )

    a_ref = reference_sign * np.sqrt(p_ref)
    dim = 2**n

    amps = np.zeros(dim, dtype=float)

    for i in range(dim):
        amps[i] = np.real(rho[reference_index, i]) / a_ref

    norm = np.linalg.norm(amps)
    if norm > 1e-14:
        amps = amps / norm

    return amps


def amplitudes_from_density_matrix_dominant_eigenvector(
    rho,
    reference_index=0,
    reference_sign=+1,
    force_real_if_close=True,
):
    """
    Khôi phục state vector từ eigenvector ứng với eigenvalue lớn nhất của rho.

    Cách này dùng được cả khi nghiệm có pha phức. Global phase được cố định
    để amplitude tại reference_index là số thực với dấu reference_sign.
    """
    if reference_sign not in [+1, -1]:
        raise ValueError("reference_sign must be +1 or -1.")

    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = int(np.argmax(eigvals))

    state = eigvecs[:, idx]

    if abs(state[reference_index]) < 1e-12:
        reference_index = int(np.argmax(np.abs(state)))

    ref_amp = state[reference_index]
    phase_fix = np.exp(-1j * np.angle(ref_amp))
    state = state * phase_fix

    if reference_sign < 0:
        state = -state

    state = state / np.linalg.norm(state)

    if force_real_if_close and np.max(np.abs(np.imag(state))) < 1e-8:
        state = np.real(state)

    return state, eigvals


def extract_vqls_solution_quantum_tomography(
    weights,
    shots=N_SHOTS,
    backend=None,
    seed=RNG_SEED,
    reference_index=0,
    reference_sign=+1,
    extraction_mode="real_reference",
    make_physical=True,
):
    """
    Khôi phục normalized VQLS state bằng Quantum State Tomography.

    extraction_mode:
        'real_reference'
            Dùng công thức từ notebook tomography:
                a_i = rho[ref, i] / sqrt(rho[ref, ref])
            Phù hợp khi nghiệm/ansatz có biên độ thực.

        'dominant_eigenvector'
            Lấy eigenvector chính của rho. Phù hợp hơn nếu nghiệm có pha phức.
    """
    state_circuit = build_vqls_state_circuit(weights)

    rho, pauli_expectations, counts_by_basis = quantum_state_tomography(
        state_circuit=state_circuit,
        shots=shots,
        backend=backend,
        seed=seed,
        make_physical=make_physical,
    )

    eig_state, eigvals = amplitudes_from_density_matrix_dominant_eigenvector(
        rho,
        reference_index=reference_index,
        reference_sign=reference_sign,
    )

    if extraction_mode == "real_reference":
        normalized_state = amplitudes_from_density_matrix_real_reference(
            rho,
            n=N_QUBITS,
            reference_index=reference_index,
            reference_sign=reference_sign,
        )
    elif extraction_mode == "dominant_eigenvector":
        normalized_state = eig_state
    else:
        raise ValueError(
            "extraction_mode phải là 'real_reference' hoặc 'dominant_eigenvector'."
        )

    normalized_state = np.asarray(normalized_state, dtype=complex)
    normalized_state = normalized_state / np.linalg.norm(normalized_state)
    probabilities = np.abs(normalized_state) ** 2

    purity = np.real(np.trace(rho @ rho))

    return {
        "state_circuit": state_circuit,
        "rho": rho,
        "pauli_expectations": pauli_expectations,
        "counts_by_basis": counts_by_basis,
        "eigenvalues": eigvals,
        "dominant_eigenvector_state": eig_state,
        "normalized_state": normalized_state,
        "prob_orig": probabilities,
        "purity": purity,
        "extraction_mode": extraction_mode,
    }


def scale_normalized_vqls_state_to_solution(
    normalized_state,
    A_matrix,
    b_vector,
):
    """
    Scale normalized quantum state |x'> thành nghiệm x ≈ k|x'>.

    Vì VQLS/HHL trả về trạng thái chuẩn hóa, hệ số k được chọn theo least-squares:
        k = <A x' | b> / <A x' | A x'>
    """
    x_prime = np.asarray(normalized_state, dtype=complex)
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
    # Classical comparison + Quantum Tomography solution recovery
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

    # Nghiệm quantum là state chuẩn hóa, nên classical cũng phải chuẩn hóa trước khi so sánh state.
    x_norm = x_classical / np.linalg.norm(x_classical)
    c_probs = np.abs(x_norm) ** 2

    # Cố định global sign/phase. Với nghiệm thực, dùng dấu của phần tử đầu tiên
    # để khớp convention của tomography real_reference.
    reference_index = 0
    first_sign_anchor = 1 if np.real(x_classical[reference_index]) >= 0 else -1

    # Nếu nghiệm/phương án ansatz thật sự phức, đổi extraction_mode thành:
    #     "dominant_eigenvector"
    # Với ansatz RY-only hiện tại, "real_reference" khớp phương pháp tomography
    # đã trình bày trước đó: a_i = rho[ref, i] / sqrt(rho[ref, ref]).
    tomography_readout = extract_vqls_solution_quantum_tomography(
        w,
        shots=N_SHOTS,
        backend=None,   # Nếu có backend thật/Aer riêng thì thay None bằng backend đó.
        seed=RNG_SEED,
        reference_index=reference_index,
        reference_sign=first_sign_anchor,
        extraction_mode="real_reference",
        make_physical=True,
    )

    x_vqls_state_tomo = tomography_readout["normalized_state"]

    x_vqls_recovered, k_coeff, b_prime = scale_normalized_vqls_state_to_solution(
        x_vqls_state_tomo,
        A_matrix,
        b_vector,
    )

    x_vqls_recovered_raw = B_VECTOR_NORM * x_vqls_recovered
    q_probs = tomography_readout["prob_orig"]

    print("\n" + "=" * 70)
    print("QUANTUM TOMOGRAPHY SOLUTION RECOVERY")
    print("=" * 70)

    print(f"Tomography extraction mode = {tomography_readout['extraction_mode']}")
    print(f"Reference index            = {reference_index}")
    print(f"Reference sign anchor      = {first_sign_anchor}")
    print(f"Number of tomography bases = {3**N_QUBITS}")
    print(f"Shots per tomography basis = {N_SHOTS}")
    print(f"Density-matrix purity Tr(rho^2) = {tomography_readout['purity']:.8f}")

    print("\nTomography eigenvalues of rho:")
    print(np.real_if_close(tomography_readout["eigenvalues"]))

    print("\nRecovered normalized VQLS state from tomography:")
    print(np.real_if_close(x_vqls_state_tomo))

    print("\nClassical normalized solution:")
    print(np.real_if_close(x_norm))

    state_fidelity = abs(np.vdot(x_norm, x_vqls_state_tomo)) ** 2

    # Sai số state sau khi chỉnh global phase để so sánh dễ nhìn.
    overlap = np.vdot(x_norm, x_vqls_state_tomo)
    if abs(overlap) > 1e-14:
        x_vqls_state_aligned = x_vqls_state_tomo * np.exp(-1j * np.angle(overlap))
    else:
        x_vqls_state_aligned = x_vqls_state_tomo

    state_error = np.linalg.norm(x_vqls_state_aligned - x_norm)

    print(f"\nState fidelity |<x_classical_norm|x_tomo>|^2 = {state_fidelity:.8f}")
    print(f"Phase-aligned normalized-state error        = {state_error:.6e}")

    print("\nRecovery coefficient k from tomography state =")
    print(k_coeff)
    print(f"  |k|    = {np.abs(k_coeff):.8f}")
    print(f"  arg(k) = {np.angle(k_coeff):.8f} rad")

    print("\nRecovered VQLS solution x_VQLS from tomography =")
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

    print("\nRecovered RAW solution from tomography VQLS:")
    print(np.real_if_close(x_vqls_recovered_raw))

    print("\nClassical RAW solution:")
    print(np.real_if_close(x_classical_raw))

    raw_abs_err = np.linalg.norm(x_vqls_recovered_raw - x_classical_raw)
    raw_rel_err = raw_abs_err / np.linalg.norm(x_classical_raw)
    raw_residual = np.linalg.norm(A_matrix @ x_vqls_recovered_raw - B_VECTOR_RAW)

    print(f"\nRAW absolute error ||x_VQLS_raw - x_classical_raw|| = {raw_abs_err:.6e}")
    print(f"RAW relative error                              = {raw_rel_err:.6e}")
    print(f"RAW residual ||A·x_VQLS_raw - b_raw||          = {raw_residual:.6e}")

    print(f"\nb' = A · x'_VQLS_tomo =")
    print(np.real_if_close(b_prime))

    print("\n" + "=" * 70)
    print("QUANTUM TOMOGRAPHY SOLUTION RECOVERY COMPLETE")
    print("=" * 70)

    print(f"\n{'Index':<8}{'Classical prob':<18}{'Tomo prob':<18}{'|Diff|':<12}")

    for i in range(dim):
        print(
            f"{i:<8}"
            f"{c_probs[i]:<18.6f}"
            f"{q_probs[i]:<18.6f}"
            f"{abs(c_probs[i] - q_probs[i]):<12.6f}"
        )

    probability_fidelity = np.sum(np.sqrt(c_probs * q_probs)) ** 2
    print(f"\nProbability fidelity from tomography: {probability_fidelity:.6f}")

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
    axes[2].set_title("VQLS tomography solution")

    plt.tight_layout()
    plt.savefig("vqls_tomography_results.png", dpi=120, bbox_inches="tight")
    print("\nPlot saved to: vqls_results.png")

    return w, cost_history, c_probs, q_probs


if __name__ == "__main__":
    main()