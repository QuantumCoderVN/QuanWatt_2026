"""
vqls/VQLS tomography on Qiskit - tổ chức lại thành 2 module chính.

Bản tổ chức lại theo yêu cầu:
    1. MODULE CỔ ĐIỂN:      classical_solver(A, b)
    2. MODULE QUANTUM:       vqls_tomography(A, b)

Đầu vào chung:
    A: ma trận vuông kích thước 2^n x 2^n
    b: vector vế phải của hệ Ax = b, chưa chuẩn hóa

Đầu ra chính:
    x_classical: nghiệm cổ điển của Ax = b
    x_vqls_tomography: nghiệm lượng tử đã được khôi phục bằng quantum tomography
                      và scale về cùng miền với b gốc.

Ghi chú:
    - Tên module quantum được đặt là vqls_tomography theo yêu cầu.
    - Logic thuật toán trong code gốc được giữ lại: Pauli decomposition,
      Hadamard test, tối ưu COBYLA, quantum state tomography, khôi phục state
      từ density matrix và scale nghiệm.
    - Các hàm phụ của phần lượng tử được đặt bên trong vqls_tomography(A, b),
      để bên ngoài chỉ còn hai solver chính.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
from qiskit.circuit.library import StatePreparation


# ============================================================
# Hyperparameters giữ từ code gốc
# ============================================================
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
# Code gốc hiện dùng ansatz thực RY-only theo N_LAYERS.
USE_COMPLEX_ANSATZ = True
N_LAYERS = 3

np.random.seed(RNG_SEED)


# ============================================================
# MODULE 1: CLASSICAL SOLVER
# ============================================================
def classical_solver(A, b):
    """
    Giải hệ tuyến tính Ax = b bằng phương pháp cổ điển.

    Đầu vào:
        A: ma trận vuông
        b: vector vế phải, chưa chuẩn hóa

    Đầu ra:
        x_classical: nghiệm cổ điển của Ax = b
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A phải là ma trận vuông.")

    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b phải là vector 1D có cùng số dòng với A.")

    try:
        x_classical = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("A bị suy biến hoặc gần suy biến. Dùng least-squares thay cho solve.")
        x_classical = np.linalg.lstsq(A, b, rcond=None)[0]

    return x_classical


# ============================================================
# MODULE 2: vqls TOMOGRAPHY SOLVER
# ============================================================
def vqls_tomography(
    A,
    b,
    shots=N_SHOTS,
    steps=STEPS,
    q_delta=Q_DELTA,
    rng_seed=RNG_SEED,
    pauli_atol=PAULI_ATOL,
    pauli_rtol=PAULI_RTOL,
    max_pauli_terms=MAX_PAULI_TERMS,
    n_layers=N_LAYERS,
    backend=None,
    reference_index=0,
    reference_sign_anchor=+1,
    extraction_mode="real_reference",
    make_physical=True,
    return_details=True,
):
    """
    Giải hệ tuyến tính Ax = b bằng module lượng tử vqls/VQLS + tomography.

    Đầu vào:
        A: ma trận vuông kích thước 2^n x 2^n
        b: vector vế phải, chưa chuẩn hóa

    Quy trình chính:
        1. Kiểm tra A và b.
        2. Chuẩn hóa b thành |b> để tạo U_b bằng StatePreparation.
        3. Phân rã A sang Pauli basis bằng SparsePauliOp.from_operator.
        4. Tối ưu cost local bằng COBYLA.
        5. Dựng circuit nghiệm |x(theta)>.
        6. Quantum state tomography để khôi phục density matrix rho.
        7. Khôi phục normalized state từ rho.
        8. Scale normalized state về nghiệm của hệ gốc Ax = b.

    extraction_mode:
        "real_reference"
            Dùng công thức a_i = rho[ref, i] / sqrt(rho[ref, ref]).
            Phù hợp khi nghiệm/ansatz có biên độ thực.

        "dominant_eigenvector"
            Lấy eigenvector chính của rho. Phù hợp hơn nếu nghiệm có pha phức.

    Đầu ra:
        Nếu return_details=False:
            x_vqls_tomography

        Nếu return_details=True:
            dict chứa nghiệm, state tomography, cost history và các chỉ số so sánh.
    """
    A_matrix = np.asarray(A, dtype=complex)
    b_vector_raw = np.asarray(b, dtype=complex)

    if A_matrix.ndim != 2 or A_matrix.shape[0] != A_matrix.shape[1]:
        raise ValueError("A phải là ma trận vuông.")

    dim = A_matrix.shape[0]
    n_qubits = int(np.log2(dim))

    if 2**n_qubits != dim:
        raise ValueError("Kích thước A phải là 2^n x 2^n để mã hóa trên n qubit.")

    if b_vector_raw.ndim != 1 or b_vector_raw.shape[0] != dim:
        raise ValueError("b phải là vector 1D có cùng số dòng với A.")

    b_norm = np.linalg.norm(b_vector_raw)

    if b_norm < 1e-15:
        raise ValueError("Vector b có norm gần 0, không thể chuẩn hóa thành quantum state |b>.")

    b_vector = b_vector_raw / b_norm

    tot_qubits = n_qubits + 1
    ancilla_idx = n_qubits
    n_params = n_qubits * n_layers

    rng = np.random.default_rng(rng_seed)

    # ============================================================
    # Các hàm phụ được gom bên trong vqls tomography module
    # ============================================================
    def infer_n_qubits_from_matrix(A_inner):
        """Kiểm tra A là ma trận vuông 2^n x 2^n và trả về n."""
        A_inner = np.asarray(A_inner, dtype=complex)

        if A_inner.ndim != 2 or A_inner.shape[0] != A_inner.shape[1]:
            raise ValueError("A_MATRIX phải là ma trận vuông.")

        dim_inner = A_inner.shape[0]
        n_qubits_inner = int(np.log2(dim_inner))

        if 2**n_qubits_inner != dim_inner:
            raise ValueError("Kích thước A_MATRIX phải là 2^n x 2^n.")

        return n_qubits_inner, A_inner

    def pauli_decompose_matrix(A_matrix_inner, atol=1e-10, rtol=1e-10, max_terms=None):
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
        n_qubits_inner, A_matrix_inner = infer_n_qubits_from_matrix(A_matrix_inner)

        op = Operator(
            A_matrix_inner,
            input_dims=(2,) * n_qubits_inner,
            output_dims=(2,) * n_qubits_inner,
        )

        A_pauli_inner = SparsePauliOp.from_operator(op, atol=atol, rtol=rtol)

        # Tùy chọn truncate theo hệ số lớn nhất để chạy thử nhanh.
        # Nếu max_terms=None thì giữ chính xác toàn bộ.
        if max_terms is not None and len(A_pauli_inner.coeffs) > max_terms:
            idx = np.argsort(np.abs(A_pauli_inner.coeffs))[::-1][:max_terms]
            A_pauli_inner = SparsePauliOp(
                A_pauli_inner.paulis[idx],
                A_pauli_inner.coeffs[idx],
            )

        pauli_labels_inner = [p.to_label() for p in A_pauli_inner.paulis]
        coeffs_inner = np.asarray(A_pauli_inner.coeffs, dtype=complex)

        return n_qubits_inner, A_matrix_inner, A_pauli_inner, pauli_labels_inner, coeffs_inner

    n_qubits_check, A_matrix, A_pauli, pauli_labels, C = pauli_decompose_matrix(
        A_matrix,
        atol=pauli_atol,
        rtol=pauli_rtol,
        max_terms=max_pauli_terms,
    )

    if n_qubits_check != n_qubits:
        raise RuntimeError("Số qubit suy ra không nhất quán sau Pauli decomposition.")

    num_pauli = len(C)

    def make_Ub_gate(b_vector_inner):
        """
        Tạo gate U_b sao cho:
            U_b |0...0> = |b>
        """
        b_vector_inner = np.asarray(b_vector_inner, dtype=complex)
        b_vector_inner = b_vector_inner / np.linalg.norm(b_vector_inner)

        return StatePreparation(b_vector_inner)

    U_B_GATE = make_Ub_gate(b_vector)

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
        apply_controlled_pauli_string(qc, pauli_labels[l], qubits, ancilla)

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

        Ansatz này phù hợp với phương pháp khôi phục dấu/pha bằng tomography.
        """
        params = np.asarray(params, dtype=float)

        for q in qubits:
            qc.h(q)

        expected = len(qubits) * n_layers

        if len(params) != expected:
            raise ValueError(
                f"Ansatz RY-only cần {expected} tham số, nhưng nhận {len(params)}."
            )

        k = 0

        for layer in range(n_layers):
            for q in qubits:
                qc.ry(params[k], q)
                k += 1

            # Entanglement chain giữa các layer.
            # Với N_LAYERS = 3, CX sẽ được thêm sau layer 0 và layer 1.
            if layer < n_layers - 1:
                for q1, q2 in zip(qubits[:-1], qubits[1:]):
                    qc.cx(q1, q2)

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
        qc = QuantumCircuit(tot_qubits)
        main_qubits = list(range(n_qubits))

        qc.h(ancilla_idx)

        if part == "Im":
            qc.p(-np.pi / 2, ancilla_idx)

        apply_variational(qc, weights, main_qubits)

        # Quan trọng: apply A_lp trước
        apply_CA(qc, lp, main_qubits, ancilla_idx)

        # B_j = U_b Z_j U_b†
        apply_U_b(qc, main_qubits, dagger=True)

        if j != -1:
            qc.cz(ancilla_idx, main_qubits[j])

        apply_U_b(qc, main_qubits, dagger=False)

        # Quan trọng: apply A_l sau
        apply_CA(qc, l, main_qubits, ancilla_idx)

        qc.h(ancilla_idx)

        return qc

    def measure_z_ancilla(weights, l, lp, j, part):
        """Measure expectation value của Z trên ancilla."""
        qc = hadamard_test(weights, l, lp, j, part)
        state = Statevector.from_instruction(qc)

        # Ancilla là qubit cuối cùng, nên trong Qiskit Pauli label,
        # Z nằm bên trái nhất.
        pauli_str = "Z" + "I" * n_qubits
        Z_obs = SparsePauliOp.from_list([(pauli_str, 1.0)])

        return state.expectation_value(Z_obs).real

    def mu(weights, l, lp, j):
        """Compute μ = Re + i·Im."""
        re = measure_z_ancilla(weights, l, lp, j, "Re")
        im = measure_z_ancilla(weights, l, lp, j, "Im")
        return re + 1j * im

    def psi_norm(weights):
        """
        Compute:
            <ψ|ψ> = <x| A†A |x>
        """
        norm = 0.0 + 0.0j

        for l in range(num_pauli):
            for lp in range(num_pauli):
                norm += np.conj(C[l]) * C[lp] * mu(weights, l, lp, -1)

        return norm.real

    def cost_local(weights):
        """Compute local cost function C_L."""
        norm = psi_norm(weights)

        if abs(norm) < 1e-12:
            return 1e6

        mu_sum = 0.0 + 0.0j

        for l in range(num_pauli):
            for lp in range(num_pauli):
                for j in range(n_qubits):
                    mu_sum += np.conj(C[l]) * C[lp] * mu(weights, l, lp, j)

        return 0.5 - 0.5 * mu_sum.real / (n_qubits * norm)

    # ============================================================
    # Optimization giữ lại phần parameter-shift cũ ở dạng comment
    # ============================================================
    # def parameter_shift_gradient(w, cost_fn):
    #     """Compute gradient bằng parameter-shift rule."""
    #     grad = np.zeros_like(w)
    #     shift = np.pi / 2
    #
    #     for i in range(len(w)):
    #         wp = w.copy()
    #         wp[i] += shift
    #
    #         wm = w.copy()
    #         wm[i] -= shift
    #
    #         grad[i] = 0.5 * (cost_fn(wp) - cost_fn(wm))
    #
    #     return grad

    # ============================================================
    # Quantum Tomography solution recovery - giữ từ code gốc
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
        return ["".join(basis) for basis in itertools.product("XYZ", repeat=n)]

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

        for i, b_char in enumerate(basis):
            qubit = n - 1 - i

            if b_char == "X":
                qc.h(qubit)
            elif b_char == "Y":
                qc.sdg(qubit)
                qc.h(qubit)
            elif b_char == "Z":
                pass
            else:
                raise ValueError(f"Unknown tomography basis character: {b_char}")

    def build_vqls_state_circuit(weights):
        """
        Circuit chuẩn bị trạng thái nghiệm |x(theta)>.

        Đây là circuit sẽ được đưa vào quantum tomography.
        """
        qc = QuantumCircuit(n_qubits)
        apply_variational(qc, weights, list(range(n_qubits)))
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

    def run_counts(qc, shots_inner=shots, backend_inner=backend, seed=rng_seed):
        """
        Chạy một circuit và trả về counts.

        Nếu backend=None thì dùng AerSimulator.
        Nếu truyền backend thật, hàm sẽ dùng backend đó.
        """
        qc_meas = add_measure_all(qc)

        if backend_inner is None:
            try:
                from qiskit_aer import AerSimulator
            except ImportError:
                raise ImportError(
                    "Bạn cần cài qiskit-aer để chạy shot-based simulation:\n"
                    "    pip install qiskit-aer\n"
                    "hoặc truyền backend thật vào vqls_tomography(..., backend=backend)."
                )

            backend_inner = AerSimulator(seed_simulator=seed)

        transpiled = transpile(qc_meas, backend_inner)
        job = backend_inner.run(transpiled, shots=int(shots_inner))
        return job.result().get_counts()

    def run_counts_batch(circuits, shots_inner=shots, backend_inner=backend, seed=rng_seed):
        """
        Chạy nhiều tomography circuits trong một batch để nhanh hơn.
        """
        measured_circuits = [add_measure_all(qc) for qc in circuits]

        if backend_inner is None:
            try:
                from qiskit_aer import AerSimulator
            except ImportError:
                raise ImportError(
                    "Bạn cần cài qiskit-aer để chạy tomography simulation:\n"
                    "    pip install qiskit-aer\n"
                    "hoặc truyền backend thật vào vqls_tomography(..., backend=backend)."
                )

            backend_inner = AerSimulator(seed_simulator=seed)

        transpiled = transpile(measured_circuits, backend_inner)
        job = backend_inner.run(transpiled, shots=int(shots_inner))
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
        shots_total = sum(counts.values())

        if shots_total == 0:
            raise ValueError("Counts rỗng, không thể ước lượng Pauli expectation.")

        expval = 0.0

        for bitstring, count in counts.items():
            eig = pauli_eigenvalue_from_bitstring(pauli_string, bitstring)
            expval += eig * count / shots_total

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
        shots_inner=shots,
        backend_inner=backend,
        seed=rng_seed,
        make_physical_inner=make_physical,
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
            shots_inner=shots_inner,
            backend_inner=backend_inner,
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

        dim_inner = 2**n
        rho = np.zeros((dim_inner, dim_inner), dtype=complex)

        for P, expval in pauli_expectations.items():
            rho += expval * pauli_matrix(P)

        rho = rho / dim_inner

        # Ép Hermitian để giảm sai số số học.
        rho = 0.5 * (rho + rho.conj().T)

        if make_physical_inner:
            rho = project_to_physical_density_matrix(rho)

        return rho, pauli_expectations, counts_by_basis

    def amplitudes_from_density_matrix_real_reference(
        rho,
        n,
        reference_index_inner=0,
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

        p_ref = np.real(rho[reference_index_inner, reference_index_inner])

        if p_ref <= 1e-14:
            raise ValueError(
                "Reference amplitude gần 0 hoặc không hợp lệ. "
                "Hãy chọn reference_index khác hoặc dùng extraction_mode='dominant_eigenvector'."
            )

        a_ref = reference_sign * np.sqrt(p_ref)
        dim_inner = 2**n

        amps = np.zeros(dim_inner, dtype=float)

        for i in range(dim_inner):
            amps[i] = np.real(rho[reference_index_inner, i]) / a_ref

        norm = np.linalg.norm(amps)
        if norm > 1e-14:
            amps = amps / norm

        return amps

    def amplitudes_from_density_matrix_dominant_eigenvector(
        rho,
        reference_index_inner=0,
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

        if abs(state[reference_index_inner]) < 1e-12:
            reference_index_inner = int(np.argmax(np.abs(state)))

        ref_amp = state[reference_index_inner]
        phase_fix = np.exp(-1j * np.angle(ref_amp))
        state = state * phase_fix

        if reference_sign < 0:
            state = -state

        state = state / np.linalg.norm(state)

        if force_real_if_close and np.max(np.abs(np.imag(state))) < 1e-8:
            state = np.real(state)

        return state, eigvals

    def extract_solution_quantum_tomography(
        weights,
        shots_inner=shots,
        backend_inner=backend,
        seed=rng_seed,
        reference_index_inner=reference_index,
        reference_sign=+1,
        extraction_mode_inner=extraction_mode,
        make_physical_inner=make_physical,
    ):
        """
        Khôi phục normalized quantum state bằng Quantum State Tomography.

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
            shots_inner=shots_inner,
            backend_inner=backend_inner,
            seed=seed,
            make_physical_inner=make_physical_inner,
        )

        eig_state, eigvals = amplitudes_from_density_matrix_dominant_eigenvector(
            rho,
            reference_index_inner=reference_index_inner,
            reference_sign=reference_sign,
        )

        if extraction_mode_inner == "real_reference":
            normalized_state = amplitudes_from_density_matrix_real_reference(
                rho,
                n=n_qubits,
                reference_index_inner=reference_index_inner,
                reference_sign=reference_sign,
            )
        elif extraction_mode_inner == "dominant_eigenvector":
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
            "extraction_mode": extraction_mode_inner,
        }

    def scale_normalized_state_to_solution(
        normalized_state,
        A_matrix_inner,
        b_vector_inner,
    ):
        """
        Scale normalized quantum state |x'> thành nghiệm x ≈ k|x'>.

        Vì vqls/VQLS trả về trạng thái chuẩn hóa, hệ số k được chọn theo least-squares:
            k = <A x' | b> / <A x' | A x'>

        Khác với code gốc ở điểm tổ chức:
            - b_vector_inner ở đây là b gốc, chưa chuẩn hóa.
            - Vì vậy x_recovered trả về trực tiếp nghiệm cùng thang đo với Ax = b.
        """
        x_prime = np.asarray(normalized_state, dtype=complex)
        b_prime = A_matrix_inner @ x_prime

        denom = np.vdot(b_prime, b_prime)

        if abs(denom) < 1e-12:
            k_coeff = 0.0
        else:
            k_coeff = np.vdot(b_prime, b_vector_inner) / denom

        x_recovered = k_coeff * x_prime

        return x_recovered, k_coeff, b_prime

    # ============================================================
    # Main optimization loop của vqls tomography module
    # ============================================================
    w_init = q_delta * rng.standard_normal(n_params)

    print(f"Number of qubits inferred from A: {n_qubits}")
    print(f"Total qubits including ancilla:   {tot_qubits}")
    print(f"Number of ansatz parameters:      {n_params}")
    print(f"Number of Pauli terms:            {num_pauli}")
    print(f"Condition number of A:            {np.linalg.cond(A_matrix):.3e}")

    print(f"\nInitial parameters:\n{w_init}")

    print("\n" + "=" * 60)
    print("Pauli decomposition of A from Qiskit")
    print("=" * 60)
    print(A_pauli)

    # Kiểm tra phân rã có khớp A không
    A_reconstructed = np.asarray(A_pauli.to_matrix(), dtype=complex)
    decomp_err = np.linalg.norm(A_reconstructed - A_matrix)

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
            "maxiter": steps,
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
    # QUANTUM TOMOGRAPHY SOLUTION RECOVERY
    # ============================================================
    reference_sign = 1 if reference_sign_anchor >= 0 else -1

    tomography_readout = extract_solution_quantum_tomography(
        w,
        shots_inner=shots,
        backend_inner=backend,
        seed=rng_seed,
        reference_index_inner=reference_index,
        reference_sign=reference_sign,
        extraction_mode_inner=extraction_mode,
        make_physical_inner=make_physical,
    )

    x_vqls_state_tomo = tomography_readout["normalized_state"]

    x_vqls_recovered, k_coeff, b_prime = scale_normalized_state_to_solution(
        x_vqls_state_tomo,
        A_matrix,
        b_vector_raw,
    )

    residual = np.linalg.norm(A_matrix @ x_vqls_recovered - b_vector_raw)

    print("\n" + "=" * 70)
    print("QUANTUM TOMOGRAPHY SOLUTION RECOVERY")
    print("=" * 70)

    print(f"Tomography extraction mode = {tomography_readout['extraction_mode']}")
    print(f"Reference index            = {reference_index}")
    print(f"Reference sign anchor      = {reference_sign}")
    print(f"Number of tomography bases = {3**n_qubits}")
    print(f"Shots per tomography basis = {shots}")
    print(f"Density-matrix purity Tr(rho^2) = {tomography_readout['purity']:.8f}")

    print("\nTomography eigenvalues of rho:")
    print(np.real_if_close(tomography_readout["eigenvalues"]))

    print("\nRecovered normalized quantum state from tomography:")
    print(np.real_if_close(x_vqls_state_tomo))

    print("\nRecovery coefficient k from tomography state =")
    print(k_coeff)
    print(f"  |k|    = {np.abs(k_coeff):.8f}")
    print(f"  arg(k) = {np.angle(k_coeff):.8f} rad")

    print("\nRecovered vqls tomography solution x_vqls_tomography for raw system Ax=b =")
    print(np.real_if_close(x_vqls_recovered))

    print(f"\nResidual check ||A·x_vqls_tomography - b|| = {residual:.6e}")
    print(f"\nA·x_vqls_tomography = {np.real_if_close(A_matrix @ x_vqls_recovered)}")
    print(f"b input            = {np.real_if_close(b_vector_raw)}")

    print(f"\nb' = A · x'_vqls_tomography =")
    print(np.real_if_close(b_prime))

    print("\n" + "=" * 70)
    print("QUANTUM TOMOGRAPHY SOLUTION RECOVERY COMPLETE")
    print("=" * 70)

    if not return_details:
        return x_vqls_recovered

    return {
        "x_vqls_tomography": x_vqls_recovered,
        "normalized_state": x_vqls_state_tomo,
        "tomography_readout": tomography_readout,
        "rho": tomography_readout["rho"],
        "eigenvalues": tomography_readout["eigenvalues"],
        "purity": tomography_readout["purity"],
        "optimized_parameters": w,
        "optimization_result": res,
        "cost_history": cost_history,
        "probability_vector": tomography_readout["prob_orig"],
        "k_coeff": k_coeff,
        "b_prime": b_prime,
        "residual": residual,
        "n_qubits": n_qubits,
        "num_pauli": num_pauli,
        "pauli_decomposition": A_pauli,
        "pauli_labels": pauli_labels,
        "pauli_coeffs": C,
        "b_norm": b_norm,
        "reference_index": reference_index,
        "reference_sign_anchor": reference_sign,
        "extraction_mode": extraction_mode,
    }


# ============================================================
# PHẦN SO SÁNH OUTPUT CỦA HAI MODULE
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 4x4 signed test matrix from code gốc
    # Symmetric real matrix, dim = 2^2, cond(A) moderate
    #
    # Designed raw solution:
    #     x_raw = [0.4, -0.3, 0.2, -0.1]
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

    print("\n" + "=" * 70)
    print("MODULE 1: CLASSICAL SOLVER")
    print("=" * 70)

    x_classical = classical_solver(A_MATRIX, B_VECTOR_RAW)

    print("\nClassical solution x_classical =")
    print(np.real_if_close(x_classical))

    print("\n" + "=" * 70)
    print("MODULE 2: vqls TOMOGRAPHY")
    print("=" * 70)

    reference_index = 0
    reference_sign_anchor = 1 if np.real(x_classical[reference_index]) >= 0 else -1

    vqls_result = vqls_tomography(
        A_MATRIX,
        B_VECTOR_RAW,
        shots=N_SHOTS,
        steps=STEPS,
        q_delta=Q_DELTA,
        rng_seed=RNG_SEED,
        backend=None,
        reference_index=reference_index,
        reference_sign_anchor=reference_sign_anchor,
        extraction_mode="real_reference",
        make_physical=True,
        return_details=True,
    )

    x_vqls = vqls_result["x_vqls_tomography"]

    print("\n" + "=" * 70)
    print("SO SÁNH OUTPUT HAI MODULE")
    print("=" * 70)

    print("\nClassical solution x_classical =")
    print(np.real_if_close(x_classical))

    print("\nRecovered vqls tomography solution x_vqls_tomography =")
    print(np.real_if_close(x_vqls))

    abs_err = np.linalg.norm(x_vqls - x_classical)
    rel_err = abs_err / np.linalg.norm(x_classical)
    residual_vqls = np.linalg.norm(A_MATRIX @ x_vqls - B_VECTOR_RAW)
    residual_classical = np.linalg.norm(A_MATRIX @ x_classical - B_VECTOR_RAW)

    print(f"\nAbsolute error ||x_vqls_tomography - x_classical|| = {abs_err:.6e}")
    print(f"Relative error                                  = {rel_err:.6e}")
    print(f"Residual vqls tomography ||A·x_vqls - b||         = {residual_vqls:.6e}")
    print(f"Residual classical ||A·x_classical - b||        = {residual_classical:.6e}")

    x_norm = x_classical / np.linalg.norm(x_classical)
    q_state = vqls_result["normalized_state"]

    state_fidelity = abs(np.vdot(x_norm, q_state)) ** 2

    # Sai số state sau khi chỉnh global phase để so sánh dễ nhìn.
    overlap = np.vdot(x_norm, q_state)
    if abs(overlap) > 1e-14:
        q_state_aligned = q_state * np.exp(-1j * np.angle(overlap))
    else:
        q_state_aligned = q_state

    state_error = np.linalg.norm(q_state_aligned - x_norm)

    print(f"\nState fidelity |<x_classical_norm|x_tomo>|^2 = {state_fidelity:.8f}")
    print(f"Phase-aligned normalized-state error        = {state_error:.6e}")

    c_probs = np.abs(x_norm) ** 2
    q_probs = vqls_result["probability_vector"]
    dim = A_MATRIX.shape[0]

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
    # Plotting giữ lại từ code gốc
    # ============================================================
    cost_history = vqls_result["cost_history"]

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
    axes[2].set_title("vqls tomography solution")

    plt.tight_layout()
    plt.savefig("vqls_tomography_results.png", dpi=120, bbox_inches="tight")
    print("\nPlot saved to: hhl_tomography_results.png")
