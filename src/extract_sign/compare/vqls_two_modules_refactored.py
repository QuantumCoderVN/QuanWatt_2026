"""
VQLS on Qiskit - Quantum Variational Linear Solver.

Bản tổ chức lại theo đúng yêu cầu:
    1. MODULE CỔ ĐIỂN: classical_solver(A, b)
    2. MODULE VQLS:    vqls_solver(A, b)

Đầu vào chung:
    A: ma trận vuông kích thước 2^n x 2^n
    b: vector vế phải của hệ Ax = b, chưa chuẩn hóa

Đầu ra chính:
    x_classical: nghiệm cổ điển của Ax = b
    x_vqls: nghiệm VQLS đã được khôi phục dấu và scale về cùng miền với b gốc

Ghi chú:
    - Logic thuật toán VQLS, Pauli decomposition, Hadamard test,
      shot-based SWAP-parity sign recovery và scale nghiệm được giữ lại.
    - Các hàm phụ của VQLS được đặt bên trong vqls_solver(A, b), để phía ngoài
      chỉ còn hai module solver chính.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
from qiskit.circuit.library import StatePreparation
import matplotlib.pyplot as plt


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
# MODULE 2: VQLS SOLVER
# ============================================================
def vqls_solver(
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
    first_sign_anchor=1,
    return_details=True,
):
    """
    Giải hệ tuyến tính Ax = b bằng VQLS.

    Đầu vào:
        A: ma trận vuông kích thước 2^n x 2^n
        b: vector vế phải, chưa chuẩn hóa

    Quy trình chính:
        1. Kiểm tra A và b.
        2. Chuẩn hóa b thành |b> để tạo U_b bằng StatePreparation.
        3. Phân rã A sang Pauli basis bằng SparsePauliOp.from_operator.
        4. Tối ưu cost local bằng COBYLA.
        5. Đo xác suất nghiệm bằng shots.
        6. Khôi phục dấu bằng SWAP-parity + Hadamard.
        7. Scale nghiệm đã khôi phục dấu về đúng hệ Ax = b gốc.

    Đầu ra:
        Nếu return_details=False:
            x_vqls

        Nếu return_details=True:
            dict chứa:
                x_vqls
                normalized_vqls_state
                optimized_parameters
                cost_history
                probability_vector
                sign_data
                k_coeff
                pauli_decomposition
                residual
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
    # Các hàm phụ được gom bên trong VQLS module
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

        Ansatz này phù hợp với phương pháp khôi phục dấu ± vì state tạo ra là real.
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

    def cyclic_left_shift_matrix_qiskit(dim_inner):
        """
        Unitary U_shift sao cho nếu state có amplitude c[i],
        output có amplitude d[i] = c[(i + 1) mod dim].

        Đây là phép dịch vòng trái trên vector nghiệm theo thứ tự index
        của Qiskit Statevector/counts:
            index i tương ứng bitstring q[n-1]...q[0].
        """
        U_shift = np.zeros((dim_inner, dim_inner), dtype=complex)

        for i in range(dim_inner):
            U_shift[i, (i + 1) % dim_inner] = 1.0

        return U_shift

    def build_vqls_probability_circuit(weights):
        """
        Circuit đo p_i = |c_i|^2 của ansatz VQLS.

        Không đọc nghiệm bằng Statevector.data.
        Khi chạy thực tế, thêm measurement rồi lấy counts.
        """
        qc = QuantumCircuit(n_qubits)
        apply_variational(qc, weights, list(range(n_qubits)))
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
        readout = n_qubits
        main_qubits = list(range(n_qubits))
        dim_inner = 2**n_qubits

        qc = QuantumCircuit(n_qubits + 1)

        # Chuẩn bị nghiệm VQLS |x(theta)>
        apply_variational(qc, weights, main_qubits)

        # Dịch vòng vector nếu cần
        if shifted:
            qc.unitary(
                cyclic_left_shift_matrix_qiskit(dim_inner),
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

    def sample_counts_from_statevector_for_debug(qc, shots_inner=shots, seed=rng_seed):
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

        rng_inner = np.random.default_rng(seed)
        samples = rng_inner.choice(len(probs), size=int(shots_inner), p=probs)

        n_bits = qc.num_qubits
        counts = {}

        for idx in samples:
            bitstring = format(int(idx), f"0{n_bits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def run_counts(qc, shots_inner=shots, backend_inner=backend, seed=rng_seed):
        """
        Chạy circuit và trả về counts.

        Nếu backend=None:
            dùng AerSimulator để đo bằng shot.

        Nếu backend != None:
            chạy trên backend bạn truyền vào.
        """
        qc_meas = add_measure_all(qc)

        if backend_inner is None:
            try:
                from qiskit_aer import AerSimulator
            except ImportError:
                raise ImportError(
                    "Bạn cần cài qiskit-aer để chạy shot-based simulation:\n"
                    "    pip install qiskit-aer\n"
                    "hoặc truyền backend thật vào vqls_solver(..., backend=backend)."
                )

            backend_inner = AerSimulator(seed_simulator=seed)

        job = backend_inner.run(qc_meas, shots=int(shots_inner))
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
            n_qubits,
        )

        parity_couple_1 = parity_interference_from_counts(
            counts_parity_shifted,
            n_qubits,
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
        shots_inner=shots,
        backend_inner=backend,
        seed=rng_seed,
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
            shots_inner=shots_inner,
            backend_inner=backend_inner,
            seed=seed,
        )

        counts_parity_0 = run_counts(
            qc_parity_0,
            shots_inner=shots_inner,
            backend_inner=backend_inner,
            seed=seed + 1,
        )

        counts_parity_1 = run_counts(
            qc_parity_1,
            shots_inner=shots_inner,
            backend_inner=backend_inner,
            seed=seed + 2,
        )

        prob_orig = counts_to_probability_vector(counts_prob, n_qubits)

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
        A_matrix_inner,
        b_vector_inner,
    ):
        """
        Dùng cùng logic scale k của code cũ, nhưng đầu vào là state đã khôi phục dấu
        từ shot + SWAP-parity, không phải Statevector.data.

        Khác với code gốc ở điểm tổ chức:
            - b_vector_inner ở đây là b gốc, chưa chuẩn hóa.
            - Vì vậy x_recovered trả về trực tiếp nghiệm cùng thang đo với Ax = b.
        """
        x_prime = np.asarray(signed_normalized_state, dtype=complex)
        b_prime = A_matrix_inner @ x_prime

        denom = np.vdot(b_prime, b_prime)

        if abs(denom) < 1e-12:
            k_coeff = 0.0
        else:
            k_coeff = np.vdot(b_prime, b_vector_inner) / denom

        x_recovered = k_coeff * x_prime

        return x_recovered, k_coeff, b_prime

    # ============================================================
    # Main optimization loop của VQLS module
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
    # SHOT-BASED SWAP-PARITY SIGN RECOVERY
    # ============================================================
    signed_readout = extract_signed_vqls_solution_shot_based(
        w,
        first_sign=first_sign_anchor,
        shots_inner=shots,
        backend_inner=backend,
        seed=rng_seed,
    )

    x_vqls_recovered, k_coeff, b_prime = scale_signed_vqls_state_to_solution(
        signed_readout["signed_normalized_state"],
        A_matrix,
        b_vector_raw,
    )

    residual = np.linalg.norm(A_matrix @ x_vqls_recovered - b_vector_raw)

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

    print("\nRecovered VQLS solution x_VQLS for raw system Ax=b =")
    print(np.real_if_close(x_vqls_recovered))

    print(f"\nResidual check ||A·x_VQLS - b|| = {residual:.6e}")
    print(f"\nA·x_VQLS = {np.real_if_close(A_matrix @ x_vqls_recovered)}")
    print(f"b input  = {np.real_if_close(b_vector_raw)}")

    print(f"\nb' = A · x'_VQLS_shot =")
    print(np.real_if_close(b_prime))

    print("\n" + "=" * 70)
    print("SHOT-BASED SOLUTION RECOVERY COMPLETE")
    print("=" * 70)

    if not return_details:
        return x_vqls_recovered

    return {
        "x_vqls": x_vqls_recovered,
        "normalized_vqls_state": signed_readout["signed_normalized_state"],
        "optimized_parameters": w,
        "optimization_result": res,
        "cost_history": cost_history,
        "probability_vector": signed_readout["prob_orig"],
        "sign_data": signed_readout,
        "k_coeff": k_coeff,
        "b_prime": b_prime,
        "residual": residual,
        "n_qubits": n_qubits,
        "num_pauli": num_pauli,
        "pauli_decomposition": A_pauli,
        "pauli_labels": pauli_labels,
        "pauli_coeffs": C,
        "b_norm": b_norm,
    }


# ============================================================
# PHẦN SO SÁNH OUTPUT CỦA HAI MODULE
# ============================================================
if __name__ == "__main__":
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

    print("\n" + "=" * 70)
    print("MODULE 1: CLASSICAL SOLVER")
    print("=" * 70)

    x_classical = classical_solver(A_MATRIX, B_VECTOR_RAW)

    print("\nClassical solution x_classical =")
    print(np.real_if_close(x_classical))

    print("\n" + "=" * 70)
    print("MODULE 2: VQLS SOLVER")
    print("=" * 70)

    vqls_result = vqls_solver(
        A_MATRIX,
        B_VECTOR_RAW,
        shots=N_SHOTS,
        steps=STEPS,
        q_delta=Q_DELTA,
        rng_seed=RNG_SEED,
        backend=None,
        first_sign_anchor=1,
        return_details=True,
    )

    x_vqls = vqls_result["x_vqls"]

    print("\n" + "=" * 70)
    print("SO SÁNH OUTPUT HAI MODULE")
    print("=" * 70)

    print("\nClassical solution x_classical =")
    print(np.real_if_close(x_classical))

    print("\nRecovered VQLS solution x_vqls =")
    print(np.real_if_close(x_vqls))

    abs_err = np.linalg.norm(x_vqls - x_classical)
    rel_err = abs_err / np.linalg.norm(x_classical)
    residual_vqls = np.linalg.norm(A_MATRIX @ x_vqls - B_VECTOR_RAW)
    residual_classical = np.linalg.norm(A_MATRIX @ x_classical - B_VECTOR_RAW)

    print(f"\nAbsolute error ||x_VQLS - x_classical|| = {abs_err:.6e}")
    print(f"Relative error                         = {rel_err:.6e}")
    print(f"Residual VQLS ||A·x_VQLS - b||        = {residual_vqls:.6e}")
    print(f"Residual classical ||A·x_classical-b||= {residual_classical:.6e}")

    x_norm = x_classical / np.linalg.norm(x_classical)
    c_probs = np.abs(x_norm) ** 2
    q_probs = vqls_result["probability_vector"]
    dim = A_MATRIX.shape[0]

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
    axes[2].set_title("VQLS solution")

    plt.tight_layout()
    plt.savefig("vqls_results.png", dpi=120, bbox_inches="tight")
    print("\nPlot saved to: vqls_results.png")
