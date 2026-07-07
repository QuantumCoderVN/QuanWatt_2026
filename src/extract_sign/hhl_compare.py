"""
================================================================================
 HHL 2x2 — So sánh 2 phương pháp khôi phục nghiệm (bản tối giản)
================================================================================
 Mục tiêu:
   Giải hệ A x = b bằng HHL trên ma trận 2x2, sau đó khôi phục nghiệm bằng:

   (1) Phương pháp ban đầu: shot + postselection + sign detection
   (2) Phương pháp Quantum Tomography: tomography trên target register sau postselection

 Chỉ vẽ 2 hình:
   - 1 hình gồm 3 biểu đồ cột nghiệm: Classical / Shot+Sign / Tomography
   - 1 hình so sánh runtime trung bình của 2 phương pháp

 Không vẽ mạch, không lưu bảng CSV.

 Cài đặt nếu cần:
   pip install qiskit qiskit-aer numpy scipy matplotlib
================================================================================
"""

import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, RYGate

try:
    from qiskit_aer import AerSimulator
except ImportError as exc:
    raise ImportError(
        "Bạn cần cài qiskit-aer để chạy shot simulator:\n"
        "    pip install qiskit-aer"
    ) from exc


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_2x2_compare")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


# =============================================================================
# THAM SỐ
# =============================================================================

n_qubits = 1
phase_qubits = 5
ancilla_qubits = 1
tot_qubits = n_qubits + phase_qubits + ancilla_qubits

SHOTS = 50_000
SEED = 2026
SUCCESS_ANCILLA_VALUE = 1
FIRST_SIGN = +1
N_RUNTIME_REPEAT = 5  # số lần lặp để lấy runtime trung bình


# =============================================================================
# BÀI TOÁN 2x2
# =============================================================================

A_matrix = np.array(
    [
        [1.5, 0.4],
        [0.4, 2.0],
    ],
    dtype=np.complex128,
)

b_vector = np.array([1.0, -0.6], dtype=np.complex128)
b_norm = b_vector / np.linalg.norm(b_vector)

eigenvalues = np.linalg.eigvalsh(A_matrix)
assert np.allclose(A_matrix, A_matrix.conj().T), "A phải là Hermitian."
assert eigenvalues.min() > 0, "A phải xác định dương."
assert eigenvalues.max() < 2 * np.pi, "Phổ của A phải nằm trong (0, 2π)."

x_classical = np.linalg.solve(A_matrix, b_vector)
x_classical_normalized = x_classical / np.linalg.norm(x_classical)

if np.real(x_classical_normalized[0]) < 0:
    FIRST_SIGN = -1

print("=" * 80)
print("BÀI TOÁN 2x2")
print("=" * 80)
print(f"A =\n{A_matrix.real}")
print(f"b = {b_vector.real}")
print(f"Trị riêng của A = {eigenvalues}")
print(f"x_classical = {x_classical}")
print(f"x_classical_normalized = {x_classical_normalized}")


# =============================================================================
# XÂY DỰNG HHL CIRCUIT
# =============================================================================

U_matrix = expm(1j * A_matrix)
assert np.allclose(U_matrix.conj().T @ U_matrix, np.eye(2 ** n_qubits)), "U phải unitary."


def controlled_unitary_power(qc, U, power, control_qubit, target_qubits):
    U_power = np.linalg.matrix_power(U, power)
    sub = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    sub.unitary(U_power, list(range(len(target_qubits))))
    qc.append(sub.to_gate().control(1), [control_qubit] + list(target_qubits))



def qpe_circuit(qc, phase_reg, target_reg):
    for q in phase_reg:
        qc.h(q)

    for i, ctrl in enumerate(reversed(phase_reg)):
        controlled_unitary_power(qc, U_matrix, 2 ** i, ctrl, target_reg)

    qc.append(QFT(num_qubits=len(phase_reg), inverse=True, do_swaps=False), phase_reg)



def control_rotation_gate(qc, control_reg, target_qubit, C):
    n = len(control_reg)
    for k in range(1, 2 ** n):
        bin_str = f"{k:0{n}b}"
        lambda_tilde = 2 * np.pi * k / (2 ** n)
        argument = C / lambda_tilde
        if abs(argument) > 1:
            continue
        theta = 2 * np.arcsin(argument)
        cry = RYGate(theta).control(n, ctrl_state=bin_str)
        qc.append(cry, list(control_reg) + [target_qubit])



def build_hhl_circuit():
    phase_reg = QuantumRegister(phase_qubits, name="phase")
    target_reg = QuantumRegister(n_qubits, name="target")
    ancilla_reg = QuantumRegister(ancilla_qubits, name="ancilla")

    qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL_2x2")

    phase_idx = list(range(phase_qubits))
    target_idx = list(range(phase_qubits, phase_qubits + n_qubits))
    ancilla_idx = list(range(phase_qubits + n_qubits, tot_qubits))

    qc_hhl.initialize(b_norm, target_idx)
    qpe_circuit(qc_hhl, phase_idx, target_idx)

    C_value = 0.75 * eigenvalues.min()
    control_rotation_gate(qc_hhl, phase_idx, ancilla_idx[0], C=C_value)

    qc_qpe_only = QuantumCircuit(phase_qubits + n_qubits, name="QPE")
    qpe_circuit(
        qc_qpe_only,
        list(range(phase_qubits)),
        list(range(phase_qubits, phase_qubits + n_qubits)),
    )
    qc_hhl.append(qc_qpe_only.inverse(), phase_idx + target_idx)

    return qc_hhl, phase_idx, target_idx, ancilla_idx


qc_hhl, phase_idx, target_idx, ancilla_idx = build_hhl_circuit()


# =============================================================================
# HELPER CHUNG
# =============================================================================


def add_measure_all(qc_no_measure):
    qc_m = qc_no_measure.copy()
    meas = ClassicalRegister(qc_m.num_qubits, "meas")
    qc_m.add_register(meas)
    qc_m.measure(list(range(qc_m.num_qubits)), list(meas))
    return qc_m



def run_circuits_counts(circuits, shots=SHOTS, seed=SEED):
    measured = [add_measure_all(qc) for qc in circuits]
    simulator = AerSimulator(seed_simulator=seed)
    transpiled = transpile(measured, simulator)
    result = simulator.run(transpiled, shots=shots).result()
    return [result.get_counts(i) for i in range(len(measured))]



def key_to_bits_little_endian(key, n_bits):
    clean = key.replace(" ", "").zfill(n_bits)
    return [int(b) for b in clean[::-1]]



def read_register_int(bits, qubit_indices):
    value = 0
    for j, q in enumerate(qubit_indices):
        value |= int(bits[q]) << j
    return value



def read_register_bits(bits, qubit_indices):
    return [int(bits[q]) for q in qubit_indices]



def is_phase_zero(bits, phase_indices):
    return all(bits[q] == 0 for q in phase_indices)



def recover_unscaled_solution_from_normalized_state(x_state_normalized):
    b_prime = A_matrix @ x_state_normalized
    k_coeff = np.vdot(b_prime, b_vector) / np.vdot(b_prime, b_prime)
    x_recovered = k_coeff * x_state_normalized
    return x_recovered, k_coeff



def real_solution_values(x):
    """Lấy giá trị nghiệm thực để vẽ cột so sánh.

    Với bài toán này nghiệm kỳ vọng là thực. Nếu có nhiễu số nhỏ ở phần ảo,
    ta chỉ vẽ phần thực để so sánh trực tiếp với nghiệm cổ điển.
    """
    return np.real(np.asarray(x, dtype=np.complex128))


# =============================================================================
# PHƯƠNG PHÁP 1: SHOT + SIGN DETECTION
# =============================================================================


def extract_target_probabilities_from_hhl_counts(counts, total_qubits, phase_indices, target_indices, ancilla_index):
    dim = 2 ** len(target_indices)
    shots = sum(counts.values())
    prob_raw = np.zeros(dim, dtype=float)
    success_count = 0

    for key, c in counts.items():
        bits = key_to_bits_little_endian(key, total_qubits)
        if not is_phase_zero(bits, phase_indices):
            continue
        if bits[ancilla_index] != SUCCESS_ANCILLA_VALUE:
            continue
        target_value = read_register_int(bits, target_indices)
        prob_raw[target_value] += c / shots
        success_count += c

    if prob_raw.sum() <= 0:
        raise RuntimeError("Không có shot nào thỏa hậu chọn phase=0 và ancilla=1.")

    return prob_raw, prob_raw / prob_raw.sum(), success_count



def build_sign_detect_circuit_2x2(qc_base_hhl, target_indices):
    qc_sd = qc_base_hhl.copy(name="HHL_sign_detect_0_1")
    sign_extra = QuantumRegister(1, "sign_extra")
    qc_sd.add_register(sign_extra)
    extra_idx = qc_sd.num_qubits - 1
    qc_sd.swap(target_indices[0], extra_idx)
    qc_sd.h(extra_idx)
    return qc_sd, extra_idx



def extract_plus_interference_probability_2x2(counts_sd, total_qubits_sd, phase_indices, target_indices, ancilla_index, extra_index):
    success_count = 0
    plus_count = 0

    for key, c in counts_sd.items():
        bits = key_to_bits_little_endian(key, total_qubits_sd)
        if not is_phase_zero(bits, phase_indices):
            continue
        if bits[ancilla_index] != SUCCESS_ANCILLA_VALUE:
            continue

        success_count += c
        target_value = read_register_int(bits, target_indices)
        if target_value == 0 and bits[extra_index] == 0:
            plus_count += c

    if success_count <= 0:
        raise RuntimeError("Sign-detect không có shot hậu chọn thành công.")

    two_p_plus = 2.0 * (plus_count / success_count)
    return two_p_plus, success_count



def recover_by_original_sign_method(qc_base_hhl, phase_indices, target_indices, ancilla_index, shots=SHOTS, seed=SEED, first_sign=FIRST_SIGN):
    start = time.perf_counter()

    qc_sd, extra_idx = build_sign_detect_circuit_2x2(qc_base_hhl, target_indices)
    counts_hhl, counts_sd = run_circuits_counts([qc_base_hhl, qc_sd], shots=shots, seed=seed)

    _, prob_norm, hhl_success_count = extract_target_probabilities_from_hhl_counts(
        counts_hhl, qc_base_hhl.num_qubits, phase_indices, target_indices, ancilla_index
    )

    abs_amplitudes = np.sqrt(np.maximum(prob_norm, 0.0))
    two_p_plus, sd_success_count = extract_plus_interference_probability_2x2(
        counts_sd, qc_sd.num_qubits, phase_indices, target_indices, ancilla_index, extra_idx
    )

    relative_sign = +1 if two_p_plus >= 1.0 else -1
    signs = np.array([+1 if first_sign >= 0 else -1, relative_sign], dtype=int)
    signs[1] *= signs[0]

    signed_state = (signs * abs_amplitudes).astype(np.complex128)
    signed_state = signed_state / np.linalg.norm(signed_state)
    x_recovered, k_coeff = recover_unscaled_solution_from_normalized_state(signed_state)

    return {
        "method": "Shot + sign",
        "normalized_state": signed_state,
        "x_recovered": x_recovered,
        "hhl_success_count": hhl_success_count,
        "sd_success_count": sd_success_count,
        "k_coeff": k_coeff,
        "runtime_sec": time.perf_counter() - start,
    }


# =============================================================================
# PHƯƠNG PHÁP 2: TOMOGRAPHY
# =============================================================================


def all_pauli_strings(n):
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)]



def unique_tomography_bases(n):
    return ["".join(b) for b in itertools.product("XYZ", repeat=n)]



def pauli_to_measurement_basis(pauli_string):
    return "".join("Z" if p == "I" else p for p in pauli_string)



def add_basis_rotation_on_target(qc, target_indices, basis):
    for j, b in enumerate(basis):
        q = target_indices[j]
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
        elif b == "Z":
            pass
        else:
            raise ValueError(f"Basis không hợp lệ: {b}")



def make_tomography_basis_circuit(qc_base_hhl, target_indices, basis):
    qc_tomo = qc_base_hhl.copy(name=f"HHL_tomo_{basis}")
    add_basis_rotation_on_target(qc_tomo, target_indices, basis)
    return qc_tomo



def pauli_eigenvalue_from_target_bits(pauli_string, target_bits):
    eig = 1
    for p, bit in zip(pauli_string, target_bits):
        if p == "I":
            continue
        eig *= +1 if bit == 0 else -1
    return eig



def estimate_pauli_expectation_postselected(pauli_string, counts, total_qubits, phase_indices, target_indices, ancilla_index):
    numerator = 0.0
    success_count = 0

    for key, c in counts.items():
        bits = key_to_bits_little_endian(key, total_qubits)
        if not is_phase_zero(bits, phase_indices):
            continue
        if bits[ancilla_index] != SUCCESS_ANCILLA_VALUE:
            continue

        target_bits = read_register_bits(bits, target_indices)
        eig = pauli_eigenvalue_from_target_bits(pauli_string, target_bits)
        numerator += eig * c
        success_count += c

    if success_count <= 0:
        raise RuntimeError("Tomography không có shot hậu chọn thành công.")

    return numerator / success_count



def single_pauli_matrix(p):
    if p == "I":
        return np.array([[1, 0], [0, 1]], dtype=np.complex128)
    if p == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if p == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if p == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(f"Pauli không hợp lệ: {p}")



def pauli_matrix(pauli_string):
    mats = [single_pauli_matrix(p) for p in pauli_string]
    result = mats[0]
    for mat in mats[1:]:
        result = np.kron(result, mat)
    return result



def project_to_physical_density_matrix(rho):
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = eigvals / eigvals.sum()
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T



def extract_real_amplitudes_from_density_matrix(rho, reference_sign=FIRST_SIGN):
    p_ref = np.real(rho[0, 0])
    if p_ref <= 0:
        raise RuntimeError("rho[0,0] <= 0, không thể dùng |0> làm reference.")

    a_ref = (+1 if reference_sign >= 0 else -1) * np.sqrt(p_ref)
    amps = np.zeros(rho.shape[0], dtype=np.complex128)
    for i in range(rho.shape[0]):
        amps[i] = np.real(rho[0, i]) / a_ref
    return amps / np.linalg.norm(amps)



def recover_by_tomography_method(qc_base_hhl, phase_indices, target_indices, ancilla_index, shots=SHOTS, seed=SEED):
    start = time.perf_counter()

    n_target = len(target_indices)
    dim = 2 ** n_target

    bases = unique_tomography_bases(n_target)
    tomo_circuits = [make_tomography_basis_circuit(qc_base_hhl, target_indices, b) for b in bases]
    counts_list = run_circuits_counts(tomo_circuits, shots=shots, seed=seed + 1000)
    counts_by_basis = dict(zip(bases, counts_list))

    pauli_expectations = {}
    for P in all_pauli_strings(n_target):
        if P == "I" * n_target:
            pauli_expectations[P] = 1.0
        else:
            basis = pauli_to_measurement_basis(P)
            pauli_expectations[P] = estimate_pauli_expectation_postselected(
                P, counts_by_basis[basis], qc_base_hhl.num_qubits, phase_indices, target_indices, ancilla_index
            )

    rho = np.zeros((dim, dim), dtype=np.complex128)
    for P, expval in pauli_expectations.items():
        rho += expval * pauli_matrix(P)
    rho = rho / dim
    rho = project_to_physical_density_matrix(rho)

    x_state = extract_real_amplitudes_from_density_matrix(rho, reference_sign=FIRST_SIGN)
    x_recovered, k_coeff = recover_unscaled_solution_from_normalized_state(x_state)

    return {
        "method": "Tomography",
        "normalized_state": x_state,
        "x_recovered": x_recovered,
        "k_coeff": k_coeff,
        "rho": rho,
        "runtime_sec": time.perf_counter() - start,
    }


# =============================================================================
# PLOT CHỈ 2 HÌNH
# =============================================================================


def plot_solution_value_comparison(x_cls, x_original, x_tomo):
    """Vẽ 1 hình duy nhất với 3 nhóm cột để so sánh giá trị nghiệm đã khôi phục.

    So sánh trực tiếp giá trị nghiệm (không phải probability):
      - nghiệm cổ điển
      - nghiệm khôi phục bằng shot + sign
      - nghiệm khôi phục bằng tomography
    """
    components = np.arange(2)
    labels = ["x0", "x1"]
    width = 0.24

    classical_vals = real_solution_values(x_cls)
    shot_vals = real_solution_values(x_original)
    tomo_vals = real_solution_values(x_tomo)

    color_classical = "#4C78A8"   # xanh dương
    color_shot = "#F58518"        # cam
    color_tomo = "#54A24B"        # xanh lá

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(components - width, classical_vals, width,
                   label="Classical", color=color_classical,
                   edgecolor="black", linewidth=1.0, alpha=0.95)
    bars2 = ax.bar(components, shot_vals, width,
                   label="Shot + sign", color=color_shot,
                   edgecolor="black", linewidth=1.0, alpha=0.95)
    bars3 = ax.bar(components + width, tomo_vals, width,
                   label="Tomography", color=color_tomo,
                   edgecolor="black", linewidth=1.0, alpha=0.95)

    ax.set_title("So sánh giá trị nghiệm sau khi khôi phục", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thành phần nghiệm")
    ax.set_ylabel("Giá trị nghiệm")
    ax.set_xticks(components)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True)

    def annotate_bars(bar_container):
        for bar in bar_container:
            h = bar.get_height()
            offset = 0.02 * max(1.0, np.max(np.abs([classical_vals, shot_vals, tomo_vals])))
            y = h + offset if h >= 0 else h - offset
            va = "bottom" if h >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, y, f"{h:.4f}",
                    ha="center", va=va, fontsize=9)

    annotate_bars(bars1)
    annotate_bars(bars2)
    annotate_bars(bars3)

    plt.tight_layout()
    save_path = out("solution_value_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Đã lưu hình nghiệm: {save_path}")



def plot_average_runtime(runtime_sign_list, runtime_tomo_list):
    methods = ["Shot + sign", "Tomography"]
    means = [np.mean(runtime_sign_list), np.mean(runtime_tomo_list)]
    stds = [np.std(runtime_sign_list), np.std(runtime_tomo_list)]
    colors = ["#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="black", linewidth=1.0, alpha=0.95)
    ax.set_title(f"Runtime trung bình sau {N_RUNTIME_REPEAT} lần chạy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Thời gian chạy (giây)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds) * 0.15 + 1e-6,
                f"{mean:.4f}s", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_path = out("runtime_average_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Đã lưu hình runtime: {save_path}")


# =============================================================================
# RUN
# =============================================================================

print("\n" + "=" * 80)
print("CHẠY MỖI PHƯƠNG PHÁP 1 LẦN ĐỂ LẤY NGHIỆM KHÔI PHỤC")
print("=" * 80)

original_result = recover_by_original_sign_method(
    qc_base_hhl=qc_hhl,
    phase_indices=phase_idx,
    target_indices=target_idx,
    ancilla_index=ancilla_idx[0],
    shots=SHOTS,
    seed=SEED,
    first_sign=FIRST_SIGN,
)

tomo_result = recover_by_tomography_method(
    qc_base_hhl=qc_hhl,
    phase_indices=phase_idx,
    target_indices=target_idx,
    ancilla_index=ancilla_idx[0],
    shots=SHOTS,
    seed=SEED,
)

print(f"x_classical                = {x_classical}")
print(f"x_recovered (shot + sign)  = {original_result['x_recovered']}")
print(f"x_recovered (tomography)   = {tomo_result['x_recovered']}")

plot_solution_value_comparison(
    x_cls=x_classical,
    x_original=original_result["x_recovered"],
    x_tomo=tomo_result["x_recovered"],
)

print("\n" + "=" * 80)
print("ĐO RUNTIME TRUNG BÌNH")
print("=" * 80)

runtime_sign_list = []
runtime_tomo_list = []

for i in range(N_RUNTIME_REPEAT):
    r1 = recover_by_original_sign_method(
        qc_base_hhl=qc_hhl,
        phase_indices=phase_idx,
        target_indices=target_idx,
        ancilla_index=ancilla_idx[0],
        shots=SHOTS,
        seed=SEED + 10 * i,
        first_sign=FIRST_SIGN,
    )
    runtime_sign_list.append(r1["runtime_sec"])

    r2 = recover_by_tomography_method(
        qc_base_hhl=qc_hhl,
        phase_indices=phase_idx,
        target_indices=target_idx,
        ancilla_index=ancilla_idx[0],
        shots=SHOTS,
        seed=SEED + 100 + 10 * i,
    )
    runtime_tomo_list.append(r2["runtime_sec"])

print(f"Runtime shot + sign (s) = {runtime_sign_list}")
print(f"Runtime tomography (s) = {runtime_tomo_list}")
print(f"Trung bình shot + sign = {np.mean(runtime_sign_list):.6f} s")
print(f"Trung bình tomography  = {np.mean(runtime_tomo_list):.6f} s")

plot_average_runtime(runtime_sign_list, runtime_tomo_list)

if np.mean(runtime_sign_list) < np.mean(runtime_tomo_list):
    print("\n=> Shot + sign chạy nhanh hơn (trung bình).")
else:
    print("\n=> Tomography chạy nhanh hơn (trung bình).")
