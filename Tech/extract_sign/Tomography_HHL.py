"""
================================================================================
 HHL Algorithm — Qiskit Implementation
================================================================================
 Giải hệ phương trình tuyến tính A·x = b bằng thuật toán Harrow-Hassidim-Lloyd
 (HHL) trên Qiskit, sau đó hồi phục nghiệm cổ điển (không chuẩn hóa) bằng
 phương pháp hệ số tỉ lệ.

 Giả định: A là ma trận Hermitian, xác định dương, có phổ nằm gọn trong
 khoảng (0, 2π). Đây là điều kiện chuẩn của HHL — ta không xử lý shift/scale
 để giữ code đơn giản, tập trung vào ý tưởng cốt lõi.

 Pipeline:
   1. Khởi tạo |b⟩ trên target register.
   2. Quantum Phase Estimation (QPE) ước lượng trị riêng của A.
   3. Controlled-RY xoay ancilla theo nghịch đảo trị riêng (1/λ).
   4. Inverse QPE để giải vướng phase register.
   5. Hậu chọn ancilla = |1⟩ → trạng thái target ∝ A⁻¹|b⟩.
   6. Hồi phục biên độ thực bằng hệ số k: x = k·x'.

 Tác giả: [Tên của bạn]
================================================================================
"""

import os
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, RYGate
from qiskit.quantum_info import Statevector


# ══════════════════════════════════════════════════════════════════════════════
#  CẤU HÌNH OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename: str) -> str:
    """Trả về đường dẫn đầy đủ tới file trong thư mục outputs/."""
    return os.path.join(OUTPUT_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  THAM SỐ THUẬT TOÁN
# ══════════════════════════════════════════════════════════════════════════════

n_qubits      = 2   # Số qubit hệ thống (encode |b⟩, kích thước 2^n_qubits = dim A)
phase_qubits  = 5   # Số qubit cho QPE — càng nhiều, ước lượng trị riêng càng chính xác
ancilla_qubit = 1   # Qubit ancilla cho phép xoay điều kiện
tot_qubits    = n_qubits + phase_qubits + ancilla_qubit


# ══════════════════════════════════════════════════════════════════════════════
#  BÀI TOÁN: Ax = b
# ══════════════════════════════════════════════════════════════════════════════
#  Chọn A Hermitian, xác định dương, phổ ⊂ (0, 2π) — thỏa mãn điều kiện HHL.
# ══════════════════════════════════════════════════════════════════════════════

A_matrix = np.array([
    [4.0,  0.4,  0.2,  0.0],
    [0.4,  5.0, -0.3,  0.1],
    [0.2, -0.3,  3.5,  0.5],
    [0.0,  0.1,  0.5,  4.5]
], dtype=np.complex128)

# Vector b có cả dấu dương và âm để test sign recovery
b_vector = np.array([0.03, -0.02, 0.04, -0.01], dtype=np.complex128)

b_norm = b_vector / np.linalg.norm(b_vector)   # |b⟩ chuẩn hóa cho mạch lượng tử

# Kiểm tra điều kiện
eigenvalues = np.linalg.eigvalsh(A_matrix)
assert np.allclose(A_matrix, A_matrix.conj().T), "A phải là Hermitian!"
assert eigenvalues.min() > 0,            "A phải xác định dương!"
assert eigenvalues.max() < 2 * np.pi,    "Phổ của A phải nằm trong (0, 2π)!"

print("=" * 70)
print(" THÔNG SỐ BÀI TOÁN")
print("=" * 70)
print(f" A =\n{A_matrix.real}")
print(f" b              = {b_vector.real}")
print(f" ||b||          = {np.linalg.norm(b_vector):.6f}")
print(f" b̂ (chuẩn hóa)  = {b_norm.real}")
print(f" Trị riêng của A = {eigenvalues}")


# ══════════════════════════════════════════════════════════════════════════════
#  TOÁN TỬ TIẾN HÓA U = exp(iA)
# ══════════════════════════════════════════════════════════════════════════════
#  QPE sẽ ước lượng pha φ của eigenvector |u⟩: U|u⟩ = e^{i·2π·φ}|u⟩.
#  Với U = e^{iA}, ta có e^{iA}|u⟩ = e^{iλ}|u⟩, tức 2π·φ = λ ⇒ φ = λ/(2π).
#  Phép đo k qubit của QPE cho λ̃ = 2π·k/2^n.
# ══════════════════════════════════════════════════════════════════════════════

U_matrix = expm(1j * A_matrix)

assert np.allclose(U_matrix.conj().T @ U_matrix, np.eye(2 ** n_qubits)), \
    "U = exp(iA) phải là unitary!"

print(f"\n U = exp(iA) =\n{np.round(U_matrix, 4)}")


# ══════════════════════════════════════════════════════════════════════════════
#  MẠCH CON: Controlled-U^(2^k)
# ══════════════════════════════════════════════════════════════════════════════

def controlled_unitary_power(qc: QuantumCircuit,
                              U: np.ndarray,
                              power: int,
                              control_qubit: int,
                              target_qubits: list) -> None:
    """Áp dụng controlled-U^power lên target_qubits, điều khiển bởi control_qubit.

    Tham số
    -------
    qc            : mạch chính.
    U             : ma trận unitary cơ sở (kích thước 2^len(target_qubits)).
    power         : số mũ — luôn là lũy thừa của 2 trong QPE.
    control_qubit : chỉ số qubit điều khiển.
    target_qubits : danh sách chỉ số qubit đích.
    """
    U_power = np.linalg.matrix_power(U, power)
    sub = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    sub.unitary(U_power, list(range(len(target_qubits))))
    qc.append(sub.to_gate().control(1), [control_qubit] + list(target_qubits))


# ══════════════════════════════════════════════════════════════════════════════
#  MẠCH CON: Quantum Phase Estimation (QPE)
# ══════════════════════════════════════════════════════════════════════════════

def qpe_circuit(qc: QuantumCircuit,
                phase_reg: list,
                target_reg: list) -> None:
    """Quantum Phase Estimation trên phase_reg, target ban đầu chứa eigenvector của U.

    Sau QPE, phase_reg chứa biểu diễn nhị phân của φ = λ/(2π) (với t = 1).
    """
    # 1. Hadamard trên phase register → siêu vị đều
    for q in phase_reg:
        qc.h(q)

    # 2. Controlled-U^{2^k} — qubit pha có trọng số cao điều khiển U^{2^(n-1)}
    for i, ctrl in enumerate(reversed(phase_reg)):
        controlled_unitary_power(qc, U_matrix, 2 ** i, ctrl, target_reg)

    # 3. Inverse QFT để chuyển từ miền pha về miền tính toán
    qc.append(QFT(num_qubits=len(phase_reg), inverse=True, do_swaps=False),
              phase_reg)


# ══════════════════════════════════════════════════════════════════════════════
#  MẠCH CON: Controlled-RY (Amplitude Encoding của 1/λ)
# ══════════════════════════════════════════════════════════════════════════════

def control_rotation_gate(qc: QuantumCircuit,
                          control_reg: list,
                          target_qubit: int,
                          C: float) -> None:
    """Xoay ancilla theo θ = 2·arcsin(C/λ̃), điều kiện trên giá trị trong phase_reg.

    Với phase_reg lưu số nguyên k tương ứng λ̃ = 2π·k/2^n (t = 1), ta có:

        θ = 2 · arcsin(C · 2^n / (2π · k))

    Hằng số C được chọn nhỏ hơn λ_min ước lượng để đảm bảo argument ≤ 1.
    Sau xoay: |ancilla=1⟩ có biên độ ∝ C/λ ⇒ amplitude của |x'⟩ ∝ A⁻¹|b⟩.
    """
    n = len(control_reg)

    for k in range(1, 2 ** n):  # bỏ k = 0 (tương ứng λ = 0, không xoay)
        bin_str  = f"{k:0{n}b}"
        argument = C * (2 ** n) / (2 * np.pi * k)

        if abs(argument) > 1:
            continue  # bỏ qua nếu argument vượt miền arcsin

        theta = 2 * np.arcsin(argument)
        cry   = RYGate(theta).control(n, ctrl_state=bin_str)
        qc.append(cry, list(control_reg) + [target_qubit])


# ══════════════════════════════════════════════════════════════════════════════
#  XÂY DỰNG MẠCH HHL
# ══════════════════════════════════════════════════════════════════════════════

phase_reg   = QuantumRegister(phase_qubits,  name='phase')
target_reg  = QuantumRegister(n_qubits,      name='target')
ancilla_reg = QuantumRegister(ancilla_qubit, name='ancilla')

qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL")

# Chỉ số qubit tuyệt đối (Qiskit dùng little-endian theo thứ tự khai báo register)
phase_idx   = list(range(phase_qubits))
target_idx  = list(range(phase_qubits, phase_qubits + n_qubits))
ancilla_idx = list(range(phase_qubits + n_qubits, tot_qubits))

# ── Bước 1: Khởi tạo |b⟩ trên target ───────────────────────────────────────────
qc_hhl.initialize(b_norm, target_idx)
qc_hhl.barrier(label="QPE")

# ── Bước 2: QPE ────────────────────────────────────────────────────────────────
qpe_circuit(qc_hhl, phase_idx, target_idx)
qc_hhl.barrier(label="AQE")

# ── Bước 3: Amplitude encoding 1/λ qua controlled-RY ──────────────────────────
#  C nhỏ hơn λ_min ước lượng (≈ 2π/2^n) để đảm bảo argument arcsin hợp lệ.
C_value = 0.9 * (2 * np.pi / 2 ** phase_qubits)
control_rotation_gate(qc_hhl, phase_idx, ancilla_idx[0], C=C_value)
qc_hhl.barrier(label="QPE†")

# ── Bước 4: Inverse QPE ────────────────────────────────────────────────────────
qc_qpe_only = QuantumCircuit(phase_qubits + n_qubits, name="QPE")
qpe_circuit(qc_qpe_only,
            list(range(phase_qubits)),
            list(range(phase_qubits, phase_qubits + n_qubits)))
inv_qpe = qc_qpe_only.inverse()
inv_qpe.name = "QPE†"
qc_hhl.append(inv_qpe, phase_idx + target_idx)

qc_hhl.barrier()


# ══════════════════════════════════════════════════════════════════════════════
#  VẼ MẠCH
# ══════════════════════════════════════════════════════════════════════════════
#  Vẽ 2 phiên bản:
#    (a) Top-level: thấy rõ các block QPE / AQE / QPE†.
#    (b) Decomposed: chi tiết từng cổng — hữu ích cho debug / báo cáo.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" VẼ MẠCH HHL")
print("=" * 70)

# (a) Mạch tổng quan — block-level
fig_top, ax_top = plt.subplots(figsize=(14, 5))
qc_hhl.draw(output='mpl',
            ax=ax_top,
            style='iqp',
            fold=-1,
            initial_state=True)
ax_top.set_title("HHL Circuit — Top-Level View", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out("hhl_circuit_toplevel.png"), dpi=200, bbox_inches='tight')
print(f" → Đã lưu: {out('hhl_circuit_toplevel.png')}")

# (b) Mạch sau khi decompose 1 lần — thấy QPE bung ra
fig_dec, ax_dec = plt.subplots(figsize=(22, 6))
qc_hhl.decompose().draw(output='mpl',
                         ax=ax_dec,
                         style='iqp',
                         fold=80,
                         initial_state=True)
ax_dec.set_title("HHL Circuit — Decomposed View", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out("hhl_circuit_decomposed.png"), dpi=200, bbox_inches='tight')
print(f" → Đã lưu: {out('hhl_circuit_decomposed.png')}")

plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MÔ PHỎNG STATEVECTOR
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" MÔ PHỎNG STATEVECTOR")
print("=" * 70)

sv       = Statevector.from_instruction(qc_hhl)
sv_array = sv.data


# ══════════════════════════════════════════════════════════════════════════════
#  TRÍCH TRẠNG THÁI |x'⟩ TỪ STATEVECTOR
# ══════════════════════════════════════════════════════════════════════════════
#  Qiskit xếp qubit theo little-endian, thứ tự register: [phase, target, ancilla].
#  Chỉ số basis state |ancilla, target, phase⟩ trong sv:
#
#       idx = (ancilla << (phase_qubits + n_qubits))
#           | (target  << phase_qubits)
#           |  phase
#
#  Sau Inverse QPE, phase trở về |0...0⟩ trên các thành phần đúng,
#  và ta hậu chọn ancilla = 1 để lấy |x'⟩ ∝ A⁻¹|b⟩.
# ══════════════════════════════════════════════════════════════════════════════

x_prime_raw = np.zeros(2 ** n_qubits, dtype=np.complex128)
phase_zero  = 0
for t in range(2 ** n_qubits):
    idx = (1 << (phase_qubits + n_qubits)) | (t << phase_qubits) | phase_zero
    x_prime_raw[t] = sv_array[idx]

print("\n[1] Trạng thái |x'⟩ trích trực tiếp từ statevector (chưa chuẩn hóa):")
print(f"    x'_raw = {x_prime_raw}")
print(f"    ||x'_raw|| = {np.linalg.norm(x_prime_raw):.6f}")
print( "    (Norm < 1 vì xác suất hậu chọn ancilla=|1⟩ < 1)")


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN A: SO SÁNH DẠNG CHUẨN HÓA
# ══════════════════════════════════════════════════════════════════════════════
#  Mục đích: kiểm tra HHL có cho ra ĐÚNG HƯỚNG của nghiệm không.
#  Cách: chuẩn hóa cả x' (lượng tử) và x_classical, rồi so sánh.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" [PHẦN A] SO SÁNH DẠNG CHUẨN HÓA (kiểm tra hướng nghiệm)")
print("=" * 70)

# Chuẩn hóa |x'⟩ lượng tử
x_prime_normalized = x_prime_raw / np.linalg.norm(x_prime_raw)

# Nghiệm cổ điển + chuẩn hóa
x_classical            = np.linalg.solve(A_matrix, b_vector)
x_classical_normalized = x_classical / np.linalg.norm(x_classical)
print (f"\n[2] Nghiệm cổ điển: x_cls = {x_classical}")
# Khử pha gauge toàn cục để so sánh công bằng
# (HHL cho |x'⟩ tới một pha toàn cục e^{iφ} — không quan sát được vật lý)
phase_gauge          = np.exp(-1j * np.angle(x_prime_normalized[0]))
x_prime_aligned      = x_prime_normalized * phase_gauge
phase_gauge_cls      = np.exp(-1j * np.angle(x_classical_normalized[0]))
x_classical_aligned  = x_classical_normalized * phase_gauge_cls

print(f"\n x' chuẩn hóa (lượng tử)   = {x_prime_aligned}")
print(f" x  chuẩn hóa (cổ điển)    = {x_classical_aligned}")

# Độ trung thực (fidelity) giữa hai trạng thái
fidelity = np.abs(np.vdot(x_prime_normalized, x_classical_normalized)) ** 2
print(f"\n Fidelity |⟨x'|x_cls⟩|²    = {fidelity:.6f}")
print(f" (1.0 = trùng khớp hoàn toàn về hướng)")

# So sánh xác suất
print(f"\n |x'_i|² (lượng tử)  = {np.abs(x_prime_normalized) ** 2}")
print(f" |x_i|²  (cổ điển)   = {np.abs(x_classical_normalized) ** 2}")


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN C: HỒI PHỤC NGHIỆM BẰNG QUANTUM TOMOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════
#  Module này THAY THẾ hoàn toàn phương pháp shot + sign-detection cũ.
#
#  Ý tưởng:
#    1. Với mỗi cơ sở tomography trên target register: basis ∈ {X,Y,Z}^n,
#       copy qc_hhl, xoay riêng các qubit target rồi đo toàn bộ mạch.
#    2. Từ counts, hậu chọn:
#           phase   = |0...0⟩
#           ancilla = |1⟩
#       Khi đó target register chứa trạng thái nghiệm chuẩn hóa |x'⟩.
#    3. Ước lượng tất cả kỳ vọng Pauli <P> của target state.
#    4. Dựng lại ma trận mật độ:
#           rho_target = (1 / 2^n) * Σ_P <P> P
#    5. Vì nghiệm HHL trong ví dụ này là nghiệm thực, lấy signed amplitudes từ:
#           rho_0x = a_0 * a_x
#       với quy ước a_0 > 0.
#    6. Dùng lại hệ số k để khôi phục nghiệm không chuẩn hóa:
#           x = k * x'
#
#  Lưu ý:
#    - Full tomography cần 3^n mạch đo, nên chỉ thực tế cho n nhỏ.
#    - Ở đây tomography được thực hiện trên TARGET sau hậu chọn, không tomography
#      toàn bộ register phase+target+ancilla.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" [PHẦN C] HỒI PHỤC NGHIỆM BẰNG QUANTUM TOMOGRAPHY")
print("=" * 70)

import itertools
import pandas as pd
from qiskit import ClassicalRegister, transpile

try:
    from qiskit_aer import AerSimulator
except ImportError as exc:
    raise ImportError(
        "Bạn cần cài qiskit-aer để chạy shot simulator:\n"
        "    pip install qiskit-aer"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
#  CẤU HÌNH TOMOGRAPHY
# ─────────────────────────────────────────────────────────────────────────────

TOMO_SHOTS = 300_000          # Tăng lên 500_000 hoặc 1_000_000 nếu nhiễu shot lớn
TOMO_REFERENCE_SIGN = +1      # Quy ước pha/dấu toàn cục: a_0 > 0
SUCCESS_ANCILLA_VALUE = 1     # HHL hậu chọn ancilla = |1⟩
TOMO_SEED = 2026
MAKE_RHO_PHYSICAL = True      # Chiếu rho về PSD trace-one để giảm nhiễu tuyến tính


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: ĐỌC BIT THEO LITTLE-ENDIAN CỦA QISKIT
# ─────────────────────────────────────────────────────────────────────────────

def key_to_bits_little_endian(key: str, n_bits: int) -> list:
    """Đổi key counts của Qiskit sang list bit theo index qubit.

    Qiskit hiển thị bitstring theo big-endian; đảo chuỗi để bits[q]
    đúng với qubit index q.
    """
    clean = key.replace(" ", "").zfill(n_bits)
    return [int(b) for b in clean[::-1]]


def read_register_int(bits: list, qubit_indices: list) -> int:
    """Đọc giá trị nguyên của một register theo little-endian Qiskit."""
    value = 0
    for j, q in enumerate(qubit_indices):
        value |= int(bits[q]) << j
    return value


def is_phase_zero(bits: list, phase_indices: list) -> bool:
    """Kiểm tra phase register = |0...0⟩."""
    return all(bits[q] == 0 for q in phase_indices)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: PAULI STRINGS VÀ MA TRẬN PAULI
# ─────────────────────────────────────────────────────────────────────────────

def all_bitstrings(n: int) -> list:
    """Trả về bitstrings theo thứ tự computational basis: 000, 001, ..."""
    return [format(i, f"0{n}b") for i in range(2 ** n)]


def all_pauli_strings(n: int) -> list:
    """Tất cả Pauli strings độ dài n trên alphabet {I, X, Y, Z}."""
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)]


def unique_tomography_bases(n: int) -> list:
    """Các cơ sở đo tomography đầy đủ: {X, Y, Z}^n."""
    return ["".join(b) for b in itertools.product("XYZ", repeat=n)]


def pauli_to_measurement_basis(pauli_string: str) -> str:
    """Đổi Pauli string thành cơ sở đo tương thích.

    Nếu ký tự là I, chọn Z vì qubit đó không ảnh hưởng eigenvalue.
    """
    return "".join("Z" if p == "I" else p for p in pauli_string)


def single_pauli_matrix(p: str) -> np.ndarray:
    """Ma trận 2x2 của I, X, Y, Z."""
    if p == "I":
        return np.array([[1, 0], [0, 1]], dtype=complex)
    if p == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if p == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if p == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError(f"Unknown Pauli: {p}")


def pauli_matrix(pauli_string: str) -> np.ndarray:
    """Ma trận Pauli tensor product ứng với pauli_string.

    pauli_string được hiểu theo thứ tự q_{n-1} ... q_1 q_0,
    khớp với thứ tự basis |000>, |001>, ...
    """
    mats = [single_pauli_matrix(p) for p in pauli_string]
    result = mats[0]
    for mat in mats[1:]:
        result = np.kron(result, mat)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  MẠCH TOMOGRAPHY: XOAY RIÊNG TARGET REGISTER
# ─────────────────────────────────────────────────────────────────────────────

def add_target_basis_rotation(qc: QuantumCircuit,
                              target_indices: list,
                              basis: str) -> None:
    """Thêm xoay để đo target theo basis X/Y/Z.

    basis string dùng thứ tự q_{n-1} ... q_1 q_0 của TARGET.
    target_indices lại là little-endian: [q_0, q_1, ..., q_{n-1}].
    Vì vậy basis[i] tác động lên target_indices[n-1-i].
    """
    n = len(target_indices)

    for i, b in enumerate(basis):
        qubit = target_indices[n - 1 - i]

        if b == "X":
            qc.h(qubit)
        elif b == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif b == "Z":
            pass
        else:
            raise ValueError(f"Unknown tomography basis character: {b}")


def make_hhl_tomography_circuit(qc_base_hhl: QuantumCircuit,
                                target_indices: list,
                                basis: str) -> QuantumCircuit:
    """Tạo một mạch tomography cho target state sau HHL.

    Mạch gồm:
      1. qc_hhl gốc,
      2. xoay target theo basis,
      3. đo toàn bộ qubit để có thể hậu chọn phase và ancilla.
    """
    qc_tomo = qc_base_hhl.copy(name=f"HHL_tomo_{basis}")
    add_target_basis_rotation(qc_tomo, target_indices, basis)

    meas = ClassicalRegister(qc_tomo.num_qubits, f"meas_{basis}")
    qc_tomo.add_register(meas)
    qc_tomo.measure(list(range(qc_tomo.num_qubits)), list(meas))

    return qc_tomo


def postselected_target_counts(counts: dict,
                               total_qubits: int,
                               phase_indices: list,
                               target_indices: list,
                               ancilla_index: int,
                               success_ancilla_value: int = 1) -> tuple:
    """Lọc counts theo phase=0...0 và ancilla=success, trả về counts target.

    target_counts dùng key bitstring theo thứ tự q_{n-1}...q_0, ví dụ "10".
    """
    n_target = len(target_indices)
    target_counts = {}
    success_count = 0

    for key, c in counts.items():
        bits = key_to_bits_little_endian(key, total_qubits)

        if not is_phase_zero(bits, phase_indices):
            continue

        if bits[ancilla_index] != success_ancilla_value:
            continue

        target_value = read_register_int(bits, target_indices)
        target_bitstring = format(target_value, f"0{n_target}b")
        target_counts[target_bitstring] = target_counts.get(target_bitstring, 0) + c
        success_count += c

    if success_count == 0:
        raise RuntimeError(
            "Không có shot nào thỏa hậu chọn phase=0 và ancilla=1. "
            "Hãy tăng TOMO_SHOTS hoặc kiểm tra C_value / mạch HHL."
        )

    return target_counts, success_count


# ─────────────────────────────────────────────────────────────────────────────
#  ƯỚC LƯỢNG <P> TỪ TARGET COUNTS ĐÃ HẬU CHỌN
# ─────────────────────────────────────────────────────────────────────────────

def pauli_eigenvalue_from_bitstring(pauli_string: str,
                                    bitstring: str) -> int:
    """Eigenvalue +/-1 của Pauli string trên một bitstring đo được.

    pauli_string và bitstring đều theo thứ tự q_{n-1}...q_0.
    Với I thì bỏ qua qubit đó.
    """
    eig = 1

    for p, bit in zip(pauli_string, bitstring):
        if p == "I":
            continue
        if bit == "0":
            eig *= +1
        elif bit == "1":
            eig *= -1
        else:
            raise ValueError(f"Invalid bit: {bit}")

    return eig


def estimate_pauli_expectation(pauli_string: str,
                               target_counts: dict) -> float:
    """Ước lượng <P> từ counts target đã hậu chọn."""
    shots = sum(target_counts.values())
    expval = 0.0

    for bitstring, count in target_counts.items():
        eig = pauli_eigenvalue_from_bitstring(pauli_string, bitstring)
        expval += eig * count / shots

    return expval


# ─────────────────────────────────────────────────────────────────────────────
#  CHIẾU RHO VỀ MA TRẬN MẬT ĐỘ VẬT LÝ
# ─────────────────────────────────────────────────────────────────────────────

def project_to_physical_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Projection đơn giản về Hermitian, PSD, trace-one.

    Đây không phải maximum-likelihood tomography đầy đủ, nhưng giúp giảm
    eigenvalue âm nhỏ do finite-shot noise.
    """
    rho = 0.5 * (rho + rho.conj().T)

    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)

    total = np.sum(eigvals)
    if total <= 0:
        raise ValueError("All eigenvalues vanished after clipping.")

    eigvals = eigvals / total
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


# ─────────────────────────────────────────────────────────────────────────────
#  FULL TOMOGRAPHY TRÊN TARGET STATE SAU HẬU CHỌN HHL
# ─────────────────────────────────────────────────────────────────────────────

def quantum_tomography_on_hhl_target(qc_base_hhl: QuantumCircuit,
                                     phase_indices: list,
                                     target_indices: list,
                                     ancilla_index: int,
                                     shots: int = TOMO_SHOTS,
                                     backend=None,
                                     seed: int = TOMO_SEED,
                                     make_physical: bool = True) -> dict:
    """Dựng rho_target bằng full Pauli-basis tomography.

    Trả về dict gồm:
      - rho
      - pauli_expectations
      - target_counts_by_basis
      - success_count_by_basis
      - counts_by_basis
    """
    n_target = len(target_indices)

    if backend is None:
        backend = AerSimulator(seed_simulator=seed)

    bases = unique_tomography_bases(n_target)
    tomography_circuits = [
        make_hhl_tomography_circuit(qc_base_hhl, target_indices, basis)
        for basis in bases
    ]

    transpiled = transpile(tomography_circuits, backend)
    result = backend.run(transpiled, shots=shots).result()

    counts_by_basis = {}
    target_counts_by_basis = {}
    success_count_by_basis = {}

    for i, basis in enumerate(bases):
        counts = result.get_counts(i)
        counts_by_basis[basis] = counts

        target_counts, success_count = postselected_target_counts(
            counts=counts,
            total_qubits=qc_base_hhl.num_qubits,
            phase_indices=phase_indices,
            target_indices=target_indices,
            ancilla_index=ancilla_index,
            success_ancilla_value=SUCCESS_ANCILLA_VALUE,
        )

        target_counts_by_basis[basis] = target_counts
        success_count_by_basis[basis] = success_count

    pauli_expectations = {}

    for P in all_pauli_strings(n_target):
        if P == "I" * n_target:
            pauli_expectations[P] = 1.0
            continue

        basis = pauli_to_measurement_basis(P)
        target_counts = target_counts_by_basis[basis]
        pauli_expectations[P] = estimate_pauli_expectation(P, target_counts)

    dim = 2 ** n_target
    rho = np.zeros((dim, dim), dtype=complex)

    for P, expval in pauli_expectations.items():
        rho += expval * pauli_matrix(P)

    rho = rho / dim
    rho = 0.5 * (rho + rho.conj().T)  # ép Hermitian số học

    if make_physical:
        rho = project_to_physical_density_matrix(rho)

    return {
        "rho": rho,
        "pauli_expectations": pauli_expectations,
        "target_counts_by_basis": target_counts_by_basis,
        "success_count_by_basis": success_count_by_basis,
        "counts_by_basis": counts_by_basis,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TRÍCH SIGNED REAL AMPLITUDES TỪ RHO
# ─────────────────────────────────────────────────────────────────────────────

def extract_real_amplitudes_from_density_matrix(rho: np.ndarray,
                                                n_target: int,
                                                reference_sign: int = +1) -> tuple:
    """Lấy biên độ thực có dấu từ rho_target.

    Giả định:
      - target state gần pure state,
      - nghiệm có biên độ thực,
      - dùng |00...0⟩ làm reference,
      - reference_sign=+1 nghĩa là a_0 > 0.
    """
    if reference_sign not in (+1, -1):
        raise ValueError("reference_sign must be +1 or -1.")

    dim = 2 ** n_target
    p_ref = float(np.real(rho[0, 0]))

    if p_ref <= 0:
        raise ValueError(
            "Reference amplitude rho[0,0] <= 0. "
            "Hãy tăng shots hoặc chọn reference basis khác."
        )

    a_ref = reference_sign * np.sqrt(p_ref)

    rows = []
    amplitudes = np.zeros(dim, dtype=np.complex128)

    for i in range(dim):
        bitstring = format(i, f"0{n_target}b")
        amplitude = np.real(rho[0, i]) / a_ref
        amplitudes[i] = amplitude

        rows.append({
            "basis_index": i,
            "bitstring": bitstring,
            "rho_0x_real": np.real(rho[0, i]),
            "rho_xx_real": np.real(rho[i, i]),
            "estimated_amplitude": amplitude,
            "magnitude_from_diagonal": np.sqrt(max(np.real(rho[i, i]), 0.0)),
            "sign": int(np.sign(amplitude)) if abs(amplitude) > 1e-12 else 0,
        })

    df = pd.DataFrame(rows)
    return amplitudes, df


# ─────────────────────────────────────────────────────────────────────────────
#  CHẠY TOMOGRAPHY VÀ KHÔI PHỤC NGHIỆM
# ─────────────────────────────────────────────────────────────────────────────

tomo_result = quantum_tomography_on_hhl_target(
    qc_base_hhl=qc_hhl,
    phase_indices=phase_idx,
    target_indices=target_idx,
    ancilla_index=ancilla_idx[0],
    shots=TOMO_SHOTS,
    seed=TOMO_SEED,
    make_physical=MAKE_RHO_PHYSICAL,
)

rho_tomo = tomo_result["rho"]

x_tomo_from_rho, df_tomo = extract_real_amplitudes_from_density_matrix(
    rho=rho_tomo,
    n_target=n_qubits,
    reference_sign=TOMO_REFERENCE_SIGN,
)

# Do finite-shot noise có thể làm norm hơi lệch 1, chuẩn hóa lại trước khi scale k.
x_tomo_signed_normalized = x_tomo_from_rho / np.linalg.norm(x_tomo_from_rho)

print(f"\n Số cơ sở tomography              = {3 ** n_qubits}")
print(f" Số shots cho mỗi cơ sở           = {TOMO_SHOTS}")
print(" Số shot hậu chọn thành công theo từng basis:")
for basis, c in tomo_result["success_count_by_basis"].items():
    print(f"    basis {basis}: {c} / {TOMO_SHOTS} = {c / TOMO_SHOTS:.6e}")

print("\n[1] Ma trận mật độ rho_target tái dựng từ tomography:")
print(np.round(rho_tomo, 6))

rho_eigvals = np.linalg.eigvalsh(rho_tomo)
purity = np.real(np.trace(rho_tomo @ rho_tomo))
print(f"\n    eigvals(rho_tomo) = {np.round(rho_eigvals, 8)}")
print(f"    Tr(rho_tomo)      = {np.trace(rho_tomo):.6f}")
print(f"    Purity Tr(rho^2)  = {purity:.6f}")

print("\n[2] Biên độ nghiệm chuẩn hóa lấy từ rho_0x = a_0 a_x:")
print(df_tomo[[
    "bitstring",
    "rho_0x_real",
    "rho_xx_real",
    "estimated_amplitude",
    "magnitude_from_diagonal",
    "sign",
]].to_string(index=False))

print("\n[3] Vector nghiệm chuẩn hóa khôi phục bằng tomography:")
print(f"    x'_tomo_raw        = {x_tomo_from_rho}")
print(f"    ||x'_tomo_raw||    = {np.linalg.norm(x_tomo_from_rho):.6f}")
print(f"    x'_tomo_normalized = {x_tomo_signed_normalized}")


# ─────────────────────────────────────────────────────────────────────────────
#  HỒI PHỤC NGHIỆM KHÔNG CHUẨN HÓA BẰNG CÙNG HỆ SỐ k
# ─────────────────────────────────────────────────────────────────────────────

b_prime_tomo = A_matrix @ x_tomo_signed_normalized

k_coeff_tomo = np.vdot(b_prime_tomo, b_vector) / np.vdot(b_prime_tomo, b_prime_tomo)

x_recovered_tomo = k_coeff_tomo * x_tomo_signed_normalized

print("\n[4] Hồi phục nghiệm không chuẩn hóa từ Quantum Tomography:")
print(f"    b'_tomo = A · x'_tomo_normalized = {b_prime_tomo}")
print(f"    k_tomo  = {k_coeff_tomo}")
print(f"\n    Nghiệm HHL từ tomography  x_HHL_tomo = {x_recovered_tomo}")
print(f"    Nghiệm cổ điển            x_cls      = {x_classical}")

abs_err_tomo = np.linalg.norm(x_recovered_tomo - x_classical)
rel_err_tomo = abs_err_tomo / np.linalg.norm(x_classical)
residual_tomo = np.linalg.norm(A_matrix @ x_recovered_tomo - b_vector)

print(f"\n    Sai số tuyệt đối  ||x_HHL_tomo - x_cls|| = {abs_err_tomo:.6e}")
print(f"    Sai số tương đối                         = {rel_err_tomo:.6e}")
print(f"    Residual         ||A·x_HHL_tomo - b||    = {residual_tomo:.6e}")

print("\n[5] So sánh với hướng nghiệm từ statevector cũ:")
print(f"    x'_statevector aligned = {x_prime_aligned}")
print(f"    x'_tomo_normalized     = {x_tomo_signed_normalized}")

fidelity_tomo = np.abs(np.vdot(
    x_tomo_signed_normalized / np.linalg.norm(x_tomo_signed_normalized),
    x_classical_normalized / np.linalg.norm(x_classical_normalized)
)) ** 2

print(f"\n    Fidelity tomography với nghiệm cổ điển = {fidelity_tomo:.6f}")

# Lưu bảng biên độ và ma trận rho để tiện dùng trong báo cáo/debug.
df_tomo.to_csv(out("hhl_tomography_amplitudes.csv"), index=False)
np.save(out("hhl_tomography_rho.npy"), rho_tomo)

print(f"\n → Đã lưu bảng biên độ: {out('hhl_tomography_amplitudes.csv')}")
print(f" → Đã lưu rho_tomo:     {out('hhl_tomography_rho.npy')}")
