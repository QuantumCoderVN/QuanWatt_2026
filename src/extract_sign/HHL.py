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
#  PHẦN C: HỒI PHỤC NGHIỆM TỪ SHOT + SIGN DETECTION
# ══════════════════════════════════════════════════════════════════════════════
#  Module này KHÔNG thay đổi qc_hhl gốc.
#
#  Ý tưởng:
#    1. Chạy qc_hhl bằng shot, hậu chọn phase=0...0 và ancilla=1.
#    2. Từ counts lấy xác suất |x_i|^2.
#    3. Lấy sqrt để có |x_i|.
#    4. Với từng cặp kề nhau (x_i, x_{i+1}), tạo mạch sign-detect:
#         - copy qc_hhl
#         - thêm 1 qubit phụ sign_extra
#         - đưa cặp cần so sánh về basis |0>, |1> của target
#         - SWAP target LSB với sign_extra
#         - Hadamard trên sign_extra
#       Khi đo sign_extra=0, xác suất chứa thông tin giao thoa:
#
#           2P(+) ≈ |x_i + x_{i+1}|^2
#
#       So sánh với |x_i|^2, |x_{i+1}|^2 để biết cùng dấu hay trái dấu.
#    5. Giả sử sign(x_0)=+1 rồi lan truyền dấu.
#    6. Dùng lại hệ số k để khôi phục nghiệm không chuẩn hóa.
#
#  Chú ý:
#    - Đo bằng shot nên cần shots đủ lớn vì xác suất hậu chọn ancilla=1 thường nhỏ.
#    - Phương pháp này khôi phục dấu tương đối cho nghiệm thực.
#      Nếu nghiệm phức, cần khôi phục pha chứ không chỉ dấu.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" [PHẦN C] HỒI PHỤC NGHIỆM BẰNG SHOT + SIGN DETECTION")
print("=" * 70)

from qiskit import ClassicalRegister, transpile

try:
    from qiskit_aer import AerSimulator
except ImportError as exc:
    raise ImportError(
        "Bạn cần cài qiskit-aer để chạy shot simulator:\n"
        "    pip install qiskit-aer"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
#  CẤU HÌNH MODULE SHOT
# ─────────────────────────────────────────────────────────────────────────────

HHL_SHOTS = 1_000_000
SIGN_FIRST_SIGN = 1          # Quy ước pha toàn cục: phần tử đầu tiên dương
SUCCESS_ANCILLA_VALUE = 1    # Trong code HHL của bạn: hậu chọn ancilla = |1>
SIGN_SEED = 2026
TOL_FACTOR = 3.0             # Dùng để giảm nhiễu shot khi quyết định dấu


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 1: CHẠY MẠCH BẰNG SHOT
# ─────────────────────────────────────────────────────────────────────────────

def run_counts_from_circuit(qc_no_measure: QuantumCircuit,
                            shots: int = HHL_SHOTS,
                            seed: int = SIGN_SEED) -> dict:
    """Copy mạch, thêm đo tất cả qubit, chạy AerSimulator và trả về counts."""
    qc_m = qc_no_measure.copy()
    meas = ClassicalRegister(qc_m.num_qubits, "meas")
    qc_m.add_register(meas)

    # Đo qubit q vào classical bit q.
    # Counts của Qiskit in bitstring theo chiều ngược, nên khi đọc sẽ reverse lại.
    qc_m.measure(list(range(qc_m.num_qubits)), list(meas))

    simulator = AerSimulator(seed_simulator=seed)
    tqc = transpile(qc_m, simulator)
    result = simulator.run(tqc, shots=shots).result()

    return result.get_counts()


def key_to_bits_little_endian(key: str, n_bits: int) -> list:
    """Đổi key counts của Qiskit sang list bit theo index qubit.

    Qiskit hiển thị counts dạng big-endian.
    Ví dụ key '1001' nghĩa là classical bits khi đọc theo qubit-index cần reverse.
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
#  MODULE 2: LẤY |x_i|^2 TỪ SHOT CỦA MẠCH HHL
# ─────────────────────────────────────────────────────────────────────────────

def extract_target_probabilities_from_hhl_counts(counts: dict,
                                                 total_qubits: int,
                                                 phase_indices: list,
                                                 target_indices: list,
                                                 ancilla_index: int,
                                                 success_ancilla_value: int = 1):
    """Lấy phân phối xác suất target sau khi hậu chọn phase=0 và ancilla=1.

    Trả về:
      - prob_raw[t]  = P(phase=0, target=t, ancilla=1)
      - prob_norm[t] = P(target=t | phase=0, ancilla=1)
      - success_count = số shot thỏa hậu chọn
    """
    dim = 2 ** len(target_indices)
    shots = sum(counts.values())

    prob_raw = np.zeros(dim, dtype=float)
    success_count = 0

    for key, c in counts.items():
        bits = key_to_bits_little_endian(key, total_qubits)

        if not is_phase_zero(bits, phase_indices):
            continue

        if bits[ancilla_index] != success_ancilla_value:
            continue

        target_value = read_register_int(bits, target_indices)
        prob_raw[target_value] += c / shots
        success_count += c

    if prob_raw.sum() == 0:
        raise RuntimeError(
            "Không có shot nào thỏa hậu chọn phase=0 và ancilla=1. "
            "Hãy tăng HHL_SHOTS hoặc kiểm tra C_value."
        )

    prob_norm = prob_raw / prob_raw.sum()

    return prob_raw, prob_norm, success_count


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3: TẠO PERMUTATION ĐƯA CẶP (i, j) VỀ BASIS (0, 1)
# ─────────────────────────────────────────────────────────────────────────────

def pair_to_front_permutation(dim: int, i: int, j: int) -> np.ndarray:
    """Tạo ma trận hoán vị P sao cho:

        P|i⟩ = |0⟩
        P|j⟩ = |1⟩

    Các basis còn lại được ánh xạ một-một vào các vị trí còn trống.
    Nhờ vậy sign-detect luôn chỉ cần giao thoa hai trạng thái |0⟩ và |1⟩.
    """
    if i == j:
        raise ValueError("i và j phải khác nhau.")
    if not (0 <= i < dim and 0 <= j < dim):
        raise ValueError("i hoặc j vượt ngoài kích thước không gian target.")

    mapping_old_to_new = {i: 0, j: 1}
    remaining_new = [k for k in range(dim) if k not in (0, 1)]

    for old in range(dim):
        if old not in mapping_old_to_new:
            mapping_old_to_new[old] = remaining_new.pop(0)

    P = np.zeros((dim, dim), dtype=np.complex128)

    for old, new in mapping_old_to_new.items():
        P[new, old] = 1.0

    return P


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 4: BUILD MẠCH SIGN-DETECT CHO MỘT CẶP BIÊN ĐỘ
# ─────────────────────────────────────────────────────────────────────────────

def build_hhl_sign_detect_for_pair(qc_base_hhl: QuantumCircuit,
                                   target_indices: list,
                                   pair_i: int,
                                   pair_j: int) -> tuple:
    """Copy qc_hhl và gắn module sign-detect cho cặp (pair_i, pair_j).

    Output:
      - qc_sd: mạch sign-detect
      - extra_idx: index của qubit phụ sign_extra
    """
    dim = 2 ** len(target_indices)

    qc_sd = qc_base_hhl.copy(name=f"HHL_sign_detect_{pair_i}_{pair_j}")

    sign_extra = QuantumRegister(1, "sign_extra")
    qc_sd.add_register(sign_extra)

    extra_idx = qc_sd.num_qubits - 1

    # Đưa cặp cần so sánh về target basis |0>, |1>.
    # Với bài toán 2x2, cặp duy nhất là (0,1), nên P là identity và có thể bỏ qua.
    if not (pair_i == 0 and pair_j == 1):
        P = pair_to_front_permutation(dim, pair_i, pair_j)
        qc_sd.unitary(P, target_indices, label=f"P({pair_i},{pair_j})->(0,1)")

    # Sau permutation, hai biên độ cần so sánh nằm ở target |0> và |1>.
    # SWAP target LSB với sign_extra, rồi H trên sign_extra tạo:
    #     amplitude(extra=0) ∝ x_i + x_j
    qc_sd.swap(target_indices[0], extra_idx)
    qc_sd.h(extra_idx)

    return qc_sd, extra_idx


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 5: ĐỌC XÁC SUẤT GIAO THOA TỪ SIGN-DETECT
# ─────────────────────────────────────────────────────────────────────────────

def extract_plus_interference_probability(counts_sd: dict,
                                          total_qubits_sd: int,
                                          phase_indices: list,
                                          target_indices: list,
                                          ancilla_index: int,
                                          extra_index: int,
                                          success_ancilla_value: int = 1):
    """Lấy đại lượng 2P(+) từ mạch sign-detect.

    Sau permutation + SWAP + H:
      - target phải là |0...0⟩
      - sign_extra = 0

    Khi đó:
        2 * P(target=0, extra=0 | phase=0, ancilla=1)
        ≈ |x_i + x_j|^2 / ||x||^2
    """
    success_count = 0
    plus_count = 0

    for key, c in counts_sd.items():
        bits = key_to_bits_little_endian(key, total_qubits_sd)

        if not is_phase_zero(bits, phase_indices):
            continue

        if bits[ancilla_index] != success_ancilla_value:
            continue

        success_count += c

        target_value = read_register_int(bits, target_indices)

        if target_value == 0 and bits[extra_index] == 0:
            plus_count += c

    if success_count == 0:
        raise RuntimeError(
            "Mạch sign-detect không có shot nào thỏa hậu chọn phase=0 và ancilla=1. "
            "Hãy tăng HHL_SHOTS."
        )

    plus_prob_conditional = plus_count / success_count
    pc_norm = 2.0 * plus_prob_conditional

    return pc_norm, plus_count, success_count


def decide_relative_sign(prob_i: float,
                         prob_j: float,
                         pc_norm: float,
                         effective_success_count: int,
                         tol_factor: float = TOL_FACTOR) -> int:
    """Quyết định hai biên độ cùng dấu hay trái dấu.

    Nếu cùng dấu:
        |x_i + x_j|^2 thường lớn hơn từng |x_i|^2, |x_j|^2.

    Nếu trái dấu:
        |x_i + x_j|^2 bị triệt tiêu một phần và thường nhỏ hơn phần tử lớn hơn.

    Trả về:
      +1 nếu cùng dấu
      -1 nếu trái dấu
    """
    tol = tol_factor / np.sqrt(max(effective_success_count, 1))

    threshold = max(prob_i, prob_j)

    if pc_norm < threshold - tol:
        return -1

    return +1


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 6: PIPELINE KHÔI PHỤC VECTOR CÓ DẤU TỪ SHOT
# ─────────────────────────────────────────────────────────────────────────────

def recover_signed_state_from_shots(qc_base_hhl: QuantumCircuit,
                                    phase_indices: list,
                                    target_indices: list,
                                    ancilla_index: int,
                                    shots: int = HHL_SHOTS,
                                    first_sign: int = 1,
                                    seed: int = SIGN_SEED):
    """Khôi phục vector nghiệm chuẩn hóa có dấu từ shot.

    Trả về dictionary gồm:
      - prob_raw
      - prob_norm
      - abs_amplitudes
      - relative_parities
      - signs
      - signed_state_normalized
      - success_count
    """
    dim = 2 ** len(target_indices)

    # 1. Chạy HHL shot để lấy |x_i|^2
    counts_hhl = run_counts_from_circuit(qc_base_hhl, shots=shots, seed=seed)

    prob_raw, prob_norm, success_count = extract_target_probabilities_from_hhl_counts(
        counts=counts_hhl,
        total_qubits=qc_base_hhl.num_qubits,
        phase_indices=phase_indices,
        target_indices=target_indices,
        ancilla_index=ancilla_index,
        success_ancilla_value=SUCCESS_ANCILLA_VALUE
    )

    abs_amplitudes = np.sqrt(np.maximum(prob_norm, 0.0))

    # 2. Chạy sign-detect cho từng cặp kề nhau: (0,1), (1,2), ...
    relative_parities = []
    sign_diagnostics = []

    for i in range(dim - 1):
        j = i + 1

        qc_sd, extra_idx = build_hhl_sign_detect_for_pair(
            qc_base_hhl=qc_base_hhl,
            target_indices=target_indices,
            pair_i=i,
            pair_j=j
        )

        counts_sd = run_counts_from_circuit(qc_sd, shots=shots, seed=seed + i + 1)

        pc_norm, plus_count, sd_success_count = extract_plus_interference_probability(
            counts_sd=counts_sd,
            total_qubits_sd=qc_sd.num_qubits,
            phase_indices=phase_indices,
            target_indices=target_indices,
            ancilla_index=ancilla_index,
            extra_index=extra_idx,
            success_ancilla_value=SUCCESS_ANCILLA_VALUE
        )

        parity = decide_relative_sign(
            prob_i=prob_norm[i],
            prob_j=prob_norm[j],
            pc_norm=pc_norm,
            effective_success_count=sd_success_count,
            tol_factor=TOL_FACTOR
        )

        relative_parities.append(parity)

        sign_diagnostics.append({
            "pair": (i, j),
            "prob_i": prob_norm[i],
            "prob_j": prob_norm[j],
            "pc_norm_2P_plus": pc_norm,
            "parity": parity,
            "plus_count": plus_count,
            "success_count": sd_success_count,
        })

    # 3. Lan truyền dấu từ first_sign
    signs = [1 if first_sign >= 0 else -1]

    for parity in relative_parities:
        signs.append(signs[-1] * parity)

    signs = np.array(signs, dtype=int)

    # 4. Gắn dấu vào biên độ
    signed_state_normalized = signs * abs_amplitudes

    return {
        "prob_raw": prob_raw,
        "prob_norm": prob_norm,
        "abs_amplitudes": abs_amplitudes,
        "relative_parities": np.array(relative_parities, dtype=int),
        "signs": signs,
        "signed_state_normalized": signed_state_normalized.astype(np.complex128),
        "success_count": success_count,
        "sign_diagnostics": sign_diagnostics,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CHẠY MODULE MỚI
# ─────────────────────────────────────────────────────────────────────────────

shot_result = recover_signed_state_from_shots(
    qc_base_hhl=qc_hhl,
    phase_indices=phase_idx,
    target_indices=target_idx,
    ancilla_index=ancilla_idx[0],
    shots=HHL_SHOTS,
    first_sign=SIGN_FIRST_SIGN,
    seed=SIGN_SEED
)

x_shot_signed_normalized = shot_result["signed_state_normalized"]

print(f"\n Số shots dùng cho mỗi mạch         = {HHL_SHOTS}")
print(f" Số shot hậu chọn HHL thành công    = {shot_result['success_count']}")
print(f" Tỉ lệ hậu chọn ước lượng           = {shot_result['success_count'] / HHL_SHOTS:.6e}")

print("\n[1] Xác suất target từ shot sau hậu chọn phase=0, ancilla=1:")
print(f"    prob_raw  = {shot_result['prob_raw']}")
print(f"    prob_norm = {shot_result['prob_norm']}")

print("\n[2] Độ lớn biên độ từ shot:")
print(f"    |x_i| = sqrt(prob_norm) = {shot_result['abs_amplitudes']}")

print("\n[3] Kết quả sign-detect từng cặp:")
for d in shot_result["sign_diagnostics"]:
    i, j = d["pair"]
    relation = "cùng dấu" if d["parity"] == 1 else "trái dấu"
    print(
        f"    Cặp ({i},{j}): "
        f"2P(+)={d['pc_norm_2P_plus']:.6f}, "
        f"p_i={d['prob_i']:.6f}, "
        f"p_j={d['prob_j']:.6f} "
        f"→ parity={d['parity']} ({relation}), "
        f"success={d['success_count']}"
    )

print("\n[4] Dấu được khôi phục:")
print(f"    signs = {shot_result['signs']}")

print("\n[5] Vector nghiệm chuẩn hóa khôi phục từ shot + sign:")
print(f"    x'_shot_signed = {x_shot_signed_normalized}")


# ─────────────────────────────────────────────────────────────────────────────
#  HỒI PHỤC NGHIỆM KHÔNG CHUẨN HÓA BẰNG CÙNG HỆ SỐ k
# ─────────────────────────────────────────────────────────────────────────────

b_prime_shot = A_matrix @ x_shot_signed_normalized

k_coeff_shot = np.vdot(b_prime_shot, b_vector) / np.vdot(b_prime_shot, b_prime_shot)

x_recovered_shot = k_coeff_shot * x_shot_signed_normalized

print("\n[6] Hồi phục nghiệm không chuẩn hóa từ shot:")
print(f"    b'_shot = A · x'_shot_signed = {b_prime_shot}")
print(f"    k_shot  = {k_coeff_shot}")
print(f"\n    Nghiệm HHL từ shot + sign  x_HHL_shot = {x_recovered_shot}")
print(f"    Nghiệm cổ điển             x_cls      = {x_classical}")

abs_err_shot = np.linalg.norm(x_recovered_shot - x_classical)
rel_err_shot = abs_err_shot / np.linalg.norm(x_classical)
residual_shot = np.linalg.norm(A_matrix @ x_recovered_shot - b_vector)

print(f"\n    Sai số tuyệt đối  ||x_HHL_shot - x_cls|| = {abs_err_shot:.6e}")
print(f"    Sai số tương đối                         = {rel_err_shot:.6e}")
print(f"    Residual         ||A·x_HHL_shot - b||    = {residual_shot:.6e}")

print("\n[7] So sánh với hướng nghiệm từ statevector cũ:")
print(f"    x'_statevector aligned = {x_prime_aligned}")
print(f"    x'_shot_signed         = {x_shot_signed_normalized}")

fidelity_shot = np.abs(np.vdot(
    x_shot_signed_normalized / np.linalg.norm(x_shot_signed_normalized),
    x_classical_normalized / np.linalg.norm(x_classical_normalized)
)) ** 2

print(f"\n    Fidelity shot-sign với nghiệm cổ điển = {fidelity_shot:.6f}")