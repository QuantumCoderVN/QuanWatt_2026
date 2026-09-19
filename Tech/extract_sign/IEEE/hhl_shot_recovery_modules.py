"""
Hai module giải hệ tuyến tính Ax = b theo đúng yêu cầu shot-recovery:
  1) classical_solver(A, b): giải cổ điển bằng numpy.linalg.solve.
  2) hhl_solver(A, b): giải bằng HHL, đo bằng shot, hậu chọn, khôi phục dấu,
     rồi khôi phục nghiệm không chuẩn hóa để so sánh trực tiếp với nghiệm cổ điển.

Điểm quan trọng:
  - hhl_solver KHÔNG lấy nghiệm từ vector trạng thái lý tưởng; toàn bộ nghiệm được khôi phục từ counts/shot.
  - Nghiệm HHL được khôi phục từ counts/shot:
      + hậu chọn phase = |0...0>, ancilla = |1>
      + lấy |x_i| từ sqrt(probability)
      + sign-detect bằng mạch giao thoa để lấy dấu tương đối
      + dùng hệ số k để khôi phục nghiệm không chuẩn hóa.

Yêu cầu:
  pip install numpy scipy qiskit qiskit-aer

Giả định của implementation HHL đơn giản này:
  - A Hermitian, xác định dương.
  - Sau scale nội bộ, phổ của A nằm trong (0, 2π).
  - Phương pháp sign-detect bên dưới khôi phục dấu tương đối cho nghiệm thực.
    Với nghiệm phức tổng quát, cần pha tương đối chứ không chỉ dấu.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import expm

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import RYGate
from qiskit.synthesis.qft import synth_qft_full


# =============================================================================
# MODULE 1 — GIẢI CỔ ĐIỂN
# =============================================================================
def classical_solver(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Giải Ax=b bằng phương pháp cổ điển."""
    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A phải là ma trận vuông.")
    if A.shape[0] != b.size:
        raise ValueError("Kích thước của A và b không khớp.")

    return np.linalg.solve(A, b)


# =============================================================================
# Các hàm phụ nội bộ cho module HHL-shot
# =============================================================================
def _validate_and_prepare_system(
    A: np.ndarray,
    b: np.ndarray,
    auto_pad: bool,
    auto_scale: bool,
) -> dict[str, Any]:
    """Kiểm tra input, padding lên 2^n, và scale phổ vào (0, 2π)."""
    A_original = np.asarray(A, dtype=np.complex128)
    b_original = np.asarray(b, dtype=np.complex128).reshape(-1)

    if A_original.ndim != 2 or A_original.shape[0] != A_original.shape[1]:
        raise ValueError("A phải là ma trận vuông.")
    if A_original.shape[0] != b_original.size:
        raise ValueError("Kích thước của A và b không khớp.")
    if np.linalg.norm(b_original) == 0:
        raise ValueError("Vector b không được là vector 0.")
    if not np.allclose(A_original, A_original.conj().T, atol=1e-10):
        raise ValueError(
            "HHL chuẩn yêu cầu A là Hermitian. "
            "Với A không Hermitian cần embedding/block-encoding riêng."
        )

    eigenvalues_original = np.linalg.eigvalsh(A_original)
    if eigenvalues_original.min() <= 0:
        raise ValueError("HHL chuẩn trong code này yêu cầu A xác định dương.")

    original_dim = A_original.shape[0]
    padded_dim = 1 << int(math.ceil(math.log2(original_dim)))

    if original_dim != padded_dim:
        if not auto_pad:
            raise ValueError("Kích thước A phải là 2^n. Bật auto_pad=True để tự padding.")

        A_padded = np.eye(padded_dim, dtype=np.complex128)
        A_padded[:original_dim, :original_dim] = A_original

        b_padded = np.zeros(padded_dim, dtype=np.complex128)
        b_padded[:original_dim] = b_original
    else:
        A_padded = A_original.copy()
        b_padded = b_original.copy()

    eigenvalues_padded = np.linalg.eigvalsh(A_padded)
    scale_factor = 1.0

    if auto_scale:
        # Đưa lambda_max xuống dưới 2π để U=exp(iA) không alias pha.
        scale_factor = max(1.0, float(eigenvalues_padded.max()) / (0.9 * 2.0 * np.pi))

    A_hhl = A_padded / scale_factor
    eigenvalues_hhl = np.linalg.eigvalsh(A_hhl)

    if eigenvalues_hhl.max() >= 2.0 * np.pi:
        raise ValueError("Phổ của A_hhl vẫn chưa nằm trong (0, 2π). Hãy bật auto_scale=True.")
    if eigenvalues_hhl.min() <= 0:
        raise ValueError("A_hhl phải xác định dương.")

    return {
        "A_original": A_original,
        "b_original": b_original,
        "A_padded": A_padded,
        "b_padded": b_padded,
        "A_hhl": A_hhl,
        "original_dim": original_dim,
        "padded_dim": padded_dim,
        "scale_factor": scale_factor,
        "eigenvalues_original": eigenvalues_original,
        "eigenvalues_hhl": eigenvalues_hhl,
    }


def _controlled_unitary_power(
    qc: QuantumCircuit,
    U: np.ndarray,
    power: int,
    control_qubit: int,
    target_qubits: list[int],
) -> None:
    """Áp dụng controlled-U^power lên target_qubits."""
    U_power = np.linalg.matrix_power(U, power)
    sub = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    sub.unitary(U_power, list(range(len(target_qubits))))
    qc.append(sub.to_gate().control(1), [control_qubit] + list(target_qubits))


def _apply_qpe(
    qc: QuantumCircuit,
    U_matrix: np.ndarray,
    phase_indices: list[int],
    target_indices: list[int],
) -> None:
    """Quantum Phase Estimation theo convention của code gốc."""
    for q in phase_indices:
        qc.h(q)

    # Giữ convention code gốc: reversed(phase), U^(2^i), inverse QFT do_swaps=False.
    for i, ctrl in enumerate(reversed(phase_indices)):
        _controlled_unitary_power(qc, U_matrix, 2**i, ctrl, target_indices)

    iqft = synth_qft_full(
        num_qubits=len(phase_indices),
        do_swaps=False,
        inverse=True
    )

    qc.append(iqft.to_gate(label="IQFT"), phase_indices)


def _apply_controlled_rotation(
    qc: QuantumCircuit,
    phase_indices: list[int],
    ancilla_index: int,
    C_value: float,
) -> None:
    """Controlled-RY theo lambda_est = 2πk/2^m."""
    m = len(phase_indices)

    for k in range(1, 2**m):
        ctrl_state = f"{k:0{m}b}"
        lambda_est = 2.0 * np.pi * k / (2**m)
        argument = C_value / lambda_est

        if abs(argument) > 1.0:
            continue

        theta = 2.0 * np.arcsin(argument)
        cry = RYGate(theta).control(m, ctrl_state=ctrl_state)
        qc.append(cry, list(phase_indices) + [ancilla_index])


def _build_hhl_circuit(
    A_hhl: np.ndarray,
    b_padded: np.ndarray,
    phase_qubits: int,
    C_value: float | None,
) -> dict[str, Any]:
    """Build mạch HHL không đo."""
    padded_dim = A_hhl.shape[0]
    n_target_qubits = int(math.log2(padded_dim))

    b_state = b_padded / np.linalg.norm(b_padded)
    U_matrix = expm(1j * A_hhl)

    if not np.allclose(U_matrix.conj().T @ U_matrix, np.eye(padded_dim), atol=1e-9):
        raise RuntimeError("U = exp(iA_hhl) không unitary trong sai số số học cho phép.")

    phase_reg = QuantumRegister(phase_qubits, "phase")
    target_reg = QuantumRegister(n_target_qubits, "target")
    ancilla_reg = QuantumRegister(1, "ancilla")
    qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL_shot")

    phase_indices = list(range(phase_qubits))
    target_indices = list(range(phase_qubits, phase_qubits + n_target_qubits))
    ancilla_index = phase_qubits + n_target_qubits

    qc_hhl.initialize(b_state, target_indices)
    _apply_qpe(qc_hhl, U_matrix, phase_indices, target_indices)

    if C_value is None:
        # Giữ lựa chọn trong code gốc: C = 0.9 * độ phân giải phase.
        phase_resolution = 2.0 * np.pi / (2**phase_qubits)
        C_value = 0.9 * phase_resolution

    _apply_controlled_rotation(qc_hhl, phase_indices, ancilla_index, C_value)

    # Inverse QPE: build lại block QPE rồi lấy inverse như code gốc.
    qc_qpe_only = QuantumCircuit(phase_qubits + n_target_qubits, name="QPE")
    qpe_phase = list(range(phase_qubits))
    qpe_target = list(range(phase_qubits, phase_qubits + n_target_qubits))
    _apply_qpe(qc_qpe_only, U_matrix, qpe_phase, qpe_target)

    inv_qpe = qc_qpe_only.inverse()
    inv_qpe.name = "QPE†"
    qc_hhl.append(inv_qpe, phase_indices + target_indices)

    return {
        "circuit": qc_hhl,
        "phase_indices": phase_indices,
        "target_indices": target_indices,
        "ancilla_index": ancilla_index,
        "n_target_qubits": n_target_qubits,
        "C_value": C_value,
    }


def _run_counts_from_circuit(
    qc_no_measure: QuantumCircuit,
    shots: int,
    seed: int | None,
) -> dict[str, int]:
    """Copy mạch, thêm đo tất cả qubit, chạy AerSimulator và trả counts."""
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError(
            "Bạn cần cài qiskit-aer để chạy HHL bằng shot:\n"
            "    pip install qiskit-aer"
        ) from exc

    qc_measured = qc_no_measure.copy()
    measured_bits = ClassicalRegister(qc_measured.num_qubits, "meas")
    qc_measured.add_register(measured_bits)
    qc_measured.measure(list(range(qc_measured.num_qubits)), list(measured_bits))

    simulator = AerSimulator(seed_simulator=seed)
    tqc = transpile(qc_measured, simulator)
    result = simulator.run(tqc, shots=shots).result()
    return result.get_counts()


def _key_to_bits_little_endian(key: str, n_bits: int) -> list[int]:
    """Đổi key counts của Qiskit sang list bit theo index qubit."""
    clean_key = key.replace(" ", "").zfill(n_bits)
    return [int(bit) for bit in clean_key[::-1]]


def _read_register_int(bits: list[int], qubit_indices: list[int]) -> int:
    """Đọc một register theo little-endian Qiskit."""
    value = 0
    for j, q in enumerate(qubit_indices):
        value |= int(bits[q]) << j
    return value


def _is_phase_zero(bits: list[int], phase_indices: list[int]) -> bool:
    return all(bits[q] == 0 for q in phase_indices)


def _extract_target_probabilities_from_hhl_counts(
    counts: dict[str, int],
    total_qubits: int,
    phase_indices: list[int],
    target_indices: list[int],
    ancilla_index: int,
    success_ancilla_value: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Lấy P(target | phase=0, ancilla=success) từ counts."""
    dim = 2 ** len(target_indices)
    shots = sum(counts.values())

    prob_raw = np.zeros(dim, dtype=float)
    success_count = 0

    for key, count in counts.items():
        bits = _key_to_bits_little_endian(key, total_qubits)

        if not _is_phase_zero(bits, phase_indices):
            continue
        if bits[ancilla_index] != success_ancilla_value:
            continue

        target_value = _read_register_int(bits, target_indices)
        prob_raw[target_value] += count / shots
        success_count += count

    if success_count == 0 or prob_raw.sum() == 0:
        raise RuntimeError(
            "Không có shot nào thỏa hậu chọn phase=0 và ancilla=1. "
            "Hãy tăng shots, tăng phase_qubits, hoặc kiểm tra C_value."
        )

    prob_norm = prob_raw / prob_raw.sum()
    return prob_raw, prob_norm, success_count


def _pair_to_front_permutation(dim: int, i: int, j: int) -> np.ndarray:
    """Tạo P sao cho P|i> = |0>, P|j> = |1>."""
    if i == j:
        raise ValueError("i và j phải khác nhau.")
    if not (0 <= i < dim and 0 <= j < dim):
        raise ValueError("i hoặc j vượt ngoài kích thước target.")

    mapping_old_to_new: dict[int, int] = {i: 0, j: 1}
    remaining_new = [k for k in range(dim) if k not in (0, 1)]

    for old in range(dim):
        if old not in mapping_old_to_new:
            mapping_old_to_new[old] = remaining_new.pop(0)

    P = np.zeros((dim, dim), dtype=np.complex128)
    for old, new in mapping_old_to_new.items():
        P[new, old] = 1.0

    return P


def _build_hhl_sign_detect_for_pair(
    qc_base_hhl: QuantumCircuit,
    target_indices: list[int],
    pair_i: int,
    pair_j: int,
) -> tuple[QuantumCircuit, int]:
    """Copy mạch HHL và gắn module sign-detect cho cặp (pair_i, pair_j)."""
    dim = 2 ** len(target_indices)
    qc_sd = qc_base_hhl.copy(name=f"HHL_sign_detect_{pair_i}_{pair_j}")

    sign_extra = QuantumRegister(1, "sign_extra")
    qc_sd.add_register(sign_extra)
    extra_index = qc_sd.num_qubits - 1

    if not (pair_i == 0 and pair_j == 1):
        P = _pair_to_front_permutation(dim, pair_i, pair_j)
        qc_sd.unitary(P, target_indices, label=f"P({pair_i},{pair_j})->(0,1)")

    # Đưa hai trạng thái |0>, |1> của target vào giao thoa trên sign_extra.
    qc_sd.swap(target_indices[0], extra_index)
    qc_sd.h(extra_index)

    return qc_sd, extra_index


def _extract_plus_interference_probability(
    counts_sd: dict[str, int],
    total_qubits_sd: int,
    phase_indices: list[int],
    target_indices: list[int],
    ancilla_index: int,
    extra_index: int,
    success_ancilla_value: int,
) -> tuple[float, int, int]:
    """Lấy 2P(+) ≈ |x_i + x_j|^2 từ mạch sign-detect."""
    success_count = 0
    plus_count = 0

    for key, count in counts_sd.items():
        bits = _key_to_bits_little_endian(key, total_qubits_sd)

        if not _is_phase_zero(bits, phase_indices):
            continue
        if bits[ancilla_index] != success_ancilla_value:
            continue

        success_count += count
        target_value = _read_register_int(bits, target_indices)

        if target_value == 0 and bits[extra_index] == 0:
            plus_count += count

    if success_count == 0:
        raise RuntimeError(
            "Mạch sign-detect không có shot nào thỏa hậu chọn phase=0 và ancilla=1. "
            "Hãy tăng shots."
        )

    plus_prob_conditional = plus_count / success_count
    two_p_plus = 2.0 * plus_prob_conditional
    return two_p_plus, plus_count, success_count


def _decide_relative_sign(
    prob_i: float,
    prob_j: float,
    two_p_plus: float,
    effective_success_count: int,
    tol_factor: float,
) -> int:
    """Trả +1 nếu cùng dấu, -1 nếu trái dấu."""
    tol = tol_factor / np.sqrt(max(effective_success_count, 1))
    threshold = max(prob_i, prob_j)

    if two_p_plus < threshold - tol:
        return -1
    return +1


def _recover_signed_state_from_shots(
    qc_base_hhl: QuantumCircuit,
    phase_indices: list[int],
    target_indices: list[int],
    ancilla_index: int,
    shots: int,
    seed: int | None,
    first_sign: int,
    success_ancilla_value: int,
    tol_factor: float,
) -> dict[str, Any]:
    """Pipeline khôi phục vector chuẩn hóa có dấu từ counts/shot."""
    dim = 2 ** len(target_indices)

    counts_hhl = _run_counts_from_circuit(qc_base_hhl, shots=shots, seed=seed)
    prob_raw, prob_norm, success_count = _extract_target_probabilities_from_hhl_counts(
        counts=counts_hhl,
        total_qubits=qc_base_hhl.num_qubits,
        phase_indices=phase_indices,
        target_indices=target_indices,
        ancilla_index=ancilla_index,
        success_ancilla_value=success_ancilla_value,
    )

    abs_amplitudes = np.sqrt(np.maximum(prob_norm, 0.0))

    relative_parities: list[int] = []
    sign_diagnostics: list[dict[str, Any]] = []

    for i in range(dim - 1):
        j = i + 1
        qc_sd, extra_index = _build_hhl_sign_detect_for_pair(
            qc_base_hhl=qc_base_hhl,
            target_indices=target_indices,
            pair_i=i,
            pair_j=j,
        )

        pair_seed = None if seed is None else seed + i + 1
        counts_sd = _run_counts_from_circuit(qc_sd, shots=shots, seed=pair_seed)

        two_p_plus, plus_count, sd_success_count = _extract_plus_interference_probability(
            counts_sd=counts_sd,
            total_qubits_sd=qc_sd.num_qubits,
            phase_indices=phase_indices,
            target_indices=target_indices,
            ancilla_index=ancilla_index,
            extra_index=extra_index,
            success_ancilla_value=success_ancilla_value,
        )

        parity = _decide_relative_sign(
            prob_i=prob_norm[i],
            prob_j=prob_norm[j],
            two_p_plus=two_p_plus,
            effective_success_count=sd_success_count,
            tol_factor=tol_factor,
        )

        relative_parities.append(parity)
        sign_diagnostics.append(
            {
                "pair": (i, j),
                "prob_i": prob_norm[i],
                "prob_j": prob_norm[j],
                "two_p_plus": two_p_plus,
                "parity": parity,
                "plus_count": plus_count,
                "success_count": sd_success_count,
            }
        )

    signs = [1 if first_sign >= 0 else -1]
    for parity in relative_parities:
        signs.append(signs[-1] * parity)

    signs_array = np.array(signs, dtype=int)
    signed_state_normalized = signs_array * abs_amplitudes

    norm = np.linalg.norm(signed_state_normalized)
    if norm == 0:
        raise RuntimeError("Vector khôi phục từ shot có norm bằng 0.")
    signed_state_normalized = signed_state_normalized / norm

    return {
        "counts_hhl": counts_hhl,
        "prob_raw": prob_raw,
        "prob_norm": prob_norm,
        "abs_amplitudes": abs_amplitudes,
        "relative_parities": np.array(relative_parities, dtype=int),
        "signs": signs_array,
        "signed_state_normalized": signed_state_normalized.astype(np.complex128),
        "success_count": success_count,
        "sign_diagnostics": sign_diagnostics,
    }


# =============================================================================
# MODULE 2 — GIẢI BẰNG HHL, KHÔI PHỤC TỪ SHOT
# =============================================================================
def hhl_solver(
    A: np.ndarray,
    b: np.ndarray,
    phase_qubits: int = 5,
    shots: int = 1_000_000,
    seed: int | None = 2026,
    first_sign: int = 1,
    success_ancilla_value: int = 1,
    tol_factor: float = 3.0,
    C_value: float | None = None,
    auto_scale: bool = True,
    auto_pad: bool = True,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Giải Ax=b bằng HHL và khôi phục nghiệm từ shot/counts.

    Parameters
    ----------
    A, b:
        Hệ tuyến tính Ax=b.
    phase_qubits:
        Số qubit phase trong QPE. Tăng lên thường cải thiện độ chính xác pha.
    shots:
        Số shot cho mỗi mạch. Vì còn hậu chọn ancilla=1, nên thường cần shots lớn.
    seed:
        Seed của AerSimulator.
    first_sign:
        Quy ước dấu toàn cục cho phần tử đầu tiên. Dấu toàn cục sẽ được hấp thụ vào k_coeff.
    success_ancilla_value:
        Code HHL này hậu chọn ancilla=1.
    tol_factor:
        Hệ số tolerance khi quyết định dấu từ shot-noise.
    C_value:
        Hằng số C trong controlled-RY. Nếu None, dùng C = 0.9 * 2π / 2^phase_qubits
        giống code gốc.
    auto_scale:
        Nếu True, tự scale A trong HHL để phổ nằm trong (0, 2π).
    auto_pad:
        Nếu True, tự padding kích thước lên 2^n.
    return_info:
        Nếu True, trả thêm dict chứa counts, xác suất, dấu, circuit, diagnostics.

    Returns
    -------
    x_hhl:
        Vector nghiệm HHL đã khôi phục không chuẩn hóa từ shot, có thể so sánh trực tiếp
        với nghiệm cổ điển.
    """
    prepared = _validate_and_prepare_system(A, b, auto_pad=auto_pad, auto_scale=auto_scale)

    circuit_data = _build_hhl_circuit(
        A_hhl=prepared["A_hhl"],
        b_padded=prepared["b_padded"],
        phase_qubits=phase_qubits,
        C_value=C_value,
    )

    shot_recovery = _recover_signed_state_from_shots(
        qc_base_hhl=circuit_data["circuit"],
        phase_indices=circuit_data["phase_indices"],
        target_indices=circuit_data["target_indices"],
        ancilla_index=circuit_data["ancilla_index"],
        shots=shots,
        seed=seed,
        first_sign=first_sign,
        success_ancilla_value=success_ancilla_value,
        tol_factor=tol_factor,
    )

    # Vector chuẩn hóa có dấu từ shot trên không gian đã padding.
    x_direction_padded = shot_recovery["signed_state_normalized"]

    # Cắt về số chiều gốc, rồi chuẩn hóa lại vì phần padding có thể nhận xác suất nhỏ do nhiễu/QPE.
    original_dim = prepared["original_dim"]
    x_direction = x_direction_padded[:original_dim].astype(np.complex128)
    direction_norm = np.linalg.norm(x_direction)
    if direction_norm == 0:
        raise RuntimeError("Phần vector nghiệm tương ứng kích thước gốc có norm bằng 0.")
    x_direction = x_direction / direction_norm

    # Khôi phục nghiệm không chuẩn hóa bằng hệ số k:
    #     x_hhl = k * x_direction
    # với k được chọn để A @ x_hhl gần b nhất theo least-squares 1D.
    A_original = prepared["A_original"]
    b_original = prepared["b_original"]
    b_from_direction = A_original @ x_direction
    denom = np.vdot(b_from_direction, b_from_direction)
    if abs(denom) < 1e-15:
        raise RuntimeError("Không thể tính k_coeff vì A @ x_direction gần bằng 0.")

    k_coeff = np.vdot(b_from_direction, b_original) / denom
    x_hhl = k_coeff * x_direction

    if not return_info:
        return x_hhl

    info = {
        "circuit": circuit_data["circuit"],
        "phase_qubits": phase_qubits,
        "shots_per_circuit": shots,
        "seed": seed,
        "n_target_qubits": circuit_data["n_target_qubits"],
        "original_dim": prepared["original_dim"],
        "padded_dim": prepared["padded_dim"],
        "scale_factor": prepared["scale_factor"],
        "C_value": circuit_data["C_value"],
        "eigenvalues_original": prepared["eigenvalues_original"],
        "eigenvalues_hhl_scaled": prepared["eigenvalues_hhl"],
        "prob_raw": shot_recovery["prob_raw"],
        "prob_norm": shot_recovery["prob_norm"],
        "abs_amplitudes": shot_recovery["abs_amplitudes"],
        "relative_parities": shot_recovery["relative_parities"],
        "signs": shot_recovery["signs"],
        "signed_state_normalized_padded": shot_recovery["signed_state_normalized"],
        "signed_state_normalized_original": x_direction,
        "success_count": shot_recovery["success_count"],
        "postselection_rate": shot_recovery["success_count"] / shots,
        "sign_diagnostics": shot_recovery["sign_diagnostics"],
        "k_coeff": k_coeff,
    }
    return x_hhl, info


# =============================================================================
# SO SÁNH OUTPUT CỦA HAI MODULE
# =============================================================================
def compare_solvers(
    A: np.ndarray,
    b: np.ndarray,
    phase_qubits: int = 5,
    shots: int = 1_000_000,
    seed: int | None = 2026,
    verbose: bool = True,
) -> dict[str, Any]:
    """Chạy module cổ điển và module HHL-shot, sau đó so sánh output."""
    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)

    x_classical = classical_solver(A, b)
    x_hhl, hhl_info = hhl_solver(
        A,
        b,
        phase_qubits=phase_qubits,
        shots=shots,
        seed=seed,
        return_info=True,
    )

    abs_error = np.linalg.norm(x_hhl - x_classical)
    rel_error = abs_error / np.linalg.norm(x_classical)
    residual_hhl = np.linalg.norm(A @ x_hhl - b)
    residual_classical = np.linalg.norm(A @ x_classical - b)

    xh_norm = x_hhl / np.linalg.norm(x_hhl)
    xc_norm = x_classical / np.linalg.norm(x_classical)
    fidelity_direction = abs(np.vdot(xh_norm, xc_norm)) ** 2

    result = {
        "x_classical": x_classical,
        "x_hhl": x_hhl,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "residual_hhl": residual_hhl,
        "residual_classical": residual_classical,
        "fidelity_direction": fidelity_direction,
        "hhl_info": hhl_info,
    }

    if verbose:
        np.set_printoptions(precision=10, suppress=True)
        print("=" * 78)
        print("SO SÁNH HAI MODULE GIẢI Ax=b")
        print("=" * 78)
        print("Module 1: classical_solver(A, b)")
        print("Module 2: hhl_solver(A, b) — phục hồi nghiệm từ shot/counts + sign-detect")
        print("-" * 78)
        print(f"A =\n{np.real_if_close(A)}")
        print(f"b = {np.real_if_close(b)}")
        print("-" * 78)
        print(f"x_classical = {np.real_if_close(x_classical)}")
        print(f"x_hhl_shot  = {np.real_if_close(x_hhl)}")
        print("-" * 78)
        print(f"Sai số tuyệt đối ||x_hhl_shot - x_classical|| = {abs_error:.6e}")
        print(f"Sai số tương đối                              = {rel_error:.6e}")
        print(f"Residual HHL-shot ||A*x_hhl_shot - b||        = {residual_hhl:.6e}")
        print(f"Residual cổ điển  ||A*x_classical - b||       = {residual_classical:.6e}")
        print(f"Fidelity hướng nghiệm                         = {fidelity_direction:.10f}")
        print("-" * 78)
        print(f"shots mỗi mạch                                = {hhl_info['shots_per_circuit']}")
        print(f"success_count hậu chọn HHL                    = {hhl_info['success_count']}")
        print(f"postselection_rate                            = {hhl_info['postselection_rate']:.6e}")
        print(f"C_value                                       = {hhl_info['C_value']:.10f}")
        print(f"scale_factor nội bộ                           = {hhl_info['scale_factor']:.10f}")
        print("-" * 78)
        print(f"prob_norm từ shot                             = {hhl_info['prob_norm']}")
        print(f"|x_i| từ shot                                 = {hhl_info['abs_amplitudes']}")
        print(f"signs khôi phục                               = {hhl_info['signs']}")
        print(f"k_coeff                                       = {hhl_info['k_coeff']}")

    return result


# =============================================================================
# VÍ DỤ CHẠY TRỰC TIẾP
# =============================================================================
if __name__ == "__main__":
    A_matrix = np.array(
        [
            [4.0, 0.4, 0.2, 0.0],
            [0.4, 5.0, -0.3, 0.1],
            [0.2, -0.3, 3.5, 0.5],
            [0.0, 0.1, 0.5, 4.5],
        ],
        dtype=np.complex128,
    )

    b_vector = np.array([0.03, -0.02, 0.04, -0.01], dtype=np.complex128)

    compare_solvers(
        A_matrix,
        b_vector,
        phase_qubits=5,
        shots=1_000,
        seed=2026,
    )
