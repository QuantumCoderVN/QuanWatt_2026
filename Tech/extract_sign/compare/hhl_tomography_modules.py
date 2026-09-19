"""
Hai module giải hệ tuyến tính Ax = b:
  1) classical_solver(A, b): giải cổ điển bằng numpy.linalg.solve.
  2) hhl_tomography_solver(A, b): giải bằng HHL, đo quantum tomography
     trên target register sau hậu chọn, rồi khôi phục nghiệm không chuẩn hóa.

Điểm quan trọng:
  - hhl_tomography_solver KHÔNG dùng Statevector để khôi phục nghiệm.
  - Nghiệm HHL được khôi phục từ counts/shot tomography:
      + chạy các mạch tomography theo basis {X,Y,Z}^n trên target
      + hậu chọn phase = |0...0>, ancilla = |1>
      + dựng rho_target từ kỳ vọng Pauli
      + trích biên độ thực có dấu từ rho_target
      + dùng hệ số k để khôi phục nghiệm không chuẩn hóa.

Yêu cầu:
  pip install numpy scipy qiskit qiskit-aer

Giả định của implementation HHL đơn giản này:
  - A Hermitian, xác định dương.
  - Sau scale nội bộ, phổ của A nằm trong (0, 2π).
  - Tomography full Pauli chỉ phù hợp cho số qubit target nhỏ vì cần 3^n mạch.
  - Hàm trích biên độ mặc định giả định nghiệm gần pure state và có biên độ thực.
    Với nghiệm phức tổng quát, cần khôi phục pha tương đối đầy đủ.
"""

from __future__ import annotations

import itertools
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
# HÀM PHỤ: CHUẨN BỊ HỆ Ax=b CHO HHL
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


# =============================================================================
# HÀM PHỤ: BUILD MẠCH HHL KHÔNG ĐO
# =============================================================================
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

    # Convention giống code gốc: reversed(phase), U^(2^i), IQFT do_swaps=False.
    for i, ctrl in enumerate(reversed(phase_indices)):
        _controlled_unitary_power(qc, U_matrix, 2**i, ctrl, target_indices)

    iqft = synth_qft_full(
        num_qubits=len(phase_indices),
        do_swaps=False,
        inverse=True,
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
    qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL")

    phase_indices = list(range(phase_qubits))
    target_indices = list(range(phase_qubits, phase_qubits + n_target_qubits))
    ancilla_index = phase_qubits + n_target_qubits

    qc_hhl.initialize(b_state, target_indices)
    _apply_qpe(qc_hhl, U_matrix, phase_indices, target_indices)

    if C_value is None:
        # Giữ lựa chọn giống code gốc: C = 0.9 * độ phân giải phase.
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


# =============================================================================
# HÀM PHỤ: ĐỌC COUNTS THEO LITTLE-ENDIAN CỦA QISKIT
# =============================================================================
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


# =============================================================================
# HÀM PHỤ: PAULI/TOMOGRAPHY
# =============================================================================
def _all_pauli_strings(n: int) -> list[str]:
    """Tất cả Pauli strings độ dài n trên alphabet {I, X, Y, Z}."""
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)]


def _unique_tomography_bases(n: int) -> list[str]:
    """Các cơ sở đo tomography đầy đủ: {X, Y, Z}^n."""
    return ["".join(b) for b in itertools.product("XYZ", repeat=n)]


def _pauli_to_measurement_basis(pauli_string: str) -> str:
    """Đổi Pauli string thành cơ sở đo tương thích; I được đo như Z."""
    return "".join("Z" if p == "I" else p for p in pauli_string)


def _single_pauli_matrix(p: str) -> np.ndarray:
    if p == "I":
        return np.array([[1, 0], [0, 1]], dtype=np.complex128)
    if p == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if p == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if p == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(f"Unknown Pauli: {p}")


def _pauli_matrix(pauli_string: str) -> np.ndarray:
    """Ma trận Pauli tensor product ứng với pauli_string.

    pauli_string được hiểu theo thứ tự q_{n-1} ... q_1 q_0,
    khớp với thứ tự computational basis |000>, |001>, ...
    """
    mats = [_single_pauli_matrix(p) for p in pauli_string]
    result = mats[0]
    for mat in mats[1:]:
        result = np.kron(result, mat)
    return result


def _add_target_basis_rotation(
    qc: QuantumCircuit,
    target_indices: list[int],
    basis: str,
) -> None:
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


def _make_hhl_tomography_circuit(
    qc_base_hhl: QuantumCircuit,
    target_indices: list[int],
    basis: str,
) -> QuantumCircuit:
    """Tạo một mạch tomography cho target state sau HHL."""
    qc_tomo = qc_base_hhl.copy(name=f"HHL_tomo_{basis}")
    _add_target_basis_rotation(qc_tomo, target_indices, basis)

    meas = ClassicalRegister(qc_tomo.num_qubits, f"meas_{basis}")
    qc_tomo.add_register(meas)
    qc_tomo.measure(list(range(qc_tomo.num_qubits)), list(meas))

    return qc_tomo


def _postselected_target_counts(
    counts: dict[str, int],
    total_qubits: int,
    phase_indices: list[int],
    target_indices: list[int],
    ancilla_index: int,
    success_ancilla_value: int,
) -> tuple[dict[str, int], int]:
    """Lọc counts theo phase=0...0 và ancilla=success, trả về counts target.

    target_counts dùng key bitstring theo thứ tự q_{n-1}...q_0, ví dụ "10".
    """
    n_target = len(target_indices)
    target_counts: dict[str, int] = {}
    success_count = 0

    for key, count in counts.items():
        bits = _key_to_bits_little_endian(key, total_qubits)

        if not _is_phase_zero(bits, phase_indices):
            continue
        if bits[ancilla_index] != success_ancilla_value:
            continue

        target_value = _read_register_int(bits, target_indices)
        target_bitstring = format(target_value, f"0{n_target}b")
        target_counts[target_bitstring] = target_counts.get(target_bitstring, 0) + count
        success_count += count

    if success_count == 0:
        raise RuntimeError(
            "Không có shot nào thỏa hậu chọn phase=0 và ancilla=1. "
            "Hãy tăng tomography_shots hoặc kiểm tra C_value / phase_qubits."
        )

    return target_counts, success_count


def _pauli_eigenvalue_from_bitstring(pauli_string: str, bitstring: str) -> int:
    """Eigenvalue +/-1 của Pauli string trên bitstring đo được."""
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


def _estimate_pauli_expectation(
    pauli_string: str,
    target_counts: dict[str, int],
) -> float:
    """Ước lượng <P> từ target counts đã hậu chọn."""
    shots = sum(target_counts.values())
    expval = 0.0

    for bitstring, count in target_counts.items():
        eig = _pauli_eigenvalue_from_bitstring(pauli_string, bitstring)
        expval += eig * count / shots

    return expval


def _project_to_physical_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Projection đơn giản về Hermitian, PSD, trace-one.

    Đây không phải maximum-likelihood tomography đầy đủ, nhưng giúp giảm
    eigenvalue âm nhỏ do finite-shot noise.
    """
    rho = 0.5 * (rho + rho.conj().T)

    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)

    total = float(np.sum(eigvals))
    if total <= 0:
        raise ValueError("All eigenvalues vanished after clipping.")

    eigvals = eigvals / total
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


def _quantum_tomography_on_hhl_target(
    qc_base_hhl: QuantumCircuit,
    phase_indices: list[int],
    target_indices: list[int],
    ancilla_index: int,
    shots: int,
    seed: int | None,
    make_physical: bool,
    success_ancilla_value: int,
    progress: bool,
) -> dict[str, Any]:
    """Dựng rho_target bằng full Pauli-basis tomography từ counts/shot."""
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError(
            "Bạn cần cài qiskit-aer để chạy HHL tomography bằng shot:\n"
            "    pip install qiskit-aer"
        ) from exc

    n_target = len(target_indices)
    bases = _unique_tomography_bases(n_target)
    backend = AerSimulator(seed_simulator=seed)

    tomography_circuits = [
        _make_hhl_tomography_circuit(qc_base_hhl, target_indices, basis)
        for basis in bases
    ]

    if progress:
        print(
            f"[TOMO] running {len(tomography_circuits)} circuits, "
            f"shots_per_basis={shots}, target_qubits={n_target}",
            flush=True,
        )
        print("[TOMO] transpiling...", flush=True)

    transpiled = transpile(tomography_circuits, backend, optimization_level=0)

    if progress:
        print("[TOMO] simulating...", flush=True)

    result = backend.run(transpiled, shots=shots).result()

    counts_by_basis: dict[str, dict[str, int]] = {}
    target_counts_by_basis: dict[str, dict[str, int]] = {}
    success_count_by_basis: dict[str, int] = {}

    for i, basis in enumerate(bases):
        counts = result.get_counts(i)
        counts_by_basis[basis] = counts

        target_counts, success_count = _postselected_target_counts(
            counts=counts,
            total_qubits=qc_base_hhl.num_qubits,
            phase_indices=phase_indices,
            target_indices=target_indices,
            ancilla_index=ancilla_index,
            success_ancilla_value=success_ancilla_value,
        )

        target_counts_by_basis[basis] = target_counts
        success_count_by_basis[basis] = success_count

        if progress:
            rate = success_count / shots
            print(f"[TOMO] basis {basis}: success={success_count}/{shots} ({rate:.3e})", flush=True)

    pauli_expectations: dict[str, float] = {}

    for pauli_string in _all_pauli_strings(n_target):
        if pauli_string == "I" * n_target:
            pauli_expectations[pauli_string] = 1.0
            continue

        measurement_basis = _pauli_to_measurement_basis(pauli_string)
        target_counts = target_counts_by_basis[measurement_basis]
        pauli_expectations[pauli_string] = _estimate_pauli_expectation(
            pauli_string,
            target_counts,
        )

    dim = 2**n_target
    rho = np.zeros((dim, dim), dtype=np.complex128)

    for pauli_string, expval in pauli_expectations.items():
        rho += expval * _pauli_matrix(pauli_string)

    rho = rho / dim
    rho = 0.5 * (rho + rho.conj().T)

    if make_physical:
        rho = _project_to_physical_density_matrix(rho)

    return {
        "rho": rho,
        "pauli_expectations": pauli_expectations,
        "target_counts_by_basis": target_counts_by_basis,
        "success_count_by_basis": success_count_by_basis,
        "counts_by_basis": counts_by_basis,
        "tomography_bases": bases,
    }


def _extract_real_amplitudes_from_density_matrix(
    rho: np.ndarray,
    reference_index: int | None,
    reference_sign: int,
) -> dict[str, Any]:
    """Lấy biên độ thực có dấu từ rho_target.

    Giả định:
      - target state gần pure state,
      - nghiệm có biên độ thực,
      - chọn một reference amplitude khác 0.

    Nếu reference_index=None, dùng basis có xác suất đường chéo lớn nhất để ổn định số.
    """
    if reference_sign not in (+1, -1):
        raise ValueError("reference_sign phải là +1 hoặc -1.")

    dim = rho.shape[0]
    diagonal = np.real(np.diag(rho))

    if reference_index is None:
        reference_index = int(np.argmax(diagonal))
    if not (0 <= reference_index < dim):
        raise ValueError("reference_index vượt ngoài kích thước rho.")

    p_ref = float(np.real(rho[reference_index, reference_index]))
    if p_ref <= 0:
        raise ValueError(
            "Reference amplitude có xác suất <= 0. "
            "Hãy tăng shots hoặc chọn reference_index khác."
        )

    a_ref = reference_sign * np.sqrt(p_ref)
    amplitudes = np.zeros(dim, dtype=np.complex128)
    diagnostics: list[dict[str, Any]] = []

    for i in range(dim):
        # Với nghiệm thực: rho[ref, i] = a_ref * a_i.
        amplitude = np.real(rho[reference_index, i]) / a_ref
        amplitudes[i] = amplitude
        diagnostics.append(
            {
                "basis_index": i,
                "rho_ref_i_real": float(np.real(rho[reference_index, i])),
                "rho_ii_real": float(np.real(rho[i, i])),
                "estimated_amplitude": float(np.real(amplitude)),
                "magnitude_from_diagonal": float(np.sqrt(max(np.real(rho[i, i]), 0.0))),
                "sign": int(np.sign(np.real(amplitude))) if abs(np.real(amplitude)) > 1e-12 else 0,
            }
        )

    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise RuntimeError("Vector amplitude trích từ rho có norm bằng 0.")

    amplitudes_normalized = amplitudes / norm

    return {
        "amplitudes_raw": amplitudes,
        "amplitudes_normalized": amplitudes_normalized,
        "reference_index": reference_index,
        "reference_probability": p_ref,
        "diagnostics": diagnostics,
    }


# =============================================================================
# MODULE 2 — GIẢI BẰNG HHL + QUANTUM TOMOGRAPHY
# =============================================================================
def hhl_tomography_solver(
    A: np.ndarray,
    b: np.ndarray,
    phase_qubits: int = 5,
    tomography_shots: int = 300_000,
    seed: int | None = 2026,
    reference_index: int | None = 0,
    reference_sign: int = +1,
    success_ancilla_value: int = 1,
    make_physical: bool = True,
    C_value: float | None = None,
    auto_scale: bool = True,
    auto_pad: bool = True,
    progress: bool = False,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Giải Ax=b bằng HHL và khôi phục nghiệm bằng quantum tomography từ counts/shot.

    Parameters
    ----------
    A, b:
        Hệ tuyến tính Ax=b.
    phase_qubits:
        Số qubit phase trong QPE.
    tomography_shots:
        Số shot cho mỗi cơ sở tomography. Tổng số shot xấp xỉ 3^n * tomography_shots.
    seed:
        Seed của AerSimulator.
    reference_index:
        Basis index dùng làm mốc để trích dấu từ rho. Nếu None, tự chọn index có
        xác suất đường chéo lớn nhất. Nếu muốn giống code gốc, để 0.
    reference_sign:
        Quy ước dấu/pha toàn cục cho reference amplitude.
    success_ancilla_value:
        Code HHL này hậu chọn ancilla=1.
    make_physical:
        Nếu True, chiếu rho về PSD trace-one để giảm nhiễu shot tuyến tính.
    C_value:
        Hằng số C trong controlled-RY. Nếu None, dùng C = 0.9 * 2π / 2^phase_qubits.
    auto_scale:
        Nếu True, tự scale A trong HHL để phổ nằm trong (0, 2π).
    auto_pad:
        Nếu True, tự padding kích thước lên 2^n.
    progress:
        Nếu True, in tiến trình chạy tomography.
    return_info:
        Nếu True, trả thêm dict chứa rho, counts, diagnostics, circuit.

    Returns
    -------
    x_hhl_tomo:
        Vector nghiệm HHL đã khôi phục không chuẩn hóa từ tomography shot,
        có thể so sánh trực tiếp với nghiệm cổ điển.
    """
    prepared = _validate_and_prepare_system(A, b, auto_pad=auto_pad, auto_scale=auto_scale)

    circuit_data = _build_hhl_circuit(
        A_hhl=prepared["A_hhl"],
        b_padded=prepared["b_padded"],
        phase_qubits=phase_qubits,
        C_value=C_value,
    )

    tomography_result = _quantum_tomography_on_hhl_target(
        qc_base_hhl=circuit_data["circuit"],
        phase_indices=circuit_data["phase_indices"],
        target_indices=circuit_data["target_indices"],
        ancilla_index=circuit_data["ancilla_index"],
        shots=tomography_shots,
        seed=seed,
        make_physical=make_physical,
        success_ancilla_value=success_ancilla_value,
        progress=progress,
    )

    rho = tomography_result["rho"]
    amplitude_result = _extract_real_amplitudes_from_density_matrix(
        rho=rho,
        reference_index=reference_index,
        reference_sign=reference_sign,
    )

    # Vector hướng nghiệm trên không gian đã padding.
    x_direction_padded = amplitude_result["amplitudes_normalized"]

    # Cắt về số chiều gốc, rồi chuẩn hóa lại vì phần padding có thể nhận xác suất nhỏ.
    original_dim = prepared["original_dim"]
    x_direction = x_direction_padded[:original_dim].astype(np.complex128)
    direction_norm = np.linalg.norm(x_direction)
    if direction_norm == 0:
        raise RuntimeError("Phần vector nghiệm tương ứng kích thước gốc có norm bằng 0.")
    x_direction = x_direction / direction_norm

    # Khôi phục nghiệm không chuẩn hóa bằng hệ số k:
    #     x_hhl_tomo = k * x_direction
    # với k được chọn để A @ x_hhl_tomo gần b nhất theo least-squares 1D.
    A_original = prepared["A_original"]
    b_original = prepared["b_original"]
    b_from_direction = A_original @ x_direction
    denom = np.vdot(b_from_direction, b_from_direction)
    if abs(denom) < 1e-15:
        raise RuntimeError("Không thể tính k_coeff vì A @ x_direction gần bằng 0.")

    k_coeff = np.vdot(b_from_direction, b_original) / denom
    x_hhl_tomo = k_coeff * x_direction

    if not return_info:
        return x_hhl_tomo

    rho_eigvals = np.linalg.eigvalsh(rho)
    purity = float(np.real(np.trace(rho @ rho)))

    info = {
        "circuit": circuit_data["circuit"],
        "phase_qubits": phase_qubits,
        "tomography_shots_per_basis": tomography_shots,
        "total_tomography_bases": len(tomography_result["tomography_bases"]),
        "total_nominal_shots": len(tomography_result["tomography_bases"]) * tomography_shots,
        "seed": seed,
        "n_target_qubits": circuit_data["n_target_qubits"],
        "original_dim": prepared["original_dim"],
        "padded_dim": prepared["padded_dim"],
        "scale_factor": prepared["scale_factor"],
        "C_value": circuit_data["C_value"],
        "eigenvalues_original": prepared["eigenvalues_original"],
        "eigenvalues_hhl_scaled": prepared["eigenvalues_hhl"],
        "rho": rho,
        "rho_eigvals": rho_eigvals,
        "rho_purity": purity,
        "pauli_expectations": tomography_result["pauli_expectations"],
        "target_counts_by_basis": tomography_result["target_counts_by_basis"],
        "success_count_by_basis": tomography_result["success_count_by_basis"],
        "tomography_bases": tomography_result["tomography_bases"],
        "amplitudes_raw_padded": amplitude_result["amplitudes_raw"],
        "amplitudes_normalized_padded": amplitude_result["amplitudes_normalized"],
        "amplitudes_normalized_original": x_direction,
        "amplitude_diagnostics": amplitude_result["diagnostics"],
        "reference_index": amplitude_result["reference_index"],
        "reference_probability": amplitude_result["reference_probability"],
        "k_coeff": k_coeff,
    }
    return x_hhl_tomo, info


# =============================================================================
# SO SÁNH OUTPUT CỦA HAI MODULE
# =============================================================================
def compare_solvers(
    A: np.ndarray,
    b: np.ndarray,
    phase_qubits: int = 5,
    tomography_shots: int = 300_000,
    seed: int | None = 2026,
    reference_index: int | None = 0,
    verbose: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    """Chạy module cổ điển và module HHL-tomography, sau đó so sánh output."""
    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)

    x_classical = classical_solver(A, b)
    x_hhl_tomo, hhl_info = hhl_tomography_solver(
        A,
        b,
        phase_qubits=phase_qubits,
        tomography_shots=tomography_shots,
        seed=seed,
        reference_index=reference_index,
        progress=progress,
        return_info=True,
    )

    abs_error = np.linalg.norm(x_hhl_tomo - x_classical)
    rel_error = abs_error / np.linalg.norm(x_classical)
    residual_hhl = np.linalg.norm(A @ x_hhl_tomo - b)
    residual_classical = np.linalg.norm(A @ x_classical - b)

    xh_norm = x_hhl_tomo / np.linalg.norm(x_hhl_tomo)
    xc_norm = x_classical / np.linalg.norm(x_classical)
    fidelity_direction = abs(np.vdot(xh_norm, xc_norm)) ** 2

    result = {
        "x_classical": x_classical,
        "x_hhl_tomography": x_hhl_tomo,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "residual_hhl_tomography": residual_hhl,
        "residual_classical": residual_classical,
        "fidelity_direction": fidelity_direction,
        "hhl_tomography_info": hhl_info,
    }

    if verbose:
        np.set_printoptions(precision=10, suppress=True)
        print("=" * 78)
        print("SO SÁNH HAI MODULE GIẢI Ax=b")
        print("=" * 78)
        print("Module 1: classical_solver(A, b)")
        print("Module 2: hhl_tomography_solver(A, b) — phục hồi nghiệm từ quantum tomography")
        print("-" * 78)
        print(f"A =\n{np.real_if_close(A)}")
        print(f"b = {np.real_if_close(b)}")
        print("-" * 78)
        print(f"x_classical     = {np.real_if_close(x_classical)}")
        print(f"x_hhl_tomography = {np.real_if_close(x_hhl_tomo)}")
        print("-" * 78)
        print(f"Sai số tuyệt đối ||x_hhl_tomo - x_classical|| = {abs_error:.6e}")
        print(f"Sai số tương đối                              = {rel_error:.6e}")
        print(f"Residual HHL-tomo ||A*x_hhl_tomo - b||        = {residual_hhl:.6e}")
        print(f"Residual cổ điển  ||A*x_classical - b||       = {residual_classical:.6e}")
        print(f"Fidelity hướng nghiệm                         = {fidelity_direction:.10f}")
        print("-" * 78)
        print(f"phase_qubits                                  = {hhl_info['phase_qubits']}")
        print(f"tomography bases                              = {hhl_info['total_tomography_bases']}")
        print(f"shots mỗi basis                               = {hhl_info['tomography_shots_per_basis']}")
        print(f"tổng shots danh nghĩa                         = {hhl_info['total_nominal_shots']}")
        print(f"reference_index                               = {hhl_info['reference_index']}")
        print(f"reference_probability                         = {hhl_info['reference_probability']:.6e}")
        print(f"rho purity Tr(rho^2)                          = {hhl_info['rho_purity']:.10f}")
        print(f"C_value                                       = {hhl_info['C_value']:.10f}")
        print(f"scale_factor nội bộ                           = {hhl_info['scale_factor']:.10f}")
        print(f"k_coeff                                       = {hhl_info['k_coeff']}")
        print("-" * 78)
        print("success_count hậu chọn theo từng basis:")
        for basis, count in hhl_info["success_count_by_basis"].items():
            rate = count / hhl_info["tomography_shots_per_basis"]
            print(f"  basis {basis}: {count} / {hhl_info['tomography_shots_per_basis']} = {rate:.6e}")
        print("-" * 78)
        print("amplitudes normalized từ tomography:")
        print(np.real_if_close(hhl_info["amplitudes_normalized_original"]))

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

    # Để test nhanh, dùng tomography_shots thấp trước.
    # Khi cần kết quả mượt hơn, tăng lên 300_000 hoặc 1_000_000.
    compare_solvers(
        A_matrix,
        b_vector,
        phase_qubits=5,
        tomography_shots=50_000,
        seed=2026,
        reference_index=0,
        verbose=True,
        progress=True,
    )
