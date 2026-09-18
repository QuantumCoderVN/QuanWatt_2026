# ============================================================
# hhl_solver.py
# HHL quantum linear system solver
# ============================================================

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, RYGate
from qiskit.quantum_info import Statevector
from typing import Tuple, Optional


def pad_to_power_of_two(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Pad system to power-of-two dimension for quantum state representation.

    Returns:
        (A_padded, b_padded, original_dimension)
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square matrix")

    if len(b) != n:
        raise ValueError("b dimension mismatch with A")

    dim = 1
    while dim < n:
        dim *= 2

    # HHL needs at least 2 dimensions for target qubits
    if dim < 2:
        dim = 2

    A_pad = np.eye(dim, dtype=complex)
    b_pad = np.zeros(dim, dtype=complex)

    A_pad[:n, :n] = A
    b_pad[:n] = b

    return A_pad, b_pad, n


def gershgorin_lambda_bound(A: np.ndarray) -> float:
    """
    Compute upper bound on eigenvalues using Gershgorin circle theorem.
    Avoids explicit eigenvalue computation.
    """
    A = np.asarray(A, dtype=complex)

    diag_abs = np.abs(np.diag(A))
    row_sum_abs = np.sum(np.abs(A), axis=1)

    radii = row_sum_abs - diag_abs
    bound = np.max(diag_abs + radii)

    if bound <= 1e-12:
        raise ValueError("Gershgorin bound too small for HHL scaling")

    return float(bound)


def recover_scaled_solution(A: np.ndarray, b: np.ndarray, x_prime: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Recover physical scale from normalized quantum solution.

    Method:
        b_prime = A @ x_prime
        k = <b_prime, b> / <b_prime, b_prime>
        x = k * x_prime

    Returns:
        (x_recovered, k_coefficient)
    """
    b_prime = A @ x_prime
    denom = np.vdot(b_prime, b_prime)

    if abs(denom) < 1e-12:
        k = 0.0
    else:
        k = np.vdot(b_prime, b) / denom

    x_recovered = k * x_prime

    return x_recovered, float(k)


def clean_real_vector(x: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Extract real part of solution, warn if imaginary part is significant.
    """
    x = np.asarray(x)

    if np.max(np.abs(np.imag(x))) > tol:
        print(f"[WARN] Solution has significant imaginary part: max={np.max(np.abs(np.imag(x))):.2e}")

    return np.real(x)


def controlled_unitary_power(qc: QuantumCircuit, U: np.ndarray, power: int,
                            control_qubit: int, target_qubits: list):
    """
    Apply controlled U^power gate to quantum circuit.
    """
    U_power = np.linalg.matrix_power(U, power)

    gate = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    gate.unitary(U_power, list(range(len(target_qubits))))

    controlled_gate = gate.to_gate().control(1)
    qc.append(controlled_gate, [control_qubit] + list(target_qubits))


def qpe_circuit(qc: QuantumCircuit, phase_reg: list, target_reg: list, U_matrix: np.ndarray):
    """
    Quantum Phase Estimation circuit.

    Steps:
        1. Apply Hadamard to phase register
        2. Apply controlled-U^{2^k} gates
        3. Apply inverse QFT to phase register
    """
    # Hadamard on phase register
    for w in phase_reg:
        qc.h(w)

    # Controlled-U^{2^k} operations
    for i, ctrl in enumerate(reversed(phase_reg)):
        power = 2 ** i
        controlled_unitary_power(qc, U_matrix, power, ctrl, target_reg)

    # Inverse QFT
    iqft = QFT(num_qubits=len(phase_reg), inverse=True, do_swaps=False)
    qc.append(iqft, phase_reg)


def control_rotation_gate(qc: QuantumCircuit, control_reg: list, target_qubit: int,
                         t: float, C: float = 1.0):
    """
    Apply controlled rotation for eigenvalue inversion.

    For each bitstring d in control register:
        phi = binary fraction
        lambda = 2*pi/t * phi
        amplitude = C / lambda
        theta = 2 * arcsin(amplitude)
    """
    n = len(control_reg)

    for d in range(1, 2**n):
        bin_str = f"{d:0{n}b}"

        phi = sum(int(bit) * 2 ** (-(j + 1)) for j, bit in enumerate(bin_str))

        if phi > 0.5:
            phi = phi - 1.0

        lam = (2 * np.pi / t) * phi

        if abs(lam) < 1e-12:
            continue

        amp = C / lam

        if abs(amp) > 1:
            continue

        theta = 2 * np.arcsin(amp)

        cry = RYGate(theta).control(n, ctrl_state=bin_str)
        qc.append(cry, list(control_reg) + [target_qubit])


def hhl_solve(A: np.ndarray, b: np.ndarray,
             phase_qubits: int = 8,
             phase_target: float = 0.40,
             C: float = 1.0,
             label: str = "HHL") -> np.ndarray:
    """
    Solve Ax=b using HHL algorithm with Qiskit statevector simulation.

    Parameters:
        A: Coefficient matrix (will be padded to power-of-two)
        b: Right-hand side vector
        phase_qubits: Number of qubits for phase estimation (default 8)
        phase_target: Target maximum phase value (default 0.40)
        C: Reciprocal rotation constant (default 1.0)
        label: Label for logging

    Returns:
        Solution vector x
    """
    A_pad, b_pad, n_original = pad_to_power_of_two(A, b)

    if np.linalg.norm(b_pad) < 1e-14:
        return np.zeros(n_original)

    dim = A_pad.shape[0]
    n_qubits = int(np.log2(dim))
    ancilla_qubit = 1
    total_qubits = n_qubits + phase_qubits + ancilla_qubit

    # Normalize b vector
    b_norm = b_pad / np.linalg.norm(b_pad)

    # Scale matrix for HHL using Gershgorin bound
    lambda_bound = gershgorin_lambda_bound(A_pad)

    t_eff = phase_target * 2 * np.pi / lambda_bound

    # Build unitary evolution operator
    U_matrix = expm(1j * A_pad * t_eff)

    # Verify unitarity
    assert np.allclose(
        U_matrix.conj().T @ U_matrix,
        np.eye(dim),
        atol=1e-8
    ), "U is not unitary"

    # Create quantum registers
    phase_reg = QuantumRegister(phase_qubits, name="phase")
    target_reg = QuantumRegister(n_qubits, name="target")
    ancilla_reg = QuantumRegister(ancilla_qubit, name="ancilla")

    qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL")

    phase_indices = list(range(phase_qubits))
    target_indices = list(range(phase_qubits, phase_qubits + n_qubits))
    ancilla_index = phase_qubits + n_qubits

    # Step 1: Initialize |b⟩ state
    qc_hhl.initialize(b_norm, target_indices)

    # Step 2: Quantum Phase Estimation
    qpe_circuit(qc_hhl, phase_indices, target_indices, U_matrix)

    # Step 3: Controlled rotation for eigenvalue inversion
    control_rotation_gate(
        qc_hhl,
        phase_indices,
        ancilla_index,
        t=t_eff,
        C=C
    )

    # Step 4: Inverse QPE
    qc_qpe_only = QuantumCircuit(phase_qubits + n_qubits, name="QPE")
    qpe_circuit(
        qc_qpe_only,
        list(range(phase_qubits)),
        list(range(phase_qubits, phase_qubits + n_qubits)),
        U_matrix
    )

    inv_qpe = qc_qpe_only.inverse()
    inv_qpe.name = "QPE†"

    qc_hhl.append(inv_qpe, phase_indices + target_indices)

    # Simulate and extract statevector
    sv = Statevector.from_instruction(qc_hhl)

    # Extract amplitude at ancilla=1, phase=0
    # Qiskit uses little-endian ordering
    raw = np.zeros(dim, dtype=complex)

    for target_state in range(dim):
        idx = 0
        idx |= target_state << phase_qubits
        idx |= 1 << (phase_qubits + n_qubits)

        raw[target_state] = sv.data[idx]

    if np.linalg.norm(raw) < 1e-14:
        print(f"[WARN] {label}: HHL postselection on ancilla=1 near zero")
        return np.zeros(n_original)

    # Recover physical scale
    x_recovered_pad, k = recover_scaled_solution(A_pad, b_pad, raw)

    x = x_recovered_pad[:n_original]

    return clean_real_vector(x)


def hhl_solve_with_diagnostics(A: np.ndarray, b: np.ndarray,
                               phase_qubits: int = 8,
                               phase_target: float = 0.40,
                               C: float = 1.0,
                               label: str = "HHL") -> dict:
    """
    Solve Ax=b using HHL with detailed diagnostics.

    Returns:
        Dictionary containing solution, scaling info, and residual
    """
    A_pad, b_pad, n_original = pad_to_power_of_two(A, b)

    if np.linalg.norm(b_pad) < 1e-14:
        return {
            "solution": np.zeros(n_original),
            "lambda_bound": 0.0,
            "t_eff": 0.0,
            "k_coefficient": 0.0,
            "residual": 0.0,
            "raw_amplitude_norm": 0.0,
        }

    dim = A_pad.shape[0]
    n_qubits = int(np.log2(dim))
    ancilla_qubit = 1

    b_norm = b_pad / np.linalg.norm(b_pad)

    lambda_bound = gershgorin_lambda_bound(A_pad)
    t_eff = phase_target * 2 * np.pi / lambda_bound

    U_matrix = expm(1j * A_pad * t_eff)

    phase_reg = QuantumRegister(phase_qubits, name="phase")
    target_reg = QuantumRegister(n_qubits, name="target")
    ancilla_reg = QuantumRegister(ancilla_qubit, name="ancilla")

    qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL")

    phase_indices = list(range(phase_qubits))
    target_indices = list(range(phase_qubits, phase_qubits + n_qubits))
    ancilla_index = phase_qubits + n_qubits

    qc_hhl.initialize(b_norm, target_indices)
    qpe_circuit(qc_hhl, phase_indices, target_indices, U_matrix)
    control_rotation_gate(qc_hhl, phase_indices, ancilla_index, t=t_eff, C=C)

    qc_qpe_only = QuantumCircuit(phase_qubits + n_qubits, name="QPE")
    qpe_circuit(
        qc_qpe_only,
        list(range(phase_qubits)),
        list(range(phase_qubits, phase_qubits + n_qubits)),
        U_matrix
    )

    inv_qpe = qc_qpe_only.inverse()
    qc_hhl.append(inv_qpe, phase_indices + target_indices)

    sv = Statevector.from_instruction(qc_hhl)

    raw = np.zeros(dim, dtype=complex)
    for target_state in range(dim):
        idx = 0
        idx |= target_state << phase_qubits
        idx |= 1 << (phase_qubits + n_qubits)
        raw[target_state] = sv.data[idx]

    raw_norm = np.linalg.norm(raw)

    if raw_norm < 1e-14:
        return {
            "solution": np.zeros(n_original),
            "lambda_bound": float(lambda_bound),
            "t_eff": float(t_eff),
            "k_coefficient": 0.0,
            "residual": float('inf'),
            "raw_amplitude_norm": float(raw_norm),
        }

    x_recovered_pad, k = recover_scaled_solution(A_pad, b_pad, raw)
    x = x_recovered_pad[:n_original]
    x = clean_real_vector(x)

    residual = np.linalg.norm(A @ x - b)

    return {
        "solution": x,
        "lambda_bound": float(lambda_bound),
        "t_eff": float(t_eff),
        "k_coefficient": float(k),
        "residual": float(residual),
        "raw_amplitude_norm": float(raw_norm),
    }
