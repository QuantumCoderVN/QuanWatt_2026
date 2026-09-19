"""Quapp function: execute the main HHL measurement circuit.

This function is intentionally fixed to the 4x4 example used in the original
research script. Quapp builds and executes the returned QuantumCircuit; this
file never selects a backend and never calls backend.run().
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate

try:
    from qiskit.synthesis.qft import synth_qft_full
except ImportError:  # Compatibility fallback for older Qiskit runtimes.
    synth_qft_full = None
    from qiskit.circuit.library import QFT


A_MATRIX = np.array(
    [
        [4.0, 0.4, 0.2, 0.0],
        [0.4, 5.0, -0.3, 0.1],
        [0.2, -0.3, 3.5, 0.5],
        [0.0, 0.1, 0.5, 4.5],
    ],
    dtype=np.complex128,
)

B_VECTOR = np.array([0.03, -0.02, 0.04, -0.01], dtype=np.complex128)

PHASE_QUBITS = 5
TARGET_QUBITS = 2
ANCILLA_QUBITS = 1
TOTAL_QUBITS = PHASE_QUBITS + TARGET_QUBITS + ANCILLA_QUBITS
C_VALUE = 0.9 * (2.0 * np.pi / (2**PHASE_QUBITS))

PHASE_INDICES = list(range(PHASE_QUBITS))
TARGET_INDICES = list(range(PHASE_QUBITS, PHASE_QUBITS + TARGET_QUBITS))
ANCILLA_INDEX = PHASE_QUBITS + TARGET_QUBITS


def _qft_circuit(num_qubits: int, inverse: bool) -> QuantumCircuit:
    if synth_qft_full is not None:
        return synth_qft_full(
            num_qubits=num_qubits,
            do_swaps=False,
            inverse=inverse,
        )
    return QFT(num_qubits=num_qubits, inverse=inverse, do_swaps=False)


def _validate_problem() -> None:
    if A_MATRIX.shape != (4, 4):
        raise ValueError("Ví dụ này yêu cầu A có kích thước 4x4.")
    if B_VECTOR.shape != (4,):
        raise ValueError("Ví dụ này yêu cầu b có 4 phần tử.")
    if not np.allclose(A_MATRIX, A_MATRIX.conj().T, atol=1e-10):
        raise ValueError("A phải là Hermitian.")
    eigenvalues = np.linalg.eigvalsh(A_MATRIX)
    if eigenvalues.min() <= 0:
        raise ValueError("A phải xác định dương.")
    if eigenvalues.max() >= 2.0 * np.pi:
        raise ValueError("Phổ của A phải nằm trong (0, 2π) với U=exp(iA).")
    if np.linalg.norm(B_VECTOR) == 0:
        raise ValueError("b không được là vector 0.")


def _controlled_unitary_power(
    circuit: QuantumCircuit,
    unitary: np.ndarray,
    power: int,
    control_qubit: int,
    target_qubits: list[int],
) -> None:
    unitary_power = np.linalg.matrix_power(unitary, power)
    subcircuit = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    subcircuit.unitary(unitary_power, list(range(len(target_qubits))))
    circuit.append(
        subcircuit.to_gate().control(1),
        [control_qubit] + list(target_qubits),
    )


def _append_qpe(
    circuit: QuantumCircuit,
    unitary: np.ndarray,
    phase_indices: list[int],
    target_indices: list[int],
) -> None:
    for qubit in phase_indices:
        circuit.h(qubit)

    for exponent, control in enumerate(reversed(phase_indices)):
        _controlled_unitary_power(
            circuit,
            unitary,
            2**exponent,
            control,
            target_indices,
        )

    circuit.append(
        _qft_circuit(len(phase_indices), inverse=True).to_gate(label="IQFT"),
        phase_indices,
    )


def _append_controlled_rotation(
    circuit: QuantumCircuit,
    phase_indices: list[int],
    ancilla_index: int,
) -> None:
    for phase_integer in range(1, 2**len(phase_indices)):
        lambda_estimate = 2.0 * np.pi * phase_integer / (2**len(phase_indices))
        argument = C_VALUE / lambda_estimate
        if abs(argument) > 1.0:
            continue

        theta = 2.0 * np.arcsin(argument)
        control_state = f"{phase_integer:0{len(phase_indices)}b}"
        controlled_ry = RYGate(theta).control(
            len(phase_indices),
            ctrl_state=control_state,
        )
        circuit.append(controlled_ry, phase_indices + [ancilla_index])


def _build_hhl_without_measurement() -> QuantumCircuit:
    _validate_problem()

    b_state = B_VECTOR / np.linalg.norm(B_VECTOR)
    unitary = expm(1j * A_MATRIX)
    if not np.allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-9):
        raise RuntimeError("U=exp(iA) không unitary trong sai số cho phép.")

    phase_register = QuantumRegister(PHASE_QUBITS, "phase")
    target_register = QuantumRegister(TARGET_QUBITS, "target")
    ancilla_register = QuantumRegister(ANCILLA_QUBITS, "ancilla")
    circuit = QuantumCircuit(
        phase_register,
        target_register,
        ancilla_register,
        name="hhl-main",
    )

    circuit.initialize(b_state, TARGET_INDICES)
    _append_qpe(circuit, unitary, PHASE_INDICES, TARGET_INDICES)
    _append_controlled_rotation(circuit, PHASE_INDICES, ANCILLA_INDEX)

    qpe_only = QuantumCircuit(PHASE_QUBITS + TARGET_QUBITS, name="QPE")
    _append_qpe(
        qpe_only,
        unitary,
        list(range(PHASE_QUBITS)),
        list(range(PHASE_QUBITS, PHASE_QUBITS + TARGET_QUBITS)),
    )
    inverse_qpe = qpe_only.inverse()
    inverse_qpe.name = "QPE-dagger"
    circuit.append(inverse_qpe, PHASE_INDICES + TARGET_INDICES)

    return circuit


def _add_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    measured = circuit.copy()
    classical = ClassicalRegister(measured.num_qubits, "meas")
    measured.add_register(classical)
    measured.measure(
        list(range(measured.num_qubits)),
        list(range(measured.num_qubits)),
    )
    return measured


def processing(invocation_input: Any) -> QuantumCircuit:
    """Build and return the fixed HHL circuit.

    ``invocation_input`` is intentionally unused in version 1 so that the
    matrix, vector, qubit layout and post-processing schema are reproducible.
    Invoke with an empty JSON object: ``{}``.
    """
    del invocation_input
    return _add_measurements(_build_hhl_without_measurement())


def post_processing(job_result: Any) -> dict[str, Any]:
    """Convert the backend result into JSON-safe counts plus fixed metadata."""
    counts = job_result.get_counts()
    if isinstance(counts, list):
        if len(counts) != 1:
            raise ValueError("Function này chỉ hỗ trợ đúng một circuit mỗi job.")
        counts = counts[0]

    clean_counts = {str(key): int(value) for key, value in counts.items()}
    shots = int(sum(clean_counts.values()))

    return {
        "schema_version": "1.0",
        "circuit_type": "hhl-main",
        "counts": clean_counts,
        "shots": shots,
        "qubit_layout": {
            "total_qubits": TOTAL_QUBITS,
            "phase_indices": PHASE_INDICES,
            "target_indices": TARGET_INDICES,
            "ancilla_index": ANCILLA_INDEX,
            "counts_endianness": "qiskit-big-endian-display",
            "measurement_mapping": "qubit-q-to-classical-bit-q",
        },
        "algorithm": {
            "phase_qubits": PHASE_QUBITS,
            "target_qubits": TARGET_QUBITS,
            "C_value": float(C_VALUE),
            "success_ancilla_value": 1,
        },
        "problem": {
            "A": A_MATRIX.real.tolist(),
            "b": B_VECTOR.real.tolist(),
        },
    }
