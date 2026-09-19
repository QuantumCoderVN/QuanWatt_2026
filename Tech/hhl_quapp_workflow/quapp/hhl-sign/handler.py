"""Quapp function: execute one HHL sign-detection circuit.

Every invocation builds the same fixed HHL circuit and adds the interference
module for one requested pair (i, j). The pair is supplied as raw JSON, for
example: {"pair_i": 0, "pair_j": 1}.
"""

from __future__ import annotations

import json
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
HHL_TOTAL_QUBITS = PHASE_QUBITS + TARGET_QUBITS + ANCILLA_QUBITS
SIGN_TOTAL_QUBITS = HHL_TOTAL_QUBITS + 1
C_VALUE = 0.9 * (2.0 * np.pi / (2**PHASE_QUBITS))

PHASE_INDICES = list(range(PHASE_QUBITS))
TARGET_INDICES = list(range(PHASE_QUBITS, PHASE_QUBITS + TARGET_QUBITS))
ANCILLA_INDEX = PHASE_QUBITS + TARGET_QUBITS
SIGN_EXTRA_INDEX = HHL_TOTAL_QUBITS


def _qft_circuit(num_qubits: int, inverse: bool) -> QuantumCircuit:
    if synth_qft_full is not None:
        return synth_qft_full(
            num_qubits=num_qubits,
            do_swaps=False,
            inverse=inverse,
        )
    return QFT(num_qubits=num_qubits, inverse=inverse, do_swaps=False)


def _parse_input(invocation_input: Any) -> dict[str, Any]:
    if invocation_input is None:
        raise ValueError('Cần input JSON, ví dụ {"pair_i": 0, "pair_j": 1}.')
    if isinstance(invocation_input, dict):
        return invocation_input
    if isinstance(invocation_input, str):
        try:
            parsed = json.loads(invocation_input)
        except json.JSONDecodeError as exc:
            raise ValueError("Raw input phải là JSON hợp lệ.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Raw input JSON phải là một object.")
        return parsed
    raise TypeError("invocation_input phải là dict hoặc chuỗi JSON.")


def _validate_problem() -> None:
    if not np.allclose(A_MATRIX, A_MATRIX.conj().T, atol=1e-10):
        raise ValueError("A phải là Hermitian.")
    eigenvalues = np.linalg.eigvalsh(A_MATRIX)
    if eigenvalues.min() <= 0:
        raise ValueError("A phải xác định dương.")
    if eigenvalues.max() >= 2.0 * np.pi:
        raise ValueError("Phổ của A phải nằm trong (0, 2π) với U=exp(iA).")
    if np.linalg.norm(B_VECTOR) == 0:
        raise ValueError("b không được là vector 0.")


def _validate_pair(pair_i: Any, pair_j: Any) -> tuple[int, int]:
    if isinstance(pair_i, bool) or isinstance(pair_j, bool):
        raise ValueError("pair_i và pair_j phải là số nguyên.")
    try:
        i = int(pair_i)
        j = int(pair_j)
    except (TypeError, ValueError) as exc:
        raise ValueError("pair_i và pair_j phải là số nguyên.") from exc

    dimension = 2**TARGET_QUBITS
    if i == j:
        raise ValueError("pair_i và pair_j phải khác nhau.")
    if not (0 <= i < dimension and 0 <= j < dimension):
        raise ValueError(f"Chỉ số cặp phải nằm trong [0, {dimension - 1}].")
    return i, j


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

    phase_register = QuantumRegister(PHASE_QUBITS, "phase")
    target_register = QuantumRegister(TARGET_QUBITS, "target")
    ancilla_register = QuantumRegister(ANCILLA_QUBITS, "ancilla")
    circuit = QuantumCircuit(
        phase_register,
        target_register,
        ancilla_register,
        name="hhl-sign-base",
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


def _pair_to_front_permutation(dimension: int, i: int, j: int) -> np.ndarray:
    mapping_old_to_new: dict[int, int] = {i: 0, j: 1}
    remaining_new = [index for index in range(dimension) if index not in (0, 1)]

    for old_index in range(dimension):
        if old_index not in mapping_old_to_new:
            mapping_old_to_new[old_index] = remaining_new.pop(0)

    permutation = np.zeros((dimension, dimension), dtype=np.complex128)
    for old_index, new_index in mapping_old_to_new.items():
        permutation[new_index, old_index] = 1.0
    return permutation


def _build_sign_circuit(pair_i: int, pair_j: int) -> QuantumCircuit:
    circuit = _build_hhl_without_measurement()
    sign_extra = QuantumRegister(1, "sign_extra")
    circuit.add_register(sign_extra)

    if not (pair_i == 0 and pair_j == 1):
        permutation = _pair_to_front_permutation(
            2**TARGET_QUBITS,
            pair_i,
            pair_j,
        )
        circuit.unitary(
            permutation,
            TARGET_INDICES,
            label=f"P({pair_i},{pair_j})->(0,1)",
        )

    circuit.swap(TARGET_INDICES[0], SIGN_EXTRA_INDEX)
    circuit.h(SIGN_EXTRA_INDEX)

    classical = ClassicalRegister(circuit.num_qubits, "meas")
    circuit.add_register(classical)
    circuit.measure(
        list(range(circuit.num_qubits)),
        list(range(circuit.num_qubits)),
    )
    circuit.name = f"hhl-sign-{pair_i}-{pair_j}"
    return circuit


def processing(invocation_input: Any) -> QuantumCircuit:
    payload = _parse_input(invocation_input)
    pair_i, pair_j = _validate_pair(payload.get("pair_i"), payload.get("pair_j"))
    return _build_sign_circuit(pair_i, pair_j)


def post_processing(job_result: Any) -> dict[str, Any]:
    """Return counts. The local controller attaches the submitted pair metadata."""
    counts = job_result.get_counts()
    if isinstance(counts, list):
        if len(counts) != 1:
            raise ValueError("Function này chỉ hỗ trợ đúng một circuit mỗi job.")
        counts = counts[0]

    clean_counts = {str(key): int(value) for key, value in counts.items()}
    shots = int(sum(clean_counts.values()))

    return {
        "schema_version": "1.0",
        "circuit_type": "hhl-sign",
        "counts": clean_counts,
        "shots": shots,
        "qubit_layout": {
            "total_qubits": SIGN_TOTAL_QUBITS,
            "phase_indices": PHASE_INDICES,
            "target_indices": TARGET_INDICES,
            "ancilla_index": ANCILLA_INDEX,
            "sign_extra_index": SIGN_EXTRA_INDEX,
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
