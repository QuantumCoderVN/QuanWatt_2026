"""Single-file Quapp PennyLane HHL function.

This file consolidates circuit construction, input validation, Quapp entry
points, probability post-processing, and the optional three-job signed-amplitude
reconstruction helper.

Quapp entry points:
    processing(invocation_input)
    post_processing(job_result)

Supported invocation modes:
    main       - HHL solution probabilities
    sign_even  - interference for pairs (x0,x1), (x2,x3), ...
    sign_odd   - interference for pairs (x1,x2), (x3,x4), ..., (xN-1,x0)

The HHL implementation follows the uploaded notebook. The sign_odd workflow
uses a cyclic permutation after HHL instead of swapping columns of A, because a
column swap generally destroys Hermiticity and makes exp(iAt) non-unitary.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pennylane as qml


ALLOWED_MODES = {"main", "sign_even", "sign_odd"}


# -----------------------------------------------------------------------------
# Input validation and problem preparation
# -----------------------------------------------------------------------------


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _coerce_json_input(invocation_input: Any) -> Dict[str, Any]:
    if invocation_input is None:
        return {}
    if not isinstance(invocation_input, dict):
        raise TypeError("Quapp input must be a JSON object.")
    return dict(invocation_input)


def validate_and_prepare(invocation_input: Any) -> Dict[str, Any]:
    """Validate Quapp JSON input and construct the internal HHL problem.

    Defaults reproduce the 2 x 2 example from the source notebook. The notebook
    performs time evolution with -A, so ``negate_matrix`` defaults to True.
    """

    payload = _coerce_json_input(invocation_input)

    matrix = np.asarray(
        payload.get("matrix", [[-5.0, 0.5], [0.5, -4.0]]),
        dtype=np.complex128,
    )
    vector = np.asarray(
        payload.get("vector", [0.1, -0.2]),
        dtype=np.complex128,
    )

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array.")

    dimension = int(matrix.shape[0])
    if not _is_power_of_two(dimension):
        raise ValueError("matrix dimension must be a power of two.")

    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise ValueError("vector length must equal the matrix dimension.")

    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise ValueError("matrix entries must be finite.")
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise ValueError("vector entries must be finite.")

    hermitian_tolerance = float(payload.get("hermitian_tolerance", 1.0e-9))
    if not np.allclose(matrix, matrix.conj().T, atol=hermitian_tolerance, rtol=0.0):
        raise ValueError("matrix must be Hermitian for this HHL implementation.")

    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 0.0:
        raise ValueError("vector must be nonzero.")
    normalized_vector = vector / vector_norm

    phase_qubits = int(payload.get("phase_qubits", 3))
    if phase_qubits < 1 or phase_qubits > 8:
        raise ValueError(
            "phase_qubits must be between 1 and 8. The inverse-eigenvalue "
            "lookup contains 2**phase_qubits - 1 controlled rotations."
        )

    evolution_time = float(payload.get("evolution_time", 1.0))
    if not math.isfinite(evolution_time) or evolution_time <= 0.0:
        raise ValueError("evolution_time must be a positive finite number.")

    negate_matrix = bool(payload.get("negate_matrix", True))
    effective_matrix = -matrix if negate_matrix else matrix

    eigenvalues = np.linalg.eigvalsh(effective_matrix)
    positivity_tolerance = float(payload.get("positivity_tolerance", 1.0e-10))
    if np.min(eigenvalues) <= positivity_tolerance:
        raise ValueError(
            "The effective matrix must be positive definite for this notebook's "
            "positive-eigenvalue inverse rotation. Change negate_matrix or "
            "rescale/shift A."
        )

    strict_phase_range = bool(payload.get("strict_phase_range", True))
    maximum_phase = float(np.max(eigenvalues) * evolution_time)
    if strict_phase_range and maximum_phase >= 2.0 * math.pi:
        raise ValueError(
            "Eigenphases wrap because max(eigenvalue) * evolution_time >= 2*pi. "
            "Reduce evolution_time or deliberately set strict_phase_range=false."
        )

    mode = str(payload.get("mode", "main")).strip().lower()
    if mode not in ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)}.")

    system_qubits = int(round(math.log2(dimension)))
    phase_wires = list(range(phase_qubits))
    target_wires = list(range(phase_qubits, phase_qubits + system_qubits))
    hhl_ancilla_wire = phase_qubits + system_qubits
    extra_wire = hhl_ancilla_wire + 1

    return {
        "matrix": matrix,
        "effective_matrix": effective_matrix,
        "vector": vector,
        "normalized_vector": normalized_vector,
        "vector_norm": vector_norm,
        "dimension": dimension,
        "system_qubits": system_qubits,
        "phase_qubits": phase_qubits,
        "evolution_time": evolution_time,
        "mode": mode,
        "phase_wires": phase_wires,
        "target_wires": target_wires,
        "hhl_ancilla_wire": hhl_ancilla_wire,
        "extra_wire": extra_wire,
        "output_wires": target_wires + [hhl_ancilla_wire, extra_wire],
        "total_wires": extra_wire + 1,
        "eigenvalues": eigenvalues,
        "negate_matrix": negate_matrix,
    }


# -----------------------------------------------------------------------------
# HHL circuit construction
# -----------------------------------------------------------------------------


def hermitian_time_evolution_unitary(
    effective_matrix: np.ndarray,
    evolution_time: float,
) -> np.ndarray:
    """Construct exp(i * effective_matrix * evolution_time)."""

    eigenvalues, eigenvectors = np.linalg.eigh(effective_matrix)
    phases = np.exp(1j * eigenvalues * evolution_time)
    unitary = (eigenvectors * phases) @ eigenvectors.conj().T

    identity = np.eye(effective_matrix.shape[0], dtype=np.complex128)
    if not np.allclose(unitary.conj().T @ unitary, identity, atol=1.0e-9):
        raise ValueError("Constructed time-evolution matrix is not unitary.")
    return unitary


def apply_qpe(
    unitary: np.ndarray,
    phase_wires: Sequence[int],
    target_wires: Sequence[int],
) -> None:
    """Apply the source notebook's quantum phase-estimation ordering."""

    for wire in phase_wires:
        qml.Hadamard(wires=wire)

    # Powers 1, 2, 4, ... are assigned to reversed phase-wire order, matching
    # the source notebook.
    for index, control_wire in enumerate(reversed(phase_wires)):
        unitary_power = np.linalg.matrix_power(unitary, 2**index)
        qml.ctrl(qml.QubitUnitary, control=control_wire)(
            unitary_power,
            wires=list(target_wires),
        )

    qml.adjoint(qml.QFT)(wires=list(phase_wires))


def apply_inverse_eigenvalue_rotation(
    control_wires: Sequence[int],
    target_wire: int,
) -> None:
    """Apply the notebook's lookup rotation for inverse encoded eigenvalues.

    For encoded integer d in {1, ..., 2**m - 1}, the circuit uses
    cos(theta_d / 2) = 1/d and postselects the HHL ancilla in |0>.
    """

    number_of_controls = len(control_wires)
    for decimal_value in range(1, 2**number_of_controls):
        control_values = [
            int(bit) for bit in f"{decimal_value:0{number_of_controls}b}"
        ]
        angle = 2.0 * math.acos(1.0 / float(decimal_value))
        qml.ctrl(
            qml.RY,
            control=list(control_wires),
            control_values=control_values,
        )(angle, wires=target_wire)


def cyclic_shift_down_unitary(dimension: int) -> np.ndarray:
    """Return the permutation |k> -> |k-1 mod dimension>."""

    permutation = np.zeros((dimension, dimension), dtype=np.complex128)
    for source in range(dimension):
        destination = (source - 1) % dimension
        permutation[destination, source] = 1.0
    return permutation


def apply_hhl_block(problem: Dict[str, Any]) -> None:
    """Prepare the HHL state before optional sign-interference operations."""

    qml.StatePrep(problem["normalized_vector"], wires=problem["target_wires"])

    unitary = hermitian_time_evolution_unitary(
        problem["effective_matrix"],
        problem["evolution_time"],
    )

    apply_qpe(unitary, problem["phase_wires"], problem["target_wires"])
    apply_inverse_eigenvalue_rotation(
        problem["phase_wires"],
        problem["hhl_ancilla_wire"],
    )
    qml.adjoint(apply_qpe)(
        unitary,
        problem["phase_wires"],
        problem["target_wires"],
    )


def build_quapp_circuit(problem: Dict[str, Any]):
    """Return an unexecuted PennyLane circuit function for Quapp."""

    def hhl_circuit():
        apply_hhl_block(problem)

        if problem["mode"] == "sign_odd":
            qml.QubitUnitary(
                cyclic_shift_down_unitary(problem["dimension"]),
                wires=problem["target_wires"],
            )

        if problem["mode"] in {"sign_even", "sign_odd"}:
            # Interfere adjacent amplitudes by moving the least-significant
            # solution qubit into the extra wire and applying a Hadamard.
            qml.SWAP(wires=[problem["target_wires"][-1], problem["extra_wire"]])
            qml.Hadamard(wires=problem["extra_wire"])

        # The extra wire is always measured. In main mode it remains |0>, which
        # keeps post_processing independent of the original invocation input.
        return qml.probs(wires=problem["output_wires"])

    return hhl_circuit


# -----------------------------------------------------------------------------
# Quapp result conversion
# -----------------------------------------------------------------------------


def _to_probability_vector(job_result: Any) -> np.ndarray:
    """Extract a normalized one-dimensional probability vector."""

    candidate = job_result

    if isinstance(candidate, dict):
        for key in ("probabilities", "probs", "result", "data"):
            if key in candidate:
                candidate = candidate[key]
                break

    if hasattr(candidate, "tolist"):
        candidate = candidate.tolist()

    if isinstance(candidate, (tuple, list)) and len(candidate) == 1:
        first = candidate[0]
        if isinstance(first, (tuple, list, np.ndarray)) or hasattr(first, "tolist"):
            candidate = first.tolist() if hasattr(first, "tolist") else first

    array = np.asarray(candidate, dtype=float)
    if array.ndim != 1:
        raise TypeError(
            "Expected one probability vector from qml.probs; "
            f"received an object with shape {array.shape}."
        )
    if array.size == 0:
        raise ValueError("The backend returned an empty probability vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError("The backend returned non-finite probabilities.")

    array = np.where(array < 0.0, np.maximum(array, -1.0e-12), array)
    array = np.clip(array, 0.0, None)

    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError("The returned probability vector has zero total weight.")
    return array / total


def transform_probability_result(job_result: Any) -> Dict[str, Any]:
    """Transform probabilities measured as [target..., hhl_ancilla, extra]."""

    probabilities = _to_probability_vector(job_result)
    width = int(round(math.log2(probabilities.size)))
    if 2**width != probabilities.size or width < 3:
        raise ValueError(
            "Probability-vector length must be 2**(system_qubits + 2)."
        )

    system_qubits = width - 2
    dimension = 2**system_qubits
    labels = [f"{index:0{width}b}" for index in range(2**width)]

    # extra is the least-significant measured bit and the HHL ancilla is the
    # second least-significant measured bit.
    ancilla_zero_indices = [
        index for index in range(probabilities.size) if ((index >> 1) & 1) == 0
    ]
    ancilla_zero_joint = probabilities[ancilla_zero_indices]
    success_probability = float(np.sum(ancilla_zero_joint))

    target_unnormalized = np.zeros(dimension, dtype=float)
    for target_index in range(dimension):
        for extra_bit in (0, 1):
            full_index = (target_index << 2) | extra_bit
            target_unnormalized[target_index] += probabilities[full_index]

    target_conditional = (
        target_unnormalized / success_probability
        if success_probability > 0.0
        else np.zeros_like(target_unnormalized)
    )

    return {
        "measurement_order": [
            f"target_{index}" for index in range(system_qubits)
        ]
        + ["hhl_ancilla", "extra"],
        "joint_probabilities": {
            label: float(probability)
            for label, probability in zip(labels, probabilities)
            if probability > 0.0
        },
        "joint_probability_vector": probabilities.tolist(),
        "ancilla_zero_success_probability": success_probability,
        "ancilla_zero_joint_probabilities": ancilla_zero_joint.tolist(),
        "target_probabilities_ancilla_zero_unnormalized": (
            target_unnormalized.tolist()
        ),
        "target_probabilities_given_ancilla_zero": target_conditional.tolist(),
        "target_basis_labels": [
            f"{index:0{system_qubits}b}" for index in range(dimension)
        ],
        "notes": (
            "For mode=main, target_probabilities_given_ancilla_zero is the "
            "normalized HHL solution probability distribution. For signed real "
            "amplitudes, invoke main, sign_even, and sign_odd separately and "
            "combine the outputs with recover_signed_solution()."
        ),
    }


# -----------------------------------------------------------------------------
# Optional local helper: recover real signs from three Quapp job outputs
# -----------------------------------------------------------------------------


def _read_vector(output: Dict[str, Any], key: str) -> np.ndarray:
    if not isinstance(output, dict) or key not in output:
        raise ValueError(f"Missing '{key}' in Quapp output.")
    vector = np.asarray(output[key], dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"'{key}' must be one-dimensional.")
    return vector


def _pair_parities(
    original_probabilities: np.ndarray,
    interference_probabilities: np.ndarray,
    zero_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Infer pair-product signs from 2*p_plus - p_i - p_j."""

    dimension = original_probabilities.size
    if interference_probabilities.size != 2 * dimension:
        raise ValueError(
            "Sign-circuit ancilla-zero vector must have length 2*dimension."
        )

    plus_probabilities = interference_probabilities[::4] * 2.0
    if plus_probabilities.size != dimension // 2:
        raise ValueError("Unexpected sign-circuit probability layout.")

    parities: List[int] = []
    cross_terms: List[float] = []
    for pair_index, doubled_plus in enumerate(plus_probabilities):
        first = original_probabilities[2 * pair_index]
        second = original_probabilities[2 * pair_index + 1]
        cross = float(doubled_plus - first - second)
        cross_terms.append(cross)
        parities.append(1 if abs(cross) <= zero_tolerance or cross > 0.0 else -1)

    return np.asarray(parities, dtype=int), np.asarray(cross_terms, dtype=float)


def recover_signed_solution(
    main_output: Dict[str, Any],
    sign_even_output: Dict[str, Any],
    sign_odd_output: Dict[str, Any],
    first_sign: int = 1,
    zero_tolerance: float = 1.0e-3,
) -> Dict[str, Any]:
    """Recover normalized real amplitudes from three processed Quapp outputs."""

    original = _read_vector(
        main_output,
        "target_probabilities_ancilla_zero_unnormalized",
    )
    even_interference = _read_vector(
        sign_even_output,
        "ancilla_zero_joint_probabilities",
    )
    odd_interference = _read_vector(
        sign_odd_output,
        "ancilla_zero_joint_probabilities",
    )

    dimension = original.size
    if dimension == 0 or not _is_power_of_two(dimension):
        raise ValueError("The solution dimension must be a nonzero power of two.")

    original_success = float(np.sum(original))
    even_success = float(np.sum(even_interference))
    odd_success = float(np.sum(odd_interference))
    if min(original_success, even_success, odd_success) <= 0.0:
        raise ValueError(
            "At least one invocation has zero HHL-ancilla success probability."
        )

    original_conditional = original / original_success
    even_conditional = even_interference / even_success
    odd_conditional = odd_interference / odd_success

    even_parities, even_cross = _pair_parities(
        original_conditional,
        even_conditional,
        zero_tolerance,
    )

    shifted_original = np.roll(original_conditional, -1)
    odd_parities, odd_cross = _pair_parities(
        shifted_original,
        odd_conditional,
        zero_tolerance,
    )

    edge_parities: List[int] = []
    for pair_index in range(max(even_parities.size, odd_parities.size)):
        if pair_index < even_parities.size:
            edge_parities.append(int(even_parities[pair_index]))
        if pair_index < odd_parities.size:
            edge_parities.append(int(odd_parities[pair_index]))

    if len(edge_parities) != dimension:
        raise ValueError("Could not construct one parity edge per amplitude.")

    signs = np.ones(dimension, dtype=int)
    signs[0] = 1 if first_sign >= 0 else -1
    for index in range(1, dimension):
        signs[index] = signs[index - 1] * edge_parities[index - 1]

    raw_amplitudes = signs * np.sqrt(np.clip(original, 0.0, None))
    normalized_amplitudes = signs * np.sqrt(
        np.clip(original_conditional, 0.0, None)
    )
    cycle_product = int(np.prod(edge_parities))

    return {
        "signs": signs.tolist(),
        "normalized_amplitudes": normalized_amplitudes.tolist(),
        "raw_postselected_amplitudes": raw_amplitudes.tolist(),
        "even_pair_cross_terms": even_cross.tolist(),
        "odd_pair_cross_terms": odd_cross.tolist(),
        "cycle_consistent": bool(cycle_product == 1),
        "cycle_parity_product": cycle_product,
        "main_success_probability": original_success,
        "sign_even_success_probability": even_success,
        "sign_odd_success_probability": odd_success,
    }


# -----------------------------------------------------------------------------
# Required Quapp entry points
# -----------------------------------------------------------------------------


def processing(invocation_input):
    """Validate input and return an unexecuted PennyLane circuit function."""

    problem = validate_and_prepare(invocation_input)
    return build_quapp_circuit(problem)


def post_processing(job_result):
    """Convert the backend probability vector into JSON-serializable output."""

    return transform_probability_result(job_result)
