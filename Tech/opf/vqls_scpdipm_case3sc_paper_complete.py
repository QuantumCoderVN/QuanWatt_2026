"""VQLS cho ma tran SC-PDIPM KKT 16 x 16 da xu ly theo paper.

File nay KHONG tu lap lai KKT 8 x 8 va KHONG dung symmetric scaling DAD.
No lay truc tiep cac ma tran tu:

    scpdipm_kkt_case3sc_paper.py

Module tren da thuc hien:
    1. Lap KKT day du theo block cua phuong trinh (2).
    2. Left-ILU: A_pre=M^{-1}A_raw, b_pre=M^{-1}b_raw.
    3. Spectral scaling: ||A_MATRIX||_2=1.
    4. Chuan hoa B_VECTOR thanh |b>.

VQLS tra search direction:
    [dX, dZ, dlambda, dmu].

Vi day la LEFT preconditioning, an khong doi. Khong co buoc x=D y.

Dat hai file trong cung mot thu muc:
    scpdipm_kkt_case3sc_paper.py
    vqls_scpdipm_case3sc_paper_complete.py

Yeu cau:
    pip install numpy scipy matplotlib qiskit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from scpdipm_kkt_case3sc_paper import (
    A_MATRIX,
    A_PRECONDITIONED,
    A_RAW,
    B_PRECONDITIONED,
    B_RAW,
    B_VECTOR,
    B_VECTOR_NORM,
    B_VECTOR_RAW,
    N_QUBITS_INPUT,
    PREPARED as PAPER_PREPARED,
    VARIABLE_LABELS,
)


# =============================================================================
# CAU HINH VQLS
# =============================================================================

RNG_SEED = 7

# He 16 chieu can ansatz manh hon he 8 chieu truoc.
N_LAYERS = 10
N_RESTARTS = 4
MAX_ITERATIONS = 1500

PAULI_ATOL = 1e-12
PAULI_RTOL = 1e-12
PRINT_ALL_PAULI_TERMS = False

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_vqls_scpdipm_paper"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(
    precision=10,
    suppress=True,
    linewidth=200,
)


# =============================================================================
# CAC DOI TUONG DU LIEU
# =============================================================================


@dataclass(frozen=True)
class PreparedSystem:
    labels: tuple[str, ...]
    A_raw: np.ndarray
    b_raw: np.ndarray
    A_preconditioned: np.ndarray
    b_preconditioned: np.ndarray
    alpha: float
    A_vqls: np.ndarray
    b_vqls_raw: np.ndarray
    beta: float
    b_state: np.ndarray
    n_qubits: int


@dataclass(frozen=True)
class VQLSResult:
    parameters: np.ndarray
    state: np.ndarray
    final_cost: float
    iterations: int
    restart: int
    cost_history: list[float]


# =============================================================================
# 1. NAP VA KIEM TRA MA TRAN TU FILE XU LY THEO PAPER
# =============================================================================


def load_paper_matrix() -> PreparedSystem:
    """Nap dung A,b da left-ILU va spectral-scale tu module matrix."""

    system = PreparedSystem(
        labels=tuple(VARIABLE_LABELS),
        A_raw=np.asarray(A_RAW, dtype=complex),
        b_raw=np.asarray(B_RAW, dtype=complex),
        A_preconditioned=np.asarray(A_PRECONDITIONED, dtype=complex),
        b_preconditioned=np.asarray(B_PRECONDITIONED, dtype=complex),
        alpha=float(PAPER_PREPARED.alpha),
        A_vqls=np.asarray(A_MATRIX, dtype=complex),
        b_vqls_raw=np.asarray(B_VECTOR_RAW, dtype=complex),
        beta=float(B_VECTOR_NORM),
        b_state=np.asarray(B_VECTOR, dtype=complex),
        n_qubits=int(N_QUBITS_INPUT),
    )

    dimension = system.A_raw.shape[0]

    if system.A_raw.shape != (dimension, dimension):
        raise ValueError("A_RAW phai la ma tran vuong.")

    if dimension != 2**system.n_qubits:
        raise ValueError("Kich thuoc A_RAW khong khop N_QUBITS_INPUT.")

    expected_vector_shape = (dimension,)

    for name, vector in (
        ("B_RAW", system.b_raw),
        ("B_PRECONDITIONED", system.b_preconditioned),
        ("B_VECTOR_RAW", system.b_vqls_raw),
        ("B_VECTOR", system.b_state),
    ):
        if vector.shape != expected_vector_shape:
            raise ValueError(f"{name} co kich thuoc khong phu hop.")

    if not np.allclose(system.A_raw, system.A_raw.conj().T, atol=1e-10):
        raise ValueError("A_RAW truoc left-ILU phai Hermitian.")

    # Left-ILU tong quat lam A_vqls khong con Hermitian; day la binh thuong.
    if np.linalg.matrix_rank(system.A_vqls) < dimension:
        raise ValueError("A_MATRIX sau left-ILU bi suy bien.")

    if abs(np.linalg.norm(system.A_vqls, 2) - 1.0) > 1e-10:
        raise ValueError("A_MATRIX chua duoc spectral-scale ve norm 1.")

    if abs(np.linalg.norm(system.b_state) - 1.0) > 1e-10:
        raise ValueError("B_VECTOR chua duoc chuan hoa thanh quantum state.")

    if np.linalg.cond(system.A_preconditioned) >= np.linalg.cond(system.A_raw):
        raise ValueError("Left-ILU khong giam condition number.")

    return system


def print_matrix_summary(system: PreparedSystem) -> None:
    print("=" * 96)
    print("IMPORTED SC-PDIPM MATRIX - LEFT ILU AS IN PAPER")
    print("=" * 96)
    print("Search-direction order:", list(system.labels))
    print(f"Matrix dimension                       = {system.A_raw.shape}")
    print(f"State qubits                           = {system.n_qubits}")
    print(f"A_raw Hermitian                        = {np.allclose(system.A_raw, system.A_raw.conj().T)}")
    print(f"A_vqls Hermitian after left-ILU        = {np.allclose(system.A_vqls, system.A_vqls.conj().T)}")
    print(f"cond_2(A_raw)                          = {np.linalg.cond(system.A_raw):.10f}")
    print(f"cond_2(M^-1 A_raw)                     = {np.linalg.cond(system.A_preconditioned):.10f}")
    print(f"Spectral scale alpha                   = {system.alpha:.12f}")
    print(f"||A_vqls||_2                           = {np.linalg.norm(system.A_vqls, 2):.12f}")
    print(f"beta=||b_vqls_raw||                    = {system.beta:.12f}")
    print(f"||B_VECTOR||                           = {np.linalg.norm(system.b_state):.12f}")
    print(f"Nonzeros A_raw                         = {np.count_nonzero(np.abs(system.A_raw) > 1e-12)}")
    print(f"Nonzeros A_vqls                        = {np.count_nonzero(np.abs(system.A_vqls) > 1e-12)}")

    # Kiem tra cac cap scaling/left-preconditioning cung nghiem.
    raw_direction = np.linalg.solve(system.A_raw, system.b_raw)
    prepared_direction = np.linalg.solve(
        system.A_vqls,
        system.b_vqls_raw,
    )
    equivalence_error = np.linalg.norm(raw_direction - prepared_direction) / np.linalg.norm(raw_direction)
    print(f"Direction equivalence error            = {equivalence_error:.12e}")


# =============================================================================
# 2. PHAN RA PAULI / LCU
# =============================================================================


def pauli_decompose_matrix(
    matrix: np.ndarray,
    atol: float = PAULI_ATOL,
    rtol: float = PAULI_RTOL,
) -> SparsePauliOp:
    dimension = matrix.shape[0]
    n_qubits = int(np.log2(dimension))

    operator = Operator(
        matrix,
        input_dims=(2,) * n_qubits,
        output_dims=(2,) * n_qubits,
    )

    return SparsePauliOp.from_operator(
        operator,
        atol=atol,
        rtol=rtol,
    ).simplify(atol=atol)


def print_pauli_decomposition(
    matrix: np.ndarray,
    pauli_operator: SparsePauliOp,
) -> None:
    reconstructed = np.asarray(pauli_operator.to_matrix(), dtype=complex)
    reconstruction_error = np.linalg.norm(reconstructed - matrix)

    print("\n" + "=" * 96)
    print("PAULI / LCU DECOMPOSITION")
    print("=" * 96)
    print(f"Number of Pauli terms                  = {len(pauli_operator.coeffs)}")
    print(f"Reconstruction error                   = {reconstruction_error:.12e}")

    if reconstruction_error > 1e-10:
        raise RuntimeError("Pauli decomposition khong chinh xac.")

    terms = pauli_operator.to_list()

    if PRINT_ALL_PAULI_TERMS:
        terms_to_print = terms
    else:
        terms_to_print = terms[:20]

    for label, coefficient in terms_to_print:
        coefficient = complex(coefficient)
        print(
            f"  ({coefficient.real:+.10e}"
            f"{coefficient.imag:+.10e}j) * {label}"
        )

    if not PRINT_ALL_PAULI_TERMS and len(terms) > len(terms_to_print):
        print(
            f"  ... con {len(terms)-len(terms_to_print)} terms; "
            "dat PRINT_ALL_PAULI_TERMS=True de in tat ca."
        )


# =============================================================================
# 3. ANSATZ QISKIT 4-QUBIT
# =============================================================================


def build_ansatz(
    parameters: np.ndarray,
    n_qubits: int,
    n_layers: int = N_LAYERS,
) -> QuantumCircuit:
    """RY+CNOT ansatz cho search direction thuc 16 chieu."""

    parameters = np.asarray(parameters, dtype=float)
    expected = n_qubits * n_layers

    if parameters.shape != (expected,):
        raise ValueError(f"Ansatz can {expected} tham so.")

    circuit = QuantumCircuit(n_qubits, name="VQLS_ansatz")
    parameter_index = 0

    for layer in range(n_layers):
        for qubit in range(n_qubits):
            circuit.ry(parameters[parameter_index], qubit)
            parameter_index += 1

        if layer % 2 == 0:
            for control in range(n_qubits - 1):
                circuit.cx(control, control + 1)
        else:
            for control in range(n_qubits - 1, 0, -1):
                circuit.cx(control, control - 1)

    return circuit


def ansatz_state(parameters: np.ndarray, n_qubits: int) -> np.ndarray:
    circuit = build_ansatz(parameters, n_qubits)
    state = Statevector.from_instruction(circuit).data
    return np.asarray(state, dtype=complex)


def build_b_preparation_circuit(b_state: np.ndarray) -> QuantumCircuit:
    n_qubits = int(np.log2(b_state.size))
    circuit = QuantumCircuit(n_qubits, name="U_b")
    circuit.append(StatePreparation(b_state), range(n_qubits))
    return circuit


# =============================================================================
# 4. GLOBAL VQLS COST CHO MA TRAN KHONG HERMITIAN
# =============================================================================


def make_global_cost_function(system: PreparedSystem):
    """C_G=<x|A^dagger(I-|b><b|)A|x>/<x|A^dagger A|x>."""

    identity = np.eye(system.A_vqls.shape[0], dtype=complex)
    projector_b = np.outer(system.b_state, system.b_state.conj())
    A_dagger_A = system.A_vqls.conj().T @ system.A_vqls
    H_global = (
        system.A_vqls.conj().T
        @ (identity - projector_b)
        @ system.A_vqls
    )

    def global_cost(parameters: np.ndarray) -> float:
        state = ansatz_state(parameters, system.n_qubits)
        numerator = float(np.vdot(state, H_global @ state).real)
        denominator = float(np.vdot(state, A_dagger_A @ state).real)

        if denominator < 1e-14:
            return 1e6

        return max(numerator / denominator, 0.0)

    return global_cost


# =============================================================================
# 5. TOI UU VQLS
# =============================================================================


def optimize_vqls(system: PreparedSystem) -> VQLSResult:
    cost_function = make_global_cost_function(system)
    rng = np.random.default_rng(RNG_SEED)
    n_parameters = system.n_qubits * N_LAYERS

    best_result: OptimizeResult | None = None
    best_restart = -1
    best_history: list[float] = []

    print("\n" + "=" * 96)
    print("VQLS OPTIMIZATION")
    print("=" * 96)
    print(f"Ansatz layers                          = {N_LAYERS}")
    print(f"Number of parameters                   = {n_parameters}")
    print(f"Restarts                               = {N_RESTARTS}")
    print(f"Maximum iterations/restart             = {MAX_ITERATIONS}")

    for restart in range(N_RESTARTS):
        initial = rng.uniform(-np.pi, np.pi, size=n_parameters)
        history: list[float] = []

        def logged_cost(parameters: np.ndarray) -> float:
            value = cost_function(parameters)
            history.append(value)
            return value

        result = minimize(
            logged_cost,
            initial,
            method="L-BFGS-B",
            options={
                "maxiter": MAX_ITERATIONS,
                "ftol": 1e-14,
                "gtol": 1e-10,
                "maxls": 50,
            },
        )

        print(
            f"Restart {restart}: cost={result.fun:.12e}, "
            f"iterations={result.nit}, success={result.success}"
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_restart = restart
            best_history = history

    if best_result is None:
        raise RuntimeError("VQLS optimizer khong tra ket qua.")

    final_state = ansatz_state(best_result.x, system.n_qubits)
    final_state /= np.linalg.norm(final_state)

    return VQLSResult(
        parameters=np.asarray(best_result.x, dtype=float),
        state=final_state,
        final_cost=float(best_result.fun),
        iterations=int(best_result.nit),
        restart=best_restart,
        cost_history=best_history,
    )


# =============================================================================
# 6. RECOVERY SEARCH DIRECTION - LEFT PRECONDITIONING KHONG DOI AN
# =============================================================================


def relative_error(approximation: np.ndarray, reference: np.ndarray) -> float:
    norm_reference = float(np.linalg.norm(reference))

    if norm_reference < 1e-14:
        return float(np.linalg.norm(approximation - reference))

    return float(np.linalg.norm(approximation - reference) / norm_reference)


def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    norm_b = float(np.linalg.norm(b))

    if norm_b < 1e-14:
        return float(np.linalg.norm(A @ x - b))

    return float(np.linalg.norm(A @ x - b) / norm_b)


def recover_and_report(
    system: PreparedSystem,
    result: VQLSResult,
) -> dict[str, np.ndarray | float | complex]:
    """Phuc hoi search direction d=[dX,dZ,dlambda,dmu]."""

    state = result.state / np.linalg.norm(result.state)

    # A_vqls q = |b>. Tim he so k phuc hoi do lon/pha cua q.
    mapped_state = system.A_vqls @ state
    denominator = np.vdot(mapped_state, mapped_state)

    if abs(denominator) < 1e-14:
        raise RuntimeError("||A_vqls|x(theta)>|| gan bang 0.")

    k_coefficient = np.vdot(mapped_state, system.b_state) / denominator
    direction_normalized_rhs = k_coefficient * state

    # b_vqls_raw=beta*|b>, nen direction=beta*q.
    search_direction = system.beta * direction_normalized_rhs

    # Left preconditioning khong doi an: KHONG NHAN D_SCALE.
    normalized_classical = np.linalg.solve(
        system.A_vqls,
        system.b_state,
    )
    prepared_classical = np.linalg.solve(
        system.A_vqls,
        system.b_vqls_raw,
    )
    raw_classical = np.linalg.solve(
        system.A_raw,
        system.b_raw,
    )

    reference_state = normalized_classical / np.linalg.norm(normalized_classical)
    state_fidelity = float(abs(np.vdot(reference_state, state)) ** 2)

    normalized_error = relative_error(
        direction_normalized_rhs,
        normalized_classical,
    )
    normalized_residual = relative_residual(
        system.A_vqls,
        direction_normalized_rhs,
        system.b_state,
    )
    prepared_error = relative_error(search_direction, prepared_classical)
    prepared_residual = relative_residual(
        system.A_vqls,
        search_direction,
        system.b_vqls_raw,
    )
    raw_error = relative_error(search_direction, raw_classical)
    raw_residual = relative_residual(
        system.A_raw,
        search_direction,
        system.b_raw,
    )

    print("\n" + "=" * 96)
    print("SC-PDIPM SEARCH-DIRECTION RECOVERY")
    print("=" * 96)
    print(f"Best restart                           = {result.restart}")
    print(f"Optimizer iterations                   = {result.iterations}")
    print(f"Final global cost                      = {result.final_cost:.12e}")
    print(f"Recovery coefficient k                 = {k_coefficient}")
    print(f"State fidelity                         = {state_fidelity:.12f}")

    print("\nRecovered VQLS search direction:")
    for label, value in zip(system.labels, search_direction, strict=True):
        value = complex(value)
        if abs(value.imag) < 1e-9:
            print(f"  {label:>12s} = {value.real: .12f}")
        else:
            print(
                f"  {label:>12s} = "
                f"{value.real: .12f}{value.imag:+.12f}j"
            )

    print("\nClassical search direction from raw KKT:")
    for label, value in zip(system.labels, raw_classical, strict=True):
        print(f"  {label:>12s} = {complex(value).real: .12f}")

    print("\nAccuracy at normalized VQLS system:")
    print(f"  Relative solution error              = {normalized_error:.12e}")
    print(f"  Relative residual                    = {normalized_residual:.12e}")

    print("\nAccuracy after left-ILU/scaling:")
    print(f"  Relative solution error              = {prepared_error:.12e}")
    print(f"  Relative residual                    = {prepared_residual:.12e}")

    print("\nAccuracy in original SC-PDIPM KKT:")
    print(f"  Relative direction error             = {raw_error:.12e}")
    print(f"  Relative residual                    = {raw_residual:.12e}")

    return {
        "state": state,
        "reference_state": reference_state,
        "direction_normalized_rhs": direction_normalized_rhs,
        "search_direction": search_direction,
        "raw_classical": raw_classical,
        "k_coefficient": k_coefficient,
        "state_fidelity": state_fidelity,
        "normalized_error": normalized_error,
        "normalized_residual": normalized_residual,
        "prepared_error": prepared_error,
        "prepared_residual": prepared_residual,
        "raw_error": raw_error,
        "raw_residual": raw_residual,
    }


# =============================================================================
# 7. VE KET QUA
# =============================================================================


def plot_results(
    system: PreparedSystem,
    result: VQLSResult,
    recovered: dict[str, np.ndarray | float | complex],
) -> Path:
    quantum_state = np.asarray(recovered["state"], dtype=complex)
    reference_state = np.asarray(recovered["reference_state"], dtype=complex)
    search_direction = np.asarray(recovered["search_direction"], dtype=complex)
    raw_classical = np.asarray(recovered["raw_classical"], dtype=complex)

    quantum_probabilities = np.abs(quantum_state) ** 2
    classical_probabilities = np.abs(reference_state) ** 2

    figure, axes = plt.subplots(1, 3, figsize=(20, 5.2))

    history = np.maximum(np.asarray(result.cost_history, dtype=float), 1e-16)
    axes[0].plot(history, color="seagreen", linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Function evaluation")
    axes[0].set_ylabel("Global VQLS cost")
    axes[0].set_title("Cost convergence")
    axes[0].grid(True, which="both", alpha=0.3)

    indices = np.arange(quantum_probabilities.size)
    width = 0.38
    probability_floor = 1e-14

    axes[1].bar(
        indices - width / 2,
        np.maximum(classical_probabilities, probability_floor),
        width,
        label="Classical",
        color="steelblue",
    )
    axes[1].bar(
        indices + width / 2,
        np.maximum(quantum_probabilities, probability_floor),
        width,
        label="VQLS",
        color="darkorange",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-14, 2.0)
    axes[1].set_xlabel("Basis index")
    axes[1].set_ylabel("Probability - log scale")
    axes[1].set_title("Normalized search-direction state")
    axes[1].legend()
    axes[1].grid(True, which="both", axis="y", alpha=0.3)

    axes[2].bar(
        indices - width / 2,
        np.real(raw_classical),
        width,
        label="Classical",
        color="steelblue",
    )
    axes[2].bar(
        indices + width / 2,
        np.real(search_direction),
        width,
        label="VQLS",
        color="darkorange",
    )
    axes[2].set_yscale("symlog", linthresh=1e-2)
    axes[2].set_xticks(indices)
    axes[2].set_xticklabels(system.labels, rotation=70, ha="right", fontsize=8)
    axes[2].set_ylabel("Search-direction component - symlog")
    axes[2].set_title("Recovered SC-PDIPM direction")
    axes[2].legend()
    axes[2].grid(True, which="both", axis="y", alpha=0.3)

    figure.tight_layout()
    output_path = OUTPUT_DIR / "vqls_scpdipm_case3sc_paper_results.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


# =============================================================================
# 8. MAIN
# =============================================================================


def main() -> None:
    np.random.seed(RNG_SEED)

    # Chi nap ma tran tu scpdipm_kkt_case3sc_paper.py.
    # Khong lap lai KKT va khong precondition lan thu hai.
    system = load_paper_matrix()
    print_matrix_summary(system)

    pauli_operator = pauli_decompose_matrix(system.A_vqls)
    print_pauli_decomposition(system.A_vqls, pauli_operator)

    b_circuit = build_b_preparation_circuit(system.b_state)
    prepared_b = Statevector.from_instruction(b_circuit).data
    b_preparation_error = np.linalg.norm(prepared_b - system.b_state)
    print(f"\nStatePreparation error                  = {b_preparation_error:.12e}")

    if b_preparation_error > 1e-10:
        raise RuntimeError("Mach U_b khong tao dung |b>.")

    result = optimize_vqls(system)
    recovered = recover_and_report(system, result)
    plot_path = plot_results(system, result, recovered)

    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
