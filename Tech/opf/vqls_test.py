"""VQLS cho he KKT DC-QOPF MATPOWER case3sc (3 bus).

Pipeline duy nhat trong file:

    du lieu case3sc
        -> KKT A_raw x = b_raw
        -> symmetric equilibration: A_pre y = b_pre, x = D y
        -> spectral scaling: A_vqls y = b_vqls_raw, ||A_vqls||_2 = 1
        -> amplitude encoding: |b> = b_vqls_raw / ||b_vqls_raw||
        -> Pauli/LCU decomposition
        -> VQLS global cost tren Qiskit Statevector
        -> phuc hoi do lon va hoan nguyen x = D y.

Day la simulator nghien cuu cho ma tran nho. Tren quantum hardware, phan global
cost can duoc thay bang cac phep do Pauli/Hadamard-test tuong ung.

Yeu cau:
    pip install numpy scipy matplotlib qiskit
"""
from load_case3 import (
    A_RAW,
    B_RAW,
    A_MATRIX,
    B_VECTOR_RAW,
    B_VECTOR_NORM,
    B_VECTOR,
    N_QUBITS_INPUT,
    VARIABLE_LABELS,
)
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


# =============================================================================
# CAU HINH
# =============================================================================

RNG_SEED = 7
N_LAYERS = 6
N_RESTARTS = 4
MAX_ITERATIONS = 700

PAULI_ATOL = 1e-12
PAULI_RTOL = 1e-12

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_vqls_qopf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(
    precision=10,
    suppress=True,
    linewidth=180,
)


# =============================================================================
# CAC DOI TUONG DU LIEU
# =============================================================================


@dataclass(frozen=True)
class PreparedSystem:
    labels: tuple[str, ...]
    A_raw: np.ndarray
    b_raw: np.ndarray
    D: np.ndarray
    A_pre: np.ndarray
    b_pre: np.ndarray
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
# 1. LAP HE KKT DC-OPF TU DU LIEU MATPOWER CASE3SC
# =============================================================================


def build_case3sc_dc_kkt() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Lap he KKT equality-constrained DC-OPF case3sc.

    Slack bus: bus 1, theta_1 = 0.

    Thu tu bien toi uu:
        z = [theta_2, theta_3, Pg_1, Pg_2, Pg_3]

    Thu tu an KKT:
        [theta_2, theta_3, Pg_1, Pg_2, Pg_3,
         lambda_1, lambda_2, lambda_3]
    """

    base_mva = 100.0
    loads_mw = np.array([110.0, 110.0, 95.0])

    # (from_bus, to_bus, reactance_pu)
    branches = (
        (1, 3, 0.62),
        (3, 2, 0.75),
        (1, 2, 0.90),
    )

    # f(Pg) = c2 Pg^2 + c1 Pg + c0
    cost_c2 = np.array([0.11, 0.085, 0.0])
    cost_c1 = np.array([5.0, 1.2, 0.0])

    n_bus = 3
    n_gen = 3

    # Bbus theo MW/rad. Moi nhanh dong gop baseMVA/x vao Laplacian.
    Bbus = np.zeros((n_bus, n_bus), dtype=float)

    for from_bus, to_bus, reactance in branches:
        i = from_bus - 1
        j = to_bus - 1
        bij = base_mva / reactance

        Bbus[i, i] += bij
        Bbus[j, j] += bij
        Bbus[i, j] -= bij
        Bbus[j, i] -= bij

    # Bo cot theta_1 vi bus 1 la slack bus.
    B_reduced = Bbus[:, 1:]

    # Moi bus co mot may phat trong case nay.
    Cg = np.eye(n_bus, n_gen)

    # B_reduced theta - Cg Pg = -Pd.
    Aeq = np.hstack((B_reduced, -Cg))

    n_angles = 2
    n_primal = n_angles + n_gen

    H = np.zeros((n_primal, n_primal), dtype=float)
    H[n_angles:, n_angles:] = np.diag(2.0 * cost_c2)

    linear_gradient = np.concatenate((np.zeros(n_angles), cost_c1))

    A_raw = np.block(
        [
            [H, Aeq.T],
            [Aeq, np.zeros((n_bus, n_bus))],
        ]
    )

    b_raw = np.concatenate((-linear_gradient, -loads_mw))

    labels = (
        "theta_2",
        "theta_3",
        "Pg_1",
        "Pg_2",
        "Pg_3",
        "lambda_1",
        "lambda_2",
        "lambda_3",
    )

    return A_raw.astype(complex), b_raw.astype(complex), labels


# =============================================================================
# 2. TIEN XU LY MA TRAN CHO VQLS
# =============================================================================


def prepare_qopf_matrix_for_vqls(
    A_raw: np.ndarray,
    b_raw: np.ndarray,
    labels: tuple[str, ...],
    eps: float = 1e-14,
) -> PreparedSystem:
    """Can bang doi xung, chuan pho va chuan hoa RHS.

    He goc:
        A_raw x = b_raw

    Dat x = D y va nhan D ben trai:
        (D A_raw D) y = D b_raw.

    Sau do chia ca hai ve cho alpha = ||D A_raw D||_2.
    """

    A_raw = np.asarray(A_raw, dtype=complex)
    b_raw = np.asarray(b_raw, dtype=complex)

    if A_raw.ndim != 2 or A_raw.shape[0] != A_raw.shape[1]:
        raise ValueError("A_raw phai la ma tran vuong.")

    dimension = A_raw.shape[0]

    if b_raw.shape != (dimension,):
        raise ValueError("Kich thuoc b_raw khong phu hop voi A_raw.")

    n_qubits = int(np.log2(dimension))

    if 2**n_qubits != dimension:
        raise ValueError("Kich thuoc A_raw phai la 2^n.")

    if not np.allclose(A_raw, A_raw.conj().T, atol=1e-10):
        raise ValueError("KKT case3sc phai Hermitian.")

    if np.linalg.matrix_rank(A_raw) < dimension:
        raise ValueError("A_raw bi suy bien.")

    row_norms = np.linalg.norm(A_raw, axis=1)

    if np.any(row_norms < eps):
        raise ValueError("A_raw co hang gan bang 0.")

    D = np.diag(1.0 / np.sqrt(row_norms))
    A_pre = D @ A_raw @ D
    b_pre = D @ b_raw

    alpha = float(np.linalg.norm(A_pre, ord=2))

    if alpha < eps:
        raise ValueError("||A_pre||_2 gan bang 0.")

    # Quan trong: chia dong thoi ca A va b cho alpha.
    A_vqls = A_pre / alpha
    b_vqls_raw = b_pre / alpha

    beta = float(np.linalg.norm(b_vqls_raw))

    if beta < eps:
        raise ValueError("||b_vqls_raw|| gan bang 0.")

    b_state = b_vqls_raw / beta

    return PreparedSystem(
        labels=labels,
        A_raw=A_raw,
        b_raw=b_raw,
        D=D,
        A_pre=A_pre,
        b_pre=b_pre,
        alpha=alpha,
        A_vqls=A_vqls,
        b_vqls_raw=b_vqls_raw,
        beta=beta,
        b_state=b_state,
        n_qubits=n_qubits,
    )


def print_preprocessing_summary(system: PreparedSystem) -> None:
    print("=" * 88)
    print("QOPF MATRIX PREPROCESSING")
    print("=" * 88)
    print("Variable order:", list(system.labels))
    print(f"Dimension                       = {system.A_raw.shape[0]}")
    print(f"Number of state qubits          = {system.n_qubits}")
    print(f"Hermitian A_raw                 = {np.allclose(system.A_raw, system.A_raw.conj().T)}")
    print(f"Hermitian A_vqls                = {np.allclose(system.A_vqls, system.A_vqls.conj().T)}")
    print(f"cond_2(A_raw)                   = {np.linalg.cond(system.A_raw):.10f}")
    print(f"cond_2(A_pre)                   = {np.linalg.cond(system.A_pre):.10f}")
    print(f"||A_vqls||_2                    = {np.linalg.norm(system.A_vqls, 2):.10f}")
    print(f"alpha                           = {system.alpha:.12f}")
    print(f"beta = ||b_vqls_raw||           = {system.beta:.12f}")
    print(f"Nonzeros in A_raw               = {np.count_nonzero(np.abs(system.A_raw) > 1e-14)}")
    print(f"Nonzeros in A_vqls              = {np.count_nonzero(np.abs(system.A_vqls) > 1e-14)}")

    print("\nA_raw =")
    print(np.real_if_close(system.A_raw))

    print("\nb_raw =")
    print(np.real_if_close(system.b_raw))

    print("\nD =")
    print(np.real_if_close(system.D))

    print("\nA_vqls =")
    print(np.real_if_close(system.A_vqls))

    print("\nb_vqls_raw =")
    print(np.real_if_close(system.b_vqls_raw))

    print("\n|b> =")
    print(np.real_if_close(system.b_state))


# =============================================================================
# 3. PHAN RA PAULI/LCU
# =============================================================================


def pauli_decompose_matrix(
    matrix: np.ndarray,
    atol: float = PAULI_ATOL,
    rtol: float = PAULI_RTOL,
) -> SparsePauliOp:
    """Phan ra chinh xac A = sum_l c_l P_l, khong truncate."""

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
    A_vqls: np.ndarray,
    pauli_operator: SparsePauliOp,
) -> None:
    reconstructed = np.asarray(pauli_operator.to_matrix(), dtype=complex)
    error = np.linalg.norm(reconstructed - A_vqls)

    print("\n" + "=" * 88)
    print("PAULI / LCU DECOMPOSITION")
    print("=" * 88)
    print(f"Number of Pauli terms            = {len(pauli_operator.coeffs)}")
    print(f"Reconstruction error             = {error:.12e}")

    if error > 1e-10:
        raise RuntimeError("Pauli decomposition khong chinh xac.")

    for label, coefficient in pauli_operator.to_list():
        coefficient = complex(coefficient)

        if abs(coefficient.imag) < 1e-12:
            coefficient_text = f"{coefficient.real:+.12f}"
        else:
            coefficient_text = (
                f"{coefficient.real:+.12f}"
                f"{coefficient.imag:+.12f}j"
            )

        print(f"  {coefficient_text} * {label}")


# =============================================================================
# 4. ANSATZ QISKIT: RY + CNOT CHO NGHIEM THUC
# =============================================================================


def build_ansatz(
    parameters: np.ndarray,
    n_qubits: int,
    n_layers: int = N_LAYERS,
) -> QuantumCircuit:
    """Hardware-efficient real ansatz.

    KKT, b va nghiem cua case nay deu thuc, nen RZ khong can thiet. Moi layer
    co RY tren tat ca qubit va mot chuoi CNOT. Huong CNOT duoc dao xen ke.
    """

    parameters = np.asarray(parameters, dtype=float)
    expected = n_qubits * n_layers

    if parameters.shape != (expected,):
        raise ValueError(f"Ansatz can {expected} tham so.")

    circuit = QuantumCircuit(n_qubits, name="VQLS_ansatz")
    index = 0

    for layer in range(n_layers):
        for qubit in range(n_qubits):
            circuit.ry(parameters[index], qubit)
            index += 1

        if n_qubits > 1:
            if layer % 2 == 0:
                for control in range(n_qubits - 1):
                    circuit.cx(control, control + 1)
            else:
                for control in range(n_qubits - 1, 0, -1):
                    circuit.cx(control, control - 1)

    return circuit


def ansatz_state(parameters: np.ndarray, n_qubits: int) -> np.ndarray:
    circuit = build_ansatz(parameters, n_qubits)
    return np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)


def build_b_preparation_circuit(b_state: np.ndarray) -> QuantumCircuit:
    """Mach U_b sao cho U_b|0...0> = |b>."""

    dimension = b_state.size
    n_qubits = int(np.log2(dimension))
    circuit = QuantumCircuit(n_qubits, name="U_b")
    circuit.append(StatePreparation(b_state), range(n_qubits))
    return circuit


# =============================================================================
# 5. GLOBAL VQLS COST THEO EFFECTIVE HAMILTONIAN H_G
# =============================================================================


def build_global_cost_operators(
    A_vqls: np.ndarray,
    b_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Tao H_G va A^dagger A.

        H_G = A^dagger (I - |b><b|) A

        C_G = <x|H_G|x> / <x|A^dagger A|x>.
    """

    identity = np.eye(A_vqls.shape[0], dtype=complex)
    projector_b = np.outer(b_state, b_state.conj())
    A_dagger_A = A_vqls.conj().T @ A_vqls
    H_global = A_vqls.conj().T @ (identity - projector_b) @ A_vqls
    return H_global, A_dagger_A


def make_global_cost_function(
    system: PreparedSystem,
):
    H_global, A_dagger_A = build_global_cost_operators(
        system.A_vqls,
        system.b_state,
    )

    def global_cost(parameters: np.ndarray) -> float:
        state = ansatz_state(parameters, system.n_qubits)
        numerator = float(np.vdot(state, H_global @ state).real)
        denominator = float(np.vdot(state, A_dagger_A @ state).real)

        if denominator < 1e-14:
            return 1e6

        cost = numerator / denominator
        return max(float(np.real(cost)), 0.0)

    return global_cost


# =============================================================================
# 6. TOI UU VQLS
# =============================================================================


def optimize_vqls(system: PreparedSystem) -> VQLSResult:
    cost_function = make_global_cost_function(system)
    rng = np.random.default_rng(RNG_SEED)
    n_parameters = system.n_qubits * N_LAYERS

    best_result: OptimizeResult | None = None
    best_restart = -1
    best_history: list[float] = []

    print("\n" + "=" * 88)
    print("VQLS OPTIMIZATION")
    print("=" * 88)
    print(f"Ansatz layers                    = {N_LAYERS}")
    print(f"Number of parameters             = {n_parameters}")
    print(f"Restarts                         = {N_RESTARTS}")
    print(f"Maximum iterations/restart       = {MAX_ITERATIONS}")

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
                "maxls": 40,
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
    final_state = final_state / np.linalg.norm(final_state)

    return VQLSResult(
        parameters=np.asarray(best_result.x, dtype=float),
        state=final_state,
        final_cost=float(best_result.fun),
        iterations=int(best_result.nit),
        restart=best_restart,
        cost_history=best_history,
    )


# =============================================================================
# 7. PHUC HOI NGHIEM QOPF/KKT
# =============================================================================


def relative_error(approximation: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))

    if denominator < 1e-14:
        return float(np.linalg.norm(approximation - reference))

    return float(np.linalg.norm(approximation - reference) / denominator)


def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(b))

    if denominator < 1e-14:
        return float(np.linalg.norm(A @ x - b))

    return float(np.linalg.norm(A @ x - b) / denominator)


def recover_and_report(
    system: PreparedSystem,
    result: VQLSResult,
) -> dict[str, np.ndarray | float | complex]:
    """Phuc hoi ba cap nghiem: z -> y -> x."""

    x_state = result.state / np.linalg.norm(result.state)

    # Cap 1: A_vqls z = |b>. VQLS chi tra huong, nen tim he so k phuc hoi.
    b_prime = system.A_vqls @ x_state
    denominator = np.vdot(b_prime, b_prime)

    if abs(denominator) < 1e-14:
        raise RuntimeError("||A_vqls |x(theta)>|| gan bang 0.")

    k_coefficient = np.vdot(b_prime, system.b_state) / denominator
    z_normalized_rhs = k_coefficient * x_state

    # Cap 2: A_vqls y = b_vqls_raw.
    y_preconditioned = system.beta * z_normalized_rhs

    # Cap 3: A_raw x = b_raw, voi x = D y.
    x_qopf = system.D @ y_preconditioned

    # Cac nghiem co dien sau chi dung de kiem chung simulator.
    z_classical = np.linalg.solve(system.A_vqls, system.b_state)
    y_classical = np.linalg.solve(system.A_vqls, system.b_vqls_raw)
    x_classical = np.linalg.solve(system.A_raw, system.b_raw)

    z_classical_state = z_classical / np.linalg.norm(z_classical)
    state_fidelity = float(abs(np.vdot(z_classical_state, x_state)) ** 2)

    metrics = {
        "normalized_error": relative_error(z_normalized_rhs, z_classical),
        "normalized_residual": relative_residual(
            system.A_vqls,
            z_normalized_rhs,
            system.b_state,
        ),
        "preconditioned_error": relative_error(
            y_preconditioned,
            y_classical,
        ),
        "preconditioned_residual": relative_residual(
            system.A_vqls,
            y_preconditioned,
            system.b_vqls_raw,
        ),
        "qopf_error": relative_error(x_qopf, x_classical),
        "qopf_residual": relative_residual(
            system.A_raw,
            x_qopf,
            system.b_raw,
        ),
    }

    print("\n" + "=" * 88)
    print("VQLS SOLUTION RECOVERY")
    print("=" * 88)
    print(f"Best restart                     = {result.restart}")
    print(f"Optimizer iterations             = {result.iterations}")
    print(f"Final global cost                = {result.final_cost:.12e}")
    print(f"Recovery coefficient k           = {k_coefficient}")
    print(f"State fidelity                   = {state_fidelity:.12f}")

    print("\nRecovered original QOPF/KKT solution:")

    for label, value in zip(system.labels, x_qopf, strict=True):
        value = complex(value)

        if abs(value.imag) < 1e-9:
            print(f"  {label:>10s} = {value.real: .12f}")
        else:
            print(
                f"  {label:>10s} = "
                f"{value.real: .12f}{value.imag:+.12f}j"
            )

    print("\nClassical original QOPF/KKT solution:")

    for label, value in zip(system.labels, x_classical, strict=True):
        value = complex(value)
        print(f"  {label:>10s} = {value.real: .12f}")

    print("\nAccuracy at normalized VQLS system:")
    print(f"  Relative solution error        = {metrics['normalized_error']:.12e}")
    print(f"  Relative residual              = {metrics['normalized_residual']:.12e}")

    print("\nAccuracy at preconditioned system:")
    print(f"  Relative solution error        = {metrics['preconditioned_error']:.12e}")
    print(f"  Relative residual              = {metrics['preconditioned_residual']:.12e}")

    print("\nAccuracy at original QOPF/KKT system:")
    print(f"  Relative solution error        = {metrics['qopf_error']:.12e}")
    print(f"  Relative residual              = {metrics['qopf_residual']:.12e}")

    return {
        "x_state": x_state,
        "z_normalized_rhs": z_normalized_rhs,
        "y_preconditioned": y_preconditioned,
        "x_qopf": x_qopf,
        "z_classical_state": z_classical_state,
        "x_classical": x_classical,
        "k_coefficient": k_coefficient,
        "state_fidelity": state_fidelity,
        **metrics,
    }


# =============================================================================
# 8. VE KET QUA
# =============================================================================


def plot_results(
    result: VQLSResult,
    recovered: dict[str, np.ndarray | float | complex],
) -> Path:
    quantum_state = np.asarray(recovered["x_state"], dtype=complex)
    classical_state = np.asarray(
        recovered["z_classical_state"],
        dtype=complex,
    )

    quantum_probabilities = np.abs(quantum_state) ** 2
    classical_probabilities = np.abs(classical_state) ** 2

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    history = np.maximum(np.asarray(result.cost_history, dtype=float), 1e-16)
    axes[0].plot(history, color="seagreen", linewidth=1.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Function evaluation")
    axes[0].set_ylabel("Global VQLS cost")
    axes[0].set_title("Cost convergence")
    axes[0].grid(True, alpha=0.3)

    indices = np.arange(quantum_probabilities.size)
    axes[1].bar(indices, classical_probabilities, color="steelblue")
    axes[1].set_xlabel("Basis index")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Classical normalized state")

    axes[2].bar(indices, quantum_probabilities, color="darkorange")
    axes[2].set_xlabel("Basis index")
    axes[2].set_ylabel("Probability")
    axes[2].set_title("VQLS normalized state")

    figure.tight_layout()
    output_path = OUTPUT_DIR / "vqls_qopf_case3sc_results.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


# =============================================================================
# 9. MAIN
# =============================================================================


def main() -> None:
    np.random.seed(RNG_SEED)

    A_raw, b_raw, labels = build_case3sc_dc_kkt()
    system = prepare_qopf_matrix_for_vqls(A_raw, b_raw, labels)
    print_preprocessing_summary(system)

    pauli_operator = pauli_decompose_matrix(system.A_vqls)
    print_pauli_decomposition(system.A_vqls, pauli_operator)

    # Kiem tra mach amplitude encoding cua |b>.
    b_circuit = build_b_preparation_circuit(system.b_state)
    prepared_b = Statevector.from_instruction(b_circuit).data
    b_preparation_error = np.linalg.norm(prepared_b - system.b_state)
    print(f"\nStatePreparation error            = {b_preparation_error:.12e}")

    if b_preparation_error > 1e-10:
        raise RuntimeError("Mach U_b khong tao dung |b>.")

    result = optimize_vqls(system)
    recovered = recover_and_report(system, result)
    plot_path = plot_results(result, recovered)

    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()