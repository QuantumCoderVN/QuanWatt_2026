"""
Benchmark two HHL solution-recovery methods on small, standard power-system
benchmark cases using a DC power-flow linear system.

Methods:
  1) HHL parity-check recovery: module hhl_shot_recovery_modules.py
  2) HHL tomography:     module hhl_tomography_modules.py

For each selected network, this script loads real MATPOWER/PYPOWER-derived data
through pandapower and constructs the reduced DC power-flow equation

    B_red @ theta_non_slack = P_red.

The slack-bus angle is removed, making the matrix nonsingular for a connected
network. The resulting matrix sizes are below 10 by default:

    case4gs -> 4 buses -> 3x3
    case5   -> 5 buses -> 4x4
    case6ww -> 6 buses -> 5x5
    case9   -> 9 buses -> 8x8

Recommended usage:

  # Install the data dependency
  pip install pandapower

  # 1) Run parity-check recovery
  python benchmark_hhl_ieee_dc_powerflow.py --method parity --phase-qubits 4 --shots 20000

  # 2) Run tomography, reusing exactly the same saved systems
  python benchmark_hhl_ieee_dc_powerflow.py --method tomography --phase-qubits 4 --tomo-shots 20000

  # Or run both methods in one command
  python benchmark_hhl_ieee_dc_powerflow.py --method both --phase-qubits 4 --shots 20000 --tomo-shots 20000

  # Replot existing CSV data only
  python benchmark_hhl_ieee_dc_powerflow.py --plot-only

Default outputs:
  benchmark_outputs_ieee_dc/

Notes:
  - The input network data are not random. They come from pandapower's standard
    MATPOWER/PYPOWER-derived test cases.
  - HHL requires dimension 2^n. The existing solver modules are expected to
    auto-pad 3x3 and 5x5 systems to 4x4 and 8x8.
  - The unknown vector is the non-slack voltage-angle vector in radians.
  - Relative-entropy metrics use Jensen-Shannon distance. The signed version
    encodes positive and negative components in separate probability bins.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "2026-07-24-dc-v6-js-dashboard"

# =============================================================================
# Output and case configuration
# =============================================================================
DEFAULT_OUTPUT_DIR = Path("benchmark_outputs_ieee_dc_js_dashboard")
DEFAULT_SYSTEM_DIR_NAME = "systems"
DEFAULT_SOLUTION_DIR_NAME = "solutions"
DEFAULT_CSV_NAME = "hhl_ieee_dc_js_dashboard_results.csv"

SUPPORTED_CASES = ("case4gs", "case5", "case6ww", "case9")
DEFAULT_CASES = list(SUPPORTED_CASES)


# =============================================================================
# Generic dimension helpers
# =============================================================================
def target_qubits_for_size(matrix_size: int) -> int:
    """Number of target qubits after padding a square system to dimension 2^n."""
    if matrix_size < 2:
        raise ValueError("matrix_size must be >= 2.")
    return int(math.ceil(math.log2(matrix_size)))


def padded_dim_for_size(matrix_size: int) -> int:
    """Smallest power-of-two dimension greater than or equal to matrix_size."""
    return 1 << target_qubits_for_size(matrix_size)


def _stable_case_seed(base_seed: int, case_name: str) -> int:
    """Deterministic per-case seed used only by the quantum measurement simulator."""
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(case_name))
    return int(base_seed + checksum)


def _to_dense_real(matrix: Any) -> np.ndarray:
    """Convert scipy sparse or ndarray input to a dense float64 array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


# =============================================================================
# Load real network data and construct DC Ax=b
# =============================================================================
def _import_pandapower_components():
    """Import pandapower components across supported 2.x/3.x package layouts."""
    try:
        import pandapower  # noqa: F401
        import pandapower.networks as pn
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pandapower is not installed in the active Python environment. "
            "Install it with: python -m pip install pandapower"
        ) from exc

    # Public documented path in current pandapower releases.
    try:
        from pandapower.converter.pypower.to_ppc import to_ppc
    except ImportError:
        # Compatibility with releases that re-exported it at converter level.
        try:
            from pandapower.converter import to_ppc
        except ImportError as exc:
            raise ImportError(
                "pandapower is installed, but to_ppc could not be imported. "
                "Expected pandapower.converter.pypower.to_ppc.to_ppc. "
                "Check the installed version with: "
                "python -c \"import pandapower; print(pandapower.__version__)\""
            ) from exc

    try:
        from pandapower.pypower.idx_bus import BUS_TYPE, REF, VA
        from pandapower.pypower.makeBdc import makeBdc
        from pandapower.pypower.makeSbus import makeSbus
    except ImportError as exc:
        raise ImportError(
            "pandapower is installed, but its PYPOWER compatibility modules "
            "could not be imported. Reinstall with: "
            "python -m pip install --upgrade --force-reinstall pandapower"
        ) from exc

    return pn, to_ppc, BUS_TYPE, REF, VA, makeBdc, makeSbus


def _call_make_bdc(make_bdc: Any, base_mva: float, bus: np.ndarray, branch: np.ndarray):
    """Call makeBdc across old and new pandapower signatures.

    Older releases use makeBdc(baseMVA, bus, branch) and return four values.
    Newer releases use makeBdc(bus, branch, ...) and return five values.
    """
    try:
        parameter_names = tuple(inspect.signature(make_bdc).parameters)
    except (TypeError, ValueError):
        parameter_names = ()

    first_parameter = parameter_names[0].lower() if parameter_names else ""
    if first_parameter.startswith("bus"):
        result = make_bdc(bus, branch)
    else:
        result = make_bdc(base_mva, bus, branch)

    if not isinstance(result, tuple) or len(result) < 4:
        raise RuntimeError(
            "Unsupported pandapower makeBdc return value. "
            f"Expected at least 4 tuple entries, received {type(result).__name__}."
        )

    Bbus, Bf, Pbusinj, Pfinj = result[:4]
    return Bbus, Bf, Pbusinj, Pfinj


def load_network(case_name: str):
    """Load one supported small benchmark network from pandapower."""
    if case_name not in SUPPORTED_CASES:
        raise ValueError(
            f"Unsupported case {case_name!r}. Supported: {', '.join(SUPPORTED_CASES)}"
        )

    pn, *_ = _import_pandapower_components()
    factory = getattr(pn, case_name, None)
    if factory is None:
        raise RuntimeError(
            f"The installed pandapower version does not provide {case_name}()."
        )
    return factory()


def build_dc_powerflow_system(
    case_name: str,
    max_matrix_size: int = 9,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the reduced DC power-flow system for one network.

    The full DC equation is

        Bbus @ Va = Pbus - Pbusinj,

    with Va in radians and power injections in per unit. The reference-bus angle
    is fixed, so its row and column are removed:

        A = Bbus[nonref, nonref]
        b = Pbus[nonref] - Pbusinj[nonref]
            - Bbus[nonref, ref] @ Va_ref.
    """
    (
        _pn,
        to_ppc,
        BUS_TYPE,
        REF,
        VA,
        makeBdc,
        makeSbus,
    ) = _import_pandapower_components()

    net = load_network(case_name)
    ppc = to_ppc(
        net,
        calculate_voltage_angles=True,
        init="flat",
        mode="pf",
        check_connectivity=True,
    )

    base_mva = float(ppc["baseMVA"])
    bus = np.asarray(ppc["bus"], dtype=np.float64)
    gen = np.asarray(ppc["gen"], dtype=np.float64)
    branch = np.asarray(ppc["branch"], dtype=np.float64)

    Bbus, _Bf, Pbusinj, _Pfinj = _call_make_bdc(
        makeBdc, base_mva, bus, branch
    )
    Bbus = _to_dense_real(Bbus)
    Pbusinj = np.asarray(Pbusinj, dtype=np.float64).reshape(-1)
    Pbus = np.asarray(makeSbus(base_mva, bus, gen)).reshape(-1).real

    ref = np.flatnonzero(bus[:, BUS_TYPE].astype(int) == int(REF))
    if ref.size != 1:
        raise ValueError(
            f"{case_name}: expected exactly one reference bus, found {ref.size}."
        )

    ref_index = int(ref[0])
    nonref = np.array([i for i in range(bus.shape[0]) if i != ref_index], dtype=int)
    va_ref = np.deg2rad(float(bus[ref_index, VA]))

    A = Bbus[np.ix_(nonref, nonref)]
    b = (
        Pbus[nonref]
        - Pbusinj[nonref]
        - Bbus[np.ix_(nonref, [ref_index])].reshape(-1) * va_ref
    )

    # Guard against tiny numerical asymmetry.
    A = 0.5 * (A + A.T)

    matrix_size = int(A.shape[0])
    if matrix_size > max_matrix_size:
        raise ValueError(
            f"{case_name}: reduced DC matrix is {matrix_size}x{matrix_size}, "
            f"which exceeds --max-matrix-size={max_matrix_size}."
        )

    rank = int(np.linalg.matrix_rank(A))
    if rank != matrix_size:
        raise np.linalg.LinAlgError(
            f"{case_name}: reduced B matrix is singular (rank {rank}/{matrix_size})."
        )

    eigvals = np.linalg.eigvalsh(A)
    metadata: dict[str, Any] = {
        "formulation": "DC power flow: B_red theta = P_red",
        "case_name": case_name,
        "number_of_buses": int(bus.shape[0]),
        "matrix_size": matrix_size,
        "reference_bus_internal_index": ref_index,
        "non_reference_bus_internal_indices": nonref.tolist(),
        "base_mva": base_mva,
        "condition_number": float(np.linalg.cond(A)),
        "matrix_rank": rank,
        "min_eigenvalue": float(np.min(eigvals)),
        "max_eigenvalue": float(np.max(eigvals)),
        "target_qubits": target_qubits_for_size(matrix_size),
        "padded_dim": padded_dim_for_size(matrix_size),
        "unknown": "non-slack voltage angles in radians",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    return A.astype(np.complex128), b.astype(np.complex128), metadata


# =============================================================================
# Save / load fixed systems so both HHL methods use identical input
# =============================================================================
def system_path(system_dir: Path, case_name: str) -> Path:
    return system_dir / f"dc_{case_name}.npz"


def save_system(path: Path, A: np.ndarray, b: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        A=A,
        b=b,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def load_system(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    A = data["A"]
    b = data["b"]
    metadata = json.loads(str(data["metadata"]))
    return A, b, metadata


def prepare_systems(
    case_names: list[str],
    system_dir: Path,
    regenerate: bool,
    max_matrix_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Create or load fixed DC systems for all selected cases."""
    systems: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    system_dir.mkdir(parents=True, exist_ok=True)

    for case_name in case_names:
        path = system_path(system_dir, case_name)
        if path.exists() and not regenerate:
            A, b, metadata = load_system(path)
        else:
            A, b, metadata = build_dc_powerflow_system(
                case_name=case_name,
                max_matrix_size=max_matrix_size,
            )
            save_system(path, A, b, metadata)
        systems[case_name] = (A, b, metadata)

    return systems


# =============================================================================
# Import and run HHL solver modules
# =============================================================================
def import_solver_module(module_name: str):
    """Import an HHL solver from this folder, parent folders, or PYTHONPATH."""
    script_dir = Path(__file__).resolve().parent
    search_dirs = [
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
        Path.cwd(),
    ]

    for directory in search_dirs:
        candidate = directory / f"{module_name}.py"
        if candidate.exists():
            directory_text = str(directory)
            if directory_text not in sys.path:
                sys.path.insert(0, directory_text)
            return importlib.import_module(module_name)

    # Fall back to an installed/importable package before raising a useful error.
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        searched = "\n".join(
            f"  - {directory / f'{module_name}.py'}" for directory in search_dirs
        )
        raise ModuleNotFoundError(
            f"Could not find {module_name!r}. Checked:\n{searched}"
        ) from exc


def classical_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Classical reference solution."""
    return np.linalg.solve(A, b)


def normalize_method_name(method: str) -> str:
    """Map CLI aliases to the method names stored in CSV and shown in plots."""
    value = method.lower().strip().replace("-", "_").replace(" ", "_")
    if value in {"shot", "parity", "parity_check"}:
        return "parity_check"
    if value == "tomography":
        return "tomography"
    raise ValueError("method must be parity/parity_check/shot or tomography.")


def display_method_name(method: str) -> str:
    return {
        "parity_check": "Parity check",
        "tomography": "Tomography",
    }.get(normalize_method_name(method), method)


def run_hhl_method(
    method: str,
    A: np.ndarray,
    b: np.ndarray,
    phase_qubits: int,
    shots: int,
    tomo_shots: int,
    seed: int | None,
    progress: bool,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Run one HHL method and measure only the solver-module runtime."""
    method = normalize_method_name(method)

    if method == "parity_check":
        module = import_solver_module("hhl_shot_recovery_modules")
        start = time.perf_counter()
        x_hhl, info = module.hhl_solver(
            A,
            b,
            phase_qubits=phase_qubits,
            shots=shots,
            seed=seed,
            return_info=True,
        )
        return x_hhl, info, time.perf_counter() - start

    if method == "tomography":
        module = import_solver_module("hhl_tomography_modules")
        start = time.perf_counter()
        x_hhl, info = module.hhl_tomography_solver(
            A,
            b,
            phase_qubits=phase_qubits,
            tomography_shots=tomo_shots,
            seed=seed,
            progress=progress,
            return_info=True,
        )
        return x_hhl, info, time.perf_counter() - start

    raise ValueError("method must be parity_check or tomography.")


# =============================================================================
# Metrics
# =============================================================================
def _align_global_phase(
    x_reference: np.ndarray,
    x_estimate: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """Align the physically irrelevant global phase/sign of x_estimate."""
    overlap = np.vdot(x_reference, x_estimate)
    if abs(overlap) <= eps:
        return np.asarray(x_estimate, dtype=np.complex128)
    return np.asarray(x_estimate, dtype=np.complex128) * np.exp(-1j * np.angle(overlap))


def _real_vector_after_phase_alignment(
    x_reference: np.ndarray,
    x_estimate: np.ndarray,
    imag_tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return real vectors after aligning the global phase of the estimate."""
    reference = np.asarray(x_reference, dtype=np.complex128).reshape(-1)
    estimate = _align_global_phase(reference, np.asarray(x_estimate).reshape(-1))

    scale = max(1.0, float(np.max(np.abs(reference))), float(np.max(np.abs(estimate))))
    if float(np.max(np.abs(reference.imag))) > imag_tolerance * scale:
        raise ValueError("Reference solution has a non-negligible imaginary component.")
    if float(np.max(np.abs(estimate.imag))) > imag_tolerance * scale:
        raise ValueError("HHL solution has a non-negligible imaginary component after phase alignment.")

    return reference.real.astype(np.float64), estimate.real.astype(np.float64)


def _amplitude_probability(vector: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Convert a vector into the Born-style distribution |x_i|^2 / ||x||^2."""
    weights = np.abs(np.asarray(vector).reshape(-1)) ** 2
    total = float(np.sum(weights))
    if total <= eps:
        raise ValueError("Cannot form a probability distribution from a near-zero vector.")
    return np.asarray(weights / total, dtype=np.float64)


def _signed_probability(vector: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Encode sign and relative magnitude as a 2N-bin probability distribution.

    The first N bins contain squared positive components and the last N bins
    contain squared negative components. The distribution therefore preserves
    component sign while remaining non-negative and normalized.
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    positive = np.square(np.clip(values, 0.0, None))
    negative = np.square(np.clip(-values, 0.0, None))
    weights = np.concatenate([positive, negative])
    total = float(np.sum(weights))
    if total <= eps:
        raise ValueError("Cannot form a signed probability distribution from a near-zero vector.")
    return weights / total


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Numerically stable KL divergence D_KL(p || q), measured in nats."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("Probability distributions must have the same shape.")

    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p = (p + eps) / float(np.sum(p + eps))
    q = (q + eps) / float(np.sum(q + eps))
    return float(np.sum(p * np.log(p / q)))


def _jensen_shannon_metrics(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Return JSD, normalized JS distance, and similarity percentage."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    midpoint = 0.5 * (p + q)
    js_divergence = 0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint)
    normalized = float(np.clip(js_divergence / np.log(2.0), 0.0, 1.0))
    js_distance = float(np.sqrt(normalized))
    similarity_percent = 100.0 * (1.0 - js_distance)
    return {
        "js_divergence": js_divergence,
        "js_distance": js_distance,
        "js_similarity_percent": similarity_percent,
    }


def compute_metrics(
    A: np.ndarray,
    b: np.ndarray,
    x_hhl: np.ndarray,
    x_classical: np.ndarray,
    eps: float = 1e-15,
) -> dict[str, float]:
    """Compute vector, residual, fidelity, and state-angle metrics."""
    norm_hhl = float(np.linalg.norm(x_hhl))
    norm_classical = float(np.linalg.norm(x_classical))
    norm_b = float(np.linalg.norm(b))

    if norm_hhl <= eps or norm_classical <= eps or norm_b <= eps:
        raise ValueError("Cannot compute metrics for a near-zero vector.")

    absolute_error = float(np.linalg.norm(x_hhl - x_classical))
    relative_error = absolute_error / norm_classical
    residual_hhl = float(np.linalg.norm(A @ x_hhl - b))
    residual_classical = float(np.linalg.norm(A @ x_classical - b))
    relative_residual_hhl = residual_hhl / norm_b

    xh_norm = x_hhl / norm_hhl
    xc_norm = x_classical / norm_classical
    overlap = float(np.clip(abs(np.vdot(xh_norm, xc_norm)), 0.0, 1.0))
    fidelity = overlap**2
    state_angle_degrees = float(np.degrees(np.arccos(overlap)))

    xc_real, xh_real = _real_vector_after_phase_alignment(x_classical, x_hhl)

    amplitude_classical = _amplitude_probability(xc_real)
    amplitude_hhl = _amplitude_probability(xh_real)
    signed_classical = _signed_probability(xc_real)
    signed_hhl = _signed_probability(xh_real)

    amplitude_js = _jensen_shannon_metrics(amplitude_classical, amplitude_hhl)
    signed_js = _jensen_shannon_metrics(signed_classical, signed_hhl)

    return {
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "residual_hhl": residual_hhl,
        "residual_classical": residual_classical,
        "relative_residual_hhl": relative_residual_hhl,
        "fidelity_direction": fidelity,
        "state_angle_degrees": state_angle_degrees,
        "kl_classical_to_hhl_amplitude": _kl_divergence(amplitude_classical, amplitude_hhl),
        "kl_classical_to_hhl_signed": _kl_divergence(signed_classical, signed_hhl),
        "js_divergence_amplitude": amplitude_js["js_divergence"],
        "js_distance_amplitude": amplitude_js["js_distance"],
        "js_similarity_amplitude_percent": amplitude_js["js_similarity_percent"],
        "js_divergence_signed": signed_js["js_divergence"],
        "js_distance_signed": signed_js["js_distance"],
        "js_similarity_signed_percent": signed_js["js_similarity_percent"],
    }


# =============================================================================
# CSV helpers
# =============================================================================
def ensure_csv_header(csv_path: Path, fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def append_result(csv_path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    ensure_csv_header(csv_path, fieldnames)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)


# =============================================================================
# Benchmark
# =============================================================================
def run_benchmark(
    method: str,
    case_names: list[str],
    output_dir: Path,
    base_seed: int,
    regenerate_systems: bool,
    max_matrix_size: int,
    phase_qubits: int,
    shots: int,
    tomo_shots: int,
    progress: bool,
) -> Path:
    """Run the selected HHL method on all selected DC power-flow systems."""
    method = normalize_method_name(method)
    output_dir.mkdir(parents=True, exist_ok=True)
    system_dir = output_dir / DEFAULT_SYSTEM_DIR_NAME
    solution_dir = output_dir / DEFAULT_SOLUTION_DIR_NAME
    solution_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / DEFAULT_CSV_NAME

    systems = prepare_systems(
        case_names=case_names,
        system_dir=system_dir,
        regenerate=regenerate_systems,
        max_matrix_size=max_matrix_size,
    )

    fieldnames = [
        "timestamp",
        "formulation",
        "method",
        "case_name",
        "number_of_buses",
        "matrix_size",
        "condition_number",
        "target_qubits",
        "padded_dim",
        "phase_qubits",
        "shots_per_circuit",
        "total_measurement_bases_or_circuits",
        "seed",
        "absolute_error",
        "relative_error",
        "residual_hhl",
        "residual_classical",
        "relative_residual_hhl",
        "fidelity_direction",
        "state_angle_degrees",
        "kl_classical_to_hhl_amplitude",
        "kl_classical_to_hhl_signed",
        "js_divergence_amplitude",
        "js_distance_amplitude",
        "js_similarity_amplitude_percent",
        "js_divergence_signed",
        "js_distance_signed",
        "js_similarity_signed_percent",
        "runtime_seconds",
        "postselection_rate",
        "success_count_summary",
        "system_file",
        "solution_file",
    ]

    for case_name in case_names:
        A, b, metadata = systems[case_name]
        matrix_size = int(A.shape[0])
        n_target = target_qubits_for_size(matrix_size)
        padded_dim = padded_dim_for_size(matrix_size)
        run_seed = _stable_case_seed(base_seed, case_name)

        print("=" * 88, flush=True)
        print(
            f"[BENCH] formulation=DC, method={method}, case={case_name}, "
            f"buses={metadata['number_of_buses']}, matrix={matrix_size}x{matrix_size}, "
            f"target_qubits={n_target}, padded_dim={padded_dim}",
            flush=True,
        )

        x_classical = classical_solve(A, b)

        try:
            x_hhl, info, runtime = run_hhl_method(
                method=method,
                A=A,
                b=b,
                phase_qubits=phase_qubits,
                shots=shots,
                tomo_shots=tomo_shots,
                seed=run_seed,
                progress=progress,
            )
            metrics = compute_metrics(A, b, x_hhl, x_classical)

            if method == "parity_check":
                shots_per_circuit = shots
                total_circuits = 1 + (padded_dim - 1)
                postselection_rate = info.get("postselection_rate", "")
                success_summary = str(info.get("success_count", ""))
            else:
                shots_per_circuit = tomo_shots
                total_circuits = 3**n_target
                success_by_basis = info.get("success_count_by_basis", {})
                if success_by_basis:
                    values = list(success_by_basis.values())
                    postselection_rate = float(np.mean(values) / tomo_shots)
                    success_summary = (
                        f"min={min(values)}, mean={np.mean(values):.2f}, max={max(values)}"
                    )
                else:
                    postselection_rate = ""
                    success_summary = ""

            solution_path = solution_dir / f"dc_{case_name}_{method}.npz"
            xc_real, xh_real = _real_vector_after_phase_alignment(x_classical, x_hhl)
            np.savez_compressed(
                solution_path,
                x_classical=np.asarray(x_classical),
                x_hhl=np.asarray(x_hhl),
                x_classical_phase_aligned_real=xc_real,
                x_hhl_phase_aligned_real=xh_real,
                amplitude_distribution_classical=_amplitude_probability(xc_real),
                amplitude_distribution_hhl=_amplitude_probability(xh_real),
                signed_distribution_classical=_signed_probability(xc_real),
                signed_distribution_hhl=_signed_probability(xh_real),
                A=np.asarray(A),
                b=np.asarray(b),
                method=method,
                case_name=case_name,
            )

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "formulation": "DC",
                "method": method,
                "case_name": case_name,
                "number_of_buses": metadata["number_of_buses"],
                "matrix_size": matrix_size,
                "condition_number": metadata["condition_number"],
                "target_qubits": n_target,
                "padded_dim": padded_dim,
                "phase_qubits": phase_qubits,
                "shots_per_circuit": shots_per_circuit,
                "total_measurement_bases_or_circuits": total_circuits,
                "seed": run_seed,
                **metrics,
                "runtime_seconds": runtime,
                "postselection_rate": postselection_rate,
                "success_count_summary": success_summary,
                "system_file": str(system_path(system_dir, case_name)),
                "solution_file": str(solution_path),
            }
            append_result(csv_path, row, fieldnames)

            print(f"[DONE] case={case_name}", flush=True)
            print(f"       state_angle     = {metrics['state_angle_degrees']:.6f} deg", flush=True)
            print(f"       relative_error  = {metrics['relative_error']:.6e}", flush=True)
            print(f"       signed_JS_dist  = {metrics['js_distance_signed']:.6f}", flush=True)
            print(f"       signed_JS_sim   = {metrics['js_similarity_signed_percent']:.3f}%", flush=True)
            print(f"       runtime         = {runtime:.3f} s", flush=True)
            print(f"       csv             = {csv_path}", flush=True)

        except Exception as exc:
            row = {name: "" for name in fieldnames}
            row.update(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "formulation": "DC",
                    "method": method,
                    "case_name": case_name,
                    "number_of_buses": metadata["number_of_buses"],
                    "matrix_size": matrix_size,
                    "condition_number": metadata["condition_number"],
                    "target_qubits": n_target,
                    "padded_dim": padded_dim,
                    "phase_qubits": phase_qubits,
                    "shots_per_circuit": shots if method == "parity_check" else tomo_shots,
                    "seed": run_seed,
                    "absolute_error": "ERROR",
                    "relative_error": "ERROR",
                    "residual_hhl": "ERROR",
                    "residual_classical": "ERROR",
                    "relative_residual_hhl": "ERROR",
                    "fidelity_direction": "ERROR",
                    "state_angle_degrees": "ERROR",
                    "kl_classical_to_hhl_amplitude": "ERROR",
                    "kl_classical_to_hhl_signed": "ERROR",
                    "js_divergence_amplitude": "ERROR",
                    "js_distance_amplitude": "ERROR",
                    "js_similarity_amplitude_percent": "ERROR",
                    "js_divergence_signed": "ERROR",
                    "js_distance_signed": "ERROR",
                    "js_similarity_signed_percent": "ERROR",
                    "runtime_seconds": "ERROR",
                    "success_count_summary": repr(exc),
                    "system_file": str(system_path(system_dir, case_name)),
                    "solution_file": "",
                }
            )
            append_result(csv_path, row, fieldnames)
            print(f"[ERROR] case={case_name}: {exc!r}", flush=True)

    return csv_path


# =============================================================================
# Plotting
# =============================================================================
def _normalize_csv_method(value: str) -> str:
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"shot", "parity", "parity_check"}:
        return "parity_check"
    if value == "tomography":
        return "tomography"
    return ""


def _load_valid_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load valid JS benchmark rows, with compatibility for older method labels."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    numeric_float_fields = (
        "condition_number",
        "absolute_error",
        "relative_error",
        "relative_residual_hhl",
        "fidelity_direction",
        "state_angle_degrees",
        "js_distance_amplitude",
        "js_similarity_amplitude_percent",
        "js_distance_signed",
        "js_similarity_signed_percent",
        "runtime_seconds",
    )
    numeric_int_fields = (
        "number_of_buses",
        "matrix_size",
        "target_qubits",
        "padded_dim",
        "total_measurement_bases_or_circuits",
    )

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                for name in numeric_int_fields:
                    row[name] = int(float(row[name]))
                for name in numeric_float_fields:
                    row[name] = float(row[name])

                method = _normalize_csv_method(row.get("method", ""))
                if not method:
                    total = row["total_measurement_bases_or_circuits"]
                    if total == row["padded_dim"]:
                        method = "parity_check"
                    elif total == 3 ** row["target_qubits"]:
                        method = "tomography"
                    else:
                        raise ValueError("Cannot infer recovery method")
                row["method"] = method
            except Exception:
                continue
            rows.append(row)
    return rows


def _latest_rows_by_method_and_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["case_name"]))
        if key not in latest or str(row["timestamp"]) >= str(latest[key]["timestamp"]):
            latest[key] = row
    return list(latest.values())


def _solution_file_from_row(row: dict[str, Any], output_dir: Path) -> Path:
    raw = str(row.get("solution_file", "")).strip()
    if raw:
        candidate = Path(raw)
        if candidate.exists():
            return candidate
        candidate_from_cwd = Path.cwd() / candidate
        if candidate_from_cwd.exists():
            return candidate_from_cwd
        candidate_from_output_parent = output_dir.parent / candidate
        if candidate_from_output_parent.exists():
            return candidate_from_output_parent

    method = _normalize_csv_method(row["method"])
    fallback = output_dir / DEFAULT_SOLUTION_DIR_NAME / f"dc_{row['case_name']}_{method}.npz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Solution NPZ not found for {row['case_name']} / {display_method_name(method)}: {fallback}"
    )


def _load_solution_distributions(
    row: dict[str, Any],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    path = _solution_file_from_row(row, output_dir)
    data = np.load(path, allow_pickle=False)

    if "amplitude_distribution_classical" in data.files:
        return {
            "x_classical": np.asarray(data["x_classical_phase_aligned_real"], dtype=float),
            "x_hhl": np.asarray(data["x_hhl_phase_aligned_real"], dtype=float),
            "amplitude_classical": np.asarray(data["amplitude_distribution_classical"], dtype=float),
            "amplitude_hhl": np.asarray(data["amplitude_distribution_hhl"], dtype=float),
            "signed_classical": np.asarray(data["signed_distribution_classical"], dtype=float),
            "signed_hhl": np.asarray(data["signed_distribution_hhl"], dtype=float),
        }

    x_classical = np.asarray(data["x_classical"])
    x_hhl = np.asarray(data["x_hhl"])
    xc_real, xh_real = _real_vector_after_phase_alignment(x_classical, x_hhl)
    return {
        "x_classical": xc_real,
        "x_hhl": xh_real,
        "amplitude_classical": _amplitude_probability(xc_real),
        "amplitude_hhl": _amplitude_probability(xh_real),
        "signed_classical": _signed_probability(xc_real),
        "signed_hhl": _signed_probability(xh_real),
    }


def _case_tick_labels(rows: list[dict[str, Any]], case_names: list[str]) -> list[str]:
    labels: list[str] = []
    for case_name in case_names:
        row = next(r for r in rows if r["case_name"] == case_name)
        labels.append(
            f"{case_name}\n{row['number_of_buses']} bus / "
            f"{row['matrix_size']}×{row['matrix_size']}"
        )
    return labels


def _method_style(method: str) -> dict[str, Any]:
    # Use Matplotlib's default color cycle to remain portable across themes.
    if method == "parity_check":
        return {"marker": "o", "linestyle": "-"}
    return {"marker": "s", "linestyle": "-"}


def _plot_similarity_axis(
    ax: Any,
    metric: str,
    title: str,
    rows: list[dict[str, Any]],
    case_names: list[str],
    by_method_case: dict[tuple[str, str], dict[str, Any]],
) -> None:
    x_positions = np.arange(len(case_names))
    for method in ("parity_check", "tomography"):
        if not any((method, case_name) in by_method_case for case_name in case_names):
            continue
        values = [
            np.nan if (method, case_name) not in by_method_case
            else float(by_method_case[(method, case_name)][metric])
            for case_name in case_names
        ]
        ax.plot(
            x_positions,
            values,
            linewidth=2.2,
            label=display_method_name(method),
            **_method_style(method),
        )
        for x, value in zip(x_positions, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.1f}%",
                    (x, value),
                    xytext=(0, 8 if method == "tomography" else -17),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(_case_tick_labels(rows, case_names), fontsize=9)
    ax.set_ylim(0.0, 102.0)
    ax.set_ylabel("Jensen–Shannon similarity (%)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def _create_individual_similarity_plot(
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    rows: list[dict[str, Any]],
    case_names: list[str],
    by_method_case: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    _plot_similarity_axis(ax, metric, title, rows, case_names, by_method_case)
    ax.set_xlabel("Standard power-system case and reduced DC matrix size")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def create_js_dashboard(
    rows: list[dict[str, Any]],
    output_dir: Path,
    dashboard_case: str,
) -> Path:
    """Create a publication-style dashboard resembling the requested illustration."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    case_names = [
        name for name in SUPPORTED_CASES
        if any(row["case_name"] == name for row in rows)
    ]
    if dashboard_case not in case_names:
        dashboard_case = case_names[-1]

    by_method_case = {(r["method"], r["case_name"]): r for r in rows}
    methods = [
        method for method in ("parity_check", "tomography")
        if (method, dashboard_case) in by_method_case
    ]
    if not methods:
        raise RuntimeError(f"No valid solution data found for dashboard case {dashboard_case}.")

    detail: dict[str, dict[str, np.ndarray]] = {
        method: _load_solution_distributions(
            by_method_case[(method, dashboard_case)], output_dir
        )
        for method in methods
    }

    classical = detail[methods[0]]
    n = classical["x_classical"].size
    x_labels = [f"x{i}" for i in range(n)]
    signed_labels = [f"+x{i}" for i in range(n)] + [f"−x{i}" for i in range(n)]

    fig = plt.figure(figsize=(20, 12.2))
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.89,
        bottom=0.085,
        hspace=0.52,
        wspace=0.55,
    )
    grid = GridSpec(2, 16, figure=fig, height_ratios=[1.15, 0.95])

    fig.suptitle(
        "HHL solution comparison using Jensen–Shannon relative entropy",
        fontsize=23,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        "DC power-flow benchmark — Parity check versus Tomography",
        ha="center",
        fontsize=13.5,
    )

    ax_formula = fig.add_subplot(grid[0, 0:4])
    ax_formula.axis("off")
    formula_text = (
        "FROM SOLUTION VECTOR TO DISTRIBUTION\n\n"
        r"Amplitude distribution" "\n"
        r"$p_i=|x_i|^2/\|x\|_2^2$" "\n\n"
        r"Signed distribution" "\n"
        r"$P_{signed}=[p_0^+,\ldots,p_{N-1}^+,p_0^-,\ldots,p_{N-1}^-]$" "\n"
        r"$p_i^+=\max(x_i,0)^2/\|x\|_2^2$" "\n"
        r"$p_i^-=\max(-x_i,0)^2/\|x\|_2^2$" "\n\n"
        r"$M=(P+Q)/2$" "\n"
        r"$D_{JS}=\frac{1}{2}D_{KL}(P\Vert M)$" "\n"
        r"$\qquad+\frac{1}{2}D_{KL}(Q\Vert M)$" "\n\n"
        r"$d_{JS}=\sqrt{D_{JS}/\ln 2}$" "\n"
        r"Similarity $=100(1-d_{JS})\%$" "\n\n"
        "Higher is better.\n"
        "Amplitude ignores sign.\n"
        "Signed distribution preserves sign."
    )
    ax_formula.text(
        0.02,
        0.98,
        formula_text,
        va="top",
        ha="left",
        fontsize=10.3,
        linespacing=1.42,
        bbox={
            "boxstyle": "round,pad=0.75",
            "facecolor": "white",
            "edgecolor": "0.65",
            "alpha": 0.98,
        },
    )

    bar_width = 0.23
    offsets = {"parity_check": 0.0, "tomography": bar_width}

    ax_amp = fig.add_subplot(grid[0, 4:9])
    positions = np.arange(n)
    ax_amp.bar(
        positions - bar_width,
        classical["amplitude_classical"],
        width=bar_width,
        label="Classical",
        alpha=0.75,
    )
    for method in methods:
        ax_amp.bar(
            positions + offsets[method],
            detail[method]["amplitude_hhl"],
            width=bar_width,
            label=display_method_name(method),
            alpha=0.88,
        )
    ax_amp.set_xticks(positions)
    ax_amp.set_xticklabels(x_labels, fontsize=8.5)
    ax_amp.set_ylabel("Probability")
    ax_amp.set_title(
        f"Amplitude distribution $|x_i|^2$ — {dashboard_case}",
        fontsize=13,
        fontweight="bold",
    )
    ax_amp.grid(True, axis="y", alpha=0.22)
    ax_amp.legend(fontsize=9, loc="upper right")

    ax_signed = fig.add_subplot(grid[0, 9:14])
    signed_positions = np.arange(2 * n)
    ax_signed.bar(
        signed_positions - bar_width,
        classical["signed_classical"],
        width=bar_width,
        label="Classical",
        alpha=0.75,
    )
    for method in methods:
        ax_signed.bar(
            signed_positions + offsets[method],
            detail[method]["signed_hhl"],
            width=bar_width,
            label=display_method_name(method),
            alpha=0.88,
        )
    ax_signed.axvline(n - 0.5, linewidth=1.2, linestyle="--", alpha=0.65)
    ax_signed.text(
        (n - 1) / 2,
        1.01,
        "Positive bins",
        transform=ax_signed.get_xaxis_transform(),
        ha="center",
        fontsize=9,
    )
    ax_signed.text(
        n + (n - 1) / 2,
        1.01,
        "Negative bins",
        transform=ax_signed.get_xaxis_transform(),
        ha="center",
        fontsize=9,
    )
    ax_signed.set_xticks(signed_positions)
    ax_signed.set_xticklabels(signed_labels, rotation=50, ha="right", fontsize=7.3)
    ax_signed.set_ylabel("Probability")
    ax_signed.set_title(
        f"Signed distribution — {dashboard_case}",
        fontsize=13,
        fontweight="bold",
    )
    ax_signed.grid(True, axis="y", alpha=0.22)
    ax_signed.legend(fontsize=8.5, loc="upper right")

    ax_summary = fig.add_subplot(grid[0, 14:16])
    ax_summary.axis("off")
    row_any = next(row for row in rows if row["case_name"] == dashboard_case)
    summary_lines = [
        "DETAIL CASE",
        dashboard_case,
        "",
        f"Buses: {row_any['number_of_buses']}",
        f"Vector size: {n}",
        f"Condition no.: {row_any['condition_number']:.2f}",
        "",
    ]
    for method in methods:
        row = by_method_case[(method, dashboard_case)]
        summary_lines.extend(
            [
                display_method_name(method),
                f"Amplitude JS: {row['js_similarity_amplitude_percent']:.1f}%",
                f"Signed JS: {row['js_similarity_signed_percent']:.1f}%",
                f"State angle: {row['state_angle_degrees']:.1f}°",
                "",
            ]
        )
    ax_summary.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=9.6,
        linespacing=1.32,
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "white",
            "edgecolor": "0.65",
            "alpha": 0.98,
        },
    )

    ax_amp_line = fig.add_subplot(grid[1, 0:8])
    _plot_similarity_axis(
        ax_amp_line,
        "js_similarity_amplitude_percent",
        "Jensen–Shannon similarity — amplitude distribution",
        rows,
        case_names,
        by_method_case,
    )

    ax_signed_line = fig.add_subplot(grid[1, 8:16])
    _plot_similarity_axis(
        ax_signed_line,
        "js_similarity_signed_percent",
        "Jensen–Shannon similarity — signed distribution",
        rows,
        case_names,
        by_method_case,
    )

    fig.text(
        0.5,
        0.045,
        "Power-system case and reduced DC matrix size",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.018,
        "Interpretation: high amplitude similarity but low signed similarity indicates sign-recovery errors. "
        "Use relative residual alongside JS similarity for equation-level validation.",
        ha="center",
        fontsize=9.8,
    )

    path = output_dir / "dc_js_relative_entropy_dashboard.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_results(
    csv_path: Path,
    output_dir: Path,
    dashboard_case: str = "case9",
    latest_only: bool = True,
) -> tuple[Path, Path, Path]:
    """Create two clean metric charts and one infographic-style dashboard."""
    rows = _load_valid_rows(csv_path)
    if latest_only:
        rows = _latest_rows_by_method_and_case(rows)
    if not rows:
        raise RuntimeError("CSV does not contain valid result rows.")

    output_dir.mkdir(parents=True, exist_ok=True)
    case_names = [
        name for name in SUPPORTED_CASES
        if any(row["case_name"] == name for row in rows)
    ]
    by_method_case = {(r["method"], r["case_name"]): r for r in rows}

    amplitude_path = _create_individual_similarity_plot(
        "js_similarity_amplitude_percent",
        "Amplitude-distribution JS similarity (%, higher is better)",
        "DC HHL Jensen–Shannon similarity — amplitude distribution",
        "dc_js_amplitude_similarity_comparison.png",
        rows,
        case_names,
        by_method_case,
        output_dir,
    )
    signed_path = _create_individual_similarity_plot(
        "js_similarity_signed_percent",
        "Signed-distribution JS similarity (%, higher is better)",
        "DC HHL Jensen–Shannon similarity — signed distribution",
        "dc_js_signed_similarity_comparison.png",
        rows,
        case_names,
        by_method_case,
        output_dir,
    )
    dashboard_path = create_js_dashboard(rows, output_dir, dashboard_case)
    return amplitude_path, signed_path, dashboard_path


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HHL parity-check recovery and tomography on reduced DC power-flow "
            "systems from small standard MATPOWER/PYPOWER-derived cases."
        )
    )
    parser.add_argument(
        "--method",
        choices=["parity", "parity_check", "shot", "tomography", "both"],
        default="parity",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(SUPPORTED_CASES),
        default=DEFAULT_CASES,
        help="Network cases to benchmark.",
    )
    parser.add_argument("--phase-qubits", type=int, default=4)
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--tomo-shots", type=int, default=20_000)
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base seed for quantum measurement simulation; network data are fixed.",
    )
    parser.add_argument(
        "--max-matrix-size",
        type=int,
        default=9,
        help="Reject a reduced system larger than this value.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--regenerate-systems",
        action="store_true",
        help="Rebuild cached NPZ systems from pandapower data.",
    )
    parser.add_argument(
        "--reset-results",
        action="store_true",
        help="Delete the existing JS CSV and saved solution vectors before a new run.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--dashboard-case",
        choices=list(SUPPORTED_CASES),
        default="case9",
        help="Case used for the detailed distribution panels in the dashboard.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build/cache DC systems and print metadata without running HHL.",
    )
    return parser.parse_args()


def main() -> None:
    print(f"[SCRIPT] {SCRIPT_VERSION}", flush=True)
    args = parse_args()
    output_dir: Path = args.output_dir
    csv_path = output_dir / DEFAULT_CSV_NAME

    if args.reset_results and not args.plot_only:
        if csv_path.exists():
            csv_path.unlink()
            print(f"[RESET] Deleted CSV: {csv_path}", flush=True)
        solution_dir = output_dir / DEFAULT_SOLUTION_DIR_NAME
        if solution_dir.exists():
            shutil.rmtree(solution_dir)
            print(f"[RESET] Deleted solutions: {solution_dir}", flush=True)

    if args.prepare_only:
        systems = prepare_systems(
            case_names=args.cases,
            system_dir=output_dir / DEFAULT_SYSTEM_DIR_NAME,
            regenerate=args.regenerate_systems,
            max_matrix_size=args.max_matrix_size,
        )
        for case_name, (A, b, metadata) in systems.items():
            print("=" * 88)
            print(f"case={case_name}")
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
            print("A =")
            print(np.real_if_close(A))
            print("b =")
            print(np.real_if_close(b))
            print("x_classical =")
            print(np.real_if_close(classical_solve(A, b)))
        return

    if not args.plot_only:
        methods = ["parity_check", "tomography"] if args.method == "both" else [normalize_method_name(args.method)]
        for method in methods:
            csv_path = run_benchmark(
                method=method,
                case_names=args.cases,
                output_dir=output_dir,
                base_seed=args.seed,
                regenerate_systems=args.regenerate_systems,
                max_matrix_size=args.max_matrix_size,
                phase_qubits=args.phase_qubits,
                shots=args.shots,
                tomo_shots=args.tomo_shots,
                progress=args.progress,
            )

    if not args.no_plot:
        try:
            amplitude_js_plot, signed_js_plot, dashboard_plot = plot_results(
                csv_path,
                output_dir,
                dashboard_case=args.dashboard_case,
            )
            print("=" * 88)
            print(f"[PLOT] Amplitude JS similarity: {amplitude_js_plot}")
            print(f"[PLOT] Signed JS similarity:    {signed_js_plot}")
            print(f"[PLOT] JS dashboard:            {dashboard_plot}")
        except Exception as exc:
            print(f"[PLOT ERROR] {exc!r}")


if __name__ == "__main__":
    main()