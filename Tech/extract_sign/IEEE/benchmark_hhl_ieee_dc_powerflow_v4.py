"""
Benchmark two HHL solution-recovery methods on small, standard power-system
benchmark cases using a DC power-flow linear system.

Methods:
  1) HHL shot-recovery: module hhl_shot_recovery_modules.py
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

  # 1) Run shot recovery
  python benchmark_hhl_ieee_dc_powerflow.py --method shot --phase-qubits 4 --shots 20000

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
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "2026-07-24-dc-v4"

# =============================================================================
# Output and case configuration
# =============================================================================
DEFAULT_OUTPUT_DIR = Path("benchmark_outputs_ieee_dc")
DEFAULT_SYSTEM_DIR_NAME = "systems"
DEFAULT_CSV_NAME = "hhl_ieee_dc_powerflow_results.csv"

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
    """Import a solver module from this script's directory or PYTHONPATH."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return importlib.import_module(module_name)


def classical_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Classical reference solution."""
    return np.linalg.solve(A, b)


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
    method = method.lower().strip()

    if method == "shot":
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

    raise ValueError("method must be 'shot' or 'tomography'.")


# =============================================================================
# Metrics
# =============================================================================
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

    return {
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "residual_hhl": residual_hhl,
        "residual_classical": residual_classical,
        "relative_residual_hhl": relative_residual_hhl,
        "fidelity_direction": fidelity,
        "state_angle_degrees": state_angle_degrees,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    system_dir = output_dir / DEFAULT_SYSTEM_DIR_NAME
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
        "runtime_seconds",
        "postselection_rate",
        "success_count_summary",
        "system_file",
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

            if method == "shot":
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
            }
            append_result(csv_path, row, fieldnames)

            print(f"[DONE] case={case_name}", flush=True)
            print(f"       state_angle     = {metrics['state_angle_degrees']:.6f} deg", flush=True)
            print(f"       relative_error  = {metrics['relative_error']:.6e}", flush=True)
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
                    "shots_per_circuit": shots if method == "shot" else tomo_shots,
                    "seed": run_seed,
                    "absolute_error": "ERROR",
                    "relative_error": "ERROR",
                    "residual_hhl": "ERROR",
                    "residual_classical": "ERROR",
                    "relative_residual_hhl": "ERROR",
                    "fidelity_direction": "ERROR",
                    "state_angle_degrees": "ERROR",
                    "runtime_seconds": "ERROR",
                    "success_count_summary": repr(exc),
                    "system_file": str(system_path(system_dir, case_name)),
                }
            )
            append_result(csv_path, row, fieldnames)
            print(f"[ERROR] case={case_name}: {exc!r}", flush=True)

    return csv_path


# =============================================================================
# Plotting
# =============================================================================
def _load_valid_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load valid rows and support CSV files created by the v3 script.

    The v3 CSV accidentally omitted the ``method`` column. For those files,
    infer the method from the circuit count:
      * shot:       total circuits == padded dimension
      * tomography: total bases    == 3 ** target_qubits
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                row["matrix_size"] = int(row["matrix_size"])
                row["number_of_buses"] = int(row["number_of_buses"])
                row["target_qubits"] = int(row["target_qubits"])
                row["padded_dim"] = int(row["padded_dim"])
                row["total_measurement_bases_or_circuits"] = int(
                    row["total_measurement_bases_or_circuits"]
                )
                row["state_angle_degrees"] = float(row["state_angle_degrees"])
                row["absolute_error"] = float(row["absolute_error"])
                row["runtime_seconds"] = float(row["runtime_seconds"])

                method = str(row.get("method", "")).strip().lower()
                if method not in {"shot", "tomography"}:
                    total = row["total_measurement_bases_or_circuits"]
                    if total == row["padded_dim"]:
                        method = "shot"
                    elif total == 3 ** row["target_qubits"]:
                        method = "tomography"
                    else:
                        raise ValueError("Cannot infer HHL recovery method")
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


def plot_results(
    csv_path: Path,
    output_dir: Path,
    latest_only: bool = True,
) -> tuple[Path, Path, Path]:
    """Plot state angle, absolute error, and runtime by network case."""
    import matplotlib.pyplot as plt

    rows = _load_valid_rows(csv_path)
    if latest_only:
        rows = _latest_rows_by_method_and_case(rows)
    if not rows:
        raise RuntimeError("CSV does not contain valid result rows.")

    output_dir.mkdir(parents=True, exist_ok=True)
    case_names = [name for name in SUPPORTED_CASES if any(r["case_name"] == name for r in rows)]
    methods = [m for m in ["shot", "tomography"] if any(r["method"] == m for r in rows)]
    x_positions = np.arange(len(case_names))

    by_method_case = {(r["method"], r["case_name"]): r for r in rows}
    tick_labels: list[str] = []
    for case_name in case_names:
        row = next(r for r in rows if r["case_name"] == case_name)
        tick_labels.append(
            f"{case_name}\n{row['number_of_buses']} bus / {row['matrix_size']}x{row['matrix_size']}"
        )

    def create_plot(metric: str, ylabel: str, title: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for method in methods:
            values = [
                np.nan
                if (method, case_name) not in by_method_case
                else float(by_method_case[(method, case_name)][metric])
                for case_name in case_names
            ]
            ax.plot(x_positions, values, marker="o", label=method)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Standard power-system case and reduced DC matrix size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    angle_path = create_plot(
        "state_angle_degrees",
        "State angle (degrees, lower is better)",
        "DC power-flow HHL recovery accuracy",
        "dc_state_angle_comparison.png",
    )
    error_path = create_plot(
        "absolute_error",
        "Absolute error ||x_HHL - x_classical||",
        "DC power-flow HHL absolute-error comparison",
        "dc_absolute_error_comparison.png",
    )
    runtime_path = create_plot(
        "runtime_seconds",
        "HHL module runtime (seconds)",
        "DC power-flow HHL runtime comparison",
        "dc_runtime_comparison.png",
    )
    return angle_path, error_path, runtime_path


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HHL shot recovery and tomography on reduced DC power-flow "
            "systems from small standard MATPOWER/PYPOWER-derived cases."
        )
    )
    parser.add_argument(
        "--method",
        choices=["shot", "tomography", "both"],
        default="shot",
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
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
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
        methods = ["shot", "tomography"] if args.method == "both" else [args.method]
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
            angle_plot, error_plot, runtime_plot = plot_results(csv_path, output_dir)
            print("=" * 88)
            print(f"[PLOT] State angle:   {angle_plot}")
            print(f"[PLOT] Absolute error:{error_plot}")
            print(f"[PLOT] Runtime:       {runtime_plot}")
        except Exception as exc:
            print(f"[PLOT ERROR] {exc!r}")


if __name__ == "__main__":
    main()