"""
Benchmark two HHL solution recovery methods:
  1) HHL shot-recovery: module hhl_shot_recovery_modules.py
  2) HHL tomography:     module hhl_tomography_modules.py

Objectives:
  - Generate linear systems Ax=b with sizes from 2x2 to 6x6.
  - Run each HHL method separately.
  - Save the absolute error ||x_HHL - x_classical|| and the HHL module runtime.
  - Plot 2 comparison charts:
      + Absolute error by the number of target qubits corresponding to the matrix size.
      + Running time by the number of target qubits corresponding to the matrix size.

Recommended usage:

  # 1) Run shot-recovery first
  python benchmark_hhl_recovery_methods.py --method shot --phase-qubits 4 --shots 20000

  # 2) Then run tomography, reusing the same matrices generated in step 1
  python benchmark_hhl_recovery_methods.py --method tomography --phase-qubits 4 --tomo-shots 20000

  # 3) Only replot the charts from the existing CSV
  python benchmark_hhl_recovery_methods.py --plot-only

Default output files are stored in:
  benchmark_outputs/

Notes:
  - HHL requires the dimension to be 2^n. For 3x3, 5x5, and 6x6 matrices,
    the current solvers will auto_pad to dimension 4 or 8. Therefore, the number
    of target qubits is ceil(log2(size)).
  - Tomography requires 3^n measurement circuits, so for size 5x5/6x6 after padding,
    n=3 target qubits are used, requiring 27 tomography bases. Start with small shots
    for testing.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# Output configuration
# =============================================================================
DEFAULT_OUTPUT_DIR = Path("benchmark_outputs")
DEFAULT_SYSTEM_DIR_NAME = "systems"
DEFAULT_CSV_NAME = "hhl_recovery_benchmark_results.csv"


# =============================================================================
# Generate / load the linear system Ax=b
# =============================================================================
def target_qubits_for_size(matrix_size: int) -> int:
    """Number of target qubits required to encode the size x size matrix after padding."""
    if matrix_size < 2:
        raise ValueError("matrix_size must be >= 2.")
    return int(math.ceil(math.log2(matrix_size)))


def padded_dim_for_size(matrix_size: int) -> int:
    """Dimension after padding to 2^n."""
    return 1 << target_qubits_for_size(matrix_size)


def generate_spd_system(
    matrix_size: int,
    seed: int,
    condition_number: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a real symmetric positive definite matrix A and a real vector b.

    A is generated in a stable way suitable for HHL demos:
      A = Q diag(lambda) Q^T,
    where lambda lies in the range [1, condition_number].

    The current HHL modules use auto_scale=True, so lambda_max < 2π is not strictly required,
    but this range still keeps the problem moderate and easy to compare.
    """
    rng = np.random.default_rng(seed)

    random_matrix = rng.normal(size=(matrix_size, matrix_size))
    q_matrix, _ = np.linalg.qr(random_matrix)

    eigenvalues = np.linspace(1.0, condition_number, matrix_size)
    A = q_matrix @ np.diag(eigenvalues) @ q_matrix.T
    A = 0.5 * (A + A.T)  # enforce numerical symmetry

    b = rng.normal(size=matrix_size)
    if np.linalg.norm(b) == 0:
        b[0] = 1.0

    return A.astype(np.complex128), b.astype(np.complex128)


def system_path(system_dir: Path, matrix_size: int) -> Path:
    return system_dir / f"system_{matrix_size}x{matrix_size}.npz"


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
    metadata_raw = str(data["metadata"])
    metadata = json.loads(metadata_raw)
    return A, b, metadata


def prepare_systems(
    sizes: list[int],
    system_dir: Path,
    base_seed: int,
    condition_number: float,
    regenerate: bool,
) -> dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Create or load fixed Ax=b systems so both methods run on the same data."""
    systems: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    system_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        path = system_path(system_dir, size)

        if path.exists() and not regenerate:
            A, b, metadata = load_system(path)
        else:
            system_seed = base_seed + size * 1009
            A, b = generate_spd_system(
                matrix_size=size,
                seed=system_seed,
                condition_number=condition_number,
            )
            metadata = {
                "matrix_size": size,
                "system_seed": system_seed,
                "condition_number_requested": condition_number,
                "target_qubits": target_qubits_for_size(size),
                "padded_dim": padded_dim_for_size(size),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_system(path, A, b, metadata)

        systems[size] = (A, b, metadata)

    return systems


# =============================================================================
# Import solver modules
# =============================================================================
def import_solver_module(module_name: str):
    """Import the solver module from the same directory as this script or from PYTHONPATH."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return importlib.import_module(module_name)


def classical_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shared classical module for computing the reference solution."""
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
    """Run one HHL method and measure runtime only around the HHL module."""
    method = method.lower().strip()

    if method == "shot":
        module = import_solver_module("hhl_shot_recovery_modules")
        solver = module.hhl_solver

        start = time.perf_counter()
        x_hhl, info = solver(
            A,
            b,
            phase_qubits=phase_qubits,
            shots=shots,
            seed=seed,
            return_info=True,
        )
        runtime = time.perf_counter() - start
        return x_hhl, info, runtime

    if method == "tomography":
        module = import_solver_module("hhl_tomography_modules")
        solver = module.hhl_tomography_solver

        start = time.perf_counter()
        x_hhl, info = solver(
            A,
            b,
            phase_qubits=phase_qubits,
            tomography_shots=tomo_shots,
            seed=seed,
            progress=progress,
            return_info=True,
        )
        runtime = time.perf_counter() - start
        return x_hhl, info, runtime

    raise ValueError("method must be 'shot' or 'tomography'.")


# =============================================================================
# Write CSV results
# =============================================================================
def ensure_csv_header(csv_path: Path, fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_result(csv_path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    ensure_csv_header(csv_path, fieldnames)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def run_benchmark(
    method: str,
    sizes: list[int],
    output_dir: Path,
    base_seed: int,
    condition_number: float,
    regenerate_systems: bool,
    phase_qubits: int,
    shots: int,
    tomo_shots: int,
    progress: bool,
) -> Path:
    """Run the benchmark for the selected method and save the CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    system_dir = output_dir / DEFAULT_SYSTEM_DIR_NAME
    csv_path = output_dir / DEFAULT_CSV_NAME

    systems = prepare_systems(
        sizes=sizes,
        system_dir=system_dir,
        base_seed=base_seed,
        condition_number=condition_number,
        regenerate=regenerate_systems,
    )

    fieldnames = [
        "timestamp",
        "method",
        "matrix_size",
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
        "fidelity_direction",
        "runtime_seconds",
        "postselection_rate",
        "success_count_summary",
        "system_file",
    ]

    for size in sizes:
        A, b, metadata = systems[size]
        n_target = target_qubits_for_size(size)
        padded_dim = padded_dim_for_size(size)
        run_seed = base_seed + size * 7919

        print("=" * 80, flush=True)
        print(
            f"[BENCH] method={method}, size={size}x{size}, "
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

            abs_error = float(np.linalg.norm(x_hhl - x_classical))
            rel_error = float(abs_error / np.linalg.norm(x_classical))
            residual_hhl = float(np.linalg.norm(A @ x_hhl - b))
            residual_classical = float(np.linalg.norm(A @ x_classical - b))

            xh_norm = x_hhl / np.linalg.norm(x_hhl)
            xc_norm = x_classical / np.linalg.norm(x_classical)
            fidelity_direction = float(abs(np.vdot(xh_norm, xc_norm)) ** 2)

            if method == "shot":
                shots_per_circuit = shots
                # 1 main HHL circuit + (padded_dim - 1) sign-detection circuits.
                total_circuits = 1 + (padded_dim - 1)
                postselection_rate = info.get("postselection_rate", "")
                success_summary = str(info.get("success_count", ""))
            else:
                shots_per_circuit = tomo_shots
                # Full tomography on n target qubits: 3^n measurement bases.
                total_circuits = 3**n_target
                success_by_basis = info.get("success_count_by_basis", {})
                if success_by_basis:
                    values = list(success_by_basis.values())
                    postselection_rate = float(np.mean(values) / tomo_shots)
                    success_summary = f"min={min(values)}, mean={np.mean(values):.2f}, max={max(values)}"
                else:
                    postselection_rate = ""
                    success_summary = ""

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "method": method,
                "matrix_size": size,
                "target_qubits": n_target,
                "padded_dim": padded_dim,
                "phase_qubits": phase_qubits,
                "shots_per_circuit": shots_per_circuit,
                "total_measurement_bases_or_circuits": total_circuits,
                "seed": run_seed,
                "absolute_error": abs_error,
                "relative_error": rel_error,
                "residual_hhl": residual_hhl,
                "residual_classical": residual_classical,
                "fidelity_direction": fidelity_direction,
                "runtime_seconds": runtime,
                "postselection_rate": postselection_rate,
                "success_count_summary": success_summary,
                "system_file": str(system_path(system_dir, size)),
            }

            append_result(csv_path, row, fieldnames)

            print(f"[DONE] size={size}x{size}", flush=True)
            print(f"       absolute_error = {abs_error:.6e}", flush=True)
            print(f"       runtime        = {runtime:.3f} s", flush=True)
            print(f"       csv            = {csv_path}", flush=True)

        except Exception as exc:
            # Still write an error row to record which size failed.
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "method": method,
                "matrix_size": size,
                "target_qubits": n_target,
                "padded_dim": padded_dim,
                "phase_qubits": phase_qubits,
                "shots_per_circuit": shots if method == "shot" else tomo_shots,
                "total_measurement_bases_or_circuits": "",
                "seed": run_seed,
                "absolute_error": "ERROR",
                "relative_error": "ERROR",
                "residual_hhl": "ERROR",
                "residual_classical": "ERROR",
                "fidelity_direction": "ERROR",
                "runtime_seconds": "ERROR",
                "postselection_rate": "",
                "success_count_summary": repr(exc),
                "system_file": str(system_path(system_dir, size)),
            }
            append_result(csv_path, row, fieldnames)
            print(f"[ERROR] size={size}x{size}: {exc!r}", flush=True)

    return csv_path


# =============================================================================
# Plot charts
# =============================================================================
def _load_valid_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["matrix_size"] = int(row["matrix_size"])
                row["target_qubits"] = int(row["target_qubits"])
                row["absolute_error"] = float(row["absolute_error"])
                row["runtime_seconds"] = float(row["runtime_seconds"])
                row["relative_error"] = float(row["relative_error"])
                row["fidelity_direction"] = float(row["fidelity_direction"])
            except Exception:
                continue
            rows.append(row)
    return rows


def _latest_rows_by_method_and_size(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["matrix_size"]))
        if key not in latest or str(row["timestamp"]) >= str(latest[key]["timestamp"]):
            latest[key] = row
    return list(latest.values())


def plot_results(csv_path: Path, output_dir: Path, latest_only: bool = True) -> tuple[Path, Path]:
    """Plot 2 charts: absolute error and runtime."""
    import matplotlib.pyplot as plt

    rows = _load_valid_rows(csv_path)
    if latest_only:
        rows = _latest_rows_by_method_and_size(rows)

    if not rows:
        raise RuntimeError("CSV does not contain any valid result rows for plotting.")

    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(row["matrix_size"]) for row in rows})
    methods = [m for m in ["shot", "tomography"] if any(row["method"] == m for row in rows)]

    # Use discrete positions to avoid overlapping points when size 3 and 4 both use 2 qubits,
    # and size 5 and 6 both use 3 qubits. Tick labels still show the corresponding qubit count.
    x_positions = np.arange(len(sizes))
    tick_labels = [f"{target_qubits_for_size(s)}q\n{s}x{s}" for s in sizes]

    by_method_size: dict[tuple[str, int], dict[str, Any]] = {
        (row["method"], int(row["matrix_size"])): row for row in rows
    }

    # Chart 1: absolute error.
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    for method in methods:
        y_values = []
        for size in sizes:
            row = by_method_size.get((method, size))
            y_values.append(np.nan if row is None else float(row["absolute_error"]))
        ax1.plot(x_positions, y_values, marker="o", label=method)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel("Number of target qubits corresponding to matrix size")
    ax1.set_ylabel("Absolute error ||x_HHL - x_classical||")
    ax1.set_title("HHL solution recovery accuracy comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()

    accuracy_plot_path = output_dir / "hhl_recovery_absolute_error_comparison.png"
    fig1.savefig(accuracy_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Chart 2: runtime.
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for method in methods:
        y_values = []
        for size in sizes:
            row = by_method_size.get((method, size))
            y_values.append(np.nan if row is None else float(row["runtime_seconds"]))
        ax2.plot(x_positions, y_values, marker="o", label=method)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel("Number of target qubits corresponding to matrix size")
    ax2.set_ylabel("HHL module running time (seconds)")
    ax2.set_title("Running time comparison of the two HHL solution recovery methods")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()

    runtime_plot_path = output_dir / "hhl_recovery_runtime_comparison.png"
    fig2.savefig(runtime_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    return accuracy_plot_path, runtime_plot_path


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HHL shot-recovery and HHL tomography from size 2x2 to 6x6."
    )

    parser.add_argument(
        "--method",
        choices=["shot", "tomography", "both"],
        default="shot",
        help="Choose the method to run. It is recommended to run shot first, then tomography.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="List of matrix sizes to benchmark.",
    )
    parser.add_argument(
        "--phase-qubits",
        type=int,
        default=4,
        help="Number of phase qubits in QPE.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=20_000,
        help="Shots for each circuit of HHL shot-recovery.",
    )
    parser.add_argument(
        "--tomo-shots",
        type=int,
        default=20_000,
        help="Shots for each tomography basis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base seed for generating systems and running the simulator.",
    )
    parser.add_argument(
        "--condition-number",
        type=float,
        default=3.0,
        help="Approximate condition number of the generated SPD matrices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saving systems, CSV, and PNG files.",
    )
    parser.add_argument(
        "--regenerate-systems",
        action="store_true",
        help="Regenerate all matrices. Do not use this option if you want tomography to reuse the matrices from shot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not plot charts after running the benchmark.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Do not run HHL; only replot charts from the existing CSV.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print more detailed progress, useful for tomography.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    csv_path = output_dir / DEFAULT_CSV_NAME

    if not args.plot_only:
        methods = ["shot", "tomography"] if args.method == "both" else [args.method]
        for method in methods:
            csv_path = run_benchmark(
                method=method,
                sizes=args.sizes,
                output_dir=output_dir,
                base_seed=args.seed,
                condition_number=args.condition_number,
                regenerate_systems=args.regenerate_systems,
                phase_qubits=args.phase_qubits,
                shots=args.shots,
                tomo_shots=args.tomo_shots,
                progress=args.progress,
            )

    if not args.no_plot:
        try:
            error_plot, runtime_plot = plot_results(csv_path=csv_path, output_dir=output_dir)
            print("=" * 80, flush=True)
            print(f"[PLOT] Saved error plot:   {error_plot}", flush=True)
            print(f"[PLOT] Saved runtime plot: {runtime_plot}", flush=True)
        except Exception as exc:
            print(f"[PLOT ERROR] Could not plot charts: {exc!r}", flush=True)


if __name__ == "__main__":
    main()