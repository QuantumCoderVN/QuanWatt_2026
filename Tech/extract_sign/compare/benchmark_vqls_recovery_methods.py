"""
Benchmark comparing two solution recovery methods:
    1. VQLS shot_recovery
    2. VQLS tomography

Objectives:
    - Run linear systems A x = b with original matrix sizes from 2x2 to 6x6.
    - For sizes that are not 2^n, automatically pad/embed them to size 2^ceil(log2(m)).
    - Save the absolute error between the classical solution and the recovered quantum solution.
    - Save the running time of the quantum module.
    - Allow running shot_recovery first, tomography afterward, then plot comparison charts.

Example usage:
    # Run VQLS shot recovery first
    python benchmark_vqls_recovery_methods.py --method shot --shots 100000 --steps 150

    # Then run VQLS tomography
    python benchmark_vqls_recovery_methods.py --method tomography --shots 100000 --steps 150

    # Replot the charts from the saved results
    python benchmark_vqls_recovery_methods.py --method plot

Important notes:
    - The current VQLS/vqls code requires matrices of size 2^n x 2^n.
    - Therefore, a 3x3 matrix will be embedded into 4x4, while 5x5 and 6x6 matrices will be embedded into 8x8.
    - When comparing solutions, only the first m elements corresponding to the original m x m system are used.
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Result file configuration
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = CURRENT_DIR / "vqls_recovery_benchmark_results.csv"
RESULTS_JSON = CURRENT_DIR / "vqls_recovery_benchmark_results.json"
ACCURACY_PLOT = CURRENT_DIR / "vqls_recovery_accuracy_comparison.png"
RUNTIME_PLOT = CURRENT_DIR / "vqls_recovery_runtime_comparison.png"

# Ensure Python can import the two refactored solver files if they are in the same directory.
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# ============================================================
# CLASSICAL MODULE
# ============================================================
def classical_solver(A, b):
    """
    Solve the linear system Ax = b using a classical method.

    If A is singular/nearly singular, use least squares.
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


# ============================================================
# Create or receive the matrix test set
# ============================================================
def make_spd_test_matrix(size, seed=0):
    """
    Create a symmetric positive definite matrix of size size x size.

    You can replace this function with your own matrix.
    The important point is that A should be invertible so the classical solution is stable.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(size, size))

    # A is SPD, with a more moderate condition number than a fully random dense matrix.
    A = M.T @ M + size * np.eye(size)
    b = rng.normal(size=size)

    return A.astype(complex), b.astype(complex)


def build_matrix_cases(sizes=range(2, 7), seed=0):
    """
    Create the benchmark problem list from 2x2 to 6x6.

    If you want to use your own matrices, modify this function as follows:

        return [
            {"name": "case_2", "A": A2, "b": b2},
            {"name": "case_3", "A": A3, "b": b3},
            ...
        ]
    """
    cases = []

    for size in sizes:
        A, b = make_spd_test_matrix(size=size, seed=seed + size)
        cases.append(
            {
                "name": f"random_spd_{size}x{size}",
                "matrix_size": size,
                "A": A,
                "b": b,
            }
        )

    return cases


# ============================================================
# Pad any m x m matrix to 2^n x 2^n
# ============================================================
def next_power_of_two(m):
    """Return dim = 2^ceil(log2(m))."""
    if m <= 0:
        raise ValueError("m must be positive.")
    return 1 << int(math.ceil(math.log2(m)))


def embed_linear_system_to_power_of_two(A, b, padding_diagonal=1.0):
    """
    Embed the original system A_m x = b_m into a system of size 2^n:

        [ A_m   0 ] [x_m]   [b_m]
        [  0    I ] [x_p] = [ 0 ]

    Then the padding solution x_p = 0, and the original solution is in the first m elements.
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    m = A.shape[0]

    if b.ndim != 1 or b.shape[0] != m:
        raise ValueError("b must be a 1D vector with the same number of rows as A.")

    dim = next_power_of_two(m)

    A_embed = padding_diagonal * np.eye(dim, dtype=complex)
    b_embed = np.zeros(dim, dtype=complex)

    A_embed[:m, :m] = A
    b_embed[:m] = b

    n_qubits = int(math.log2(dim))

    return A_embed, b_embed, dim, n_qubits


# ============================================================
# VQLS shot recovery MODULE
# ============================================================
def vqls_shot_recovery_module(A, b, shots, steps, rng_seed, backend=None):
    """
    Call the refactored VQLS shot_recovery module from file:
        vqls_two_modules_refactored.py
    """
    from vqls_two_modules_refactored import vqls_solver

    result = vqls_solver(
        A,
        b,
        shots=shots,
        steps=steps,
        rng_seed=rng_seed,
        backend=backend,
        return_details=True,
    )

    return result["x_vqls"], result


# ============================================================
# VQLS tomography MODULE
# ============================================================
def vqls_tomography_module(A, b, shots, steps, rng_seed, backend=None):
    """
    Call the refactored vqls_tomography module from file:
        vqls_tomography_two_modules_refactored.py

    The original function name is vqls_tomography based on the previous requirement, but in this benchmark
    the method label is set to vqls_tomography to match the comparison objective.
    """
    from vqls_tomography_two_modules_refactored import vqls_tomography

    result = vqls_tomography(
        A,
        b,
        shots=shots,
        steps=steps,
        rng_seed=rng_seed,
        backend=backend,
        return_details=True,
    )

    return result["x_vqls_tomography"], result


# ============================================================
# Run one benchmark case
# ============================================================
def run_one_case(case, method, shots, steps, rng_seed, backend=None):
    """
    Run one matrix A, b with one quantum recovery method.

    Returns:
        result dict to save to CSV/JSON.
    """
    name = case.get("name", "unnamed_case")
    A_original = np.asarray(case["A"], dtype=complex)
    b_original = np.asarray(case["b"], dtype=complex)
    matrix_size = int(case.get("matrix_size", A_original.shape[0]))

    A_embed, b_embed, embedded_dim, n_qubits = embed_linear_system_to_power_of_two(
        A_original,
        b_original,
    )

    x_classical = classical_solver(A_original, b_original)

    if method == "shot":
        method_label = "vqls_shot_recovery"
        solver = vqls_shot_recovery_module
    elif method == "tomography":
        method_label = "vqls_tomography"
        solver = vqls_tomography_module
    else:
        raise ValueError("method must be 'shot' or 'tomography'.")

    started = time.perf_counter()
    status = "ok"
    error_message = ""
    x_quantum_original = None
    quantum_details = None

    try:
        x_quantum_padded, quantum_details = solver(
            A_embed,
            b_embed,
            shots=shots,
            steps=steps,
            rng_seed=rng_seed,
            backend=backend,
        )
        runtime_seconds = time.perf_counter() - started
        x_quantum_original = np.asarray(x_quantum_padded[:matrix_size], dtype=complex)
    except Exception as exc:
        runtime_seconds = time.perf_counter() - started
        status = "failed"
        error_message = repr(exc)

    if status == "ok":
        abs_error = float(np.linalg.norm(x_quantum_original - x_classical))
        rel_error = float(abs_error / max(np.linalg.norm(x_classical), 1e-15))
        residual = float(np.linalg.norm(A_original @ x_quantum_original - b_original))
        accuracy_neg_log10 = float(-np.log10(abs_error + 1e-16))
    else:
        abs_error = np.nan
        rel_error = np.nan
        residual = np.nan
        accuracy_neg_log10 = np.nan

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case_name": name,
        "method": method_label,
        "matrix_size": matrix_size,
        "embedded_dim": embedded_dim,
        "n_qubits": n_qubits,
        "shots": int(shots),
        "steps": int(steps),
        "rng_seed": int(rng_seed),
        "abs_error_l2": abs_error,
        "rel_error_l2": rel_error,
        "accuracy_neg_log10_abs_error": accuracy_neg_log10,
        "residual_l2": residual,
        "runtime_seconds": float(runtime_seconds),
        "status": status,
        "error_message": error_message,
    }

    # Save some vectors for debugging in JSON, but do not include them in CSV to keep it compact.
    row_json_extra = {
        "x_classical_real_if_close": np.real_if_close(x_classical).tolist(),
        "x_quantum_real_if_close": (
            None if x_quantum_original is None else np.real_if_close(x_quantum_original).tolist()
        ),
    }

    return row, row_json_extra, quantum_details


# ============================================================
# Save benchmark results
# ============================================================
def append_rows_to_csv(rows, csv_path=RESULTS_CSV):
    """Append results to CSV."""
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    file_exists = Path(csv_path).exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def append_rows_to_json(rows_with_extra, json_path=RESULTS_JSON):
    """Append more complete results to JSON."""
    json_path = Path(json_path)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.extend(rows_with_extra)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Run multiple benchmark cases
# ============================================================
def run_benchmark(method, matrix_cases=None, shots=10**6, steps=150, rng_seed=0, backend=None):
    """
    Run the benchmark for one method.

    method:
        'shot' or 'tomography'
    """
    if matrix_cases is None:
        matrix_cases = build_matrix_cases(sizes=range(2, 7), seed=rng_seed)

    rows = []
    rows_with_extra = []

    for case in matrix_cases:
        print("\n" + "=" * 80)
        print(f"Running method={method} | case={case.get('name', 'unnamed')} | size={case['A'].shape[0]}x{case['A'].shape[1]}")
        print("=" * 80)

        row, extra, _details = run_one_case(
            case=case,
            method=method,
            shots=shots,
            steps=steps,
            rng_seed=rng_seed,
            backend=backend,
        )

        rows.append(row)
        row_full = dict(row)
        row_full.update(extra)
        rows_with_extra.append(row_full)

        print(f"status        = {row['status']}")
        print(f"matrix_size   = {row['matrix_size']}x{row['matrix_size']}")
        print(f"embedded_dim  = {row['embedded_dim']} -> n_qubits = {row['n_qubits']}")
        print(f"abs_error_l2  = {row['abs_error_l2']}")
        print(f"rel_error_l2  = {row['rel_error_l2']}")
        print(f"runtime_sec   = {row['runtime_seconds']:.6f}")
        if row["status"] != "ok":
            print(f"error_message = {row['error_message']}")

    append_rows_to_csv(rows)
    append_rows_to_json(rows_with_extra)

    print("\nSaved results:")
    print(f"  CSV : {RESULTS_CSV}")
    print(f"  JSON: {RESULTS_JSON}")

    return rows


# ============================================================
# Read the latest results for plotting
# ============================================================
def load_latest_successful_results(csv_path=RESULTS_CSV):
    """
    Read the CSV and get the latest record for each pair:
        (method, matrix_size)
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"No result file found: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    latest = {}

    for row in all_rows:
        if row.get("status") != "ok":
            continue

        key = (row["method"], int(row["matrix_size"]))
        latest[key] = row

    rows = []
    for row in latest.values():
        parsed = dict(row)
        parsed["matrix_size"] = int(parsed["matrix_size"])
        parsed["embedded_dim"] = int(parsed["embedded_dim"])
        parsed["n_qubits"] = int(parsed["n_qubits"])
        parsed["abs_error_l2"] = float(parsed["abs_error_l2"])
        parsed["rel_error_l2"] = float(parsed["rel_error_l2"])
        parsed["accuracy_neg_log10_abs_error"] = float(parsed["accuracy_neg_log10_abs_error"])
        parsed["residual_l2"] = float(parsed["residual_l2"])
        parsed["runtime_seconds"] = float(parsed["runtime_seconds"])
        rows.append(parsed)

    rows.sort(key=lambda r: (r["method"], r["matrix_size"]))
    return rows


# ============================================================
# Plot comparison charts
# ============================================================
def plot_benchmark_results(csv_path=RESULTS_CSV):
    """
    Plot 2 charts:
        1. Accuracy: accuracy = -log10(abs_error_l2 + 1e-16)
           Higher values are better.
        2. Running time by the number of embedded qubits.
    """
    rows = load_latest_successful_results(csv_path)

    if not rows:
        raise ValueError("No successful results available for plotting.")

    method_order = ["vqls_shot_recovery", "vqls_tomography"]
    method_labels = {
        "vqls_shot_recovery": "VQLS shot recovery",
        "vqls_tomography": "VQLS tomography",
    }

    # ---------------- Accuracy plot ----------------
    plt.figure(figsize=(8, 5))

    for method in method_order:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue

        method_rows.sort(key=lambda r: r["matrix_size"])
        x = [r["n_qubits"] for r in method_rows]
        y = [r["accuracy_neg_log10_abs_error"] for r in method_rows]

        plt.plot(x, y, marker="o", linewidth=2, label=method_labels[method])

        for r in method_rows:
            plt.annotate(
                f"{r['matrix_size']}x{r['matrix_size']}",
                (r["n_qubits"], r["accuracy_neg_log10_abs_error"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    plt.xlabel("Number of embedded qubits n, with dim = 2^n")
    plt.ylabel("Accuracy = -log10(||x_quantum - x_classical||₂)")
    plt.title("Solution recovery accuracy comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    # ---------------- Runtime plot ----------------
    plt.figure(figsize=(8, 5))

    for method in method_order:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue

        method_rows.sort(key=lambda r: r["matrix_size"])
        x = [r["n_qubits"] for r in method_rows]
        y = [r["runtime_seconds"] for r in method_rows]

        plt.plot(x, y, marker="o", linewidth=2, label=method_labels[method])

        for r in method_rows:
            plt.annotate(
                f"{r['matrix_size']}x{r['matrix_size']}",
                (r["n_qubits"], r["runtime_seconds"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    plt.xlabel("Number of embedded qubits n, with dim = 2^n")
    plt.ylabel("Running time of the quantum VQLS module (seconds)")
    plt.title("Running time comparison of the two solution recovery methods")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUNTIME_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plots:")
    print(f"  Accuracy: {ACCURACY_PLOT}")
    print(f"  Runtime : {RUNTIME_PLOT}")

    return ACCURACY_PLOT, RUNTIME_PLOT


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark VQLS shot recovery vs VQLS tomography."
    )

    parser.add_argument(
        "--method",
        choices=["shot", "tomography", "both", "plot"],
        default="shot",
        help="Choose the method to run: shot, tomography, both, or plot.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10**6,
        help="Number of shots for each circuit.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Number of COBYLA optimization iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for test matrices and parameter initialization.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.method == "shot":
        run_benchmark(
            method="shot",
            shots=args.shots,
            steps=args.steps,
            rng_seed=args.seed,
        )
        plot_benchmark_results()

    elif args.method == "tomography":
        run_benchmark(
            method="tomography",
            shots=args.shots,
            steps=args.steps,
            rng_seed=args.seed,
        )
        plot_benchmark_results()

    elif args.method == "both":
        run_benchmark(
            method="shot",
            shots=args.shots,
            steps=args.steps,
            rng_seed=args.seed,
        )
        run_benchmark(
            method="tomography",
            shots=args.shots,
            steps=args.steps,
            rng_seed=args.seed,
        )
        plot_benchmark_results()

    elif args.method == "plot":
        plot_benchmark_results()


if __name__ == "__main__":
    main()