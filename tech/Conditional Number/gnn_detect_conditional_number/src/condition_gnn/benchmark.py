from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .data import GraphSample, collate_graphs, load_split
from .inference import _inference_sample, load_trained_model
from .gershgorin import gershgorin_kappa2
from .labels import two_norm_value
from .metrics import condition_number_metrics
from .train import resolve_device


def sample_matrix(sample: GraphSample) -> sp.csr_matrix:
    rows = sample.edge_index[0].numpy()
    cols = sample.edge_index[1].numpy()
    values = sample.edge_value.numpy().astype(np.float64)
    n = sample.x.shape[0]
    return sp.csr_matrix((values, (rows, cols)), shape=(n, n))


def hager_higham_kappa1(matrix: sp.spmatrix) -> float:
    a = matrix.tocsc().astype(np.float64)
    lu = spla.splu(a)
    inverse = spla.LinearOperator(
        a.shape,
        matvec=lu.solve,
        rmatvec=lambda x: lu.solve(x, trans="T"),
        matmat=lu.solve,
        rmatmat=lambda x: lu.solve(x, trans="T"),
        dtype=np.float64,
    )
    return float(spla.norm(a, ord=1) * spla.onenormest(inverse))


def lanczos_kappa2(
    matrix: sp.spmatrix,
    iterations: int,
    seed: int,
    device: torch.device,
    matrix_type: str = "spd",
) -> float:
    """Approximate kappa_2 with Torch LOBPCG.

    For SPD matrices this estimates eigenvalue ratio directly. For general
    matrices it estimates the eigenvalue ratio of A.T @ A and takes a square root.
    """
    working = matrix if matrix_type == "spd" else (matrix.T @ matrix).tocsr()
    coo = working.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).to(device=device)
    values = torch.from_numpy(coo.data).to(device=device, dtype=torch.float64)
    sparse = torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randn(
        (working.shape[0], 1), generator=generator, device=device, dtype=torch.float64
    )
    largest = torch.lobpcg(
        sparse, k=1, X=initial.clone(), niter=iterations, largest=True
    )[0][0]
    smallest = torch.lobpcg(
        sparse, k=1, X=initial.clone(), niter=iterations, largest=False
    )[0][0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ratio = float((largest / smallest).item())
    if matrix_type == "general":
        ratio = float(np.sqrt(ratio))
    return ratio


def exact_condition_number(
    matrix: sp.spmatrix,
    norm: int,
    device: torch.device,
) -> float:
    dense = torch.from_numpy(matrix.toarray()).to(device=device, dtype=torch.float64)
    value = torch.linalg.cond(dense, p=norm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(value.item())


def torch_hager_higham_kappa1(
    matrix: sp.spmatrix,
    device: torch.device,
    max_iterations: int = 10,
) -> float:
    """Classic Hager inverse 1-norm iteration using Torch dense LU solves."""
    a = torch.from_numpy(matrix.toarray()).to(device=device, dtype=torch.float64)
    lu, pivots = torch.linalg.lu_factor(a)
    n = a.shape[0]
    x = torch.full((n, 1), 1.0 / n, device=device, dtype=torch.float64)
    estimate = 0.0
    previous_index = -1
    for _ in range(max_iterations):
        y = torch.linalg.lu_solve(lu, pivots, x)
        estimate = max(estimate, float(torch.linalg.vector_norm(y, ord=1).item()))
        signs = torch.sign(y)
        signs[signs == 0] = 1.0
        z = torch.linalg.lu_solve(lu, pivots, signs, left=True, adjoint=True)
        index = int(torch.argmax(torch.abs(z)).item())
        if index == previous_index or float(torch.abs(z[index]).item()) <= float((z.T @ x).item()):
            break
        x.zero_()
        x[index] = 1.0
        previous_index = index
    norm_1 = float(torch.linalg.matrix_norm(a, ord=1).item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return norm_1 * estimate


def _timed(call: Callable[[], float], repeats: int) -> tuple[float, float]:
    durations: list[float] = []
    value = float("nan")
    for _ in range(repeats):
        start = time.perf_counter()
        value = float(call())
        durations.append(1000.0 * (time.perf_counter() - start))
    return value, float(np.mean(durations))


@torch.no_grad()
def _gnn_prediction(
    matrix: sp.spmatrix,
    model: torch.nn.Module,
    scheme: int,
    norm: int,
    device: torch.device,
    matrix_type: str = "general",
) -> float:
    graph = collate_graphs([_inference_sample(matrix)]).to(device)
    raw = model(graph)
    log_kappa = raw
    if scheme == 1:
        if norm == 1:
            forward_norm = float(spla.norm(matrix, ord=1))
        else:
            forward_norm = two_norm_value(matrix)
        log_kappa = log_kappa + np.log10(forward_norm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(10.0 ** log_kappa.item())


def _row(
    norm: int,
    method: str,
    truths: np.ndarray,
    estimates: np.ndarray,
    durations: np.ndarray,
) -> dict[str, Any]:
    metrics = condition_number_metrics(truths, estimates)
    return {
        "norm": norm,
        "method": method,
        "time_mean_ms": float(durations.mean()),
        "time_std_ms": float(durations.std()),
        "lre_mean_percent": metrics["paper_lre_mean_percent"],
        "lre_max_percent": metrics["paper_lre_max_percent"],
        "lre_below_0_5_percent": metrics["paper_lre_below_0_5_percent"],
        "lre_below_1_percent": metrics["paper_lre_below_1_percent"],
        "relative_error_mean": metrics["relative_error_mean"],
        "relative_error_median": metrics["relative_error_median"],
        "relative_error_p95": metrics["relative_error_p95"],
        "relative_error_below_0_5_percent": metrics["relative_error_below_0_5_percent"],
        "accuracy_count_below_0_5": metrics["accuracy_count_below_0_5"],
        "sample_count": metrics["sample_count"],
        "finite_prediction_count": metrics["finite_prediction_count"],
        "finite_prediction_percent": metrics["finite_prediction_percent"],
        "accuracy_mean": metrics["accuracy_mean"],
        "accuracy_median": metrics["accuracy_median"],
        "accuracy_min": metrics["accuracy_min"],
        "accuracy_mean_percent": metrics["accuracy_mean_percent"],
        "factor_error_median": metrics["factor_error_median"],
        "factor_error_p95": metrics["factor_error_p95"],
    }


def benchmark(config: dict[str, Any], norm: int, scheme: int) -> list[dict[str, Any]]:
    device = resolve_device(str(config.get("device", "auto")))
    test_samples, metadata = load_split(Path(config["data_dir"]) / "test.pt")
    settings = config.get("benchmark", {})
    matrix_type = str(metadata.get("matrix_type", config.get("generation", {}).get("matrix_type", "spd")))
    limit = min(int(settings.get("max_matrices", len(test_samples))), len(test_samples))
    repeats = int(settings.get("repeats", 1))
    include_exact = bool(settings.get("include_exact_timing", False))
    samples = test_samples[:limit]
    matrices = [sample_matrix(sample) for sample in samples]
    truths = np.array(
        [float(sample.kappa1 if norm == 1 else sample.kappa2) for sample in samples]
    )

    checkpoint = Path(config["output_dir"]) / f"norm_{norm}_scheme_{scheme}.pt"
    model, checkpoint_scheme, checkpoint_norm, _ = load_trained_model(checkpoint, device)
    if (checkpoint_scheme, checkpoint_norm) != (scheme, norm):
        raise ValueError("Checkpoint norm/scheme does not match benchmark request")

    estimates: list[float] = []
    durations: list[float] = []
    for matrix in matrices:
        value, elapsed = _timed(
            lambda matrix=matrix: _gnn_prediction(
                matrix, model, scheme, norm, device, matrix_type=matrix_type
            ),
            repeats,
        )
        estimates.append(value)
        durations.append(elapsed)
    rows = [
        _row(norm, "GNN", truths, np.asarray(estimates), np.asarray(durations))
    ]

    if norm == 2 and bool(settings.get("include_gershgorin", True)):
        estimates, durations = [], []
        for matrix in matrices:
            value, elapsed = _timed(
                lambda matrix=matrix: gershgorin_kappa2(matrix, matrix_type=matrix_type), repeats
            )
            estimates.append(value)
            durations.append(elapsed)
        rows.append(
            _row(norm, "Gershgorin", truths, np.asarray(estimates), np.asarray(durations))
        )

    if norm == 1:
        estimates, durations = [], []
        for matrix in matrices:
            value, elapsed = _timed(lambda matrix=matrix: hager_higham_kappa1(matrix), repeats)
            estimates.append(value)
            durations.append(elapsed)
        rows.append(
            _row(norm, "Hager-Higham (SciPy)", truths, np.asarray(estimates), np.asarray(durations))
        )
        if bool(settings.get("include_torch_hager", True)):
            estimates, durations = [], []
            for matrix in matrices:
                value, elapsed = _timed(
                    lambda matrix=matrix: torch_hager_higham_kappa1(matrix, device), repeats
                )
                estimates.append(value)
                durations.append(elapsed)
            rows.append(
                _row(
                    norm,
                    "Hager-Higham (Torch)",
                    truths,
                    np.asarray(estimates),
                    np.asarray(durations),
                )
            )
    else:
        for iterations in settings.get("lanczos_iterations", [5, 10]):
            estimates, durations = [], []
            for index, matrix in enumerate(matrices):
                value, elapsed = _timed(
                    lambda matrix=matrix, index=index, iterations=int(iterations): lanczos_kappa2(
                        matrix, iterations, int(config["seed"]) + index, device, matrix_type=matrix_type
                    ),
                    repeats,
                )
                estimates.append(value)
                durations.append(elapsed)
            rows.append(
                _row(
                    norm,
                    f"Lanczos/LOBPCG (Torch, iter={iterations})",
                    truths,
                    np.asarray(estimates),
                    np.asarray(durations),
                )
            )

    if include_exact:
        exact_times = []
        exact_values = []
        for matrix in matrices:
            value, elapsed = _timed(
                lambda matrix=matrix: exact_condition_number(matrix, norm, device), repeats
            )
            exact_values.append(value)
            exact_times.append(elapsed)
        rows.append(_row(norm, "Exact dense", truths, np.asarray(exact_values), np.asarray(exact_times)))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"norm_{norm}_scheme_{scheme}_benchmark"
    payload = {
        "metadata": metadata,
        "norm": norm,
        "scheme": scheme,
        "matrix_type": matrix_type,
        "matrix_count": limit,
        "rows": rows,
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    return rows
