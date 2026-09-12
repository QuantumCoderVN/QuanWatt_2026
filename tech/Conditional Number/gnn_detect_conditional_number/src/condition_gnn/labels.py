from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def _torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def extreme_eigenvalues_spd(
    matrix: sp.spmatrix,
    dense_max_n: int = 256,
    tolerance: float = 1e-8,
    method: str = "auto",
    device: str = "auto",
) -> tuple[float, float]:
    """Return smallest/largest eigenvalues for an SPD matrix."""
    a = matrix.astype(np.float64).tocsr()
    n = a.shape[0]
    selected = "dense" if method == "auto" and n <= dense_max_n else method
    if selected == "auto":
        selected = "sparse"
    if selected == "torch":
        target = _torch_device(device)
        dense = torch.from_numpy(a.toarray()).to(device=target, dtype=torch.float64)
        eigenvalues = torch.linalg.eigvalsh(dense)
        lambda_min = float(eigenvalues[0].item())
        lambda_max = float(eigenvalues[-1].item())
    elif selected == "dense":
        eigenvalues = np.linalg.eigvalsh(a.toarray())
        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])
    elif selected == "sparse":
        lambda_max = float(
            spla.eigsh(a, k=1, which="LA", return_eigenvectors=False, tol=tolerance)[0]
        )
        lambda_min = float(
            spla.eigsh(
                a,
                k=1,
                sigma=0.0,
                which="LM",
                return_eigenvectors=False,
                tol=tolerance,
            )[0]
        )
    else:
        raise ValueError("2-norm label method must be auto, dense, sparse, or torch")

    if not np.isfinite(lambda_min) or not np.isfinite(lambda_max) or lambda_min <= 0.0:
        raise ValueError(
            f"Invalid SPD spectrum: lambda_min={lambda_min}, lambda_max={lambda_max}"
        )
    return lambda_min, lambda_max


def extreme_singular_values(
    matrix: sp.spmatrix,
    dense_max_n: int = 512,
    tolerance: float = 1e-8,
    method: str = "auto",
    device: str = "auto",
) -> tuple[float, float]:
    """Return smallest/largest singular values for a general nonsingular matrix."""
    a = matrix.astype(np.float64).tocsr()
    n = a.shape[0]
    selected = "dense" if method == "auto" and n <= dense_max_n else method
    if selected == "auto":
        selected = "sparse"

    if selected == "torch":
        target = _torch_device(device)
        dense = torch.from_numpy(a.toarray()).to(device=target, dtype=torch.float64)
        singular_values = torch.linalg.svdvals(dense)
        sigma_max = float(torch.max(singular_values).item())
        sigma_min = float(torch.min(singular_values).item())
    elif selected == "dense":
        singular_values = np.linalg.svd(a.toarray(), compute_uv=False)
        sigma_max = float(np.max(singular_values))
        sigma_min = float(np.min(singular_values))
    elif selected == "sparse":
        # For large matrices, use singular values of A directly. ARPACK can be
        # fragile for the smallest singular value; if it fails and the matrix is
        # still moderate, fall back to dense SVD for label generation.
        try:
            sigma_max = float(
                spla.svds(a, k=1, which="LM", return_singular_vectors=False, tol=tolerance)[0]
            )
            sigma_min = float(
                spla.svds(a, k=1, which="SM", return_singular_vectors=False, tol=tolerance)[0]
            )
        except Exception:
            if n <= max(dense_max_n, 2000):
                singular_values = np.linalg.svd(a.toarray(), compute_uv=False)
                sigma_max = float(np.max(singular_values))
                sigma_min = float(np.min(singular_values))
            else:
                raise
    else:
        raise ValueError("2-norm label method must be auto, dense, sparse, or torch")

    if not np.isfinite(sigma_min) or not np.isfinite(sigma_max) or sigma_min <= 0.0:
        raise ValueError(
            f"Invalid singular spectrum: sigma_min={sigma_min}, sigma_max={sigma_max}"
        )
    return sigma_min, sigma_max


def spectral_values_2norm(
    matrix: sp.spmatrix,
    matrix_type: str,
    dense_max_n: int = 512,
    tolerance: float = 1e-8,
    method: str = "auto",
    device: str = "auto",
) -> tuple[float, float]:
    """Return the values needed for kappa_2.

    For SPD matrices these are lambda_min/lambda_max. For general matrices these
    are sigma_min/sigma_max. The rest of the project stores them under the legacy
    names lambda_min/lambda_max to keep checkpoint compatibility.
    """
    if matrix_type == "spd":
        return extreme_eigenvalues_spd(matrix, dense_max_n, tolerance, method, device)
    if matrix_type == "general":
        return extreme_singular_values(matrix, dense_max_n, tolerance, method, device)
    raise ValueError("matrix_type must be 'spd' or 'general'")


def two_norm_value(matrix: sp.spmatrix, dense_max_n: int = 2048) -> float:
    """Robust ||A||_2 for SPD or general sparse matrices."""
    a = matrix.astype(np.float64).tocsr()
    n = a.shape[0]
    if n <= dense_max_n:
        return float(np.linalg.svd(a.toarray(), compute_uv=False)[0])
    try:
        return float(spla.svds(a, k=1, which="LM", return_singular_vectors=False)[0])
    except Exception:
        # Power iteration on A^T A as a final fallback.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        x /= np.linalg.norm(x)
        value = 0.0
        ata = (a.T @ a).tocsr()
        for _ in range(300):
            y = ata @ x
            y_norm = np.linalg.norm(y)
            if not np.isfinite(y_norm) or y_norm == 0.0:
                break
            x = y / y_norm
            value = float(np.sqrt(max(x @ (ata @ x), 0.0)))
        if value <= 0.0 or not np.isfinite(value):
            raise RuntimeError("Could not estimate ||A||_2")
        return value


def one_norm_labels(
    matrix: sp.spmatrix,
    method: str = "auto",
    dense_max_n: int = 512,
    device: str = "auto",
) -> tuple[float, float, float]:
    """Return ``(||A||_1, ||A^-1||_1, kappa_1(A))``."""
    a = matrix.astype(np.float64).tocsc()
    norm_1 = float(spla.norm(a, ord=1))
    selected = "exact" if method == "auto" and a.shape[0] <= dense_max_n else method
    if selected == "auto":
        selected = "estimate"

    if selected == "torch":
        target = _torch_device(device)
        dense = torch.from_numpy(a.toarray()).to(device=target, dtype=torch.float64)
        inverse_norm = float(torch.linalg.matrix_norm(torch.linalg.inv(dense), ord=1).item())
    elif selected == "exact":
        inverse_norm = float(np.linalg.norm(np.linalg.inv(a.toarray()), ord=1))
    elif selected == "estimate":
        lu = spla.splu(a)
        inverse = spla.LinearOperator(
            a.shape,
            matvec=lu.solve,
            rmatvec=lambda x: lu.solve(x, trans="T"),
            matmat=lu.solve,
            rmatmat=lambda x: lu.solve(x, trans="T"),
            dtype=np.float64,
        )
        inverse_norm = float(spla.onenormest(inverse))
    else:
        raise ValueError("1-norm label method must be auto, exact, estimate, or torch")

    if not np.isfinite(inverse_norm) or inverse_norm <= 0.0:
        raise ValueError(f"Invalid inverse 1-norm: {inverse_norm}")
    return norm_1, inverse_norm, norm_1 * inverse_norm
