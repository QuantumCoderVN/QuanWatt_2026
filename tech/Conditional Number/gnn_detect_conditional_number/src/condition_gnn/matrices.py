from __future__ import annotations

import math
from typing import Callable

import numpy as np
import scipy.sparse as sp


SPD_FAMILIES = (
    "poisson",
    "anisotropic",
    "high_contrast",
    "random_spd",
    "tridiagonal",
)

GENERAL_FAMILIES = (
    "general_diagonal_dominant",
    "general_scaled_diagonal_dominant",
    "general_sparse_random",
)

# Backward-compatible default used by the paper-style SPD experiments.
FAMILIES = SPD_FAMILIES
ALL_FAMILIES = SPD_FAMILIES + GENERAL_FAMILIES


def _grid_size(target_n: int) -> int:
    return max(2, int(round(math.sqrt(target_n))))


def poisson_2d(target_n: int, rng: np.random.Generator) -> sp.csr_matrix:
    del rng
    m = _grid_size(target_n)
    t = sp.diags((-np.ones(m - 1), 2.0 * np.ones(m), -np.ones(m - 1)), (-1, 0, 1))
    eye = sp.eye(m, format="csr")
    return (sp.kron(eye, t) + sp.kron(t, eye)).tocsr()


def anisotropic_diffusion(target_n: int, rng: np.random.Generator) -> sp.csr_matrix:
    m = _grid_size(target_n)
    epsilon = 10.0 ** rng.uniform(-8.0, -2.0)
    t = sp.diags((-np.ones(m - 1), 2.0 * np.ones(m), -np.ones(m - 1)), (-1, 0, 1))
    eye = sp.eye(m, format="csr")
    return (epsilon * sp.kron(eye, t) + sp.kron(t, eye)).tocsr()


def high_contrast_diffusion(target_n: int, rng: np.random.Generator) -> sp.csr_matrix:
    base = poisson_2d(target_n, rng)
    contrast_exp = rng.uniform(6.0, 13.0)
    coefficients = 10.0 ** rng.uniform(0.0, contrast_exp, size=base.shape[0])
    scaling = sp.diags(np.sqrt(coefficients), format="csr")
    return (scaling @ base @ scaling).tocsr()


def random_spd(target_n: int, rng: np.random.Generator) -> sp.csr_matrix:
    n = target_n
    density = min(0.25, max(0.05, 6.0 / max(n, 1)))
    b = sp.random(
        n,
        n,
        density=density,
        format="csr",
        random_state=rng,
        data_rvs=lambda size: rng.uniform(0.0, 1.0, size),
    )
    c = ((b + b.T) * 0.5).tocsr()
    row_sum = np.asarray(np.abs(c).sum(axis=1)).ravel()
    d = c + sp.diags(1.5 * row_sum + 1.0, format="csr")

    target_kappa = 10.0 ** rng.uniform(2.0, 7.0)
    scales = np.geomspace(1.0, math.sqrt(target_kappa), n)
    rng.shuffle(scales)
    s = sp.diags(scales, format="csr")
    return (s @ d @ s).tocsr()


def symmetric_tridiagonal(target_n: int, rng: np.random.Generator) -> sp.csr_matrix:
    n = target_n
    alpha = rng.uniform(0.1, 0.9)
    return sp.diags(
        (-alpha * np.ones(n - 1), 2.0 * np.ones(n), -alpha * np.ones(n - 1)),
        (-1, 0, 1),
        format="csr",
    )


def general_diagonal_dominant(target_n: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Sparse nonsymmetric, strictly row-diagonally-dominant matrix.

    This family is nonsingular but not symmetric positive definite in general.
    It is useful for testing kappa_2(A) through singular values rather than SPD eigenvalues.
    """
    n = target_n
    density = min(0.30, max(0.08, 8.0 / max(n, 1)))
    entries = rng.uniform(-1.0, 1.0, size=(n, n))
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    off = entries * mask
    row_abs_sum = np.sum(np.abs(off), axis=1)
    margin = rng.uniform(0.2, 2.0, size=n)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    diagonal = signs * (row_abs_sum + margin)
    return sp.csr_matrix(off + np.diag(diagonal))


def general_scaled_diagonal_dominant(target_n: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Nonsymmetric matrix with controlled row/column scaling.

    The base matrix is nonsingular. Left and right diagonal scalings create a wider
    condition-number range while keeping the sparse pattern simple.
    """
    base = general_diagonal_dominant(target_n, rng).tocsc()
    n = base.shape[0]
    spread = rng.uniform(1.0, 5.0)
    left = 10.0 ** rng.uniform(-0.5 * spread, 0.5 * spread, size=n)
    right = 10.0 ** rng.uniform(-0.5 * spread, 0.5 * spread, size=n)
    return (sp.diags(left, format="csr") @ base @ sp.diags(right, format="csr")).tocsr()


def general_sparse_random(target_n: int, rng: np.random.Generator) -> sp.csr_matrix:
    """General sparse nonsymmetric matrix made nonsingular by a diagonal shift."""
    n = target_n
    density = min(0.25, max(0.06, 7.0 / max(n, 1)))
    b = sp.random(
        n,
        n,
        density=density,
        format="csr",
        random_state=rng,
        data_rvs=lambda size: rng.normal(0.0, 1.0, size),
    )
    b.setdiag(0.0)
    b.eliminate_zeros()
    row_abs_sum = np.asarray(np.abs(b).sum(axis=1)).ravel()
    margin = rng.uniform(0.5, 2.5, size=n)
    diag = row_abs_sum + margin
    a = (b + sp.diags(diag, format="csr")).tocsr()

    # Mild column scaling to create a range of singular-value condition numbers.
    spread = rng.uniform(0.0, 4.0)
    col_scale = 10.0 ** rng.uniform(-0.5 * spread, 0.5 * spread, size=n)
    return (a @ sp.diags(col_scale, format="csr")).tocsr()


GENERATORS: dict[str, Callable[[int, np.random.Generator], sp.csr_matrix]] = {
    "poisson": poisson_2d,
    "anisotropic": anisotropic_diffusion,
    "high_contrast": high_contrast_diffusion,
    "random_spd": random_spd,
    "tridiagonal": symmetric_tridiagonal,
    "general_diagonal_dominant": general_diagonal_dominant,
    "general_scaled_diagonal_dominant": general_scaled_diagonal_dominant,
    "general_sparse_random": general_sparse_random,
}


def default_families_for_matrix_type(matrix_type: str) -> tuple[str, ...]:
    if matrix_type == "spd":
        return SPD_FAMILIES
    if matrix_type == "general":
        return GENERAL_FAMILIES
    raise ValueError("matrix_type must be 'spd' or 'general'")


def infer_matrix_type_from_family(family: str) -> str:
    if family in SPD_FAMILIES:
        return "spd"
    if family in GENERAL_FAMILIES:
        return "general"
    raise ValueError(f"Unknown matrix family: {family}")


def generate_matrix(family: str, n: int, rng: np.random.Generator) -> sp.csr_matrix:
    try:
        matrix = GENERATORS[family](n, rng)
    except KeyError as exc:
        raise ValueError(f"Unknown matrix family: {family}") from exc
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix.astype(np.float64)
