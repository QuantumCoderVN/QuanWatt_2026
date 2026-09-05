from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def gershgorin_kappa2_bounds_spd(matrix: sp.spmatrix) -> tuple[float, float, float]:
    """Estimate kappa_2(A) for a sparse symmetric/SPD matrix using Gershgorin.

    For row i, r_i = sum_{j != i} |a_ij|. For an SPD matrix,
    lambda_min >= min_i(a_ii - r_i) and lambda_max <= max_i(a_ii + r_i).
    If the lower bound is nonpositive, return infinity and count it as failed.
    """
    a = matrix.tocsr().astype(np.float64)
    diagonal = a.diagonal()
    radii = np.asarray(abs(a).sum(axis=1)).ravel() - np.abs(diagonal)

    lambda_lower = float(np.min(diagonal - radii))
    lambda_upper = float(np.max(diagonal + radii))

    if not np.isfinite(lambda_lower) or not np.isfinite(lambda_upper):
        return float("inf"), lambda_lower, lambda_upper
    if lambda_lower <= 0.0:
        return float("inf"), lambda_lower, lambda_upper

    return float(lambda_upper / lambda_lower), lambda_lower, lambda_upper


def gershgorin_kappa2_bounds_normal(matrix: sp.spmatrix) -> tuple[float, float, float]:
    """Estimate kappa_2(A) for a general matrix via Gershgorin on A.T @ A.

    Singular values satisfy sigma_i(A)^2 = lambda_i(A.T A), so
    kappa_2(A) = sqrt(kappa_2(A.T A)). The bound can be very loose because
    forming A.T A squares the condition number and increases fill-in.
    """
    a = matrix.tocsr().astype(np.float64)
    normal = (a.T @ a).tocsr()
    kappa_squared, lambda_lower, lambda_upper = gershgorin_kappa2_bounds_spd(normal)
    if not np.isfinite(kappa_squared):
        return float("inf"), lambda_lower, lambda_upper
    return float(np.sqrt(kappa_squared)), lambda_lower, lambda_upper


def gershgorin_kappa2_bounds(
    matrix: sp.spmatrix,
    matrix_type: str = "spd",
) -> tuple[float, float, float]:
    if matrix_type == "spd":
        return gershgorin_kappa2_bounds_spd(matrix)
    if matrix_type == "general":
        return gershgorin_kappa2_bounds_normal(matrix)
    raise ValueError("matrix_type must be 'spd' or 'general'")


def gershgorin_kappa2(matrix: sp.spmatrix, matrix_type: str = "spd") -> float:
    """Return only the Gershgorin kappa_2 estimate."""
    value, _, _ = gershgorin_kappa2_bounds(matrix, matrix_type=matrix_type)
    return value
