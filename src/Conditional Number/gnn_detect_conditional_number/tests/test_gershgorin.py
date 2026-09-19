import numpy as np
import scipy.sparse as sp

from condition_gnn.gershgorin import gershgorin_kappa2_bounds


def test_gershgorin_returns_expected_tridiagonal_bound():
    matrix = sp.diags(
        diagonals=[-1.0 * np.ones(4), 3.0 * np.ones(5), -1.0 * np.ones(4)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    estimate, lower, upper = gershgorin_kappa2_bounds(matrix)
    assert np.isclose(lower, 1.0)
    assert np.isclose(upper, 5.0)
    assert np.isclose(estimate, 5.0)


def test_gershgorin_infinite_when_lower_bound_is_nonpositive():
    matrix = sp.csr_matrix([[1.0, -2.0], [-2.0, 1.0]])
    estimate, lower, upper = gershgorin_kappa2_bounds(matrix)
    assert lower <= 0.0
    assert np.isinf(estimate)
    assert np.isfinite(upper)
