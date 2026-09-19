import numpy as np

from condition_gnn.labels import extreme_eigenvalues_spd, one_norm_labels
from condition_gnn.matrices import FAMILIES, generate_matrix


def test_all_generators_produce_spd_matrices() -> None:
    rng = np.random.default_rng(3)
    for family in FAMILIES:
        matrix = generate_matrix(family, 36, rng)
        lambda_min, lambda_max = extreme_eigenvalues_spd(matrix, dense_max_n=256)
        assert lambda_min > 0.0
        assert lambda_max >= lambda_min


def test_exact_one_norm_label_for_diagonal_matrix() -> None:
    import scipy.sparse as sp

    matrix = sp.diags([1.0, 2.0, 4.0], format="csr")
    norm_1, inverse_norm_1, kappa1 = one_norm_labels(matrix, method="exact")
    assert norm_1 == 4.0
    assert inverse_norm_1 == 1.0
    assert kappa1 == 4.0
