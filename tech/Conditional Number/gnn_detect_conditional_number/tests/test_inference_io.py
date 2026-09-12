from pathlib import Path

import numpy as np
import scipy.sparse as sp

from condition_gnn.inference import load_matrix


def test_load_sparse_npz(tmp_path: Path) -> None:
    expected = sp.diags([1.0, 2.0, 3.0], format="csr")
    path = tmp_path / "matrix.npz"
    sp.save_npz(path, expected)
    actual = load_matrix(path)
    assert actual.shape == (3, 3)
    assert np.array_equal(actual.toarray(), expected.toarray())

