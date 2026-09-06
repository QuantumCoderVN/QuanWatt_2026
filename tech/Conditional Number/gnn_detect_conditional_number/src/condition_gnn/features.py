from __future__ import annotations

import numpy as np
import scipy.sparse as sp


EPS = 1e-10
GLOBAL_FEATURE_DIM = 29
NODE_FEATURE_DIM = 2
EDGE_FEATURE_DIM = 1


def _stats(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


def _log10(value: float | np.ndarray, offset: float = EPS) -> float | np.ndarray:
    return np.log10(np.asarray(value) + offset)


def extract_global_features(matrix: sp.spmatrix) -> np.ndarray:
    a = matrix.tocsr()
    n = a.shape[0]
    nnz = a.nnz
    abs_a = abs(a)
    diagonal = np.abs(a.diagonal())
    row_sums = np.asarray(abs_a.sum(axis=1)).ravel()
    col_sums = np.asarray(abs_a.sum(axis=0)).ravel()
    row_nnz = np.diff(a.indptr).astype(np.float64)
    nz_values = np.abs(a.data)

    norm_1 = float(col_sums.max(initial=0.0))
    norm_inf = float(row_sums.max(initial=0.0))
    norm_f = float(np.linalg.norm(a.data))

    diag_mean, diag_std, diag_min, diag_max = _stats(diagonal)
    nonempty_row_nnz = row_nnz[row_nnz > 0]
    rsp_mean, rsp_std, rsp_min, rsp_max = _stats(nonempty_row_nnz)
    nz_mean, nz_std, nz_min, nz_max = _stats(nz_values)

    dominance = diagonal / (row_sums + EPS)
    dom_mean, dom_std, dom_min, dom_max = _stats(dominance)

    gershgorin = np.maximum(row_sums - diagonal, 0.0)
    gers_mean = float(np.mean(gershgorin))
    gers_max = float(np.max(gershgorin, initial=0.0))
    normalized_gers = float(np.mean(gershgorin / (diagonal + EPS)))

    features = np.array(
        [
            np.log10(n + 1.0),
            np.log10(nnz + 1.0),
            nnz / float(n * n),
            _log10(diag_mean),
            _log10(diag_std),
            _log10(diag_min),
            _log10(diag_max),
            _log10(diag_max / (diag_min + EPS)),
            _log10(norm_1),
            _log10(norm_inf),
            _log10(norm_f),
            _log10(norm_1 / (norm_inf + EPS)),
            dom_mean,
            dom_min,
            dom_max,
            dom_std,
            np.log10(rsp_mean + 1.0),
            np.log10(rsp_std + 1.0),
            np.log10(rsp_max + 1.0),
            np.log10(rsp_min + 1.0),
            rsp_std / (rsp_mean + 1.0),
            _log10(nz_mean),
            _log10(nz_std),
            _log10(nz_min),
            _log10(nz_max),
            _log10(nz_max / (nz_min + EPS)),
            _log10(gers_mean),
            _log10(gers_max),
            normalized_gers,
        ],
        dtype=np.float32,
    )
    if features.shape != (GLOBAL_FEATURE_DIM,):
        raise RuntimeError(f"Expected {GLOBAL_FEATURE_DIM} global features, got {features.shape}")
    return features


def extract_graph(matrix: sp.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = matrix.tocsr()
    n = a.shape[0]
    diagonal = np.abs(a.diagonal())
    row_nnz = np.diff(a.indptr)
    node_features = np.column_stack(
        (
            np.log10(diagonal + EPS),
            np.log10(row_nnz + 1.0),
        )
    ).astype(np.float32)

    coo = a.tocoo()
    edge_index = np.vstack((coo.row, coo.col)).astype(np.int64)
    edge_features = np.log10(np.abs(coo.data) + EPS).reshape(-1, 1).astype(np.float32)
    if node_features.shape != (n, NODE_FEATURE_DIM):
        raise RuntimeError("Unexpected node feature shape")
    return node_features, edge_index, edge_features
