import numpy as np

from condition_gnn.features import extract_global_features, extract_graph
from condition_gnn.matrices import poisson_2d


def test_feature_shapes_and_finiteness() -> None:
    matrix = poisson_2d(25, np.random.default_rng(0))
    global_features = extract_global_features(matrix)
    nodes, edges, edge_features = extract_graph(matrix)

    assert global_features.shape == (29,)
    assert nodes.shape == (25, 2)
    assert edges.shape[0] == 2
    assert edge_features.shape == (edges.shape[1], 1)
    assert np.isfinite(global_features).all()
    assert np.isfinite(nodes).all()

