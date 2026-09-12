import numpy as np
import torch

from condition_gnn.benchmark import sample_matrix
from condition_gnn.data import collate_graphs, matrix_to_sample
from condition_gnn.matrices import poisson_2d, symmetric_tridiagonal
from condition_gnn.model import ConditionGNN, compute_input_statistics


def test_forward_and_backward() -> None:
    rng = np.random.default_rng(1)
    samples = [
        matrix_to_sample(poisson_2d(25, rng), "poisson", 256),
        matrix_to_sample(symmetric_tridiagonal(31, rng), "tridiagonal", 256),
    ]
    batch = collate_graphs(samples)
    model = ConditionGNN(
        hidden_dim=16,
        gcn_layers=2,
        head_dims=(16,),
        statistics=compute_input_statistics(samples),
    )
    prediction = model(batch)
    assert prediction.shape == (2,)
    loss = torch.nn.functional.mse_loss(prediction, batch.target(2, 1))
    loss.backward()
    assert torch.isfinite(loss)


def test_edge_aware_model_uses_edge_attributes() -> None:
    rng = np.random.default_rng(3)
    samples = [matrix_to_sample(symmetric_tridiagonal(20, rng), "tridiagonal", 256)]
    statistics = compute_input_statistics(samples)
    graph = collate_graphs(samples)
    altered_graph = collate_graphs(samples)
    altered_graph.edge_attr = altered_graph.edge_attr + 1.0

    torch.manual_seed(3)
    model = ConditionGNN(
        hidden_dim=16,
        gcn_layers=1,
        head_dims=(16,),
        use_edge_attr=True,
        statistics=statistics,
    )

    assert not torch.allclose(model(graph), model(altered_graph))


def test_sample_preserves_signed_sparse_matrix() -> None:
    rng = np.random.default_rng(9)
    matrix = poisson_2d(25, rng)
    sample = matrix_to_sample(matrix, "poisson", 256)
    recovered = sample_matrix(sample)
    assert np.allclose(recovered.toarray(), matrix.toarray())
