from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import GraphBatch, GraphSample
from .features import EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM


class GCNLayer(nn.Module):
    """Symmetrically normalized GCN layer matching equation (13) in the paper."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        degree = torch.bincount(dst, minlength=x.shape[0]).to(x.dtype).clamp_min_(1.0)
        normalization = degree[src].rsqrt() * degree[dst].rsqrt()
        messages = x[src] * normalization.unsqueeze(1)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        return self.linear(aggregated)


class EdgeGCNLayer(nn.Module):
    """GCN-style message passing that conditions messages on matrix entries."""

    def __init__(
        self,
        width: int,
        edge_dim: int = EDGE_FEATURE_DIM,
        edge_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        hidden = edge_hidden_dim or width
        self.message_mlp = nn.Sequential(
            nn.Linear(width + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, width),
        )
        self.update = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_index
        degree = torch.bincount(dst, minlength=x.shape[0]).to(x.dtype).clamp_min_(1.0)
        normalization = degree[src].rsqrt() * degree[dst].rsqrt()
        message_input = torch.cat((x[src], edge_attr), dim=1)
        messages = self.message_mlp(message_input) * normalization.unsqueeze(1)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        return self.update(aggregated)


def graph_pool(x: Tensor, batch: Tensor, graph_count: int) -> tuple[Tensor, Tensor]:
    sums = torch.zeros(graph_count, x.shape[1], device=x.device, dtype=x.dtype)
    sums.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=graph_count).to(x.dtype).unsqueeze(1)
    means = sums / counts.clamp_min(1.0)

    maxima = torch.full_like(sums, -torch.inf)
    index = batch.unsqueeze(1).expand_as(x)
    maxima.scatter_reduce_(0, index, x, reduce="amax", include_self=True)
    return means, maxima


@dataclass
class InputStatistics:
    node_mean: Tensor
    node_std: Tensor
    edge_mean: Tensor
    edge_std: Tensor
    global_mean: Tensor
    global_std: Tensor


def compute_input_statistics(samples: list[GraphSample]) -> InputStatistics:
    nodes = torch.cat([sample.x for sample in samples], dim=0)
    edges = torch.cat([sample.edge_attr for sample in samples], dim=0)
    globals_ = torch.stack([sample.global_features for sample in samples])
    return InputStatistics(
        node_mean=nodes.mean(dim=0),
        node_std=nodes.std(dim=0).clamp_min(1e-6),
        edge_mean=edges.mean(dim=0),
        edge_std=edges.std(dim=0).clamp_min(1e-6),
        global_mean=globals_.mean(dim=0),
        global_std=globals_.std(dim=0).clamp_min(1e-6),
    )


class ConditionGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 112,
        gcn_layers: int = 2,
        head_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        use_edge_attr: bool = True,
        edge_hidden_dim: int = 0,
        residual: bool = True,
        statistics: InputStatistics | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gcn_layer_count = gcn_layers
        self.head_dims = head_dims
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.edge_hidden_dim = edge_hidden_dim
        self.residual = residual

        self.node_encoder = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        if use_edge_attr:
            self.gcn_layers = nn.ModuleList(
                EdgeGCNLayer(hidden_dim, EDGE_FEATURE_DIM, edge_hidden_dim)
                for _ in range(gcn_layers)
            )
        else:
            self.gcn_layers = nn.ModuleList(GCNLayer(hidden_dim) for _ in range(gcn_layers))
        self.global_encoder = nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim)

        layers: list[nn.Module] = []
        input_dim = 3 * hidden_dim
        for width in head_dims:
            layers.extend((nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)))
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.prediction_head = nn.Sequential(*layers)

        if statistics is None:
            statistics = InputStatistics(
                torch.zeros(NODE_FEATURE_DIM),
                torch.ones(NODE_FEATURE_DIM),
                torch.zeros(EDGE_FEATURE_DIM),
                torch.ones(EDGE_FEATURE_DIM),
                torch.zeros(GLOBAL_FEATURE_DIM),
                torch.ones(GLOBAL_FEATURE_DIM),
            )
        self.register_buffer("node_mean", statistics.node_mean.clone())
        self.register_buffer("node_std", statistics.node_std.clone())
        self.register_buffer("edge_mean", statistics.edge_mean.clone())
        self.register_buffer("edge_std", statistics.edge_std.clone())
        self.register_buffer("global_mean", statistics.global_mean.clone())
        self.register_buffer("global_std", statistics.global_std.clone())

    def forward(self, graph: GraphBatch) -> Tensor:
        x = (graph.x - self.node_mean) / self.node_std
        x = F.relu(self.node_encoder(x))
        edge_attr = (graph.edge_attr - self.edge_mean) / self.edge_std
        for layer in self.gcn_layers:
            if self.use_edge_attr:
                update = layer(x, graph.edge_index, edge_attr)
            else:
                update = layer(x, graph.edge_index)
            x = F.relu(x + update) if self.residual else F.relu(update)

        graph_count = graph.global_features.shape[0]
        mean_pool, max_pool = graph_pool(x, graph.batch, graph_count)
        global_features = (graph.global_features - self.global_mean) / self.global_std
        global_embedding = F.relu(self.global_encoder(global_features))
        representation = torch.cat((mean_pool, max_pool, global_embedding), dim=1)
        return self.prediction_head(representation).squeeze(1)

    def configuration(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "gcn_layers": self.gcn_layer_count,
            "head_dims": list(self.head_dims),
            "dropout": self.dropout,
            "use_edge_attr": self.use_edge_attr,
            "edge_hidden_dim": self.edge_hidden_dim,
            "residual": self.residual,
        }
