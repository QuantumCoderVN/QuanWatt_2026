from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from .features import extract_global_features, extract_graph
from .labels import one_norm_labels, spectral_values_2norm
from .matrices import FAMILIES, default_families_for_matrix_type, generate_matrix, infer_matrix_type_from_family


@dataclass
class GraphSample:
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    edge_value: Tensor
    global_features: Tensor
    lambda_min: Tensor
    lambda_max: Tensor
    kappa2: Tensor
    norm_1: Tensor
    inverse_norm_1: Tensor
    kappa1: Tensor
    family: str
    matrix_type: str = "spd"

    @property
    def target(self, norm: int, scheme: int) -> Tensor:
        if norm == 1:
            value = self.inverse_norm_1 if scheme == 1 else self.kappa1
        elif norm == 2:
            value = 1.0 / self.lambda_min if scheme == 1 else self.kappa2
        else:
            raise ValueError("norm must be 1 or 2")
        return torch.log10(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "edge_value": self.edge_value,
            "global_features": self.global_features,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "kappa2": self.kappa2,
            "norm_1": self.norm_1,
            "inverse_norm_1": self.inverse_norm_1,
            "kappa1": self.kappa1,
            "family": self.family,
            "matrix_type": self.matrix_type,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GraphSample":
        if "matrix_type" not in values:
            values = dict(values)
            values["matrix_type"] = "spd"
        return cls(**values)


@dataclass
class GraphBatch:
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    edge_value: Tensor
    global_features: Tensor
    batch: Tensor
    lambda_min: Tensor
    lambda_max: Tensor
    kappa2: Tensor
    norm_1: Tensor
    inverse_norm_1: Tensor
    kappa1: Tensor
    families: list[str]
    matrix_types: list[str]

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            edge_value=self.edge_value.to(device),
            global_features=self.global_features.to(device),
            batch=self.batch.to(device),
            lambda_min=self.lambda_min.to(device),
            lambda_max=self.lambda_max.to(device),
            kappa2=self.kappa2.to(device),
            norm_1=self.norm_1.to(device),
            inverse_norm_1=self.inverse_norm_1.to(device),
            kappa1=self.kappa1.to(device),
            families=self.families,
            matrix_types=self.matrix_types,
        )

    def target(self, norm: int, scheme: int) -> Tensor:
        if norm == 1:
            value = self.inverse_norm_1 if scheme == 1 else self.kappa1
        elif norm == 2:
            value = 1.0 / self.lambda_min if scheme == 1 else self.kappa2
        else:
            raise ValueError("norm must be 1 or 2")
        return torch.log10(value)

    def condition_number(self, norm: int) -> Tensor:
        return self.kappa1 if norm == 1 else self.kappa2

    def forward_norm(self, norm: int) -> Tensor:
        return self.norm_1 if norm == 1 else self.lambda_max


def collate_graphs(samples: list[GraphSample]) -> GraphBatch:
    xs: list[Tensor] = []
    edges: list[Tensor] = []
    edge_attrs: list[Tensor] = []
    edge_values: list[Tensor] = []
    batches: list[Tensor] = []
    node_offset = 0
    for graph_id, sample in enumerate(samples):
        xs.append(sample.x)
        edges.append(sample.edge_index + node_offset)
        edge_attrs.append(sample.edge_attr)
        edge_values.append(sample.edge_value)
        batches.append(torch.full((sample.x.shape[0],), graph_id, dtype=torch.long))
        node_offset += sample.x.shape[0]

    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_attr=torch.cat(edge_attrs, dim=0),
        edge_value=torch.cat(edge_values, dim=0),
        global_features=torch.stack([sample.global_features for sample in samples]),
        batch=torch.cat(batches),
        lambda_min=torch.stack([sample.lambda_min for sample in samples]),
        lambda_max=torch.stack([sample.lambda_max for sample in samples]),
        kappa2=torch.stack([sample.kappa2 for sample in samples]),
        norm_1=torch.stack([sample.norm_1 for sample in samples]),
        inverse_norm_1=torch.stack([sample.inverse_norm_1 for sample in samples]),
        kappa1=torch.stack([sample.kappa1 for sample in samples]),
        families=[sample.family for sample in samples],
        matrix_types=[sample.matrix_type for sample in samples],
    )


def matrix_to_sample(
    matrix: Any,
    family: str,
    dense_label_max_n: int,
    one_norm_method: str = "auto",
    two_norm_method: str = "auto",
    label_device: str = "auto",
    matrix_type: str | None = None,
) -> GraphSample:
    resolved_matrix_type = matrix_type or infer_matrix_type_from_family(family)
    spectral_min, spectral_max = spectral_values_2norm(
        matrix,
        matrix_type=resolved_matrix_type,
        dense_max_n=dense_label_max_n,
        method=two_norm_method,
        device=label_device,
    )
    norm_1, inverse_norm_1, kappa1 = one_norm_labels(
        matrix,
        method=one_norm_method,
        dense_max_n=dense_label_max_n,
        device=label_device,
    )
    node_features, edge_index, edge_features = extract_graph(matrix)
    edge_value = matrix.tocoo().data.astype(np.float32)
    global_features = extract_global_features(matrix)
    return GraphSample(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_features),
        edge_value=torch.from_numpy(edge_value),
        global_features=torch.from_numpy(global_features),
        lambda_min=torch.tensor(spectral_min, dtype=torch.float32),
        lambda_max=torch.tensor(spectral_max, dtype=torch.float32),
        kappa2=torch.tensor(spectral_max / spectral_min, dtype=torch.float32),
        norm_1=torch.tensor(norm_1, dtype=torch.float32),
        inverse_norm_1=torch.tensor(inverse_norm_1, dtype=torch.float32),
        kappa1=torch.tensor(kappa1, dtype=torch.float32),
        family=family,
        matrix_type=resolved_matrix_type,
    )


def generate_split(
    count: int,
    n_min: int,
    n_max: int,
    seed: int,
    dense_label_max_n: int,
    families: Iterable[str] | None = None,
    one_norm_method: str = "auto",
    two_norm_method: str = "auto",
    label_device: str = "auto",
    matrix_type: str = "spd",
) -> list[GraphSample]:
    rng = np.random.default_rng(seed)
    family_list = tuple(families) if families is not None else default_families_for_matrix_type(matrix_type)
    if not family_list:
        raise ValueError("At least one matrix family is required")
    samples: list[GraphSample] = []
    attempts = 0
    max_attempts = max(20, 10 * count)
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        family = family_list[len(samples) % len(family_list)]
        n = int(rng.integers(n_min, n_max + 1))
        try:
            matrix = generate_matrix(family, n, rng)
            family_type = infer_matrix_type_from_family(family)
            if matrix_type != family_type:
                raise ValueError(
                    f"family {family!r} belongs to matrix_type={family_type!r}, "
                    f"but config requested matrix_type={matrix_type!r}"
                )
            samples.append(
                matrix_to_sample(
                    matrix,
                    family,
                    dense_label_max_n,
                    one_norm_method,
                    two_norm_method,
                    label_device,
                    matrix_type=matrix_type,
                )
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
    if len(samples) != count:
        raise RuntimeError(f"Generated {len(samples)}/{count} samples after {attempts} attempts")
    rng.shuffle(samples)
    return samples


def save_split(path: str | Path, samples: list[GraphSample], metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"metadata": metadata, "samples": [sample.as_dict() for sample in samples]},
        destination,
    )


def load_split(path: str | Path) -> tuple[list[GraphSample], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return [GraphSample.from_dict(item) for item in payload["samples"]], payload["metadata"]
