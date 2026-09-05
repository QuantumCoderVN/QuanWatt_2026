from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .data import GraphSample, collate_graphs
from .features import extract_global_features, extract_graph
from .labels import two_norm_value
from .model import ConditionGNN
from .train import resolve_device


def load_matrix(path: str | Path) -> sp.csr_matrix:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        matrix = sp.load_npz(source)
    elif suffix == ".npy":
        matrix = sp.csr_matrix(np.load(source))
    elif suffix in {".mtx", ".mm"}:
        matrix = sp.csr_matrix(scipy.io.mmread(source))
    else:
        raise ValueError("Supported matrix formats are .npz, .npy, .mtx, and .mm")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {matrix.shape}")
    matrix = matrix.astype(np.float64).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _inference_sample(matrix: sp.spmatrix) -> GraphSample:
    node_features, edge_index, edge_features = extract_graph(matrix)
    placeholder = torch.tensor(float("nan"), dtype=torch.float32)
    return GraphSample(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_features),
        edge_value=torch.from_numpy(matrix.tocoo().data.astype(np.float32)),
        global_features=torch.from_numpy(extract_global_features(matrix)),
        lambda_min=placeholder,
        lambda_max=placeholder,
        kappa2=placeholder,
        norm_1=placeholder,
        inverse_norm_1=placeholder,
        kappa1=placeholder,
        family="inference",
        matrix_type="general",
    )


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ConditionGNN, int, int, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = ConditionGNN(
        hidden_dim=int(model_config["hidden_dim"]),
        gcn_layers=int(model_config["gcn_layers"]),
        head_dims=tuple(int(value) for value in model_config["head_dims"]),
        dropout=float(model_config["dropout"]),
        use_edge_attr=bool(model_config.get("use_edge_attr", False)),
        edge_hidden_dim=int(model_config.get("edge_hidden_dim", 0)),
        residual=bool(model_config.get("residual", False)),
    ).to(device)
    result = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed_missing = {"edge_mean", "edge_std"}
    if set(result.missing_keys) - allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint architecture does not match model config: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    model.eval()
    return model, int(checkpoint["scheme"]), int(checkpoint.get("norm", 2)), checkpoint


@torch.no_grad()
def predict_condition_number(
    matrix: sp.spmatrix,
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, scheme, norm, checkpoint = load_trained_model(checkpoint_path, device)
    graph = collate_graphs([_inference_sample(matrix)]).to(device)
    raw_prediction = float(model(graph).item())

    forward_norm: float | None = None
    if scheme == 1:
        if norm == 1:
            forward_norm = float(spla.norm(matrix, ord=1))
        else:
            forward_norm = two_norm_value(matrix)
        log10_kappa = float(np.log10(forward_norm) + raw_prediction)
    else:
        log10_kappa = raw_prediction

    return {
        "condition_number_estimate": float(10.0**log10_kappa),
        "log10_condition_number_estimate": log10_kappa,
        "norm": norm,
        "scheme": scheme,
        "forward_norm_used": forward_norm,
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "training_metrics": checkpoint.get("metrics", {}),
    }


def predict_file(
    matrix_path: str | Path,
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    result = predict_condition_number(load_matrix(matrix_path), checkpoint_path, device_name)
    print(json.dumps(result, indent=2))
    return result


def predict_kappa2(
    matrix: sp.spmatrix,
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Backward-compatible alias; rejects a checkpoint trained for kappa_1."""
    result = predict_condition_number(matrix, checkpoint_path, device_name)
    if result["norm"] != 2:
        raise ValueError("The supplied checkpoint predicts the 1-norm condition number")
    result["kappa2_estimate"] = result["condition_number_estimate"]
    result["log10_kappa2_estimate"] = result["log10_condition_number_estimate"]
    return result
