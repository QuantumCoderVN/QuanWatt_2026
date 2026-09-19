from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import GraphSample, collate_graphs, load_split
from .metrics import regression_metrics
from .model import ConditionGNN, compute_input_statistics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def predict_log_kappa(
    model: ConditionGNN,
    samples: list[GraphSample],
    scheme: int,
    norm: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    model.eval()
    predictions: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []
    elapsed = 0.0
    for batch in loader:
        batch = batch.to(device)
        start = time.perf_counter()
        raw_prediction = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        if scheme == 1:
            log_prediction = torch.log10(batch.forward_norm(norm)) + raw_prediction
        else:
            log_prediction = raw_prediction
        predictions.append(log_prediction.cpu())
        truths.append(torch.log10(batch.condition_number(norm)).cpu())
    return (
        torch.cat(truths).numpy(),
        torch.cat(predictions).numpy(),
        1000.0 * elapsed / max(len(samples), 1),
    )


def _loss_on_split(
    model: ConditionGNN,
    samples: list[GraphSample],
    scheme: int,
    norm: int,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    criterion = nn.MSELoss(reduction="sum")
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            total += float(criterion(model(batch), batch.target(norm, scheme)).item())
    return total / len(samples)


def train_model(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(config["device"])
    data_dir = Path(config["data_dir"])
    train_samples, _ = load_split(data_dir / "train.pt")
    validation_samples, _ = load_split(data_dir / "validation.pt")
    test_samples, _ = load_split(data_dir / "test.pt")

    model_config = config["model"]
    statistics = compute_input_statistics(train_samples)
    model = ConditionGNN(
        hidden_dim=int(model_config["hidden_dim"]),
        gcn_layers=int(model_config["gcn_layers"]),
        head_dims=tuple(int(value) for value in model_config["head_dims"]),
        dropout=float(model_config["dropout"]),
        use_edge_attr=bool(model_config.get("use_edge_attr", True)),
        edge_hidden_dim=int(model_config.get("edge_hidden_dim", 0)),
        residual=bool(model_config.get("residual", True)),
        statistics=statistics,
    ).to(device)
    scheme = int(config["scheme"])
    norm = int(config.get("norm", 2))
    training = config["training"]
    batch_size = int(training["batch_size"])
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_samples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        generator=loader_generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(training["lr_patience"]),
    )
    criterion = nn.MSELoss()
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch.target(norm, scheme))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            running_loss += float(loss.item()) * batch.kappa2.shape[0]

        train_loss = running_loss / len(train_samples)
        validation_loss = _loss_on_split(
            model, validation_samples, scheme, norm, batch_size, device
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6g} "
            f"validation_mse={validation_loss:.6g}"
        )

        if validation_loss < best_validation - 1e-12:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(training["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    true_log, predicted_log, mean_inference_ms = predict_log_kappa(
        model, test_samples, scheme, norm, batch_size, device
    )
    metrics = regression_metrics(true_log, predicted_log)
    metrics["mean_model_inference_ms_per_matrix"] = mean_inference_ms
    metrics["parameter_count"] = float(sum(parameter.numel() for parameter in model.parameters()))
    metrics["best_validation_mse"] = best_validation

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"norm_{norm}_scheme_{scheme}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.configuration(),
            "scheme": scheme,
            "norm": norm,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    (output_dir / f"norm_{norm}_scheme_{scheme}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / f"norm_{norm}_scheme_{scheme}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics
