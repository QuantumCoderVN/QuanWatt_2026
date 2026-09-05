from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_results(config: dict[str, Any]) -> list[Path]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    found_history = False
    for row, norm in enumerate((1, 2)):
        for column, scheme in enumerate((1, 2)):
            path = output_dir / f"norm_{norm}_scheme_{scheme}_history.json"
            axis = axes[row, column]
            if not path.exists():
                axis.set_visible(False)
                continue
            history = json.loads(path.read_text(encoding="utf-8"))
            epochs = [entry["epoch"] for entry in history]
            axis.plot(epochs, [entry["train_mse"] for entry in history], label="Train")
            axis.plot(
                epochs,
                [entry["validation_mse"] for entry in history],
                label="Validation",
            )
            axis.set_yscale("log")
            axis.set_title(f"{norm}-norm, Scheme {scheme}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("MSE")
            axis.grid(alpha=0.25)
            axis.legend()
            found_history = True
    if found_history:
        path = output_dir / "training_curves.png"
        figure.savefig(path, dpi=180)
        written.append(path)
    plt.close(figure)

    labels: list[str] = []
    accuracy: list[float] = []
    lre: list[float] = []
    factor_median: list[float] = []
    within_factor_2: list[float] = []
    within_factor_10: list[float] = []
    for norm in (1, 2):
        for scheme in (1, 2):
            path = output_dir / f"norm_{norm}_scheme_{scheme}_metrics.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            labels.append(f"N{norm}/S{scheme}")
            accuracy.append(float(metrics.get("accuracy_mean_percent", np.nan)))
            lre.append(float(metrics["paper_lre_mean_percent"]))
            factor_median.append(float(metrics["factor_error_median"]))
            within_factor_2.append(float(metrics["within_factor_2_percent"]))
            within_factor_10.append(float(metrics["within_factor_10_percent"]))

    if labels:
        x = np.arange(len(labels))
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes[0, 0].bar(x, accuracy, color="#2b6f9f")
        axes[0, 0].set_title("Relative Error < 0.5")
        axes[0, 0].set_ylabel("Matrices (%)")
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].grid(axis="y", alpha=0.25)

        axes[0, 1].bar(x, lre, color="#d06b32")
        axes[0, 1].set_title("Mean LRE")
        axes[0, 1].set_ylabel("LRE (%)")
        axes[0, 1].grid(axis="y", alpha=0.25)

        axes[1, 0].bar(x, factor_median, color="#5f8f3f")
        axes[1, 0].axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
        axes[1, 0].set_title("Median Factor Error")
        axes[1, 0].set_ylabel("Multiplicative factor")
        axes[1, 0].grid(axis="y", alpha=0.25)

        width = 0.35
        axes[1, 1].bar(x - width / 2, within_factor_2, width, label="within x2")
        axes[1, 1].bar(x + width / 2, within_factor_10, width, label="within x10")
        axes[1, 1].set_title("Prediction Coverage")
        axes[1, 1].set_ylabel("Matrices (%)")
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].legend()
        axes[1, 1].grid(axis="y", alpha=0.25)

        for axis in axes.ravel():
            axis.set_xticks(x, labels)

        path = output_dir / "training_metrics_summary.png"
        figure.savefig(path, dpi=180)
        written.append(path)
        plt.close(figure)

    labels: list[str] = []
    runtimes: list[float] = []
    errors: list[float] = []
    accuracies: list[float] = []
    for norm in (1, 2):
        for scheme in (1, 2):
            path = output_dir / f"norm_{norm}_scheme_{scheme}_benchmark.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    labels.append(f"N{norm}/S{scheme}\n{row['method']}")
                    runtimes.append(float(row["time_mean_ms"]))
                    errors.append(float(row["lre_mean_percent"]))
                    accuracies.append(float(row.get("accuracy_mean_percent", np.nan)))

    if labels:
        x = np.arange(len(labels))
        figure, axes = plt.subplots(3, 1, figsize=(max(12, len(labels) * 0.8), 11))
        axes[0].bar(x, runtimes, color="#2b6f9f")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Mean runtime (ms, log scale)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(x, errors, color="#d06b32")
        axes[1].set_ylabel("Mean LRE (%)")
        axes[1].grid(axis="y", alpha=0.25)
        axes[2].bar(x, accuracies, color="#5f8f3f")
        axes[2].set_ylabel("Relative error < 0.5 (%)")
        axes[2].set_xticks(x, labels, rotation=45, ha="right")
        axes[2].grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / "benchmark_summary.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        written.append(path)
        plt.close(figure)

    for path in written:
        print(path)
    return written
