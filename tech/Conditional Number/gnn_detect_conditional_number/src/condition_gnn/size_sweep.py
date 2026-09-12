from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .benchmark import _gnn_prediction, sample_matrix
from .data import generate_split
from .gershgorin import gershgorin_kappa2
from .matrices import default_families_for_matrix_type
from .metrics import condition_number_metrics
from .inference import load_trained_model
from .train import resolve_device


def _metrics_row(
    *,
    method: str,
    size: int,
    count: int,
    truths: np.ndarray,
    estimates: np.ndarray,
    family_group: str = "mixed",
    families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    metrics = condition_number_metrics(truths, estimates)
    return {
        "method": method,
        "family_group": family_group,
        "families": ",".join(families or ()),
        "matrix_size": size,
        "count": count,
        "finite_prediction_count": metrics["finite_prediction_count"],
        "finite_prediction_percent": metrics["finite_prediction_percent"],
        "relative_error_mean": metrics["relative_error_mean"],
        "relative_error_median": metrics["relative_error_median"],
        "relative_error_p95": metrics["relative_error_p95"],
        "relative_error_below_0_5_percent": metrics["relative_error_below_0_5_percent"],
        "accuracy_count_below_0_5": metrics["accuracy_count_below_0_5"],
        "accuracy_mean": metrics["accuracy_mean"],
        "accuracy_median": metrics["accuracy_median"],
        "accuracy_min": metrics["accuracy_min"],
        "accuracy_mean_percent": metrics["accuracy_mean_percent"],
        "paper_lre_mean_percent": metrics["paper_lre_mean_percent"],
        "factor_error_median": metrics["factor_error_median"],
        "factor_error_p95": metrics["factor_error_p95"],
        "within_factor_2_percent": metrics["within_factor_2_percent"],
        "within_factor_10_percent": metrics["within_factor_10_percent"],
    }


def run_size_sweep(config: dict[str, Any], norm: int, scheme: int) -> list[dict[str, Any]]:
    """Evaluate one trained checkpoint and Gershgorin on the same fixed-size datasets."""
    if norm != 2:
        raise ValueError("Gershgorin size-sweep comparison is implemented for norm=2 only")

    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = output_dir / f"norm_{norm}_scheme_{scheme}.pt"
    model, checkpoint_scheme, checkpoint_norm, _ = load_trained_model(checkpoint, device)
    if (checkpoint_scheme, checkpoint_norm) != (scheme, norm):
        raise ValueError("Checkpoint norm/scheme does not match size-sweep request")

    generation = config["generation"]
    settings = config.get("size_sweep", {})
    matrix_type = str(settings.get("matrix_type", generation.get("matrix_type", "spd")))
    n_min = int(settings.get("n_min", 20))
    n_max = int(settings.get("n_max", 100))
    count_per_size = int(settings.get("count_per_size", 300))
    families = settings.get(
        "families", generation.get("families", default_families_for_matrix_type(matrix_type))
    )
    family_groups = settings.get("family_groups")
    if family_groups:
        family_groups = {str(name): list(group) for name, group in family_groups.items()}
    else:
        family_groups = {"mixed": list(families)}
    dense_label_max_n = int(generation["dense_label_max_n"])
    one_norm_method = str(generation.get("one_norm_method", "auto"))
    two_norm_method = str(generation.get("two_norm_method", "auto"))
    label_device = str(generation.get("label_device", config.get("device", "auto")))
    base_seed = int(settings.get("seed", int(config["seed"]) + 50_000))

    rows: list[dict[str, Any]] = []
    for size in range(n_min, n_max + 1):
        for group_index, (family_group, group_families) in enumerate(family_groups.items()):
            print(
                f"Size sweep: generating {count_per_size} test matrices "
                f"with n={size}, family_group={family_group}, families={group_families}"
            )
            samples = generate_split(
                count=count_per_size,
                n_min=size,
                n_max=size,
                seed=base_seed + 10_000 * group_index + size,
                dense_label_max_n=dense_label_max_n,
                families=group_families,
                one_norm_method=one_norm_method,
                two_norm_method=two_norm_method,
                label_device=label_device,
                matrix_type=matrix_type,
            )
            truths = np.array([float(sample.kappa2) for sample in samples], dtype=np.float64)
            matrices = [sample_matrix(sample) for sample in samples]

            gnn_estimates = np.array(
                [
                    _gnn_prediction(matrix, model, scheme, norm, device, matrix_type=matrix_type)
                    for matrix in matrices
                ],
                dtype=np.float64,
            )
            gershgorin_estimates = np.array(
                [gershgorin_kappa2(matrix, matrix_type=matrix_type) for matrix in matrices],
                dtype=np.float64,
            )

            rows.append(
                _metrics_row(
                    method="GNN",
                    family_group=family_group,
                    families=group_families,
                    size=size,
                    count=count_per_size,
                    truths=truths,
                    estimates=gnn_estimates,
                )
            )
            rows.append(
                _metrics_row(
                    method="Gershgorin",
                    family_group=family_group,
                    families=group_families,
                    size=size,
                    count=count_per_size,
                    truths=truths,
                    estimates=gershgorin_estimates,
                )
            )

    output_tag = str(settings.get("output_tag", "")).strip()
    stem_name = f"norm_{norm}_scheme_{scheme}_size_sweep"
    if output_tag:
        stem_name += f"_{output_tag}"
    stem = output_dir / stem_name
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "norm": norm,
                "scheme": scheme,
                "n_min": n_min,
                "n_max": n_max,
                "count_per_size": count_per_size,
                "matrix_type": matrix_type,
                "families": list(families),
                "family_groups": family_groups,
                "accuracy_definition": "100 * mean(abs(true - predicted) / true < 0.5), denominator = all samples",
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def _draw_accuracy_plot(
        *,
        selected_family_groups: list[str],
        output_path: Path,
        title_suffix: str,
    ) -> None:
        figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for family_group in selected_family_groups:
            for method in ("GNN", "Gershgorin"):
                group = [
                    row
                    for row in rows
                    if row["method"] == method and row["family_group"] == family_group
                ]
                sizes = np.array([row["matrix_size"] for row in group], dtype=np.float64)
                accuracy = np.array(
                    [row["accuracy_mean_percent"] for row in group], dtype=np.float64
                )
                label = method if len(selected_family_groups) == 1 else f"{method} - {family_group}"
                axis.plot(sizes, accuracy, marker="o", linewidth=2.0, label=label)

        train_max_size = int(generation["n_max"])
        axis.axvline(
            train_max_size,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
            label=f"train max n={train_max_size}",
        )
        axis.set_xlabel("Matrix size n")
        axis.set_ylabel("Matrices with relative error < 0.5 (%)")
        axis.set_title(
            f"GNN vs Gershgorin Accuracy by Matrix Size ({matrix_type}{title_suffix}), "
            f"norm={norm}, scheme={scheme}"
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.savefig(output_path, dpi=180)
        plt.close(figure)

    plot_paths: dict[str, str] = {}

    # Combined plot, useful when several family groups are requested.
    combined_plot = stem.with_suffix(".png")
    _draw_accuracy_plot(
        selected_family_groups=list(family_groups),
        output_path=combined_plot,
        title_suffix="",
    )
    plot_paths["combined"] = str(combined_plot)

    # Separate plot for each 100% family dataset, e.g. random_spd and tridiagonal.
    for family_group in family_groups:
        safe_family_name = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in str(family_group)
        )
        family_plot = stem.parent / f"{stem.name}_{safe_family_name}.png"
        _draw_accuracy_plot(
            selected_family_groups=[family_group],
            output_path=family_plot,
            title_suffix=f", {family_group}",
        )
        plot_paths[str(family_group)] = str(family_plot)

    print(json.dumps({"rows": rows, "plots": plot_paths}, indent=2))
    return rows
