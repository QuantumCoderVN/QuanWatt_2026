from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .data import generate_split, save_split
from .benchmark import benchmark
from .inference import predict_file
from .matrices import FAMILIES, default_families_for_matrix_type
from .plotting import plot_results
from .size_sweep import run_size_sweep
from .train import train_model


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_data(config: dict[str, Any]) -> None:
    data_dir = Path(config["data_dir"])
    generation = config["generation"]
    for offset, (split, count) in enumerate(
        (
            ("train", generation["train_count"]),
            ("validation", generation["validation_count"]),
            ("test", generation["test_count"]),
        )
    ):
        print(f"Generating {split}: {count} matrices")
        matrix_type = str(generation.get("matrix_type", "spd"))
        families = generation.get("families", default_families_for_matrix_type(matrix_type))
        samples = generate_split(
            count=int(count),
            n_min=int(generation["n_min"]),
            n_max=int(generation["n_max"]),
            seed=int(config["seed"]) + 1000 * offset,
            dense_label_max_n=int(generation["dense_label_max_n"]),
            families=families,
            one_norm_method=str(generation.get("one_norm_method", "auto")),
            two_norm_method=str(generation.get("two_norm_method", "auto")),
            label_device=str(generation.get("label_device", config.get("device", "auto"))),
            matrix_type=matrix_type,
        )
        save_split(
            data_dir / f"{split}.pt",
            samples,
            metadata={
                "split": split,
                "count": len(samples),
                "n_min": int(generation["n_min"]),
                "n_max": int(generation["n_max"]),
                "matrix_type": matrix_type,
                "families": list(families),
                "one_norm_method": str(generation.get("one_norm_method", "auto")),
                "two_norm_method": str(generation.get("two_norm_method", "auto")),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "generate",
            "train",
            "benchmark",
            "plot",
            "size-sweep",
            "reproduce",
            "all",
            "predict",
        ),
    )
    parser.add_argument("--config", help="YAML experiment configuration")
    parser.add_argument("--scheme", type=int, choices=(1, 2), help="Override prediction scheme")
    parser.add_argument("--norm", type=int, choices=(1, 2), help="Override condition-number norm")
    parser.add_argument("--matrix", help="Sparse .npz, dense .npy, or Matrix Market input")
    parser.add_argument("--checkpoint", help="Trained .pt checkpoint for prediction")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "predict":
        if not args.matrix or not args.checkpoint:
            raise SystemExit("predict requires --matrix and --checkpoint")
        predict_file(args.matrix, args.checkpoint, args.device)
        return
    if not args.config:
        raise SystemExit(f"{args.command} requires --config")
    config = load_config(args.config)
    if args.scheme is not None:
        config["scheme"] = args.scheme
    if args.norm is not None:
        config["norm"] = args.norm
    if args.command in ("generate", "all", "reproduce"):
        generate_data(config)
    if args.command in ("train", "all"):
        train_model(config)
    if args.command == "benchmark":
        benchmark(config, int(config.get("norm", 2)), int(config.get("scheme", 1)))
    if args.command == "plot":
        plot_results(config)
    if args.command == "size-sweep":
        run_size_sweep(config, int(config.get("norm", 2)), int(config.get("scheme", 1)))
    if args.command == "reproduce":
        for norm in (1, 2):
            for scheme in (1, 2):
                run_config = dict(config)
                run_config["norm"] = norm
                run_config["scheme"] = scheme
                train_model(run_config)
                benchmark(run_config, norm, scheme)
        plot_results(config)


if __name__ == "__main__":
    main()
