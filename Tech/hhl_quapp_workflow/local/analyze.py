"""Analyze 16x16 JSON results already downloaded from Quapp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_workflow import (
    SIGN_PAIRS,
    analyze_results,
    create_plots,
    print_summary,
    write_json,
)


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_directory",
        nargs="?",
        default="input_16x16",
        help="Thư mục chứa hhl_result.json và 15 sign JSON.",
    )
    parser.add_argument(
        "--output-directory",
        default="output_16x16",
        help="Thư mục lưu summary và figures.",
    )
    args = parser.parse_args()

    input_directory = Path(args.input_directory).resolve()
    output_directory = Path(args.output_directory).resolve()

    hhl = read_json(input_directory / "hhl_result.json")
    signs = [
        read_json(input_directory / f"sign_{i}_{j}_result.json")
        for i, j in SIGN_PAIRS
    ]

    analysis = analyze_results(hhl, signs)
    write_json(output_directory / "analysis" / "summary.json", analysis)
    create_plots(analysis, output_directory / "figures")
    print_summary(analysis)
    print(f"\nĐã lưu kết quả tại: {output_directory}")


if __name__ == "__main__":
    main()