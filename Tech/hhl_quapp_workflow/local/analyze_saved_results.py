"""Phân tích các JSON đã tải từ Quapp mà không chạy lại quantum jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_workflow import (
    analyze_results,
    create_plots,
    print_summary,
    write_json,
)


EXPECTED_FILES = [
    "hhl_result.json",
    "sign_0_1_result.json",
    "sign_1_2_result.json",
    "sign_2_3_result.json",
]


def read_json(path: Path) -> dict[str, Any]:
    """Đọc một file JSON và kiểm tra dữ liệu cấp cao nhất là object."""

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    if not path.is_file():
        raise ValueError(f"Đường dẫn không phải file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"File JSON không hợp lệ: {path}\n"
            f"Dòng {exc.lineno}, cột {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Nội dung file {path.name} phải là một JSON object."
        )

    return payload


def validate_input_directory(input_directory: Path) -> None:
    """Kiểm tra thư mục input có đủ bốn file cần thiết."""

    if not input_directory.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục input: {input_directory}"
        )

    if not input_directory.is_dir():
        raise ValueError(
            f"Đường dẫn input không phải thư mục: {input_directory}"
        )

    missing_files = [
        filename
        for filename in EXPECTED_FILES
        if not (input_directory / filename).is_file()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {input_directory / filename}"
            for filename in missing_files
        )
        raise FileNotFoundError(
            "Thiếu các file JSON sau:\n"
            f"{missing_text}"
        )


def validate_payloads(
    hhl: dict[str, Any],
    signs: list[dict[str, Any]],
) -> None:
    """Kiểm tra sơ bộ loại circuit và counts."""

    if hhl.get("circuit_type") != "hhl-main":
        raise ValueError(
            "hhl_result.json không có circuit_type='hhl-main'."
        )

    if "counts" not in hhl:
        raise ValueError("hhl_result.json không chứa trường counts.")

    for index, sign in enumerate(signs):
        pair_names = ["0-1", "1-2", "2-3"]

        if sign.get("circuit_type") != "hhl-sign":
            raise ValueError(
                f"sign_{pair_names[index]}_result.json không có "
                "circuit_type='hhl-sign'."
            )

        if "counts" not in sign:
            raise ValueError(
                f"sign_{pair_names[index]}_result.json không chứa counts."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Đọc bốn JSON kết quả từ Quapp, khôi phục nghiệm HHL "
            "và tạo biểu đồ."
        )
    )

    parser.add_argument(
        "input_directory",
        nargs="?",
        default="input",
        help=(
            "Thư mục chứa hhl_result.json và ba sign result. "
            "Mặc định: input"
        ),
    )

    parser.add_argument(
        "--output-directory",
        default="output",
        help=(
            "Thư mục lưu summary.json và biểu đồ. "
            "Mặc định: output"
        ),
    )

    args = parser.parse_args()

    input_directory = Path(args.input_directory).resolve()
    output_directory = Path(args.output_directory).resolve()

    try:
        validate_input_directory(input_directory)

        print("=" * 78)
        print("ĐỌC KẾT QUẢ QUAPP")
        print("=" * 78)
        print(f"Input : {input_directory}")
        print(f"Output: {output_directory}")
        print("-" * 78)

        hhl = read_json(input_directory / "hhl_result.json")

        signs = [
            read_json(input_directory / "sign_0_1_result.json"),
            read_json(input_directory / "sign_1_2_result.json"),
            read_json(input_directory / "sign_2_3_result.json"),
        ]

        validate_payloads(hhl, signs)

        print(
            f"HHL shots      : "
            f"{sum(int(v) for v in hhl['counts'].values())}"
        )

        for pair, sign in zip(
            ["0-1", "1-2", "2-3"],
            signs,
            strict=True,
        ):
            total_shots = sum(
                int(value)
                for value in sign["counts"].values()
            )
            print(f"Sign {pair} shots : {total_shots}")

        print("-" * 78)
        print("Đang khôi phục nghiệm...")

        analysis = analyze_results(hhl, signs)

        summary_path = (
            output_directory
            / "analysis"
            / "summary.json"
        )

        figures_directory = output_directory / "figures"

        write_json(summary_path, analysis)
        create_plots(analysis, figures_directory)
        print_summary(analysis)

        print()
        print("ĐÃ TẠO CÁC FILE")
        print("-" * 78)
        print(f"Summary:")
        print(f"  {summary_path}")
        print()
        print("Biểu đồ:")
        print(
            f"  {figures_directory / 'solution_comparison.png'}"
        )
        print(
            f"  {figures_directory / 'probability_comparison.png'}"
        )
        print(
            f"  {figures_directory / 'sign_diagnostics.png'}"
        )

        return 0

    except Exception as exc:
        print()
        print("=" * 78, file=sys.stderr)
        print("PHÂN TÍCH THẤT BẠI", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())