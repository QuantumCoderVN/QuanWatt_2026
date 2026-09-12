"""Execute the two Quapp handlers with Aer before deploying them.

This is only a local validation utility. Aer is never imported by a Quapp
handler and never runs inside processing().
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from qiskit import transpile
from qiskit_aer import AerSimulator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "offline_smoke_results"
SHOTS = 2_000


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Không load được module {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def execute(module: Any, input_payload: dict[str, Any], seed: int) -> dict[str, Any]:
    circuit = module.processing(input_payload)
    simulator = AerSimulator(seed_simulator=seed)
    compiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=SHOTS).result()
    return module.post_processing(result)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hhl = load_module("quapp_hhl_handler", ROOT / "quapp" / "hhl-main" / "handler.py")
    sign = load_module("quapp_sign_handler", ROOT / "quapp" / "hhl-sign" / "handler.py")

    results = {
        "hhl": execute(hhl, {}, 2026),
        "sign": {},
    }
    for offset, pair in enumerate(((0, 1), (1, 2), (2, 3)), start=1):
        i, j = pair
        results["sign"][f"{i}_{j}"] = execute(
            sign,
            {"pair_i": i, "pair_j": j},
            2026 + offset,
        )

    with (OUTPUT / "all_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"Đã lưu kết quả smoke test: {OUTPUT / 'all_results.json'}")


if __name__ == "__main__":
    main()
