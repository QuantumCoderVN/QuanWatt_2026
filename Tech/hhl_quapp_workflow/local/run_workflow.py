"""Run the multi-job HHL workflow against Quapp and analyze results locally."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv


DIMENSION = 16
A_MATRIX = 4.0 * np.eye(DIMENSION, dtype=np.float64)
for index in range(DIMENSION - 1):
    A_MATRIX[index, index + 1] = -0.5
    A_MATRIX[index + 1, index] = -0.5

B_VECTOR = np.array(
    [
        0.045, -0.05, 0.05, -0.05,
        0.05, -0.05, 0.05, -0.05,
        0.05, -0.05, 0.05, -0.05,
        0.05, -0.05, 0.05, -0.045,
    ],
    dtype=np.float64,
)
SIGN_PAIRS = [(index, index + 1) for index in range(DIMENSION - 1)]
TERMINAL_FAILURE_STATES = {"ERROR", "IN_DEAD_LETTER_QUEUE", "CANCELLED", "FAILED"}


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuappConfig:
    base_url: str
    token: str
    project_id: str
    tenant_id: str
    device_id: int
    hhl_function: str
    sign_function: str
    shots: int
    poll_seconds: float
    job_timeout_seconds: float

    @classmethod
    def from_environment(cls) -> "QuappConfig":
        required = {
            "QUAPP_BASE_URL": os.getenv("QUAPP_BASE_URL", "").strip(),
            "QUAPP_API_TOKEN": os.getenv("QUAPP_API_TOKEN", "").strip(),
            "QUAPP_PROJECT_ID": os.getenv("QUAPP_PROJECT_ID", "").strip(),
            "QUAPP_TENANT_ID": os.getenv("QUAPP_TENANT_ID", "").strip(),
            "QUAPP_DEVICE_ID": os.getenv("QUAPP_DEVICE_ID", "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise WorkflowError("Thiếu biến môi trường: " + ", ".join(missing))

        try:
            device_id = int(required["QUAPP_DEVICE_ID"])
            shots = int(os.getenv("QUAPP_SHOTS", "100000"))
            poll_seconds = float(os.getenv("QUAPP_POLL_SECONDS", "3"))
            timeout_seconds = float(os.getenv("QUAPP_JOB_TIMEOUT_SECONDS", "3600"))
        except ValueError as exc:
            raise WorkflowError("Device, shots, poll và timeout phải là số hợp lệ.") from exc

        if shots <= 0:
            raise WorkflowError("QUAPP_SHOTS phải lớn hơn 0.")

        return cls(
            base_url=required["QUAPP_BASE_URL"].rstrip("/"),
            token=required["QUAPP_API_TOKEN"],
            project_id=required["QUAPP_PROJECT_ID"],
            tenant_id=required["QUAPP_TENANT_ID"],
            device_id=device_id,
            hhl_function=os.getenv("QUAPP_HHL_FUNCTION", "hhl-main-circuit"),
            sign_function=os.getenv("QUAPP_SIGN_FUNCTION", "hhl-sign-circuit"),
            shots=shots,
            poll_seconds=poll_seconds,
            job_timeout_seconds=timeout_seconds,
        )


class QuappClient:
    def __init__(self, config: QuappConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "X-Project-Id": config.project_id,
                "X-Tenant-Id": config.tenant_id,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def invoke(
        self,
        function_name: str,
        input_payload: dict[str, Any],
        description: str,
    ) -> tuple[str, dict[str, Any]]:
        url = f"{self.config.base_url}/api/v1/third-party/function/invoke"
        body = {
            "functionName": function_name,
            "deviceId": self.config.device_id,
            "description": description,
            "shots": self.config.shots,
            "input": input_payload,
        }

        response = self._request_with_retry("POST", url, json=body)
        payload = self._json_response(response)
        job_id = payload.get("data")
        if not isinstance(job_id, str) or not job_id:
            raise WorkflowError(f"Invoke response không có job_id hợp lệ: {payload}")
        return job_id, payload

    def get_job_detail(self, job_id: str) -> dict[str, Any]:
        # The public documentation currently shows /v1/... while an example route
        # contains /api/v1/.... Try the documented path first and then the prefixed
        # path so the client remains usable across deployments.
        paths = [
            f"/v1/third-party/jobs/{job_id}/detail",
            f"/api/v1/third-party/jobs/{job_id}/detail",
        ]
        last_error: Exception | None = None

        for path in paths:
            try:
                response = self._request_with_retry(
                    "GET",
                    f"{self.config.base_url}{path}",
                    retry_not_found=False,
                )
                if response.status_code == 404:
                    continue
                return self._json_response(response)
            except (requests.RequestException, WorkflowError) as exc:
                last_error = exc

        raise WorkflowError(
            f"Không đọc được job detail cho {job_id}. Lỗi cuối: {last_error}"
        )

    def wait_until_done(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.job_timeout_seconds
        last_status = None

        while time.monotonic() < deadline:
            envelope = self.get_job_detail(job_id)
            data = envelope.get("data", envelope)
            if not isinstance(data, dict):
                raise WorkflowError(f"Job detail có cấu trúc không hợp lệ: {envelope}")

            status = str(data.get("status", "UNKNOWN")).upper()
            if status != last_status:
                print(f"  job {job_id}: {status}", flush=True)
                last_status = status

            if status == "DONE":
                return envelope
            if status in TERMINAL_FAILURE_STATES:
                raise WorkflowError(
                    f"Job {job_id} kết thúc với trạng thái {status}: {envelope}"
                )
            time.sleep(self.config.poll_seconds)

        raise WorkflowError(
            f"Job {job_id} vượt timeout {self.config.job_timeout_seconds} giây."
        )

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        retry_not_found: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        delay = 2.0
        for attempt in range(6):
            try:
                response = self.session.request(method, url, timeout=60, **kwargs)
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue

            retryable = response.status_code in {429, 500, 502, 503, 504}
            if retry_not_found and response.status_code == 404:
                retryable = True

            if retryable and attempt < 5:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else delay
                time.sleep(sleep_seconds)
                delay = min(delay * 2.0, 30.0)
                continue

            if response.status_code >= 400 and response.status_code != 404:
                raise WorkflowError(
                    f"HTTP {response.status_code} từ {url}: {response.text[:2000]}"
                )
            return response

        raise WorkflowError(f"Không thể gọi {url} sau nhiều lần thử.")

    @staticmethod
    def _json_response(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise WorkflowError(
                f"Response không phải JSON: HTTP {response.status_code} "
                f"{response.text[:1000]}"
            ) from exc
        if not isinstance(payload, dict):
            raise WorkflowError(f"Response JSON phải là object: {payload}")
        return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"Không thể JSON serialize kiểu {type(value).__name__}")


def extract_function_result(job_envelope: dict[str, Any]) -> dict[str, Any]:
    data = job_envelope.get("data", job_envelope)
    if not isinstance(data, dict):
        raise WorkflowError("Job detail không chứa data object.")

    result = data.get("jobResult")
    if result is None:
        result = data.get("result")
    if result is None and isinstance(data.get("histogram"), dict):
        # Fallback for deployments exposing counts only as histogram.
        result = {
            "counts": data["histogram"],
            "shots": data.get("shots", sum(data["histogram"].values())),
        }

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError as exc:
            raise WorkflowError("jobResult là chuỗi nhưng không phải JSON hợp lệ.") from exc

    if not isinstance(result, dict):
        raise WorkflowError(f"Không tìm thấy jobResult dạng object: {data}")
    if "counts" not in result:
        # Some backends may place counts directly in jobResult.
        if all(isinstance(value, (int, float)) for value in result.values()):
            result = {"counts": result, "shots": int(sum(result.values()))}
        else:
            raise WorkflowError(f"jobResult không có counts: {result}")
    return result


def key_to_bits_little_endian(key: str, number_of_bits: int) -> list[int]:
    clean = key.replace(" ", "").zfill(number_of_bits)
    if len(clean) != number_of_bits or any(bit not in "01" for bit in clean):
        raise WorkflowError(f"Counts key không hợp lệ: {key!r}")
    return [int(bit) for bit in clean[::-1]]


def read_register_int(bits: list[int], qubit_indices: list[int]) -> int:
    value = 0
    for position, qubit_index in enumerate(qubit_indices):
        value |= int(bits[qubit_index]) << position
    return value


def extract_hhl_probabilities(result: dict[str, Any]) -> dict[str, Any]:
    counts = {str(key): int(value) for key, value in result["counts"].items()}
    layout = result.get("qubit_layout", {})
    total_qubits = int(layout.get("total_qubits", 10))
    phase_indices = [int(x) for x in layout.get("phase_indices", [0, 1, 2, 3, 4])]
    target_indices = [int(x) for x in layout.get("target_indices", [5, 6, 7, 8])]
    ancilla_index = int(layout.get("ancilla_index", 9))
    success_ancilla = int(result.get("algorithm", {}).get("success_ancilla_value", 1))

    shots = sum(counts.values())
    dimension = 2 ** len(target_indices)
    raw = np.zeros(dimension, dtype=float)
    success_count = 0

    for key, count in counts.items():
        bits = key_to_bits_little_endian(key, total_qubits)
        if any(bits[index] != 0 for index in phase_indices):
            continue
        if bits[ancilla_index] != success_ancilla:
            continue

        target_value = read_register_int(bits, target_indices)
        raw[target_value] += count / shots
        success_count += count

    if success_count == 0 or raw.sum() <= 0:
        raise WorkflowError(
            "Không có shot HHL thỏa phase=0 và ancilla=1. "
            "Tăng shots hoặc kiểm tra C_value/backend."
        )

    normalized = raw / raw.sum()
    return {
        "prob_raw": raw,
        "prob_norm": normalized,
        "abs_amplitudes": np.sqrt(np.maximum(normalized, 0.0)),
        "success_count": success_count,
        "shots": shots,
        "postselection_rate": success_count / shots,
    }


def extract_sign_interference(result: dict[str, Any]) -> dict[str, Any]:
    counts = {str(key): int(value) for key, value in result["counts"].items()}
    layout = result.get("qubit_layout", {})
    total_qubits = int(layout.get("total_qubits", 11))
    phase_indices = [int(x) for x in layout.get("phase_indices", [0, 1, 2, 3, 4])]
    target_indices = [int(x) for x in layout.get("target_indices", [5, 6, 7, 8])]
    ancilla_index = int(layout.get("ancilla_index", 9))
    extra_index = int(layout.get("sign_extra_index", 10))
    success_ancilla = int(result.get("algorithm", {}).get("success_ancilla_value", 1))

    success_count = 0
    plus_count = 0

    for key, count in counts.items():
        bits = key_to_bits_little_endian(key, total_qubits)
        if any(bits[index] != 0 for index in phase_indices):
            continue
        if bits[ancilla_index] != success_ancilla:
            continue

        success_count += count
        target_value = read_register_int(bits, target_indices)
        if target_value == 0 and bits[extra_index] == 0:
            plus_count += count

    if success_count == 0:
        raise WorkflowError(
            "Sign job không có shot thỏa phase=0 và ancilla=1. Tăng shots."
        )

    two_p_plus = 2.0 * plus_count / success_count
    return {
        "two_p_plus": two_p_plus,
        "plus_count": plus_count,
        "success_count": success_count,
        "postselection_rate": success_count / sum(counts.values()),
    }


def decide_relative_sign(
    probability_i: float,
    probability_j: float,
    two_p_plus: float,
    effective_success_count: int,
    tolerance_factor: float = 3.0,
) -> tuple[int, float, float]:
    """Infer sign from the interference cross term.

    2P(+) = p_i + p_j + 2*x_i*x_j for a real normalized solution.
    Therefore the sign is determined by the sign of
    2P(+) - (p_i + p_j), with a shot-noise tolerance.
    """
    baseline = probability_i + probability_j
    delta = two_p_plus - baseline
    tolerance = tolerance_factor / np.sqrt(max(effective_success_count, 1))
    if delta > tolerance:
        parity = 1
    elif delta < -tolerance:
        parity = -1
    else:
        raise WorkflowError(
            "Không đủ shot để xác định dấu tương đối: "
            f"delta={delta:.6g}, tolerance={tolerance:.6g}, "
            f"successful_shots={effective_success_count}."
        )
    return parity, delta, tolerance


def analyze_results(
    hhl_result: dict[str, Any],
    sign_results: list[dict[str, Any]],
) -> dict[str, Any]:
    hhl = extract_hhl_probabilities(hhl_result)
    probabilities = hhl["prob_norm"]
    absolute_amplitudes = hhl["abs_amplitudes"]

    parities: list[int] = []
    sign_diagnostics: list[dict[str, Any]] = []

    for pair, result in zip(SIGN_PAIRS, sign_results, strict=True):
        i, j = pair
        interference = extract_sign_interference(result)
        parity, delta, tolerance = decide_relative_sign(
            probabilities[i],
            probabilities[j],
            interference["two_p_plus"],
            interference["success_count"],
        )
        parities.append(parity)
        sign_diagnostics.append(
            {
                "pair": [i, j],
                "prob_i": probabilities[i],
                "prob_j": probabilities[j],
                "two_p_plus": interference["two_p_plus"],
                "baseline_p_i_plus_p_j": probabilities[i] + probabilities[j],
                "cross_term_delta": delta,
                "tolerance": tolerance,
                "parity": parity,
                "plus_count": interference["plus_count"],
                "success_count": interference["success_count"],
                "postselection_rate": interference["postselection_rate"],
            }
        )

    signs = [1]
    for parity in parities:
        signs.append(signs[-1] * parity)
    signs_array = np.asarray(signs, dtype=int)

    signed_direction = signs_array * absolute_amplitudes
    signed_direction = signed_direction / np.linalg.norm(signed_direction)

    a_times_direction = A_MATRIX @ signed_direction
    denominator = np.vdot(a_times_direction, a_times_direction)
    if abs(denominator) < 1e-15:
        raise WorkflowError("Không thể tính k vì A @ x_direction gần bằng 0.")

    k_coefficient = np.vdot(a_times_direction, B_VECTOR) / denominator
    x_hhl = np.real_if_close(k_coefficient * signed_direction).astype(float)
    x_classical = np.linalg.solve(A_MATRIX, B_VECTOR)

    absolute_error = np.linalg.norm(x_hhl - x_classical)
    relative_error = absolute_error / np.linalg.norm(x_classical)
    residual_hhl = np.linalg.norm(A_MATRIX @ x_hhl - B_VECTOR)
    residual_classical = np.linalg.norm(A_MATRIX @ x_classical - B_VECTOR)
    fidelity = abs(
        np.vdot(
            x_hhl / np.linalg.norm(x_hhl),
            x_classical / np.linalg.norm(x_classical),
        )
    ) ** 2

    return {
        "problem": {
            "A": A_MATRIX,
            "b": B_VECTOR,
            "eigenvalues": np.linalg.eigvalsh(A_MATRIX),
        },
        "hhl_measurement": hhl,
        "sign_diagnostics": sign_diagnostics,
        "relative_parities": np.asarray(parities, dtype=int),
        "signs": signs_array,
        "signed_direction": signed_direction,
        "k_coefficient": float(np.real_if_close(k_coefficient)),
        "x_hhl": x_hhl,
        "x_classical": x_classical,
        "metrics": {
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "residual_hhl": residual_hhl,
            "residual_classical": residual_classical,
            "fidelity_direction": fidelity,
        },
    }


def create_plots(analysis: dict[str, Any], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    indices = np.arange(len(analysis["x_classical"]))

    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.36
    axis.bar(indices - width / 2, analysis["x_classical"], width, label="Classical")
    axis.bar(indices + width / 2, analysis["x_hhl"], width, label="HHL shot + sign")
    axis.set_xticks(indices, [f"x{i}" for i in indices])
    axis.set_ylabel("Giá trị nghiệm")
    axis.set_title("So sánh nghiệm cổ điển và HHL")
    axis.axhline(0, linewidth=0.8)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "solution_comparison.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    classical_direction = analysis["x_classical"] / np.linalg.norm(analysis["x_classical"])
    axis.bar(
        indices - width / 2,
        np.abs(classical_direction) ** 2,
        width,
        label="Classical normalized",
    )
    axis.bar(
        indices + width / 2,
        analysis["hhl_measurement"]["prob_norm"],
        width,
        label="HHL counts",
    )
    axis.set_xticks(indices, [f"|{i}⟩" for i in indices])
    axis.set_ylabel("Xác suất")
    axis.set_title("Phân phối xác suất target sau hậu chọn")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "probability_comparison.png", dpi=200)
    plt.close(figure)

    diagnostics = analysis["sign_diagnostics"]
    labels = [f"{item['pair'][0]}-{item['pair'][1]}" for item in diagnostics]
    deltas = [item["cross_term_delta"] for item in diagnostics]
    tolerances = [item["tolerance"] for item in diagnostics]

    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(labels))
    axis.bar(positions, deltas, label="2P(+) - (p_i+p_j)")
    axis.errorbar(
        positions,
        np.zeros(len(labels)),
        yerr=tolerances,
        fmt="none",
        capsize=5,
        label="Ngưỡng nhiễu shot",
    )
    axis.axhline(0, linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Cặp biên độ")
    axis.set_ylabel("Cross term")
    axis.set_title("Chẩn đoán dấu tương đối")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "sign_diagnostics.png", dpi=200)
    plt.close(figure)


def print_summary(analysis: dict[str, Any]) -> None:
    np.set_printoptions(precision=10, suppress=True)
    print("\n" + "=" * 78)
    print("KẾT QUẢ WORKFLOW HHL NHIỀU JOB")
    print("=" * 78)
    print(f"x_classical = {analysis['x_classical']}")
    print(f"x_hhl       = {analysis['x_hhl']}")
    print(f"signs       = {analysis['signs']}")
    print(f"k            = {analysis['k_coefficient']:.12g}")
    print(f"postselection HHL = {analysis['hhl_measurement']['postselection_rate']:.6e}")
    print("-" * 78)
    for name, value in analysis["metrics"].items():
        print(f"{name:24s} = {value:.10e}")
    print("=" * 78)


def run_quapp_workflow(config: QuappConfig, run_directory: Path) -> dict[str, Any]:
    client = QuappClient(config)
    jobs_directory = run_directory / "jobs"
    results_directory = run_directory / "results"

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device_id": config.device_id,
        "shots_per_job": config.shots,
        "functions": {
            "hhl": config.hhl_function,
            "sign": config.sign_function,
        },
        "jobs": [],
    }

    print("Đang invoke HHL main job...")
    hhl_job_id, hhl_invoke_response = client.invoke(
        config.hhl_function,
        {},
        "HHL main counts for 16x16 fixed matrix",
    )
    write_json(jobs_directory / "hhl_invoke_response.json", hhl_invoke_response)
    hhl_detail = client.wait_until_done(hhl_job_id)
    write_json(jobs_directory / "hhl_job_detail.json", hhl_detail)
    hhl_result = extract_function_result(hhl_detail)
    write_json(results_directory / "hhl_result.json", hhl_result)
    manifest["jobs"].append({"kind": "hhl", "job_id": hhl_job_id})

    sign_results: list[dict[str, Any]] = []
    for pair_index, (pair_i, pair_j) in enumerate(SIGN_PAIRS):
        # Keep invocation frequency comfortably below Quapp's 3/10-second limit.
        # Apply the delay before every sign invocation, including the first one.
        del pair_index
        time.sleep(3.5)

        print(f"Đang invoke sign job cho cặp ({pair_i}, {pair_j})...")
        job_id, invoke_response = client.invoke(
            config.sign_function,
            {"pair_i": pair_i, "pair_j": pair_j},
            f"HHL sign detection for pair ({pair_i},{pair_j})",
        )
        stem = f"sign_{pair_i}_{pair_j}"
        write_json(jobs_directory / f"{stem}_invoke_response.json", invoke_response)
        detail = client.wait_until_done(job_id)
        write_json(jobs_directory / f"{stem}_job_detail.json", detail)
        result = extract_function_result(detail)
        wrapped_result = {
            "submitted_pair": [pair_i, pair_j],
            **result,
        }
        write_json(results_directory / f"{stem}_result.json", wrapped_result)
        sign_results.append(wrapped_result)
        manifest["jobs"].append(
            {"kind": "sign", "pair": [pair_i, pair_j], "job_id": job_id}
        )

    write_json(run_directory / "manifest.json", manifest)
    return analyze_results(hhl_result, sign_results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Invoke sixteen Quapp jobs, recover HHL solution and draw plots."
    )
    parser.add_argument(
        "--output-root",
        default="runs",
        help="Thư mục gốc lưu raw jobs, JSON kết quả và biểu đồ.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Tên run. Mặc định dùng timestamp UTC.",
    )
    arguments = parser.parse_args()

    load_dotenv()
    config = QuappConfig.from_environment()
    run_name = arguments.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = Path(arguments.output_root).resolve() / run_name
    run_directory.mkdir(parents=True, exist_ok=False)

    try:
        analysis = run_quapp_workflow(config, run_directory)
        write_json(run_directory / "analysis" / "summary.json", analysis)
        create_plots(analysis, run_directory / "figures")
        print_summary(analysis)
        print(f"\nĐã lưu toàn bộ run tại: {run_directory}")
        return 0
    except Exception as exc:
        write_json(
            run_directory / "failure.json",
            {"error_type": type(exc).__name__, "message": str(exc)},
        )
        print(f"Workflow thất bại: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())