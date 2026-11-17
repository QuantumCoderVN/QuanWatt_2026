"""Smoke-test Quapp HHL function (same invoke/poll flow as the backend client).

Uses only the Python standard library.

    conda activate scquan
    cd app
    python scripts/quapp_smoke.py             # invoke Quapp and wait
    python scripts/quapp_smoke.py --local     # run handler.py locally if present
    python scripts/quapp_smoke.py --input my_input.json --shots 1

Requires ``.env`` in the ``app/`` root:
    QUAPP_BASE_URL, QUAPP_ACCESS_TOKEN (or QUAPP_API_TOKEN), QUAPP_TENANT_ID,
    QUAPP_PROJECT_ID, QUAPP_FUNCTION_NAME, QUAPP_DEVICE_ID
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent

# Input mặc định đúng với contract của handler.processing()
DEFAULT_INPUT = {
    "A": [[2.0, 0.0], [0.0, 1.0]],
    "b": [1.0, 1.0],
    "phase_qubits": 4,
    "phase_target": 0.40,
    "C": 0.25,
}

FAILURE_STATES = {"ERROR", "FAILED", "CANCELLED", "IN_DEAD_LETTER_QUEUE"}


# --------------------------------------------------------------------------- .env
def load_env(path: Path) -> None:
    """Đọc file .env (KEY=VALUE) vào os.environ, không ghi đè biến đã có."""
    if not path.exists():
        sys.exit(f"Không tìm thấy {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env(name: str, *fallbacks: str) -> str:
    for key in (name, *fallbacks):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    sys.exit(f"Thiếu biến {name} trong .env")


# ------------------------------------------------------------------------- client
class QuappClient:
    def __init__(self) -> None:
        self.base_url = env("QUAPP_BASE_URL").rstrip("/")
        token = env("QUAPP_ACCESS_TOKEN", "QUAPP_API_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "X-Tenant-Id": env("QUAPP_TENANT_ID"),
            "X-Project-Id": env("QUAPP_PROJECT_ID"),
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Cloudflare của Quapp chặn UA mặc định "Python-urllib/x.y" (Error 1010)
            "User-Agent": "quapp-test-client/1.0",
        }

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = Request(self.base_url + path, data=data, headers=self.headers, method=method)
        try:
            with urlopen(req, timeout=60) as resp:
                raw = resp.read()
                try:
                    return resp.status, json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    # Không phải JSON (vd. trang HTML của web UI) -> coi như sai path
                    return 404, {}
        except HTTPError as exc:
            text = exc.read().decode(errors="replace")[:2000]
            if exc.code == 404:
                return 404, {}
            if exc.code in (401, 403) and "cloudflare" not in text.lower():
                sys.exit(f"HTTP {exc.code}: token hết hạn/không hợp lệ hoặc sai tenant/project.\n{text}")
            sys.exit(f"HTTP {exc.code} {method} {path}\n{text}")
        except URLError as exc:
            sys.exit(f"Không kết nối được {self.base_url}: {exc.reason}")

    def invoke(self, function_name: str, device_id: int, shots: int, input: dict,
               description: str = "local test from python_env.py") -> str:
        status, payload = self._request("POST", "/api/v1/third-party/function/invoke", {
            "functionName": function_name,
            "deviceId": device_id,
            "description": description,
            "shots": shots,
            "input": input,
        })
        job_id = payload.get("data")
        if status == 404 or not isinstance(job_id, str) or not job_id:
            sys.exit(f"Invoke không trả về job id (HTTP {status}): {payload}")
        return job_id

    def job_detail(self, job_id: str) -> dict:
        # Tài liệu Quapp dùng /v1/..., một số ví dụ dùng /api/v1/... -> thử cả hai.
        for path in (f"/api/v1/third-party/jobs/{job_id}/detail",
                     f"/v1/third-party/jobs/{job_id}/detail"):
            status, payload = self._request("GET", path)
            if status != 404:
                return payload
        return {}  # chưa tìm thấy job (vừa tạo), poll lại

    def wait_for_result(self, job_id: str, poll: float = 3.0, timeout: float = 600) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            envelope = self.job_detail(job_id)
            job = envelope.get("data", envelope) if isinstance(envelope, dict) else {}
            status = str(job.get("status", "PENDING")).upper()
            if status != last:
                print(f"  job {job_id}: {status}", flush=True)
                last = status
            if status == "DONE" or status in FAILURE_STATES:
                return job
            time.sleep(poll)
        sys.exit(f"Job {job_id} quá thời gian chờ {timeout}s")


# -------------------------------------------------------------------------- main
def run_local(payload: dict) -> dict:
    """Run handler.py locally: processing -> (skip device) -> post_processing."""
    # Prefer a local handler.py next to this script, else the Quapp module in backend/
    for candidate in (HERE, APP_ROOT / "backend"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        import handler  # type: ignore
    except ImportError:
        from backend import hhl_solver_quapp as handler  # type: ignore

    circuit = handler.processing(payload)
    print("processing() trả về:", type(circuit).__name__, f"({circuit.num_qubits} qubit)")
    return handler.post_processing(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="chạy handler.py trên máy, không gọi API")
    parser.add_argument("--input", type=Path, help="file JSON input (mặc định: ma trận 2x2 mẫu)")
    parser.add_argument("--shots", type=int, default=1, help="shots cho circuit placeholder (mặc định 1)")
    parser.add_argument("--timeout", type=float, default=600, help="giây chờ job tối đa")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text()) if args.input else dict(DEFAULT_INPUT)
    payload.setdefault("request_id", f"local-{uuid.uuid4().hex[:8]}")

    if args.local:
        result = run_local(payload)
        print(json.dumps(result, indent=2))
        return

    load_env(APP_ROOT / ".env")
    client = QuappClient()
    function_name = env("QUAPP_FUNCTION_NAME")
    device_id = int(env("QUAPP_DEVICE_ID"))
    print(f"Invoke '{function_name}' trên device {device_id}, request_id={payload['request_id']}")

    job_id = client.invoke(function_name=function_name, device_id=device_id,
                           shots=args.shots, input=payload)
    job = client.wait_for_result(job_id, timeout=args.timeout)

    result = job.get("jobResult")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass
    print("status:", job.get("status"))
    print(json.dumps(result if result is not None else job, indent=2, ensure_ascii=False))

    if job.get("status", "").upper() != "DONE":
        sys.exit(1)
    if isinstance(result, dict) and result.get("request_id") not in (None, payload["request_id"]):
        print("CẢNH BÁO: request_id trong kết quả không khớp input!")


if __name__ == "__main__":
    main()
