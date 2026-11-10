"""HHL solver using Quapp-hosted statevector simulation.

Keep the existing FDLS-facing hhl_solve and hhl_solve_with_diagnostics APIs.
Quapp confirmed that processing() and post_processing() run in the same process:
processing() classically simulates the unmeasured HHL circuit, caches its
successful branch, and returns a measured one-qubit placeholder circuit.
post_processing() returns the cached JSON result, not device amplitudes/counts.
No magnitude, parity, support-detection, or sign-reconstruction jobs are used.

Deploy this module with processing/post_processing as the function entry points.
If uploaded as an additional file, import these entries from actual handler.py.
Dependencies: numpy, scipy, qiskit. No credentials are sent to the function.
Server configuration: QUAPP_BASE_URL (HTTPS origin), QUAPP_API_TOKEN,
QUAPP_PROJECT_ID, QUAPP_TENANT_ID, QUAPP_DEVICE_ID, QUAPP_FUNCTION_NAME.
Optional: QUAPP_PLACEHOLDER_SHOTS (default 1), QUAPP_SOLVE_TIMEOUT_SECONDS (120),
QUAPP_JOB_TIMEOUT_SECONDS (90), QUAPP_POLL_SECONDS (2),
QUAPP_HTTP_TIMEOUT_SECONDS (15), QUAPP_JOB_DETAIL_PATH.
Former parity/shot-allocation environment settings are no longer used.
Placeholder shots do not determine the accuracy of statevector simulation.

The module cache rejects overlapping processing calls rather than overwriting
another invocation's result. Quapp must keep each processing/post_processing
pair in the same process and serialize it there. If a platform failure prevents
post_processing from running, recycle that function process to clear its slot.
A request ID echoed in the result also prevents accepting a stale cached result.

Execution is labeled quapp_statevector (classical circuit simulation on Quapp).
Remote failures retain the authorized, explicit local_statevector_fallback.
Invalid inputs/configuration raise. Finite phase estimation and the reciprocal
lookup's spectral filter still affect accuracy despite exact amplitude access.
No result files are persisted. The existing dashboard currently drops execution
metadata; exposing it requires a separate frontend/backend integration change.
This synchronous solver preserves the app API; app hosting, authentication,
background execution, and whole-analysis deadline handling remain separate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import warnings
import uuid
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from typing import Any, Callable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, RYGate


LOGGER = logging.getLogger(__name__)
MAX_SHOTS = 1_000_000
_CACHE_LOCK = threading.Lock()
_cache: Optional[dict] = None
_CONFIG: Optional["QuappConfig"] = None
_RUNNER: Optional[Callable[[dict, int, float], dict]] = None
_LAST = threading.local()
_RATE_LOCK = threading.Lock()
_RATE_WINDOWS: dict[str, deque] = {}


class QuappConfigurationError(ValueError):
    """The operator must supply a setting; no remote run has been attempted."""


class QuappExecutionError(RuntimeError):
    """Remote execution or statevector result recovery could not complete."""


@dataclass(frozen=True)
class QuappConfig:
    base_url: str
    token: str = field(repr=False)
    project_id: str
    tenant_id: str
    device_id: int
    function_name: str
    placeholder_shots: int = 1
    solve_timeout_seconds: float = 120.0
    job_timeout_seconds: float = 90.0
    poll_seconds: float = 2.0
    http_timeout_seconds: float = 15.0
    job_detail_path: str = "/v1/third-party/jobs/{job_id}/detail"

    def __post_init__(self):
        parts = urlsplit(self.base_url)
        if (parts.scheme != "https" or not parts.netloc or parts.username
                or parts.password or parts.query or parts.fragment
                or parts.path not in ("", "/")):
            raise QuappConfigurationError("QUAPP_BASE_URL must be the explicit HTTPS API origin.")
        for name in ("token", "project_id", "tenant_id", "function_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise QuappConfigurationError(f"Missing Quapp {name}.")
        for name in ("device_id", "placeholder_shots"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise QuappConfigurationError(f"{name} must be a positive integer.")
        if self.placeholder_shots > MAX_SHOTS:
            raise QuappConfigurationError("Placeholder shots exceed the 1,000,000-shot ceiling.")
        for name in ("solve_timeout_seconds", "job_timeout_seconds",
                     "poll_seconds", "http_timeout_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise QuappConfigurationError(f"{name} must be positive and finite.")
        if (not self.job_detail_path.startswith("/") or
                self.job_detail_path.count("{job_id}") != 1 or
                "{" in self.job_detail_path.replace("{job_id}", "") or
                "}" in self.job_detail_path.replace("{job_id}", "") or
                "?" in self.job_detail_path or "#" in self.job_detail_path):
            raise QuappConfigurationError("job_detail_path must be an absolute path with one {job_id}.")

    @classmethod
    def from_environment(cls):
        names = {
            "base_url": "QUAPP_BASE_URL", "token": "QUAPP_API_TOKEN",
            "project_id": "QUAPP_PROJECT_ID", "tenant_id": "QUAPP_TENANT_ID",
            "device_id": "QUAPP_DEVICE_ID", "function_name": "QUAPP_FUNCTION_NAME",
        }
        values = {key: os.environ.get(env, "").strip() for key, env in names.items()}
        missing = [names[key] for key, value in values.items() if not value]
        if missing:
            raise QuappConfigurationError("Configure these server-side settings: " + ", ".join(missing))
        defaults = {
            "placeholder_shots": ("QUAPP_PLACEHOLDER_SHOTS", "1", int),
            "solve_timeout_seconds": ("QUAPP_SOLVE_TIMEOUT_SECONDS", "120", float),
            "job_timeout_seconds": ("QUAPP_JOB_TIMEOUT_SECONDS", "90", float),
            "poll_seconds": ("QUAPP_POLL_SECONDS", "2", float),
            "http_timeout_seconds": ("QUAPP_HTTP_TIMEOUT_SECONDS", "15", float),
        }
        try:
            values["device_id"] = int(values["device_id"])
            for key, (env, default, cast) in defaults.items():
                values[key] = cast(os.environ.get(env, default))
        except ValueError as exc:
            raise QuappConfigurationError("Invalid numeric Quapp setting.") from exc
        if os.environ.get("QUAPP_JOB_DETAIL_PATH"):
            values["job_detail_path"] = os.environ["QUAPP_JOB_DETAIL_PATH"]
        return cls(**values)


def configure_quapp(config: QuappConfig, runner: Optional[Callable] = None):
    """Configure at server startup, before accepting requests.

    Optional runner(payload, raw_shots, absolute_monotonic_deadline) lets Quapp
    supply a native job adapter instead of REST. It must return the same cached
    statevector-branch envelope as post_processing(), respect deadlines, and not
    duplicate jobs. raw_shots are only for the placeholder circuit.
    """
    if not isinstance(config, QuappConfig):
        raise QuappConfigurationError("config must be QuappConfig.")
    if runner is not None and not callable(runner):
        raise QuappConfigurationError("runner must be callable.")
    global _CONFIG, _RUNNER
    _CONFIG, _RUNNER = config, runner


def get_last_execution_summary() -> dict:
    """Small summary for the current worker thread; not a global run archive."""
    return dict(getattr(_LAST, "summary", {}))


def pad_to_power_of_two(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validate a real symmetric system and pad with an uncoupled identity block."""
    A, b = np.asarray(A, dtype=complex), np.asarray(b, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix.")
    if b.ndim != 1 or b.size != A.shape[0]:
        raise ValueError("b must be a vector with the same dimension as A.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain finite values.")
    if np.any(A.imag != 0) or np.any(b.imag != 0):
        raise ValueError("The QuanWatt FDLS solver currently requires real A and b.")
    if not np.allclose(A, A.T, atol=1e-10, rtol=0):
        raise ValueError("A must be symmetric for unitary HHL evolution.")
    # Symmetrize only floating-point roundoff already accepted above.
    A = (A.real + A.real.T) / 2
    n = A.shape[0]
    dim = max(2, 1 << (n - 1).bit_length())
    padded_A, padded_b = np.eye(dim), np.zeros(dim)
    padded_A[:n, :n], padded_b[:n] = A, b.real
    return padded_A, padded_b, n


def gershgorin_lambda_bound(A: np.ndarray) -> float:
    bound = float(np.max(np.sum(np.abs(A), axis=1)))
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid Gershgorin spectral bound.")
    return bound


def recover_scaled_solution(A, b, x_prime):
    """Least-squares scalar k=(Ax)^T b / ||Ax||²; fixes the arbitrary global sign."""
    projected = np.asarray(A) @ np.asarray(x_prime)
    denominator = np.vdot(projected, projected).real
    if denominator <= 0 or not math.isfinite(float(denominator)):
        raise QuappExecutionError("Cannot restore scale from a zero solution direction.")
    coefficient = np.vdot(projected, b) / denominator
    if abs(coefficient.imag) > 1e-8 * max(1, abs(coefficient.real)):
        raise QuappExecutionError("Recovered scale is not real.")
    return coefficient.real * np.asarray(x_prime), float(coefficient.real)


def clean_real_vector(x, tol=1e-8):
    if np.max(np.abs(np.asarray(x).imag)) > tol:
        raise QuappExecutionError("The reconstructed HHL state is not real within tolerance.")
    return np.asarray(x).real


def _settings(phase_qubits, phase_target, C):
    if isinstance(phase_qubits, bool) or not isinstance(phase_qubits, (int, np.integer)) or not 1 <= phase_qubits <= 8:
        raise ValueError("phase_qubits must be an integer from 1 to 8.")
    # Signed phase decoding aliases eigenvalues if the maximum phase reaches 1/2.
    if not math.isfinite(phase_target) or not 0 < phase_target < 0.5:
        raise ValueError("Signed-phase HHL requires 0 < phase_target < 0.5.")
    if not math.isfinite(C) or C <= 0:
        raise ValueError("C must be positive and finite.")


@lru_cache(maxsize=16)
def _evolution(matrix_rows, t):
    """Reuse Hamiltonian evolution across changing FDLS right-hand sides."""
    return expm(1j * np.asarray(matrix_rows) * t)


def controlled_unitary_power(qc, U, power, control_qubit, target_qubits):
    gate = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    gate.unitary(np.linalg.matrix_power(U, power), range(len(target_qubits)))
    qc.append(gate.to_gate().control(1), [control_qubit] + list(target_qubits))


def qpe_circuit(qc, phase_reg, target_reg, U_matrix):
    for wire in phase_reg:
        qc.h(wire)
    # Reversed powers match the existing no-swaps IQFT convention.
    for exponent, control in enumerate(reversed(phase_reg)):
        controlled_unitary_power(qc, U_matrix, 2**exponent, control, target_reg)
    qc.append(QFT(len(phase_reg), inverse=True, do_swaps=False).to_gate(), phase_reg)


def control_rotation_gate(qc, control_reg, target_qubit, t, C=1.0):
    """Exact lookup: controlled Ry(2 asin(C/lambda_d)) for each phase code.

    As in the app, code zero and bins with |lambda_d| < C receive no rotation.
    This is an explicit spectral filter, not exact inversion of those bins.
    The half-turn code is negative under the signed binary convention.
    """
    size = 2**len(control_reg)
    rotations = 0
    for code in range(1, size):
        phase = code / size
        if code >= size // 2:
            phase -= 1
        eigenvalue = 2 * np.pi * phase / t
        amplitude = C / eigenvalue
        if abs(amplitude) > 1:
            continue
        qc.append(RYGate(2 * np.arcsin(amplitude)).control(len(control_reg), ctrl_state=code),
                  list(control_reg) + [target_qubit])
        rotations += 1
    if not rotations:
        raise ValueError("C filters every representable phase bin; reduce C or change phase precision.")


def _build_hhl(A, b, phase_qubits, phase_target, C):
    bound = gershgorin_lambda_bound(A)
    t = phase_target * 2 * np.pi / bound
    target_qubits = int(math.log2(len(b)))
    phase = list(range(phase_qubits))
    target = list(range(phase_qubits, phase_qubits + target_qubits))
    ancilla = phase_qubits + target_qubits
    qc = QuantumCircuit(QuantumRegister(phase_qubits, "phase"),
                        QuantumRegister(target_qubits, "target"), QuantumRegister(1, "ancilla"))
    norm = float(np.linalg.norm(b))
    if norm == 0:
        raise ValueError("A quantum circuit cannot initialize the zero vector.")
    qc.initialize(b / norm, target)
    U = _evolution(tuple(tuple(float(x) for x in row) for row in A), t)
    qpe = QuantumCircuit(phase_qubits + target_qubits, name="QPE")
    qpe_circuit(qpe, phase, target, U)
    qc.append(qpe.to_gate(), phase + target)
    control_rotation_gate(qc, phase, ancilla, t, C)
    qc.append(qpe.inverse().to_gate(), phase + target)
    return qc, target, bound, t


def _statevector_branch(A, b, phase_qubits, phase_target, C):
    """Simulate HHL and retain phase=0, ancilla=1 amplitudes, without measuring.

    Qiskit qubit zero is the least significant statevector-index bit. Our layout
    is phase bits, then system bits, then the inversion ancilla. Preserve the
    unnormalized branch for postselection probability and existing k semantics.
    """
    from qiskit.quantum_info import Statevector
    circuit, _, _, _ = _build_hhl(A, b, phase_qubits, phase_target, C)
    state = Statevector.from_instruction(circuit).data
    ancilla = phase_qubits + int(math.log2(len(b)))
    indices = (np.arange(len(b), dtype=np.int64) << phase_qubits) | (1 << ancilla)
    branch = state[indices]
    probability = float(np.vdot(branch, branch).real)
    if not math.isfinite(probability) or probability <= 0:
        raise QuappExecutionError("HHL has zero or invalid postselection probability.")
    return branch, probability


def processing(invocation_input: Any) -> QuantumCircuit:
    """Quapp-confirmed statevector/cache workflow; device runs a placeholder."""
    global _cache
    if isinstance(invocation_input, str):
        invocation_input = json.loads(invocation_input)
    if not isinstance(invocation_input, dict):
        raise ValueError("Quapp input must be a JSON object.")
    required = ("A", "b", "phase_qubits", "phase_target", "C", "request_id")
    if any(key not in invocation_input for key in required):
        raise ValueError("Input requires A, b, phase_qubits, phase_target, C, request_id.")
    request_id = invocation_input["request_id"]
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a nonempty string.")
    A, b, original = pad_to_power_of_two(invocation_input["A"], invocation_input["b"])
    m = invocation_input["phase_qubits"]
    phase_target, C = float(invocation_input["phase_target"]), float(invocation_input["C"])
    _settings(m, phase_target, C)
    # Lock spans both entries; a second invocation must never overwrite this
    # invocation's result. A plain Lock can be released by post_processing even
    # if Quapp uses a different thread in the same process.
    if not _CACHE_LOCK.acquire(blocking=False):
        raise QuappExecutionError("Function cache is busy; serialize invocations per process.")
    try:
        branch, probability = _statevector_branch(A, b, m, phase_target, C)
        _cache = {"schema_version": "quanwatt-hhl-statevector-1",
                  "request_id": request_id, "execution_mode": "quapp_statevector",
                  "original_dimension": original, "padded_dimension": len(b),
                  "branch_real": branch.real.tolist(),
                  "branch_imag": branch.imag.tolist(),
                  "postselection_rate": probability,
                  "raw_amplitude_norm": math.sqrt(probability)}
        # Measurements make the placeholder valid on Quapp's qasm_simulator;
        # its counts have no relationship to the HHL amplitudes cached above.
        placeholder = QuantumCircuit(1)
        placeholder.measure_all()
        return placeholder
    except Exception:
        _cache = None
        _CACHE_LOCK.release()
        raise


def post_processing(job_result: Any) -> dict:
    """Return the cached simulation result; deliberately ignore device counts."""
    global _cache
    if _cache is None:
        raise QuappExecutionError("No cached HHL result; processing must run in the same process.")
    result = _cache
    _cache = None
    _CACHE_LOCK.release()
    return result


def _pause(seconds, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise QuappExecutionError("Quapp solve deadline reached.")
    time.sleep(min(seconds, remaining))


class _QuappClient:
    def __init__(self, config):
        self.config = config

    def _request(self, method, path, deadline, payload=None):
        cfg = self.config
        headers = {"Authorization": "Bearer " + cfg.token, "X-Project-Id": cfg.project_id,
                   "X-Tenant-Id": cfg.tenant_id, "Content-Type": "application/json", "Accept": "application/json"}
        request = Request(cfg.base_url.rstrip("/") + path, headers=headers, method=method,
                          data=None if payload is None else json.dumps(payload, allow_nan=False).encode())
        retry_delay = cfg.poll_seconds
        while True:
            if method == "POST":
                self._throttle(deadline)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QuappExecutionError("Quapp request deadline reached.")
            try:
                with urlopen(request, timeout=min(cfg.http_timeout_seconds, remaining)) as response:
                    result = json.loads(response.read().decode())
                if not isinstance(result, dict):
                    raise QuappExecutionError("Quapp returned a non-object response.")
                return result
            except HTTPError as exc:
                # A rejected 429 is safe to retry. A POST timeout/5xx is ambiguous:
                # do not submit a duplicate billable job without an idempotency API.
                retryable = exc.code == 429 or (method == "GET" and exc.code in (500, 502, 503, 504))
                if retryable:
                    delay = retry_delay
                    try:
                        delay = max(delay, float(exc.headers.get("Retry-After", "0")))
                    except ValueError:
                        pass
                    _pause(delay, deadline)
                    retry_delay = min(30.0, retry_delay * 2)
                    continue
                raise QuappExecutionError(f"Quapp {method} returned HTTP {exc.code}.") from exc
            except (URLError, TimeoutError, OSError) as exc:
                if method == "GET":
                    _pause(retry_delay, deadline)
                    retry_delay = min(30.0, retry_delay * 2)
                    continue
                raise QuappExecutionError("Quapp submission outcome is unknown; POST was not repeated.") from exc
            except (ValueError, UnicodeError) as exc:
                raise QuappExecutionError("Quapp returned invalid JSON.") from exc

    def _throttle(self, deadline):
        # Token-wide rolling window across threads in this Python process.
        # Other workers still handle the platform's explicit 429 rejection.
        key = self.config.token
        while True:
            with _RATE_LOCK:
                window = _RATE_WINDOWS.setdefault(key, deque())
                now = time.monotonic()
                while window and now - window[0] >= 10:
                    window.popleft()
                if len(window) < 3:
                    window.append(now)
                    return
                delay = max(0.01, 10 - (now - window[0]))
            _pause(delay, deadline)

    def run(self, payload, shots, deadline):
        job_deadline = min(deadline, time.monotonic() + self.config.job_timeout_seconds)
        submitted = self._request("POST", "/api/v1/third-party/function/invoke", job_deadline,
                                  {"functionName": self.config.function_name,
                                   "deviceId": self.config.device_id, "shots": shots,
                                   "description": "QuanWatt HHL statevector simulation", "input": payload})
        job_id = submitted.get("data")
        if not isinstance(job_id, str) or not job_id:
            raise QuappExecutionError("Quapp invoke response lacks a job ID.")
        path = self.config.job_detail_path.replace("{job_id}", quote(job_id, safe=""))
        while True:
            envelope = self._request("GET", path, job_deadline)
            data = envelope.get("data")
            if not isinstance(data, dict):
                raise QuappExecutionError("Quapp job detail lacks a data object.")
            status = str(data.get("status", "")).upper()
            if status == "DONE":
                result = data.get("jobResult")
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except ValueError as exc:
                        raise QuappExecutionError("Invalid jobResult JSON.") from exc
                if not isinstance(result, dict):
                    raise QuappExecutionError("Completed Quapp job has no cached statevector result.")
                return {**result, "job_id": job_id}
            if status in ("ERROR", "IN_DEAD_LETTER_QUEUE", "FAILED", "CANCELLED"):
                raise QuappExecutionError(f"Quapp job {job_id} ended with {status}.")
            if status not in ("NEW", "INITIALIZING", "IN_QUEUE", "RUNNING"):
                raise QuappExecutionError(f"Unknown Quapp job status: {status!r}.")
            _pause(self.config.poll_seconds, job_deadline)


def _decode_remote_branch(result, payload, dimension, original):
    """Validate schema, invocation identity, dimensions, and complex amplitudes."""
    if (not isinstance(result, dict) or
            result.get("schema_version") != "quanwatt-hhl-statevector-1" or
            result.get("execution_mode") != "quapp_statevector" or
            result.get("request_id") != payload["request_id"] or
            result.get("padded_dimension") != dimension or
            result.get("original_dimension") != original):
        raise QuappExecutionError("Quapp returned an incompatible or stale statevector result.")
    try:
        real = np.asarray(result["branch_real"], dtype=float)
        imag = np.asarray(result["branch_imag"], dtype=float)
    except (KeyError, ValueError, TypeError) as exc:
        raise QuappExecutionError("Invalid statevector-branch arrays.") from exc
    if (real.shape != (dimension,) or imag.shape != (dimension,) or
            not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag))):
        raise QuappExecutionError("Statevector branch has invalid dimensions or nonfinite values.")
    raw = real + 1j * imag
    probability = float(np.vdot(raw, raw).real)
    if not 0 < probability <= 1 + 1e-8:
        raise QuappExecutionError("Invalid HHL postselection probability.")
    return raw, {"raw_amplitude_norm": math.sqrt(probability),
                 "postselection_rate": probability}


def hhl_solve_with_diagnostics(A: np.ndarray, b: np.ndarray, phase_qubits: int = 8,
                               phase_target: float = 0.40, C: float = 1.0,
                               label: str = "HHL", *, config: Optional[QuappConfig] = None,
                               runner: Optional[Callable] = None) -> dict:
    """Solve Ax=b via Quapp-hosted statevector simulation, preserving app outputs.

    Uses one invocation per nonzero system and no measurement-based readout.
    The returned k multiplies the unnormalized successful HHL branch, as before.
    Remote execution failures trigger the explicitly labeled local fallback.
    """
    _LAST.summary = {}
    padded_A, padded_b, original = pad_to_power_of_two(A, b)
    _settings(phase_qubits, phase_target, C)
    bound = gershgorin_lambda_bound(padded_A)
    t = phase_target * 2 * np.pi / bound
    if np.linalg.norm(padded_b) == 0:
        summary = {"execution_mode": "zero_rhs", "requested_raw_shots": 0,
                   "jobs": [], "fallback_used": False}
        _LAST.summary = summary
        return {"solution": np.zeros(original), "lambda_bound": bound, "t_eff": t,
                "k_coefficient": 0.0, "residual": 0.0, "raw_amplitude_norm": 0.0, **summary}
    cfg = config or _CONFIG or QuappConfig.from_environment()
    if not isinstance(cfg, QuappConfig):
        raise QuappConfigurationError("config must be QuappConfig.")
    adapter = runner or _RUNNER or _QuappClient(cfg).run
    if not callable(adapter):
        raise QuappConfigurationError("runner must be callable.")
    payload = {"A": np.asarray(A).real.tolist(), "b": np.asarray(b).real.tolist(),
               "phase_qubits": int(phase_qubits), "phase_target": float(phase_target),
               "C": float(C), "request_id": uuid.uuid4().hex}
    jobs = [{"setting": "statevector", "requested_shots": cfg.placeholder_shots,
             "status": "submitted"}]
    accounting = {"requested_raw_shots": cfg.placeholder_shots, "jobs": jobs,
                  "readout_strategy": "direct_statevector",
                  "shots_purpose": "placeholder_only"}
    deadline = time.monotonic() + cfg.solve_timeout_seconds
    try:
        result = adapter(payload, cfg.placeholder_shots, deadline)
        raw, info = _decode_remote_branch(result, payload, len(padded_b), original)
        jobs[0].update(job_id=result.get("job_id"), status="completed")
        # Validate the real solution direction within the fallback boundary too.
        raw = clean_real_vector(raw)
        execution = {"execution_mode": "quapp_statevector", "fallback_used": False}
    except QuappExecutionError as exc:
        jobs[0].update(status="failed_or_unknown")
        execution = {"execution_mode": "local_statevector_fallback", "fallback_used": True,
                     "fallback_reason": str(exc)}
        message = f"{label}: LOCAL STATEVECTOR FALLBACK; remote result unavailable: {exc}"
        LOGGER.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _LAST.summary = {**accounting, **execution}
        branch, probability = _statevector_branch(padded_A, padded_b, phase_qubits, phase_target, C)
        raw = clean_real_vector(branch)
        info = {"raw_amplitude_norm": math.sqrt(probability), "postselection_rate": probability}
    recovered, coefficient = recover_scaled_solution(padded_A, padded_b, raw)
    solution = clean_real_vector(recovered[:original])
    summary = {**accounting, **info, **execution}
    _LAST.summary = summary
    LOGGER.info("%s execution=%s placeholder_shots=%s", label,
                execution["execution_mode"], cfg.placeholder_shots)
    return {"solution": solution, "lambda_bound": bound, "t_eff": t,
            "k_coefficient": coefficient,
            "residual": float(np.linalg.norm(np.asarray(A) @ solution - np.asarray(b))), **summary}


def hhl_solve(A: np.ndarray, b: np.ndarray, phase_qubits: int = 8,
              phase_target: float = 0.40, C: float = 1.0, label: str = "HHL", *,
              config: Optional[QuappConfig] = None, runner: Optional[Callable] = None) -> np.ndarray:
    """Compatible vector-returning app entry point; fallback also emits a warning."""
    return hhl_solve_with_diagnostics(A, b, phase_qubits, phase_target, C, label,
                                      config=config, runner=runner)["solution"]
