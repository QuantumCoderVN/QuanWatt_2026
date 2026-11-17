# ============================================================
# api.py
# FastAPI proxy: FDLS on proxy, each Ax=b via Quapp HHL
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import asyncio
import json
import numpy as np
import os
import queue
import threading
import time

from .network import default_case
from .solver import solve_power_flow
from .config import (
    get_default_solver_settings,
    get_case_limits,
    MIN_BUSES, MAX_BUSES,
)
from .api_models import numpy_to_serializable


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE from .env into os.environ (do not override existing)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_APP_ROOT = Path(__file__).resolve().parents[1]
_load_dotenv(_APP_ROOT / ".env")
_load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(
    title="HHL Power-Flow API",
    description="Power-flow via Quapp HHL (FDLS on proxy)",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BusData(BaseModel):
    bus_id: int = Field(..., ge=1)
    bus_type: int = Field(..., ge=1, le=3)
    pd: float = 0.0
    qd: float = 0.0
    gs: float = 0.0
    bs: float = 0.0
    vm: float = Field(..., gt=0)
    va: float = 0.0


class GenData(BaseModel):
    gen_bus: int = Field(..., ge=1)
    pg: float = 0.0
    qg: float = 0.0
    vg: float = Field(..., gt=0)


class BranchData(BaseModel):
    f_bus: int = Field(..., ge=1)
    t_bus: int = Field(..., ge=1)
    r: float = 0.0
    x: float = 0.0
    b: float = 0.0
    tap: float = 0.0
    shift: float = 0.0
    status: int = Field(..., ge=0, le=1)


class CaseRequest(BaseModel):
    base_mva: float = Field(..., gt=0)
    bus: List[BusData]
    gen: List[GenData] = []
    branch: List[BranchData]


class SolverSettingsRequest(BaseModel):
    phase_qubits: int = Field(8, ge=4, le=8)
    max_iter: int = Field(8, ge=1, le=12)
    tolerance: float = Field(1e-7, gt=0)
    phase_target: float = Field(0.40, gt=0, lt=1)
    C: float = Field(1.0, gt=0)
    use_adaptive_c: bool = False


class SolveRequest(BaseModel):
    case: CaseRequest
    settings: Optional[SolverSettingsRequest] = None


def _prepare_solve(request: SolveRequest) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Validate request and build internal case + settings."""
    print(f"📊 Converting case data...")
    print(f"   - Buses: {len(request.case.bus)}")
    print(f"   - Branches: {len(request.case.branch)}")
    print(f"   - Generators: {len(request.case.gen)}")

    bus_array = np.array([
        [
            bus.bus_id,
            bus.bus_type,
            bus.pd,
            bus.qd,
            bus.gs,
            bus.bs,
            bus.vm,
            bus.va,
        ]
        for bus in request.case.bus
    ], dtype=float)

    gen_array = np.array([
        [gen.gen_bus, gen.pg, gen.qg, gen.vg]
        for gen in request.case.gen
    ], dtype=float) if request.case.gen else np.array([]).reshape(0, 4)

    branch_array = np.array([
        [
            branch.f_bus,
            branch.t_bus,
            branch.r,
            branch.x,
            branch.b,
            branch.tap,
            branch.shift,
            branch.status,
        ]
        for branch in request.case.branch
    ], dtype=float)

    case = {
        "base_mva": request.case.base_mva,
        "bus": bus_array,
        "gen": gen_array,
        "branch": branch_array,
    }

    n_buses = len(bus_array)
    print(f"✅ Case assembled: {n_buses} buses")

    if n_buses < MIN_BUSES or n_buses > MAX_BUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Bus count must be between {MIN_BUSES} and {MAX_BUSES}, got {n_buses}"
        )

    settings = get_default_solver_settings()
    if request.settings:
        settings.update({
            "phase_qubits": request.settings.phase_qubits,
            "max_iter": request.settings.max_iter,
            "tolerance": request.settings.tolerance,
            "phase_target": request.settings.phase_target,
            "C": request.settings.C,
        })

    use_adaptive_c = request.settings.use_adaptive_c if request.settings else False

    print(f"\n⚙️  SOLVER SETTINGS:")
    print(f"   - Phase Qubits: {settings['phase_qubits']}")
    print(f"   - Max Iterations: {settings['max_iter']}")
    print(f"   - Tolerance: {settings['tolerance']:.2e}")
    print(f"   - Phase Target: {settings['phase_target']}")
    print(f"   - C: {settings['C']}")
    print(f"   - Adaptive C: {use_adaptive_c}")

    return case, settings, use_adaptive_c


@app.get("/api/health")
async def health_check():
    """Health check endpoint (proxy liveness)."""
    return {
        "status": "healthy",
        "service": "hhl-power-flow-proxy",
        "version": "1.1.0",
        "timestamp": time.time(),
    }


@app.get("/api/case")
async def get_default_case():
    """Get the default IEEE-14-derived 3-bus case and system limits."""
    case = default_case()
    limits = get_case_limits()
    settings = get_default_solver_settings()

    bus_list = []
    for bus_row in case["bus"]:
        bus_list.append({
            "bus_id": int(bus_row[0]),
            "bus_type": int(bus_row[1]),
            "pd": float(bus_row[2]),
            "qd": float(bus_row[3]),
            "gs": float(bus_row[4]),
            "bs": float(bus_row[5]),
            "vm": float(bus_row[6]),
            "va": float(bus_row[7]),
        })

    gen_list = []
    for gen_row in case["gen"]:
        gen_list.append({
            "gen_bus": int(gen_row[0]),
            "pg": float(gen_row[1]),
            "qg": float(gen_row[2]),
            "vg": float(gen_row[3]),
        })

    branch_list = []
    for branch_row in case["branch"]:
        branch_list.append({
            "f_bus": int(branch_row[0]),
            "t_bus": int(branch_row[1]),
            "r": float(branch_row[2]),
            "x": float(branch_row[3]),
            "b": float(branch_row[4]),
            "tap": float(branch_row[5]),
            "shift": float(branch_row[6]),
            "status": int(branch_row[7]),
        })

    return {
        "case": {
            "base_mva": float(case["base_mva"]),
            "bus": bus_list,
            "gen": gen_list,
            "branch": branch_list,
        },
        "limits": limits,
        "default_settings": settings,
    }


@app.post("/api/solve")
async def solve_case(request: SolveRequest):
    """
    Solve power flow: FDLS on this proxy, each linear system via Quapp HHL.

    Returns a single HHL SolverResult (no classical comparison).
    """
    print("\n" + "=" * 80)
    print("🔵 NEW REQUEST: POST /api/solve")
    print("=" * 80)

    try:
        case, settings, use_adaptive_c = _prepare_solve(request)

        print(f"\n🚀 Starting HHL power flow (Quapp)...")

        result, error = solve_power_flow(
            case=case,
            settings=settings,
            use_adaptive_c=use_adaptive_c,
        )

        if error:
            print(f"\n❌ SOLVER ERROR: {error}")
            raise HTTPException(status_code=400, detail=error)

        payload = numpy_to_serializable(result.to_dict())

        diag = result.diagnostics or {}
        print(f"\n✅ SOLVE COMPLETE!")
        print(f"   HHL: {result.iterations} iterations, {result.elapsed_time:.2f}s")
        print(f"   Status: {result.status}  Converged: {result.converged}")
        print(f"   Quapp solves: {diag.get('quapp_solve_count')}  fallback: {diag.get('fallback_solve_count')}")
        print("=" * 80 + "\n")

        return payload

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ INTERNAL ERROR: {str(e)}")
        print("=" * 80 + "\n")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/solve/stream")
async def solve_case_stream(request: SolveRequest):
    """
    Same solve as /api/solve, but streams SSE progress events then a final result.

    Event types:
      data: {"type":"status", "phase":..., "message":..., ...}
      data: {"type":"result", "data": <SolverResult>}
      data: {"type":"error", "message":...}
    """
    print("\n" + "=" * 80)
    print("🔵 NEW REQUEST: POST /api/solve/stream")
    print("=" * 80)

    case, settings, use_adaptive_c = _prepare_solve(request)
    progress_q: queue.Queue = queue.Queue()

    def emit(event: Dict[str, Any]) -> None:
        progress_q.put(event)

    def worker() -> None:
        try:
            print(f"\n🚀 Starting HHL power flow (Quapp, streamed)...")
            result, error = solve_power_flow(
                case=case,
                settings=settings,
                use_adaptive_c=use_adaptive_c,
                on_progress=emit,
            )
            if error:
                print(f"\n❌ SOLVER ERROR: {error}")
                progress_q.put({"type": "error", "message": error})
            else:
                payload = numpy_to_serializable(result.to_dict())
                diag = result.diagnostics or {}
                print(f"\n✅ SOLVE COMPLETE!")
                print(f"   HHL: {result.iterations} iterations, {result.elapsed_time:.2f}s")
                print(f"   Status: {result.status}  Converged: {result.converged}")
                print(
                    f"   Quapp solves: {diag.get('quapp_solve_count')}  "
                    f"fallback: {diag.get('fallback_solve_count')}"
                )
                print("=" * 80 + "\n")
                progress_q.put({"type": "result", "data": payload})
        except Exception as exc:
            print(f"\n❌ INTERNAL ERROR: {exc}")
            print("=" * 80 + "\n")
            progress_q.put({"type": "error", "message": f"Internal error: {exc}"})
        finally:
            progress_q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_generator():
        while True:
            item = await asyncio.to_thread(progress_q.get)
            if item is None:
                break
            yield f"data: {json.dumps(item, allow_nan=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/limits")
async def get_limits():
    """Get system limits and constraints."""
    return get_case_limits()


try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
