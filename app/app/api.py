# ============================================================
# api.py
# FastAPI backend for HHL Power-Flow Dashboard
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import time

from .network import default_case, validate_case, prepare_case_indices
from .solver import solve_power_flow
from .config import (
    get_default_solver_settings,
    get_case_limits,
    validate_solver_settings,
    MIN_BUSES, MAX_BUSES,
)
from .api_models import numpy_to_serializable

app = FastAPI(
    title="HHL Power-Flow API",
    description="Quantum HHL solver for power flow analysis",
    version="1.0.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

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


# ============================================================
# Endpoints
# ============================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hhl-power-flow",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.get("/api/case")
async def get_default_case():
    """
    Get the default IEEE-14-derived 3-bus case and system limits.
    """
    case = default_case()
    limits = get_case_limits()
    settings = get_default_solver_settings()

    # Convert case to API format
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
    Solve power flow using HHL and Classical methods.

    Returns comparison results with convergence history,
    voltages, powers, and error metrics.
    """
    try:
        # Convert request to internal case format
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

        # Validate bus count
        n_buses = len(bus_array)
        if n_buses < MIN_BUSES or n_buses > MAX_BUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Bus count must be between {MIN_BUSES} and {MAX_BUSES}, got {n_buses}"
            )

        # Prepare settings
        settings = get_default_solver_settings()
        if request.settings:
            settings.update({
                "phase_qubits": request.settings.phase_qubits,
                "max_iter": request.settings.max_iter,
                "tolerance": request.settings.tolerance,
                "phase_target": request.settings.phase_target,
                "C": request.settings.C,
            })

        # Solve
        comparison, error = solve_power_flow(
            case=case,
            settings=settings,
            use_adaptive_c=request.settings.use_adaptive_c if request.settings else False,
        )

        if error:
            raise HTTPException(status_code=400, detail=error)

        # Convert to JSON-serializable format
        result = comparison.to_dict()
        result = numpy_to_serializable(result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/limits")
async def get_limits():
    """Get system limits and constraints."""
    limits = get_case_limits()
    return limits


# ============================================================
# Application Entry Point
# ============================================================

# For Vercel deployment with Mangum
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
