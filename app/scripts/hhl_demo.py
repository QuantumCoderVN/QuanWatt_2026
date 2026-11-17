#!/usr/bin/env python3
"""Standalone HHL power-flow demo (FDLS locally; each Ax=b via Quapp).

Usage (from repo ``app/`` directory):

    conda activate scquan
    python scripts/hhl_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from backend.network import default_case
from backend.solver import solve_power_flow
from backend.config import get_default_solver_settings


def print_separator(char="=", width=80):
    print(char * width)


def print_section(title):
    print("\n")
    print_separator()
    print(title)
    print_separator()


def print_case_info(case):
    print_section("CASE INFORMATION")
    print(f"Base MVA: {case['base_mva']}")
    print(f"Number of buses: {len(case['bus'])}")
    print(f"Number of branches: {len(case['branch'])}")
    print(f"Slack buses: {case['slack']}")
    print(f"PV buses: {case['pv']}")
    print(f"PQ buses: {case['pq']}")


def print_solver_results(result):
    print_section(f"{result.name.upper()} RESULTS")

    print(f"\nStatus: {result.status}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final mismatch: {result.final_mismatch:.10e} p.u.")
    print(f"Elapsed time: {result.elapsed_time:.4f} seconds")

    print("\nVoltage Magnitudes:")
    for bus_res in result.bus_results:
        print(f"  Bus {bus_res.bus_id}: {bus_res.vm:.10f} p.u.")

    print("\nVoltage Angles (degrees):")
    for bus_res in result.bus_results:
        print(f"  Bus {bus_res.bus_id}: {bus_res.va_deg:.10f}°")

    print("\nPower Injections:")
    for bus_res in result.bus_results:
        print(f"  Bus {bus_res.bus_id}: P={bus_res.p_calc:.4f} MW, Q={bus_res.q_calc:.4f} MVAr")

    if result.first_delta_theta is not None:
        print("\nFirst Δθ solution:")
        print(f"  {np.array(result.first_delta_theta)}")

    if result.first_delta_v is not None:
        print("\nFirst ΔV solution:")
        print(f"  {np.array(result.first_delta_v)}")

    if result.diagnostics:
        print("\nHHL Diagnostics:")
        if "bprime_scaling" in result.diagnostics:
            print("  B' matrix:")
            for key, val in result.diagnostics["bprime_scaling"].items():
                print(f"    {key}: {val}")
        if "bdouble_scaling" in result.diagnostics:
            print("  B'' matrix:")
            for key, val in result.diagnostics["bdouble_scaling"].items():
                print(f"    {key}: {val}")


def main():
    print_separator("=")
    print("HHL POWER FLOW - STANDALONE DEMONSTRATION")
    print_separator("=")

    case = default_case()
    print_case_info(case)

    settings = get_default_solver_settings()

    print_section("SOLVER SETTINGS")
    print(f"Phase qubits: {settings['phase_qubits']}")
    print(f"Max iterations: {settings['max_iter']}")
    print(f"Tolerance: {settings['tolerance']:.1e}")
    print(f"Phase target: {settings['phase_target']}")
    print(f"C constant: {settings['C']}")

    print_section("RUNNING HHL POWER FLOW")
    print("FDLS on host; each Ax=b via Quapp HHL...")

    result, error = solve_power_flow(case, settings)

    if error:
        print(f"\n[ERROR] {error}")
        sys.exit(1)

    print_solver_results(result)

    print("\n")
    print_separator("=")
    print("DEMONSTRATION COMPLETE")
    print_separator("=")


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True)
    main()
