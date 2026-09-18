#!/usr/bin/env python3
# ============================================================
# hhl.py
# Standalone HHL power-flow demonstration
# ============================================================

"""
Standalone HHL Power Flow Script

This script runs the HHL quantum linear solver on the IEEE-14-derived 3-bus
test case and compares it with the classical solver.

Usage:
    python app/hhl.py
"""

import numpy as np
import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir.parent))

from network import default_case
from solver import solve_power_flow
from config import get_default_solver_settings


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


def print_comparison(comparison):
    print_section("COMPARISON: HHL vs CLASSICAL")

    print("\nVoltage Magnitude Errors:")
    for i, error in enumerate(comparison.voltage_magnitude_errors):
        print(f"  Bus {comparison.hhl_result.bus_results[i].bus_id}: {error:.10e} p.u.")
    print(f"  Maximum: {comparison.max_vm_error:.10e} p.u.")

    print("\nVoltage Angle Errors:")
    for i, error in enumerate(comparison.voltage_angle_errors):
        print(f"  Bus {comparison.hhl_result.bus_results[i].bus_id}: {error:.10e}°")
    print(f"  Maximum: {comparison.max_va_error:.10e}°")

    if comparison.first_theta_error is not None:
        print(f"\nFirst Δθ error: {comparison.first_theta_error:.10e}")

    if comparison.first_v_error is not None:
        print(f"First ΔV error: {comparison.first_v_error:.10e}")

    # Check reference baseline
    baseline_vm_tolerance = 1e-7
    baseline_va_tolerance = 1e-7

    print("\nReference Validation:")
    if comparison.max_vm_error < baseline_vm_tolerance:
        print(f"  ✓ Voltage magnitude within baseline tolerance ({baseline_vm_tolerance})")
    else:
        print(f"  ✗ Voltage magnitude exceeds baseline tolerance ({baseline_vm_tolerance})")

    if comparison.max_va_error < baseline_va_tolerance:
        print(f"  ✓ Voltage angle within baseline tolerance ({baseline_va_tolerance})")
    else:
        print(f"  ✗ Voltage angle exceeds baseline tolerance ({baseline_va_tolerance})")

    if (comparison.hhl_result.converged and
        comparison.hhl_result.iterations <= 8 and
        comparison.hhl_result.final_mismatch < 1e-7):
        print(f"  ✓ HHL converged within 8 iterations with mismatch < 1e-7")
    else:
        print(f"  ✗ HHL convergence criteria not met")


def main():
    print_separator("=")
    print("HHL POWER FLOW - STANDALONE DEMONSTRATION")
    print_separator("=")

    # Load default case
    case = default_case()
    print_case_info(case)

    # Get default settings
    settings = get_default_solver_settings()

    print_section("SOLVER SETTINGS")
    print(f"Phase qubits: {settings['phase_qubits']}")
    print(f"Max iterations: {settings['max_iter']}")
    print(f"Tolerance: {settings['tolerance']:.1e}")
    print(f"Phase target: {settings['phase_target']}")
    print(f"C constant: {settings['C']}")

    # Solve
    print_section("RUNNING SOLVERS")
    print("This may take a few moments...")

    comparison, error = solve_power_flow(case, settings)

    if error:
        print(f"\n[ERROR] {error}")
        sys.exit(1)

    # Print results
    print_solver_results(comparison.classical_result)
    print_solver_results(comparison.hhl_result)
    print_comparison(comparison)

    print("\n")
    print_separator("=")
    print("DEMONSTRATION COMPLETE")
    print_separator("=")
    print("\nOutputs saved to: outputs/ (if plotting enabled)")


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True)
    main()
