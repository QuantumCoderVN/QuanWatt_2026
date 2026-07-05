# ============================================================
# main.py
# ============================================================

import numpy as np

from ieee_case import (
    load_ieee14_reduced_3bus,
    make_ybus,
    make_B_matrices,
    print_ieee_data,
)

from quantum_solvers import (
    classical_solve,
    hhl_solve,
    vqls_solve,
)

from fdls import fdls_power_flow

from plotting import (
    plot_loss,
    plot_voltage_magnitude,
    plot_voltage_angle,
)


def print_linear_solution_summary(results):
    print("\n" + "=" * 80)
    print("FIRST ITERATION LINEAR SOLVER RESULTS")
    print("=" * 80)

    for r in results:
        print("\n" + "-" * 80)
        print(r["name"])
        print("-" * 80)

        print("First Δθ solution:")
        print(r["first_theta_solution"])

        print("First ΔV solution:")
        print(r["first_v_solution"])


def print_final_summary(results):
    print("\n" + "=" * 80)
    print("FINAL FDLS RESULTS")
    print("=" * 80)

    for r in results:
        print("\n" + "-" * 80)
        print(r["name"])
        print("-" * 80)

        print("Converged:", r["converged"])
        print("Iterations:", r["iterations"])
        print("Final loss:", r["final_loss"])

        print("\nVm:")
        print(r["vm"])

        print("\nVa rad:")
        print(r["va"])

        print("\nVa degree:")
        print(r["va_deg"])

        print("\nP_calc:")
        print(r["P_calc"])

        print("\nQ_calc:")
        print(r["Q_calc"])


def main():
    np.set_printoptions(precision=10, suppress=True)

    case = load_ieee14_reduced_3bus()

    Ybus = make_ybus(case)
    Bprime, Bdouble = make_B_matrices(case, Ybus)

    # In data IEEE
    print_ieee_data(case, Ybus, Bprime, Bdouble)

    solvers = [
        ("Classical", classical_solve),
        ("HHL", hhl_solve),
        ("VQLS", vqls_solve),
    ]

    results = []

    for name, solver in solvers:
        print("\n" + "#" * 80)
        print(f"RUNNING FDLS WITH {name}")
        print("#" * 80)

        result = fdls_power_flow(
            case=case,
            Ybus=Ybus,
            Bprime=Bprime,
            Bdouble=Bdouble,
            solver=solver,
            name=name,
        )

        results.append(result)

    # In nghiệm tuyến tính vòng đầu:
    # Δθ và ΔV của cả Classical, HHL, VQLS
    print_linear_solution_summary(results)

    # In nghiệm FDLS cuối cùng:
    # Vm, Va của cả Classical, HHL, VQLS
    print_final_summary(results)

    # Vẽ đồ thị loss
    plot_loss(results)

    # Vẽ biểu đồ nghiệm
    plot_voltage_magnitude(results)
    plot_voltage_angle(results)


if __name__ == "__main__":
    main()