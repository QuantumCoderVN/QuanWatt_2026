# ============================================================
# main_qopf3.py
# Run Classical SC-PDIPM and VQLS SC-PDIPM, then compare
# ============================================================

import traceback
import numpy as np

import config_qopf3 as cfg

from case3_dc_opf import load_case3_dc_opf_from_ieee14
from kkt_builder import build_dc_opf_qp
from sc_pdipm_solver import solve_opf_scpdipm, VQLSResidualError

from plot_utils import (
    plot_vqls_loss,
    plot_solution_comparison,
    plot_pdipm_convergence,
    plot_kkt_condition_history,
)


def get_cfg(name, default):
    return getattr(cfg, name, default)


def print_matrix(name, A):
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(np.array2string(np.asarray(A), precision=6, suppress_small=True))
    print()


def print_case_info(case):
    print("=" * 80)
    print("IEEE14-subcase-3bus information")
    print("=" * 80)
    print("case name        :", case["name"])
    print("baseMVA          :", case["baseMVA"])
    print("selected buses   :", case["selected_buses"])
    print("slack bus number :", case["slack_bus_number"])
    print("number of buses  :", case["bus"].shape[0])
    print("number of gens   :", case["gen"].shape[0])
    print("number of branch :", case["branch"].shape[0])
    print()


def print_qp_info(qp):
    print_matrix("H", qp["H"])
    print_matrix("f", qp["f"])
    print_matrix("Aeq", qp["Aeq"])
    print_matrix("beq", qp["beq"])
    print_matrix("Aineq branch only", qp["Aineq"])
    print_matrix("bineq branch only", qp["bineq"])
    print_matrix("lb", qp["lb"])
    print_matrix("ub", qp["ub"])

    nX = qp["nvar"]
    nZ = qp["Aineq"].shape[0] + 2 * nX
    nlambda = qp["Aeq"].shape[0]
    nmu = nZ
    total = nX + nZ + nlambda + nmu

    padded = 1
    while padded < total:
        padded *= 2

    print("=" * 80)
    print("Expected SC-PDIPM KKT size")
    print("=" * 80)
    print("nX      :", nX)
    print("nZ      :", nZ)
    print("nlambda :", nlambda)
    print("nmu     :", nmu)
    print("total KKT dimension:", total)
    print("VQLS padded dimension:", padded)
    print("VQLS required qubits :", int(np.log2(padded)))
    print()


def print_result(result):
    qp = result["qp"]
    X = result["X"]

    print("=" * 80)
    print(f"Final result: {result['method']}")
    print("=" * 80)

    print("converged    :", result["converged"])
    print("iterations   :", result["iterations"])
    print("linear solver:", result["linear_solver"])
    print()

    print("X variables:")
    for name, value in zip(qp["variable_names"], X):
        print(f"{name:28s}: {value: .12f}")

    print()
    print("Cost:")
    print(result["cost"])

    print()
    print("Final conditions:")
    print("feascond:", result["feascond"])
    print("gradcond:", result["gradcond"])
    print("compcond:", result["compcond"])
    print("gamma   :", result["gamma"])

    print()
    print("Active inequalities:")
    if len(result["active_inequalities"]) == 0:
        print("None")
    else:
        for name in result["active_inequalities"]:
            print(" -", name)

    print()
    print("Physical interpretation:")
    for k, v in result["parsed"].items():
        print(f"{k:32s}: {v}")

    print()


def print_power_balance(result):
    qp = result["qp"]
    X = result["X"]

    lhs = qp["Aeq"] @ X
    rhs = qp["beq"]
    mismatch = lhs - rhs

    print("=" * 80)
    print(f"Power balance check: {result['method']}")
    print("=" * 80)

    for i in range(len(rhs)):
        print(
            f"bus_internal_{i}: "
            f"lhs={lhs[i]: .12f}, "
            f"rhs={rhs[i]: .12f}, "
            f"mismatch={mismatch[i]: .3e}"
        )

    print()


def compare_results(classical_result, vqls_result):
    names = classical_result["qp"]["variable_names"]

    X_classical = np.asarray(classical_result["X"], dtype=float)
    X_vqls = np.asarray(vqls_result["X"], dtype=float)

    print("=" * 80)
    print("Classical vs VQLS final solution comparison")
    print("=" * 80)

    print(f"{'variable':28s} {'classical':>15s} {'vqls':>15s} {'abs_error':>15s}")

    for name, a, b in zip(names, X_classical, X_vqls):
        print(f"{name:28s} {a:15.8f} {b:15.8f} {abs(a-b):15.6e}")

    print()
    print("Cost classical:", classical_result["cost"])
    print("Cost VQLS     :", vqls_result["cost"])
    print("Cost error    :", abs(classical_result["cost"] - vqls_result["cost"]))

    print()
    print("X error norm:")
    print(np.linalg.norm(X_classical - X_vqls))

    print()
    print("Classical final conditions:")
    print(
        "feas=", classical_result["feascond"],
        "grad=", classical_result["gradcond"],
        "comp=", classical_result["compcond"],
    )

    print("VQLS final conditions:")
    print(
        "feas=", vqls_result["feascond"],
        "grad=", vqls_result["gradcond"],
        "comp=", vqls_result["compcond"],
    )
    print()


def main():
    run_classical = get_cfg("RUN_CLASSICAL_BASELINE", True)
    run_vqls = get_cfg("RUN_VQLS_BASELINE", True)
    print_matrix_detail = get_cfg("PRINT_MATRIX_DETAIL", True)

    case = load_case3_dc_opf_from_ieee14()

    print_case_info(case)

    qp = build_dc_opf_qp(case)

    if print_matrix_detail:
        print_qp_info(qp)

    classical_result = None
    vqls_result = None

    # --------------------------------------------------------
    # Run Classical SC-PDIPM
    # --------------------------------------------------------
    if run_classical:
        print("=" * 80)
        print("RUN 1: SC-PDIPM with Classical linear solver")
        print("=" * 80)

        classical_result = solve_opf_scpdipm(
            case,
            linear_solver="classical",
        )

        print_result(classical_result)
        print_power_balance(classical_result)

        plot_pdipm_convergence(
            classical_result,
            filename="classical_pdipm_convergence.png",
        )

        plot_kkt_condition_history(
            classical_result,
            filename="classical_kkt_condition_history.png",
        )
    # --------------------------------------------------------
    # Run VQLS SC-PDIPM
    # --------------------------------------------------------
    if run_vqls:
        print("=" * 80)
        print("RUN 2: SC-PDIPM with VQLS linear solver")
        print("=" * 80)

        try:
            vqls_result = solve_opf_scpdipm(
                case,
                linear_solver="vqls",
            )

            print_result(vqls_result)
            print_power_balance(vqls_result)

            plot_pdipm_convergence(
                vqls_result,
                filename="vqls_pdipm_convergence.png",
            )

            plot_vqls_loss(
                vqls_result.get("vqls_loss_history", []),
                filename="vqls_loss.png",
            )

        except VQLSResidualError as e:
            print("=" * 80)
            print("VQLS SC-PDIPM stopped safely")
            print("=" * 80)
            print("Reason:")
            print(e)
            print()

            info = e.info

            print("VQLS diagnostic:")
            print("loss              :", info.get("loss"))
            print("residual          :", info.get("residual"))
            print("relative residual :", info.get("relative_residual"))
            print("padded dim        :", info.get("padded_dim"))
            print("n_qubits          :", info.get("n_qubits"))
            print()

            plot_vqls_loss(
                info.get("loss_history", []),
                filename="vqls_loss_failed_iteration.png",
            )

        except Exception as e:
            print("=" * 80)
            print("VQLS SC-PDIPM failed")
            print("=" * 80)
            print("Error:")
            print(e)
            print()
            print("Traceback:")
            traceback.print_exc()
            print()

    # --------------------------------------------------------
    # Compare final solutions
    # --------------------------------------------------------
    if classical_result is not None and vqls_result is not None:
        compare_results(classical_result, vqls_result)

        plot_solution_comparison(
            classical_result,
            vqls_result,
            filename="solution_comparison.png",
        )


if __name__ == "__main__":
    main()