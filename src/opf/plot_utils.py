# ============================================================
# plot_utils.py
# Plot VQLS loss and compare Classical vs VQLS SC-PDIPM results
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

import config_qopf3 as cfg


def get_cfg(name, default):
    return getattr(cfg, name, default)


def ensure_output_dir():
    output_dir = get_cfg("OUTPUT_DIR", "outputs_qopf3")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_or_show(filename):
    save_figures = get_cfg("SAVE_FIGURES", True)
    output_dir = ensure_output_dir()

    if save_figures:
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=200)
        print(f"Saved: {path}")

    plt.show()


def plot_vqls_loss(loss_history, filename="vqls_loss.png"):
    """
    Plot concatenated VQLS loss history collected across all SC-PDIPM KKT solves.

    Note:
        If VQLS is called at multiple SC-PDIPM iterations, this plot concatenates
        all optimizer loss values into one long curve.
    """

    if loss_history is None or len(loss_history) == 0:
        print("No VQLS loss history to plot.")
        return

    loss_history = np.asarray(loss_history, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=2)
    plt.yscale("log")
    plt.xlabel("VQLS optimizer function evaluation")
    plt.ylabel("VQLS global loss")
    plt.title("VQLS loss during SC-PDIPM KKT solves")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_or_show(filename)


def plot_solution_comparison(
    classical_result,
    vqls_result,
    filename="solution_comparison.png",
):
    """
    Compare final SC-PDIPM primal solution:

        X = [Pg variables, theta variables]

    Classical and VQLS are two complete SC-PDIPM runs.
    """

    names = classical_result["qp"]["variable_names"]

    classical_x = np.asarray(classical_result["X"], dtype=float)
    vqls_x = np.asarray(vqls_result["X"], dtype=float)

    if classical_x.shape != vqls_x.shape:
        raise ValueError("Classical and VQLS solution vectors have different shapes.")

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, classical_x, width, label="Classical")
    plt.bar(x + width / 2, vqls_x, width, label="VQLS")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Value")
    plt.title("Classical vs VQLS SC-PDIPM final solution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_or_show(filename)


def plot_pdipm_convergence(
    result,
    filename="pdipm_convergence.png",
):
    """
    Plot SC-PDIPM convergence:
        - feasibility condition
        - stationarity condition
        - complementarity gap
    """

    history = result.get("history", [])

    if len(history) == 0:
        print("No SC-PDIPM history to plot.")
        return

    iterations = np.array([h["it"] for h in history], dtype=int)
    feas = np.array([h["feascond"] for h in history], dtype=float)
    grad = np.array([h["gradcond"] for h in history], dtype=float)
    comp = np.array([h["compcond"] for h in history], dtype=float)

    eps = 1e-30
    feas = np.maximum(feas, eps)
    grad = np.maximum(grad, eps)
    comp = np.maximum(comp, eps)

    plt.figure(figsize=(9, 5.5))

    plt.plot(iterations, feas, marker="o", linewidth=2, label="Feasibility")
    plt.plot(iterations, grad, marker="s", linewidth=2, label="Stationarity")
    plt.plot(iterations, comp, marker="^", linewidth=2, label="Complementarity")

    # Tolerance lines
    tol_feas = get_cfg("PDIPM_TOL_FEAS", 1e-9)
    tol_grad = get_cfg("PDIPM_TOL_GRAD", 1e-9)
    tol_comp = get_cfg("PDIPM_TOL_COMP", 1e-9)

    plt.axhline(tol_feas, linestyle="--", linewidth=1, alpha=0.7, label="feas tol")
    plt.axhline(tol_grad, linestyle=":", linewidth=1, alpha=0.7, label="grad tol")
    plt.axhline(tol_comp, linestyle="-.", linewidth=1, alpha=0.7, label="comp tol")

    plt.yscale("log")
    plt.xlabel("SC-PDIPM iteration")
    plt.ylabel("Condition value")
    plt.title(f"SC-PDIPM convergence: {result['method']}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_or_show(filename)

def plot_kkt_condition_history(
    result,
    filename="kkt_condition_history.png",
):
    """
    Plot KKT condition number during SC-PDIPM.
    This explains why VQLS/HHL becomes difficult near the solution.
    """

    history = result.get("history", [])

    if len(history) == 0:
        print("No SC-PDIPM history to plot.")
        return

    iterations = np.array([h["it"] for h in history], dtype=int)
    cond = np.array([h["kkt_cond"] for h in history], dtype=float)

    cond = np.maximum(cond, 1e-30)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, cond, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("SC-PDIPM iteration")
    plt.ylabel("cond(KKT)")
    plt.title(f"KKT condition number: {result['method']}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_or_show(filename)
def plot_linear_residual_history(
    result,
    filename="linear_residual_history.png",
):
    """
    Optional plot:
        residual of each KKT linear solve during SC-PDIPM.

    Useful for checking whether VQLS solves KKT systems well enough.
    """

    history = result.get("history", [])

    rows = [
        h for h in history
        if h.get("linear_relative_residual", None) is not None
    ]

    if len(rows) == 0:
        print("No linear residual history to plot.")
        return

    iterations = np.array([h["it"] for h in rows], dtype=int)
    rel = np.array([h["linear_relative_residual"] for h in rows], dtype=float)

    eps = 1e-30
    rel = np.maximum(rel, eps)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, rel, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("SC-PDIPM iteration")
    plt.ylabel("Relative KKT solve residual")
    plt.title(f"KKT linear-solve residual: {result['method']}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_or_show(filename)