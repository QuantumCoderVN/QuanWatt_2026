# ============================================================
# plotting.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def plot_loss(results):
    plt.figure(figsize=(8, 5))

    for r in results:
        plt.plot(
            range(len(r["loss_history"])),
            r["loss_history"],
            marker="o",
            label=r["name"]
        )

    plt.xlabel("FDLS iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("FDLS loss comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fdls_loss_comparison.png")
    plt.savefig(path, dpi=150)
    print("\nSaved:", path)


def plot_voltage_magnitude(results):
    nb = len(results[0]["vm"])
    x = np.arange(nb)
    width = 0.25

    plt.figure(figsize=(9, 5))

    for i, r in enumerate(results):
        plt.bar(
            x + i * width,
            r["vm"],
            width=width,
            label=r["name"]
        )

    plt.xticks(x + width, [f"Bus {i+1}" for i in range(nb)])
    plt.ylabel("Vm [p.u.]")
    plt.title("Voltage magnitude comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "vm_solution_comparison.png")
    plt.savefig(path, dpi=150)
    print("Saved:", path)


def plot_voltage_angle(results):
    nb = len(results[0]["va_deg"])
    x = np.arange(nb)
    width = 0.25

    plt.figure(figsize=(9, 5))

    for i, r in enumerate(results):
        plt.bar(
            x + i * width,
            r["va_deg"],
            width=width,
            label=r["name"]
        )

    plt.xticks(x + width, [f"Bus {i+1}" for i in range(nb)])
    plt.ylabel("Va [degree]")
    plt.title("Voltage angle comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "va_solution_comparison.png")
    plt.savefig(path, dpi=150)
    print("Saved:", path)