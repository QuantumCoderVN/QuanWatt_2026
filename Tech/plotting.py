# ============================================================
# plotting.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def plot_loss(results):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from config import OUTPUT_DIR

    plt.figure(figsize=(9, 5.5))

    styles = {
        "Classical": {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.2,
            "markersize": 7,
            "zorder": 3,
        },
        "HHL": {
            "marker": "s",
            "linestyle": "--",
            "linewidth": 2.0,
            "markersize": 6,
            "zorder": 5,
        },
        "VQLS": {
            "marker": "^",
            "linestyle": ":",
            "linewidth": 2.0,
            "markersize": 6,
            "zorder": 4,
        },
    }

    # Dịch rất nhẹ vị trí marker theo trục x để các điểm không che nhau.
    x_offsets = {
        "Classical": -0.04,
        "HHL": 0.00,
        "VQLS": 0.04,
    }

    print("\n" + "=" * 80)
    print("LOSS HISTORY USED FOR PLOTTING")
    print("=" * 80)

    for r in results:
        name = r["name"]
        y = np.array(r["loss_history"], dtype=float)
        x = np.arange(len(y), dtype=float)

        print(f"\n{name} loss_history:")
        print(y)

        style = styles.get(name, {})
        offset = x_offsets.get(name, 0.0)

        plt.plot(
            x + offset,
            y,
            label=f"{name} final={r['final_loss']:.2e}",
            **style,
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