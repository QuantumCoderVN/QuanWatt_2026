# QuanWatt 2026

QuanWatt explores how quantum computing can support power-system analysis. The current focus is **power flow**: calculating bus voltages and voltage angles in an electrical network.

The project compares classical solvers with two quantum algorithms: **HHL** (Harrow–Hassidim–Lloyd) and **VQLS** (Variational Quantum Linear Solver). It includes a small 3-bus demonstration derived from IEEE 14-bus data, a web dashboard, and research experiments for quantum solution recovery and optimal power flow.

The demos run quantum simulations locally; no quantum hardware account is required.

## Quick start: run the HHL demo

The executable entry points are Python scripts. Start by opening a terminal in the repository root (`QuanWatt_2026`).

### 1. Set up Python

Use Python 3.12 with pip. Create and activate a virtual environment:

#### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the demo dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Run the script

```bash
python app/hhl.py
```

The demo runs classical and HHL power flow on the default 3-bus network. Results appear in the terminal, including:

- Convergence status and iteration count.
- Bus voltage magnitudes and angles.
- Power injections and differences between the two solvers.

To change solver parameters, edit [app/config.py](app/config.py). The default network is defined in [app/network.py](app/network.py).

### 3. Run it again

For later runs, activate the existing virtual environment and run `python app/hhl.py` from the repository root. Press `Ctrl+C` to stop a running calculation.

## Run HHL and VQLS with `src/main.py`

After completing the Python setup above, run these commands from the repository root:

```bash
cd src
python main.py
```

[src/main.py](src/main.py) runs classical, HHL, and VQLS solvers in sequence on the same 3-bus power-flow case. One command runs all three; no solver argument is needed. The terminal shows each solver's convergence, voltage magnitudes, and voltage angles. It also saves these plots in `src/outputs/`:

- `fdls_loss_comparison.png` — convergence comparison.
- `vm_solution_comparison.png` — bus voltage magnitudes.
- `va_solution_comparison.png` — bus voltage angles.

Edit [src/config.py](src/config.py) to adjust the iteration limit, tolerance, HHL phase qubits, or VQLS settings. Quantum simulations can take longer than the classical calculation.

## Run the condition-number GNN

The [condition_gnn package](<tech/Conditional Number/gnn_detect_conditional_number/src/condition_gnn/>) uses a graph neural network to estimate matrix condition numbers, which describe how sensitive a linear system is to changes in its inputs.

### 1. Install the package

With your Python virtual environment activated, start from the repository root:

```bash
cd "tech/Conditional Number/gnn_detect_conditional_number"
python -m pip install -e .
```

This installs the package and its dependencies, including PyTorch, and makes the `condition-gnn` command available. Keep this directory as your working directory for the commands below.

### 2. Generate data and train a small model

```bash
condition-gnn all --config configs/smoke.yaml --norm 2 --scheme 2
```

This generates small symmetric positive-definite matrices and trains a model to predict the 2-norm condition number directly. The smoke configuration uses four training epochs and is intended to check the workflow.

Generated datasets are saved in `artifacts/smoke/data/`. The model checkpoint, test metrics, and training history are saved in `artifacts/smoke/results/`.

### 3. Benchmark and plot the results

```bash
condition-gnn benchmark --config configs/smoke.yaml --norm 2 --scheme 2
condition-gnn plot --config configs/smoke.yaml
```

The benchmark compares the trained model with classical estimators. CSV results and PNG plots are saved in `artifacts/smoke/results/`.

### 4. Predict a matrix condition number

After training, replace `path/to/matrix.npy` with your own dense NumPy matrix file:

```bash
condition-gnn predict --matrix "path/to/matrix.npy" --checkpoint artifacts/smoke/results/norm_2_scheme_2.pt
```

Sparse SciPy `.npz` and Matrix Market files are also supported. Use matrices similar to the training data when evaluating this small demonstration model.

Run `condition-gnn --help` to see the available commands. For more configurations and experiments, see the [condition-number guide](<tech/Conditional Number/README.md>).

## Run the web dashboard

The dashboard lets you edit the network, run power-flow analysis, and compare HHL and classical results in your browser. It requires Python 3.12 and Node.js 20 or newer.

Follow the [dashboard setup guide](app/README.md) to install its dependencies and start the backend and frontend.

## Project folders

| Folder | Contents |
| --- | --- |
| [app/](app/) | HHL demo, solver modules, and web dashboard. |
| [src/](src/README.md) | Classical, HHL, and VQLS experiments. |
| [Tech/](Tech/) | Supporting research code and notebooks. |
| [pitch/](pitch/) | Product and presentation materials. |

For more experiments, see the [source guide](src/README.md), [sign-recovery guide](src/extract_sign/README.md), and [QuApp workflow](src/hhl_quapp_workflow/README.md).

## Troubleshooting

- **Missing Python module:** Activate your virtual environment and rerun `python -m pip install -r requirements.txt` from the repository root.
- **Script not found:** Run `python app/hhl.py` from the repository root, or `python main.py` from `src/`.
- **`condition-gnn` command not found:** Activate the environment used to install the package and rerun `python -m pip install -e .` from `tech/Conditional Number/gnn_detect_conditional_number/`.
- **Simulation takes a while:** Start with the default 3-bus case and check the terminal for progress.
