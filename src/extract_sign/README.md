# Quantum Solution Sign-Recovery Experiments

This directory studies how to reconstruct signed real-valued solutions from HHL and VQLS output. Quantum measurements directly expose probabilities, which correspond to squared amplitude magnitudes and do not by themselves retain the sign of each real component. The scripts here compare interference/parity measurements with quantum state tomography as ways to recover that missing information.

## Top-level files

| File | Purpose |
|---|---|
| `HHL.py` | Standalone Qiskit HHL experiment for a small Hermitian positive-definite system, including solution-scale recovery. |
| `Tomography_HHL.py` | HHL implementation with postselected target-register tomography. |
| `VQLS.py` | General VQLS prototype using automatic Pauli decomposition through Qiskit. |
| `Tomography_VQLS.py` | VQLS reconstruction based on quantum state tomography. |
| `hhl_compare.py` | Compares shot-based sign recovery with tomography on a 2-by-2 HHL problem. |
| `vqls_compare.py` | Compares parity-based and tomography-based signed recovery on a 4-by-4 VQLS problem. Supports replotting saved results. |

The PNG files located directly in this directory are generated result figures from earlier experiment runs.

## Subdirectories

### `compare/`

Contains modularized implementations and benchmark drivers for comparing HHL and VQLS recovery methods on small synthetic linear systems.

Important files include:

- `hhl_shot_recovery_modules.py`: helpers for shot-based HHL recovery;
- `hhl_tomography_modules.py`: helpers for tomography-based HHL recovery;
- `vqls_two_modules_refactored.py`: modular VQLS recovery without full tomography;
- `vqls_tomography_two_modules_refactored.py`: modular tomography-based VQLS recovery;
- `benchmark_hhl_recovery_methods.py`: HHL comparison benchmark; and
- `benchmark_vqls_recovery_methods.py`: VQLS comparison benchmark.

#### `compare/benchmark_outputs/`

Generated benchmark results. CSV files store metrics, PNG files visualize error and runtime, and `systems/` stores the synthetic linear systems as NumPy `.npz` files.

### `IEEE/`

Applies HHL recovery and comparison methods to AC and DC power-flow systems built from Pandapower IEEE cases. The versioned benchmark scripts represent different experiment stages, case selections, metrics, and dashboard layouts.

The principal generated-result directories are:

- `benchmark_outputs_ieee_dc/`: baseline DC power-flow benchmark;
- `benchmark_outputs_ieee_dc_js_dashboard/`: Jensen-Shannon similarity dashboard;
- `benchmark_outputs_ieee_dc_js_runtime_5cases/`: five-case runtime and similarity experiment; and
- `benchmark_outputs_ieee_dc_js_runtime_7cases/`: extended seven-case experiment.

Inside those directories:

- `systems/` contains prepared DC power-flow matrices and vectors;
- `solutions/` contains saved parity- or tomography-recovered solutions;
- CSV files contain aggregated metrics; and
- PNG files contain comparison plots and dashboards.

### `output/`

Generated data and figures from `vqls_compare.py`. The directory includes saved NumPy results, a JSON summary, optimization convergence, signed-solution comparisons, and recovery-runtime plots. These saved results allow figures to be redrawn without rerunning the VQLS optimization.

### `outputs_2x2_compare/`

Generated plots from `hhl_compare.py`, including signed solution-value and average-runtime comparisons for the two recovery methods.

### `__pycache__/`

Automatically generated Python bytecode. It is not part of the experiment source.

## Typical usage

Run scripts from this directory so their local paths resolve consistently:

```bash
cd src/extract_sign
python hhl_compare.py
python vqls_compare.py
```

To redraw the saved VQLS figures without rerunning optimization:

```bash
python vqls_compare.py --plot-only
```

The IEEE and benchmark scripts have experiment-specific parameters in their source. Inspect the selected script's argument parser and constants before launching a potentially expensive run.

## Dependencies

The scripts use different subsets of:

```text
numpy
scipy
matplotlib
pandas
qiskit
qiskit-aer
pandapower
```

## Research cautions

- HHL examples assume a compatible Hermitian system and suitable eigenvalue scaling.
- Shot counts can be high, and tomography requires many measurement settings.
- Several benchmark scripts execute expensive work and generate or overwrite result artifacts.
- Versioned IEEE scripts are separate experiment variants rather than a stable public API.
- When comparing methods, distinguish normalized state fidelity from the error in the rescaled physical solution.
