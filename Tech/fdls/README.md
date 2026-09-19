# Fast Decoupled Load Flow with Classical and Quantum Solvers

This directory is a self-contained experiment for solving a small AC power-flow problem with the Fast Decoupled Load Flow (FDLS) method. The same FDLS iteration is executed with three linear-system backends:

- a direct classical NumPy solver;
- the Harrow-Hassidim-Lloyd (HHL) algorithm simulated with Qiskit; and
- a Variational Quantum Linear Solver (VQLS).

The goal is to compare solver behavior inside an identical power-flow workflow. This is research and demonstration code, not a production power-system solver.

## Files

| File or directory | Purpose |
|---|---|
| `main.py` | Main entry point. Builds the test system, runs all three solver variants, prints summaries, and creates plots. |
| `config.py` | Configuration for FDLS convergence, HHL, VQLS, logging, and output paths. |
| `ieee_case.py` | Defines the reduced three-bus case and builds `Ybus`, `B'`, `B''`, specified powers, and initial voltages. |
| `fdls.py` | Implements the FDLS iteration and calls the selected linear solver for each update. |
| `quantum_solvers.py` | Implements the classical, HHL, and VQLS solver interfaces and shared preprocessing utilities. |
| `plotting.py` | Generates convergence and voltage-comparison figures. |
| `outputs/` | Destination for generated plots. |
| `__pycache__/` | Automatically generated Python bytecode; not source code. |

## Computation flow

`main.py` loads a reduced network derived from the IEEE 14-bus system. The case has one slack bus, one PV bus, and one PQ bus. It constructs the admittance and decoupled matrices, then calls `fdls_power_flow` once for each solver.

Each FDLS iteration evaluates the active- and reactive-power mismatches and solves:

```text
B'  Delta-theta = Delta-P / |V|
B'' Delta-V     = Delta-Q / |V|
```

The resulting angle and voltage-magnitude corrections are applied until the configured tolerance or maximum number of iterations is reached.

## Run

Run the script from this directory because it uses local imports and a relative output path:

```bash
cd src/fdls
python main.py
```

The principal dependencies are:

```text
numpy
scipy
matplotlib
qiskit
```

## Outputs

The run writes comparison plots under `outputs/`, including:

- FDLS loss versus iteration;
- final voltage magnitudes; and
- final voltage angles.

It also prints the first linear-system corrections and final convergence results for all solver backends.

## Notes

- The code pads linear systems to power-of-two dimensions when required by the quantum representation.
- HHL and VQLS return normalized quantum-state information, so the code restores a physical solution scale during postprocessing.
- Quantum simulation and variational optimization may be much slower than a direct classical solve for these small matrices.
- A second copy of these modules exists directly under `src/`. Changes made here do not automatically update the root copy.
