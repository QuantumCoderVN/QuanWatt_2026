# HHL Multi-Job Workflow for QuApp

This subproject splits the original HHL experiment into deployable QuApp circuit functions and a local controller:

1. `hhl-main-circuit`: one job that measures the main HHL circuit to estimate `|x_i|^2`.
2. `hhl-sign-circuit`: independent jobs for selected index pairs to recover relative signs.
3. Local controller: invokes and polls jobs, stores JSON, performs postselection, reconstructs signs and scale, compares against a classical solution, and generates plots.

Each deployable QuApp function directory contains a `handler.py` and its own `requirements.txt`.

## Directory structure

```text
hhl_quapp_workflow/
|-- quapp/
|   |-- hhl-main/
|   |   |-- handler.py
|   |   `-- requirements.txt
|   `-- hhl-sign/
|       |-- handler.py
|       `-- requirements.txt
|-- local/
|   |-- run_workflow.py
|   |-- analyze_saved_results.py
|   |-- analyze.py
|   |-- offline_smoke_test.py
|   |-- requirements.txt
|   |-- requirements-dev.txt
|   |-- input/
|   |-- input_16x16/
|   |-- output/
|   `-- output_16x16/
`-- samples/
```

## `quapp/`

Contains the code deployed to QuApp.

### `quapp/hhl-main/`

`handler.py` builds the main HHL circuit. Its measurements are used to estimate the probability of each target-register basis state after the required postselection. The function-specific dependencies are listed in `requirements.txt`.

### `quapp/hhl-sign/`

`handler.py` builds an interference circuit for a requested pair of solution indices. The additional sign qubit allows the local analysis to infer whether two real solution components have equal or opposite signs. Its dependencies are listed in the adjacent `requirements.txt`.

## `local/`

Contains the workstation-side controller, offline validation, and analysis tools.

- `run_workflow.py` invokes the QuApp functions, polls jobs, saves raw and extracted responses, reconstructs the signed solution, and creates plots.
- `analyze_saved_results.py` repeats reconstruction and plotting from saved result JSON without consuming additional quantum jobs.
- `analyze.py` analyzes the checked-in input/output layout.
- `offline_smoke_test.py` executes the handler logic with Qiskit Aer before deployment.
- `requirements.txt` contains controller dependencies.
- `requirements-dev.txt` adds packages required for local simulation and development.

### `local/input/` and `local/input_16x16/`

Saved QuApp result JSON used as analysis inputs for the small and 16-by-16 workflows. Files named `hhl_result.json` contain the main job result; files named `sign_I_J_result.json` contain sign-interference results for a pair of indices.

### `local/output/` and `local/output_16x16/`

Generated analysis artifacts. Each output directory contains:

- `analysis/summary.json`: reconstructed solution and quality metrics; and
- `figures/`: solution, probability, and sign-diagnostic plots.

### `local/__pycache__/`

Automatically generated Python bytecode, not maintained source code.

## `samples/`

Contains example request payloads for the main HHL circuit and sign circuits, plus expected JSON result schemas. Use these samples to understand and validate the QuApp function contract.

## Fixed demonstration problem

The checked-in small workflow uses:

```text
A = [[4.0,  0.4,  0.2,  0.0],
     [0.4,  5.0, -0.3,  0.1],
     [0.2, -0.3,  3.5,  0.5],
     [0.0,  0.1,  0.5,  4.5]]

b = [0.03, -0.02, 0.04, -0.01]
```

The authoritative matrix and vector are defined in the handlers and controller. Keep all deployed and local copies synchronized when changing the demonstration problem.

## Local setup

```bash
cd src/hhl_quapp_workflow/local
python -m venv .venv
```

Activate the environment, then install the controller dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from the provided example if present, then configure the QuApp API URL, token, function identifiers, device, shot count, and polling parameters. Do not commit API tokens.

Run the remote workflow:

```bash
python run_workflow.py
```

Analyze previously saved results without invoking QuApp:

```bash
python analyze_saved_results.py PATH_TO_SAVED_RUN
```

## Offline smoke test

Install development dependencies and run the handlers with Aer before deployment:

```bash
pip install -r requirements-dev.txt
python offline_smoke_test.py
```

The deployable handlers do not depend on Aer and do not execute a backend themselves; they return circuits for the hosted runtime.

## Postprocessing summary

For the main HHL result, the controller keeps shots satisfying the configured phase-register and ancilla postselection conditions. It groups the remaining counts by target state, normalizes the conditional probabilities, and estimates amplitude magnitudes with:

```text
|x_i| = sqrt(p_i)
```

For a sign job involving indices `i` and `j`, interference provides a cross term. Its sign indicates whether the two components have equal or opposite relative signs. A tolerance region is used when shot noise makes the decision uncertain.

After recovering a normalized signed direction, the controller restores the physical scale with a least-squares scalar and compares the result with `numpy.linalg.solve(A, b)` using absolute error, relative error, residual, and direction fidelity.

## Shot-count considerations

The workflow uses one main job plus multiple sign jobs. The total number of shots is therefore the number of jobs multiplied by the shots per job. Postselection can retain only a small fraction of the shots, so evaluate the successful-shot count in addition to the requested total.
