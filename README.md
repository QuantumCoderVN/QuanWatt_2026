# Quantum Computing for End-to-End Power Systems

## Project vision

This project develops an end-to-end quantum computing framework for power-system analysis and operation. Its long-term objective is to identify, build, and validate the parts of the power-system computational stack where quantum computing can provide a practical advantage, while retaining classical computing, high-performance computing (HPC), and artificial intelligence (AI) wherever they remain more effective.

The program covers four connected application areas:

- **Quantum Power Flow (QPF):** hybrid and quantum linear-system methods for AC and DC power-flow analysis.
- **Quantum State Estimation (QSE):** quantum-assisted reconstruction of the operating state of the grid from noisy and incomplete measurements.
- **Quantum Optimization:** unit commitment, optimal power flow, topology control, contingency response, and other constrained grid decisions.
- **Dynamic and operational analytics:** extending the same hybrid architecture toward contingency screening, uncertainty analysis, and time-dependent grid studies.

The immediate minimum viable product (MVP) is **Power Flow Analysis using QPF**, with the quantum linear solve implemented and benchmarked through the Variational Quantum Linear Solver (VQLS) and the Harrow-Hassidim-Lloyd (HHL) algorithm. The MVP is designed as a research-to-product testbed: it provides a complete workflow for converting a power-network case into a quantum-compatible linear system, selecting an appropriate solver, executing the solve, extracting useful grid quantities, and comparing the result with a trusted classical baseline.

## Why power flow is the starting point

Power flow is a foundational calculation in grid planning and operation. Given network parameters, generation, and demand, it determines bus voltage magnitudes and phase angles, line flows, and power mismatches. It is repeatedly invoked inside state estimation, contingency analysis, probabilistic studies, optimal power flow, and security assessment.

Increasing renewable generation, storage, electric vehicles, converter-based resources, and operating uncertainty greatly increase the number of scenarios that must be evaluated. A useful quantum approach must therefore do more than solve a small linear system: it must fit into the iterative and operational structure of a real power-flow workflow.

Our initial formulation focuses on Newton-Raphson and fast-decoupled load flow. At each iteration, the classical nonlinear problem is reduced to one or more linear systems,

$$
A_k\Delta x_k=b_k,
$$

where $A_k$ is a Jacobian or susceptance-derived matrix, $b_k$ is the active/reactive power mismatch, and $\Delta x_k$ updates the voltage angles or magnitudes. Fast-decoupled power flow is especially attractive because its matrices are structured, sparse, and often reused over multiple iterations or scenarios. These properties can reduce repeated quantum data-loading and matrix-encoding costs.

## MVP: hybrid QPF with VQLS and HHL

The MVP follows a hybrid loop:

1. Parse and validate a power-network case.
2. Run classical preprocessing, scaling, and initialization.
3. Construct the power-flow mismatch vector and linearized system.
4. Analyze matrix size, sparsity, symmetry, conditioning, and expected precision.
5. Route the solve to a classical, HPC, VQLS, HHL, or future SQD backend.
6. Execute the selected solver and estimate relevant observables or solution components.
7. Update the voltage state and test power-flow convergence.
8. Verify the final result against physical constraints and a classical reference.

### VQLS path

VQLS is the near-term path. It replaces the deep phase-estimation structure of HHL with a parameterized quantum circuit and a classical optimizer. The project uses it to investigate shallow-circuit QPF, noise-aware circuit design, problem-informed ansatz construction, measurement reduction, and warm starts across related power-flow iterations or scenarios.

VQLS is not assumed to be automatically advantageous. Its performance depends on ansatz expressibility, optimization landscape, measurement cost, noise, and scaling. The MVP therefore measures total time to an application-relevant answer, not only circuit depth or state fidelity.

### HHL path

HHL is the fault-tolerant and algorithmic reference path. For a sparse, well-conditioned linear system and an output expressible as an efficiently measured property of $|x\rangle$, quantum linear-system algorithms can offer favorable dependence on the system dimension. HHL also exposes the main engineering bottlenecks clearly: state preparation, Hamiltonian simulation or block encoding, phase estimation, reciprocal-eigenvalue rotation, postselection or amplitude amplification, condition-number dependence, and output extraction.

The project includes circuit-level work on quantum phase estimation, approximate controlled rotations for $1/\lambda$, success probability, sign recovery, and comparison with classical solutions. HHL is treated as a roadmap toward fault-tolerant QPF rather than a claim of near-term production speedup.

## Near-to-far-term algorithm roadmap

The research program connects three solver families rather than treating them as isolated alternatives:

| Horizon | Primary method | Role in the project | Main research questions |
| --- | --- | --- | --- |
| Near term | SQD / sample-based quantum diagonalization | Use quantum samples with classical subspace construction and diagonalization | How should power-system structure, symmetry, sparsity, and postselection guide bitstring filtering and subspace selection? |
| Near term | VQLS | Noise-resilient variational QPF and hardware experiments | Which ansatz, cost function, initialization, and measurement strategy remain stable as the grid grows? |
| Fault-tolerant | HHL and improved quantum linear solvers | Long-term complexity and resource benchmark | When do encoding, conditioning, reciprocal rotation, success probability, and readout preserve an end-to-end advantage? |

SQD provides a bridge between near-term sampling and structured classical computation. For linear solving, the project studies reformulations such as positive-semidefinite or ground-state problems whose solution encodes the desired vector. Power-network structure can then inform sample filtering, constraint preservation, and subspace expansion. The aim is to learn which ingredients remain useful when moving from SQD to variational and ultimately fault-tolerant solvers.

## AI + HPC + QC orchestration

The proposed system is heterogeneous by design. It does not send every case to a quantum processor.

### AI problem classifier and router

An AI model uses inexpensive features and historical benchmark data to predict the most suitable computational route. Candidate features include:

- network size and topology;
- matrix sparsity, symmetry, diagonal dominance, and block structure;
- estimated condition number and spectral bounds;
- required accuracy and convergence tolerance;
- number of repeated right-hand sides or related scenarios;
- target output: full state, selected buses, line flows, risk score, or aggregate observable;
- state-preparation, circuit-depth, sampling, and queue-time estimates;
- hardware noise and available quantum/HPC resources.

The router can choose classical sparse solvers for easy or latency-critical cases, HPC for large classical batches and quantum support tasks, VQLS/SQD for suitable near-term experiments, and HHL-class algorithms when fault-tolerant resource estimates indicate a credible benefit. Uncertain cases can be sent to more than one backend to improve the training data and maintain operational reliability.

### HPC as a quantum accelerator

HPC supports rather than merely competes with the quantum component. Its roles include matrix assembly and decomposition, preconditioning, partitioning, circuit compilation, parameter optimization, batched observable evaluation, noise simulation and mitigation, SQD subspace diagonalization, and verification. It can also distribute independent power-flow scenarios and quantum measurement groups.

This creates a co-processing workflow in which each resource handles the task matching its strengths: AI selects and adapts the route, HPC performs large-scale classical support, and QC targets the structured kernel or observable for which quantum representation is valuable.

## Signature research direction: application-aware quantum-state extraction

Quantum linear solvers normally return an amplitude-encoded state,

$$
|x\rangle=\frac{1}{\|x\|}\sum_i x_i|i\rangle,
$$

rather than a classical list of all entries of $x$. Reconstructing every amplitude by full tomography generally requires resources proportional to the output dimension and can remove the speedup sought from the quantum linear solve. This input/output bottleneck is a long-standing obstacle to practical quantum linear-system applications.

A defining contribution of this project is to develop **application-aware state-extraction techniques** for power systems. The goal is not unrestricted full-vector readout at exponentially low cost. Instead, we exploit specific conditions under which the operator needs only a structured subset or function of the solution, for example:

- voltage or angle changes at selected critical buses;
- line-flow, overload, or security indicators;
- inner products, norms, averages, residuals, or threshold tests;
- sparse or compressible solution updates;
- known reference amplitudes and interference-based recovery of relative signs;
- repeated solves where a previous classical or quantum solution supplies prior information;
- grid partitions in which only boundary variables must be exchanged.

Candidate techniques include direct observable estimation, amplitude estimation, reference-state interference, classical shadows or targeted tomography, compressed recovery under sparsity assumptions, and hybrid reconstruction constrained by the power-flow equations. The extraction method will be selected jointly with the solver and target operational output.

Any acceleration claim will include the full cost of state preparation, solver execution, postselection, measurements, sign recovery, normalization, classical reconstruction, and verification. The intended advantage is therefore **end-to-end and conditional**: it applies when the matrix, input access, desired output, accuracy, and hardware regime satisfy explicit criteria.

## Technical work packages

### 1. Power-system formulation

- AC/DC and fast-decoupled power-flow models.
- Sparse Jacobian and susceptance-matrix construction.
- Hermitian embedding, normalization, scaling, and preconditioning.
- Warm starts and matrix reuse across iterations and scenarios.
- Extension of reusable components to QSE and optimization.

### 2. Instance characterization and solver selection

- Exact and inexpensive condition-number estimation.
- Gershgorin spectral bounds for suitable matrix classes.
- Hager-Higham 1-norm estimation and bounds relating $\kappa_1$ and $\kappa_2$.
- Tests on sparse, dense, SPD, diagonally dominant, and representative grid matrices.
- Accuracy metrics that report invalid/infinite bounds separately and avoid biased comparisons.

### 3. Quantum linear solvers

- VQLS ansatz, cost-function, optimizer, and measurement studies.
- HHL phase estimation, reciprocal rotation, postselection, and resource analysis.
- Approximate controlled-rotation designs that avoid exponentially compiled lookup logic.
- SQD sampling, postselection, bitstring filtering, subspace construction, and ground-state reformulations.
- Noise mitigation, error suppression, and hardware-aware compilation.

### 4. Output extraction and validation

- Real-amplitude and relative-sign extraction with a reference amplitude.
- Targeted tomography and observable-based readout.
- Application-aware reconstruction and uncertainty estimates.
- Residual checks $\|A\hat{x}-b\|$, power-balance validation, and convergence tests.
- Comparison against trusted sparse classical solvers.

### 5. Hybrid platform and benchmarking

- A common backend interface for classical, HPC, SQD, VQLS, and HHL solvers.
- AI routing with confidence and fallback policies.
- Simulation, noisy simulation, and quantum-hardware execution.
- Reproducible benchmarks and resource estimation.
- Monitoring of accuracy, latency, energy, cost, and operational usefulness.

## Evaluation framework

The project separates four levels of success:

1. **Numerical correctness:** the recovered result satisfies the linear-system residual and matches a classical reference within tolerance.
2. **Power-system correctness:** the resulting voltages and flows satisfy power balance, operational limits, and iterative convergence requirements.
3. **Quantum-resource viability:** circuit width/depth, shots, state preparation, postselection, mitigation, and classical support remain within the target hardware regime.
4. **End-to-end advantage:** total time, cost, energy, or scenario throughput improves over the best relevant classical/HPC baseline for a clearly defined task.

Core benchmark variables include bus count, sparsity, condition number, solver tolerance, power-flow iterations, circuit resources, optimizer iterations, number of shots, success probability, extraction cost, residual error, and total wall-clock time. Results will be reported by matrix family and grid case so that favorable structured examples are not mistaken for general performance.

## Development milestones

### Phase 1 - Reproducible MVP

- Implement classical AC/DC and fast-decoupled baselines.
- Integrate VQLS and HHL for small power-flow linear systems.
- Add conditioning diagnostics and solver-independent validation.
- Recover selected grid outputs and compare them with classical solutions.

### Phase 2 - Near-term hybrid experiments

- Add noise models, mitigation, and real-hardware tests.
- Develop problem-informed VQLS ansatz and warm-start strategies.
- Implement SQD-based linear-system experiments and grid-aware filtering.
- Benchmark targeted state extraction against full tomography.

### Phase 3 - Intelligent heterogeneous execution

- Train the AI solver router on accumulated benchmark data.
- Distribute preprocessing, optimization, measurements, and verification through HPC.
- Add confidence-aware fallback to classical methods.
- Demonstrate repeated-scenario and contingency-analysis workflows.

### Phase 4 - Scalable advantage assessment

- Perform logical and physical resource estimates for fault-tolerant HHL-class QPF.
- Validate the conditions under which application-aware extraction preserves a speedup.
- Extend the platform to QSE and quantum optimization.
- Demonstrate an end-to-end advantage on a precisely defined power-system workload.

## Scientific position

This project does not assume that quantum computing will replace established power-system solvers. Modern sparse classical and HPC methods are strong baselines, especially when a full classical solution is required. The research question is narrower and more useful:

> For which power-system instances and outputs can a carefully co-designed AI-HPC-QC workflow deliver a verified end-to-end advantage?

The project addresses that question across the full computational chain. Its central hypothesis is that advantage will come from co-design: exploiting grid structure, selecting the right instance, using the right near- or far-term solver, accelerating quantum execution with HPC, and extracting only the classical information required for an operational decision.

## Current source base

The project builds on the original HHL linear-system algorithm, QPF formulations based on fast-decoupled and Newton-Raphson power flow, noise-resilient VQLS-QPF approaches, real-hardware QPF experiments, and recent work connecting QPF theory with resource estimation and output downloading. The accompanying notebooks implement HHL components, controlled inverse-eigenvalue rotations, solution-state inspection, sign-detection experiments, and classical comparisons.

## Collaboration scope

Relevant collaboration areas include power-system modeling, sparse numerical linear algebra, quantum algorithms, circuit compilation, error mitigation, quantum measurement and tomography, HPC orchestration, machine-learning routing, and fault-tolerant resource estimation. Contributions should state the target grid case, solver assumptions, output requirement, classical baseline, and complete resource accounting.

---

**Project focus:** Quantum computing for end-to-end power systems  
**MVP:** Hybrid Quantum Power Flow using VQLS and HHL  
**Algorithm roadmap:** SQD $\rightarrow$ VQLS $\rightarrow$ HHL / fault-tolerant quantum linear solvers  
**System architecture:** AI routing + HPC acceleration + quantum computation  
**Signature research:** application-aware quantum-state extraction for preserving end-to-end advantage
