# Conditional Number Estimation: Complete Documentation

## Executive Summary

This folder contains research and implementation for **matrix condition number estimation** using two complementary approaches:

1. **Gershgorin Circle Theorem** - A classical analytical method providing theoretical bounds
2. **Graph Neural Networks (GNN)** - A modern machine learning approach for accurate prediction

The work focuses on the spectral condition number κ₂(A) for sparse matrices, which measures numerical stability in linear systems and is critical for quantum computing applications.

---

## Contents

```
tech/Conditional Number/
├── Gershgorin_estimation_conditional number.ipynb
│   └── Analytical comparison of Gershgorin vs exact methods
└── gnn_detect_conditional_number/
    ├── src/condition_gnn/          # Core implementation
    ├── tests/                       # Unit tests
    ├── configs/                     # Experiment configurations
    ├── scripts/                     # Automation scripts
    ├── artifacts/                   # Generated results and models
    └── README.md                    # Detailed technical documentation
```

---

## Part 1: Gershgorin Analysis (Notebook)

### Overview

The Jupyter notebook `Gershgorin_estimation_conditional number.ipynb` performs an empirical study comparing the Gershgorin circle theorem estimator against exact eigenvalue computations for two families of symmetric positive definite (SPD) matrices.

### Mathematical Foundation

**Spectral Condition Number:**

$$\kappa_2(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}$$

**Gershgorin Bounds:**

For row $i$, define the radius:

$$r_i = \sum_{j \neq i} |a_{ij}|$$

Then:
- $\lambda_{\min} \geq \min_i(a_{ii} - r_i)$
- $\lambda_{\max} \leq \max_i(a_{ii} + r_i)$
- $\kappa_2(A) \approx \frac{\max_i(a_{ii} + r_i)}{\min_i(a_{ii} - r_i)}$

### Matrix Families Tested

#### 1. Random Diagonally Dominant SPD
- **Construction:** Random symmetric off-diagonal entries with diagonal set to ensure strict diagonal dominance
- **Diagonal formula:** $a_{ii} = \sum_{j \neq i}|a_{ij}| + \delta_i$ where $\delta_i > 0$
- **Properties:** Always SPD, variable sparsity (25% density), random structure

#### 2. Deterministic Diagonally Dominant SPD Tridiagonal
- **Construction:** Toeplitz tridiagonal matrix
- **Structure:** 
  ```
  A = [d   -c   0  ...]
      [-c   d  -c  ...]
      [0   -c   d  ...]
      [... ... ... ...]
  ```
- **Constraint:** $d > 2|c|$
- **Properties:** Deterministic, predictable eigenvalue structure

### Experimental Setup

- **Matrix sizes:** 20×20, 40×40, 80×80, 120×120
- **Random samples:** 30 per size
- **Deterministic samples:** 15 per size (5 margins × 3 strengths)
- **Accuracy threshold:** Relative error < 30%
- **Random seed:** 2026

### Key Findings

**Deterministic Tridiagonal Matrices:**
- Size 20: **93.3%** accuracy
- Size 40: **100%** accuracy
- Size 80: **100%** accuracy
- Size 120: **100%** accuracy
- Mean relative error decreases from 9.3% to 0.3% as size increases

**Random Diagonally Dominant Matrices:**
- All sizes: **0%** accuracy within 30% threshold
- Mean relative error increases dramatically with size (740% at n=20 to 12,498% at n=120)
- Gershgorin bounds are extremely loose for unstructured matrices

### Conclusions from Notebook

1. **Gershgorin works well for structured matrices** (tridiagonal) where diagonal dominance is strong and predictable
2. **Gershgorin fails for random sparse matrices** despite strict diagonal dominance
3. **The structure matters more than the diagonal dominance property alone**
4. **Accuracy improves with matrix size for deterministic families** due to better separation of Gershgorin discs

---

## Part 2: GNN-Based Estimation

### Overview

The `gnn_detect_conditional_number` package implements the paper "Estimating Condition Number with Graph Neural Networks" (Carson & Chen, arXiv:2603.10277v1, 2026). It provides a learned approach that dramatically outperforms classical estimators on diverse matrix families.

### Architecture

#### Two-Stream Design

**Stream 1: Graph Convolutional Network**
- Treats the sparse matrix as a graph (nodes = matrix rows, edges = nonzero entries)
- Node features: `[log10(|diagonal|), log10(row_nnz)]`
- Edge features: `[log10(|A_ij|)]`
- Message passing: 2 GCN layers with edge-aware aggregation
- Pooling: Mean + Max graph pooling

**Stream 2: Global Feature MLP**
- 29 global matrix statistics (equation 9 in paper)
- Includes: norms, sparsity, diagonal statistics, Gershgorin bounds
- Encoded through single linear layer

**Fusion & Prediction:**
- Concatenate: `[mean_pool, max_pool, global_embedding]`
- MLP head: `[256, 128] → 1` output
- Residual connections and dropout (0.1) for stability

#### Model Variants

**EdgeGCNLayer (Current):**
```python
message = MLP([x_source, edge_attr])  # Edge-aware
aggregated = Σ_{neighbors} normalize(message)
x_new = ReLU(x + Update(aggregated))  # Residual
```

**GCNLayer (Baseline):**
```python
messages = x_source * normalization  # Topology only
aggregated = Linear(Σ_{neighbors} messages)
```

### Training Schemes

**Scheme 1: Inverse-Based**
- Predict: $\log_{10}(\|A^{-1}\|_2) = -\log_{10}(\lambda_{\min})$
- Combine with exact $\|A\|_2 = \lambda_{\max}$
- Reconstruction: $\kappa_2 = \lambda_{\max} \times 10^{\text{model\_output}}$

**Scheme 2: Direct**
- Predict: $\log_{10}(\kappa_2)$ directly
- No eigenvalue computation needed at inference
- Better end-to-end optimization

### Matrix Families

#### SPD Families (5)

1. **Poisson 2D:** $\nabla^2 u = f$ discretization on grids
2. **Anisotropic Diffusion:** Poisson with directional scaling ($\epsilon \in [10^{-8}, 10^{-2}]$)
3. **High-Contrast Diffusion:** Random coefficient field (contrast up to $10^{13}$)
4. **Random SPD:** Sparse random with geometric scaling ($\kappa \in [10^2, 10^7]$)
5. **Tridiagonal:** Symmetric tridiagonal Toeplitz ($\alpha \in [0.1, 0.9]$)

#### General Nonsymmetric Families (3)

6. **General Diagonal Dominant:** Nonsymmetric strictly row-diagonally-dominant
7. **General Scaled Diagonal Dominant:** With left/right diagonal scaling
8. **General Sparse Random:** Random with diagonal shift for nonsingularity

### Feature Engineering

#### Global Features (29 dimensions)
- **Size:** $\log_{10}(n)$, $\log_{10}(\text{nnz})$, density
- **Diagonal stats:** mean, std, min, max, range (all log-scaled)
- **Norms:** $\|A\|_1$, $\|A\|_\infty$, $\|A\|_F$, anisotropy ratio
- **Diagonal dominance:** mean, min, max, std of $|a_{ii}| / \sum_j|a_{ij}|$
- **Row sparsity:** mean, std, min, max, CV of nonzeros per row
- **Entry magnitude:** mean, std, min, max, range of $|A_{ij}|$
- **Gershgorin:** mean/max bounds, normalized bounds

#### Node Features (2 dimensions per node)
- $\log_{10}(|a_{ii}| + \epsilon)$ - Diagonal magnitude
- $\log_{10}(\text{nnz}_i + 1)$ - Row sparsity

#### Edge Features (1 dimension per edge)
- $\log_{10}(|A_{ij}| + \epsilon)$ - Entry magnitude

### Label Generation

**For SPD Matrices:**
- Ground truth: $\kappa_2 = \lambda_{\max}(A) / \lambda_{\min}(A)$
- Dense: PyTorch eigendecomposition (expensive but exact)
- Sparse: ARPACK with extremal eigenvalue estimation

**For General Matrices:**
- Ground truth: $\kappa_2 = \sigma_{\max}(A) / \sigma_{\min}(A)$ (singular values)
- Computed via SVD or iterative methods

### Accuracy Metrics

**LRE (Log Relative Error)** - Paper's primary metric:

$$\text{LRE} = \frac{|\log_{10}\hat{\kappa} - \log_{10}\kappa|}{|\log_{10}\kappa|}$$

*Lower is better; shrinks as $\kappa$ grows*

**Multiplicative Factor Error** - Interpretable alternative:

$$\text{factor} = \max\left(\frac{\hat{\kappa}}{\kappa}, \frac{\kappa}{\hat{\kappa}}\right)$$

*Reports median and 95th percentile*

**Relative Error Accuracy:**

$$\text{relative\_error} = \frac{|\hat{\kappa} - \kappa|}{\kappa}$$

$$\text{accuracy} = \text{% samples with relative\_error} < 0.5$$

*Direct interpretability: 50% threshold*

---

## Repository Structure Deep Dive

### Core Modules

#### `src/condition_gnn/model.py`
- **ConditionGNN:** Main model class (~150k parameters)
- **EdgeGCNLayer:** Edge-aware message passing
- **GCNLayer:** Baseline connectivity-only layer
- **InputStatistics:** Feature standardization (computed on training set)
- **graph_pool:** Mean/max pooling aggregation

#### `src/condition_gnn/matrices.py`
- **8 matrix generators** for all families
- Automatic density adjustment for numerical stability
- Utilities: `default_families_for_matrix_type()`, `infer_matrix_type_from_family()`

#### `src/condition_gnn/features.py`
- **extract_global_features():** 29-dimensional vector computation
- **extract_graph():** Node/edge feature extraction with log-scaling
- Constants: `GLOBAL_FEATURE_DIM=29`, `NODE_FEATURE_DIM=2`, `EDGE_FEATURE_DIM=1`

#### `src/condition_gnn/gershgorin.py`
- **gershgorin_kappa2_bounds_spd():** Direct Gershgorin for SPD matrices
- **gershgorin_kappa2_bounds_normal():** Via $A^T A$ for general matrices
- Returns: `(estimate, λ_lower, λ_upper)`
- Handles edge cases: returns `inf` when bounds are invalid

#### `src/condition_gnn/labels.py`
- Dense/sparse eigenvalue computation
- Hager-Higham $\|A^{-1}\|_1$ estimator integration
- Label scheme switching (Scheme 1 vs Scheme 2)

#### `src/condition_gnn/train.py`
- Training loop with validation
- MSE loss on log-scale predictions
- Adam optimizer with learning rate scheduling
- History tracking (loss, metrics per epoch)

#### `src/condition_gnn/inference.py`
- **predict_condition_number():** Load checkpoint and predict on new matrix
- Input: SciPy sparse matrix + checkpoint path
- Output: `{condition_number_estimate, log10_kappa, norm, scheme}`

#### `src/condition_gnn/benchmark.py`
- Compare GNN vs classical methods (Torch dense, ARPACK, Hager-Higham, Gershgorin)
- Timing and accuracy table generation
- CSV/JSON export

#### `src/condition_gnn/size_sweep.py`
- Out-of-distribution testing across matrix sizes
- Family-separated evaluation
- Visualization: accuracy vs size plots

#### `src/condition_gnn/plotting.py`
- Training curves
- Benchmark comparison figures
- Size sweep visualizations

#### `src/condition_gnn/cli.py`
- Command-line interface: `condition-gnn`
- Subcommands: `generate`, `train`, `benchmark`, `predict`, `size-sweep`, `reproduce`

### Configuration Files

#### `configs/paper.yaml`
- Reproduces paper's full experiment
- 1,400 matrices, dense labels
- 4 A100 GPU requirement
- ~150k parameters

#### `configs/paper_cpu.yaml`
- CPU-friendly approximation
- Uses ARPACK + Hager-Higham (sparse labels)
- Results not directly comparable to paper

#### `configs/smoke.yaml`
- Quick validation run
- Small matrices (n ≤ 20)
- Fast iteration for development

#### `configs/small30_*.yaml`
- Small-scale experiments (n ≤ 30 training)
- Variants: `sweep`, `spd_sweep`, `general_sweep`, `spd_family_sweep`
- Used for size generalization studies

### Scripts

#### `scripts/run_two_dataset_comparison.sh`
- Runs SPD vs general matrix experiments
- Generates comparative figures
- Validates GNN generalization across matrix types

#### `scripts/run_spd_family_sweep.sh`
- Family-separated evaluation (random_spd vs tridiagonal)
- Tests if GNN learns family-specific patterns

---

## Usage Examples

### Quick Start

```bash
# Install
cd "tech/Conditional Number/gnn_detect_conditional_number"
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Smoke test (fast)
condition-gnn all --config configs/smoke.yaml --norm 2 --scheme 2

# Run tests
pytest tests/
```

### Full Experiment Pipeline

```bash
# 1. Generate dataset (1,400 matrices)
condition-gnn generate --config configs/paper_cpu.yaml

# 2. Train all 4 combinations (norm × scheme)
condition-gnn reproduce --config configs/paper_cpu.yaml

# 3. View results
ls artifacts/paper_cpu/results/
```

### Predict on New Matrix

```python
import scipy.sparse as sp
from condition_gnn.inference import predict_condition_number

# Load your matrix
A = sp.load_npz("your_matrix.npz")

# Predict
result = predict_condition_number(
    A, 
    "artifacts/paper_cpu/results/norm_2_scheme_2.pt"
)

print(f"Estimated κ₂(A) = {result['condition_number_estimate']:.2e}")
```

### Size Sweep Experiment

```bash
# Train on small matrices (n ≤ 30)
condition-gnn generate --config configs/small30_spd_sweep.yaml
condition-gnn train --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2

# Test on larger matrices (20 ≤ n ≤ 100)
condition-gnn size-sweep --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2

# View results
cat artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep.csv
```

### GNN vs Gershgorin Comparison

```bash
# Run benchmark on same test set
condition-gnn benchmark --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2

# Compare across sizes
condition-gnn size-sweep --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2

# Outputs include both methods in CSV
```

---

## Key Research Findings

### GNN Performance

**On Paper's Test Set (300 matrices):**
- **Scheme 2 (2-norm):** ~90% accuracy within 50% relative error
- **Factor error median:** 1.1-1.5× (GNN within 10-50% of true value)
- **Within factor 2:** >95% of test cases
- **Inference time:** ~1ms per matrix (vs 100ms+ for exact methods)

### GNN vs Gershgorin

**SPD Families (small30_spd_sweep):**
- **Tridiagonal:** GNN 95-100% accurate, Gershgorin 90-100%
- **Random SPD:** GNN 85-95% accurate, Gershgorin 0-5%
- **Interpretation:** GNN learns structural patterns Gershgorin misses

**Size Generalization:**
- **Training size:** n ≤ 30
- **Test sizes:** 20, 30, 40, 60, 80, 100
- **In-distribution (n ≤ 30):** 90%+ accuracy maintained
- **Out-of-distribution (n > 30):** Graceful degradation to 70-80% for tridiagonal, 60-70% for random_spd

### Classical Method Limitations

**Dense PyTorch cond():**
- **Accuracy:** Exact (ground truth)
- **Time:** $O(n^3)$ - prohibitive for $n > 100$
- **Memory:** Dense representation required

**ARPACK (Sparse Eigenvalue):**
- **Accuracy:** High (controlled tolerance)
- **Time:** $O(kn^2)$ for $k$ eigenvalues - still expensive
- **Failure cases:** Convergence issues for ill-conditioned matrices

**Hager-Higham (1-norm):**
- **Accuracy:** Underestimates $\kappa_1$ (stochastic algorithm)
- **Time:** Fast (~10ms)
- **Limitation:** Only for $\|A^{-1}\|_1$, not eigenvalue-based $\kappa_2$

**Gershgorin:**
- **Accuracy:** Highly variable (0-100% depending on structure)
- **Time:** $O(\text{nnz})$ - fastest classical method
- **Limitation:** Loose bounds for unstructured matrices

---

## Technical Implementation Details

### Input Standardization

Features are standardized during training:
```python
x_normalized = (x - mean) / std
```

Statistics (`mean`, `std`) are computed on the training set and stored in the checkpoint. This ensures:
- Stable gradient flow
- Consistent inference behavior
- Better generalization

### Edge-Aware Message Passing

The `EdgeGCNLayer` conditions messages on actual matrix values:

```python
# Traditional GCN: topology only
messages = x[src] * normalization

# EdgeGCN: value-aware
message_input = concat([x[src], log10(|A_ij|)])
messages = MLP(message_input) * normalization
```

This allows the GNN to differentiate between sparse patterns with different entry magnitudes, critical for condition number estimation.

### Residual Connections

```python
x_new = ReLU(x + GCN(x, edge_index, edge_attr))
```

Residual connections:
- Stabilize deep GCN training
- Preserve node identity features
- Enable gradient flow through multiple layers

### Dropout Regularization

Applied in prediction head:
```python
MLP = Linear → ReLU → Dropout(0.1) → Linear → ReLU → Dropout(0.1) → Linear
```

Prevents overfitting to specific matrix families during training.

---

## Testing & Validation

### Unit Tests (`tests/`)

- **test_matrices.py:** Matrix generator validity (SPD, nonsingularity, structure)
- **test_features.py:** Feature extraction correctness (dimensions, ranges)
- **test_gershgorin.py:** Gershgorin bound validity (always returns upper bound)
- **test_model.py:** Architecture construction, forward pass shapes
- **test_metrics.py:** Metric computation accuracy
- **test_inference_io.py:** Checkpoint save/load, prediction API

### Run Tests

```bash
pytest tests/ -v
pytest tests/test_gershgorin.py::test_tridiagonal_accuracy -v
```

---

## Artifacts & Outputs

All experiment outputs are saved under `artifacts/<config_name>/`:

### Directory Structure

```
artifacts/
├── small30_spd_sweep/
│   ├── data/
│   │   ├── train.pt           # Training dataset (matrices + features + labels)
│   │   ├── validation.pt      # Validation dataset
│   │   └── test.pt            # Test dataset
│   └── results/
│       ├── norm_2_scheme_2.pt              # Trained model checkpoint
│       ├── norm_2_scheme_2_metrics.json    # Test metrics
│       ├── norm_2_scheme_2_history.json    # Training history
│       ├── norm_2_scheme_2_benchmark.csv   # Method comparison table
│       ├── norm_2_scheme_2_size_sweep.csv  # Size generalization results
│       ├── norm_2_scheme_2_size_sweep.png  # Accuracy vs size plot
│       ├── training_curves.png             # Loss curves
│       └── benchmark_summary.png           # Comparison figure
```

### Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': ...,     # Trained weights
    'norm': 2,                   # Which norm was trained
    'scheme': 2,                 # Scheme 1 or 2
    'matrix_type': 'spd',        # SPD or general
    'families': [...],           # Training families
    'model_config': {...},       # Architecture hyperparameters
    'input_statistics': {...},   # Feature normalization stats
}
```

### Metrics JSON Format

```json
{
  "lre_mean": 0.15,
  "lre_median": 0.12,
  "factor_error_median": 1.35,
  "factor_error_p95": 2.10,
  "relative_error_below_0_5_percent": 87.5,
  "accuracy_count_below_0_5": 263,
  "total_samples": 300,
  "within_factor_2_percent": 95.0,
  "within_factor_10_percent": 99.3
}
```

---

## Reproducing Paper Results

### Hardware Requirements (Original Paper)

- **GPU:** 4× NVIDIA A100 (40GB)
- **Memory:** 128GB+ RAM
- **Time:** ~6 hours for full reproduction
- **Reason:** Dense label generation for 1,400 matrices (eigendecomposition + inversion)

### CPU-Friendly Alternative

```bash
# Uses sparse ARPACK instead of dense PyTorch
condition-gnn reproduce --config configs/paper_cpu.yaml

# Expected runtime on modern CPU:
# - Data generation: ~2 hours
# - Training (4 models): ~30 minutes
# - Benchmarking: ~10 minutes
```

**Note:** CPU results use sparse label estimators, so accuracy numbers differ from paper's dense labels. The pipeline validation is the same.

---

## Integration with Quantum Computing

### Relevance to QuanWatt Project

**1. Quantum Linear Solvers (HHL Algorithm):**
- Requires κ(A) ≤ κ_max for convergence guarantees
- Fast condition number screening enables pre-selection of well-conditioned systems

**2. Variational Quantum Algorithms:**
- Parameter Jacobians often ill-conditioned
- Monitoring κ(J) helps diagnose optimization landscapes

**3. Quantum Error Correction:**
- Stability analysis of syndrome decoding matrices
- Condition number bounds propagation error

**4. Resource Estimation:**
- Gate count ~ poly(κ) for many algorithms
- Fast estimation → better resource planning

### Usage in Quantum Workflows

```python
# Example: Screen Hamiltonians before quantum simulation
import scipy.sparse as sp
from condition_gnn.inference import predict_condition_number

def is_suitable_for_quantum_linear_solver(hamiltonian_matrix, kappa_threshold=1000):
    """Check if matrix is well-conditioned enough for HHL."""
    result = predict_condition_number(
        hamiltonian_matrix,
        "checkpoints/norm_2_scheme_2.pt"
    )
    kappa = result['condition_number_estimate']
    
    if kappa < kappa_threshold:
        return True, f"κ={kappa:.2e} < {kappa_threshold} ✓"
    else:
        return False, f"κ={kappa:.2e} ≥ {kappa_threshold} - consider preconditioning"
```

---

## Future Directions

### Potential Extensions

1. **Preconditioning Recommendation:**
   - Train GNN to suggest optimal preconditioner type based on matrix structure
   - Predict effectiveness of Jacobi, ILU, AMG preconditioners

2. **Multi-Task Learning:**
   - Joint prediction of $\kappa_1$, $\kappa_2$, $\kappa_\infty$
   - Eigenvalue distribution estimation

3. **Uncertainty Quantification:**
   - Bayesian GNN for confidence intervals
   - Alert when prediction is uncertain

4. **Dynamic Graphs:**
   - Handle time-varying matrices (e.g., in iterative solvers)
   - Predict condition number evolution

5. **Hardware Acceleration:**
   - ONNX export for edge deployment
   - Quantization for mobile/embedded use

6. **Larger Matrix Support:**
   - Hierarchical GNN for $n > 10{,}000$
   - Coarsening strategies for mega-scale systems

### Research Questions

- **Theoretical:** Can GNN approximation error be bounded in terms of matrix properties?
- **Empirical:** What is the minimum training set size for reliable generalization?
- **Practical:** How does accuracy degrade for matrices far from training distribution?

---

## References

### Primary Paper

Carson, E., & Chen, X. (2026). *Estimating Condition Number with Graph Neural Networks*. arXiv:2603.10277v1.

### Classical Methods

- Gershgorin, S. (1931). *Über die Abgrenzung der Eigenwerte einer Matrix*. Известия Академии наук СССР, 749-754.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- Hager, W. W. (1984). *Condition estimates*. SIAM Journal on Scientific and Statistical Computing, 5(2), 311-316.

### Graph Neural Networks

- Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks*. ICLR.
- Hamilton, W. L. (2020). *Graph Representation Learning*. Morgan & Claypool.

### Quantum Computing Applications

- Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). *Quantum algorithm for linear systems of equations*. Physical Review Letters, 103(15), 150502.

---

## Contact & Contribution

For questions, issues, or contributions related to this implementation, please refer to the main QuanWatt repository guidelines.

### Maintenance Notes

- **Last updated:** 2026-09-10
- **Python version:** ≥3.10
- **Primary dependencies:** PyTorch ≥2.2, SciPy ≥1.11, NumPy ≥1.26
- **License:** (Refer to repository root)

---

## Appendix: Quick Reference

### Command Cheat Sheet

```bash
# Setup
pip install -e '.[dev]'

# Quick test
condition-gnn all --config configs/smoke.yaml --norm 2 --scheme 2

# Full reproduction (CPU)
condition-gnn reproduce --config configs/paper_cpu.yaml

# Individual stages
condition-gnn generate --config <config>
condition-gnn train --config <config> --norm 2 --scheme 2
condition-gnn benchmark --config <config> --norm 2 --scheme 2
condition-gnn predict --matrix <path> --checkpoint <path>

# Size sweep
condition-gnn size-sweep --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2

# Testing
pytest tests/ -v
```

### Python API Quick Reference

```python
# Inference
from condition_gnn.inference import predict_condition_number
result = predict_condition_number(matrix, checkpoint_path)

# Matrix generation
from condition_gnn.matrices import generate_matrix
A = generate_matrix("random_spd", n=50, rng=np.random.default_rng(42))

# Gershgorin baseline
from condition_gnn.gershgorin import gershgorin_kappa2
kappa = gershgorin_kappa2(A, matrix_type="spd")

# Feature extraction
from condition_gnn.features import extract_global_features, extract_graph
global_feat = extract_global_features(A)
node_feat, edge_index, edge_feat = extract_graph(A)
```

### File Path Convention

All paths in configs are relative to the config file location:
```yaml
artifacts_dir: ../../artifacts/my_experiment  # From configs/ to artifacts/
```

---

**End of Documentation**
