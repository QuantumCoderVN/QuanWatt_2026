# Condition GNN 2026 Reimplementation

Python reimplementation of Carson and Chen, *Estimating Condition Number with
Graph Neural Networks* (arXiv:2603.10277v1, 2026), covering both the 1-norm and
2-norm condition numbers of sparse SPD matrices.

\[
\kappa_2(A)=\lambda_{\max}(A)/\lambda_{\min}(A).
\]

## What is implemented

- Five training matrix families from the paper: Poisson, anisotropic diffusion,
  high-contrast diffusion, random ill-conditioned SPD, and symmetric
  tridiagonal matrices.
- The 29 global features described in equation (9).
- Node features and graph construction from equations (10)-(11).
- The two-stream architecture: GCN node encoder plus global-feature MLP,
  mean/max graph pooling, and an MLP prediction head.
- Scheme 1: predict `log10(||A^-1||_2) = -log10(lambda_min)` and combine with
  the exact `||A||_2 = lambda_max`.
- Scheme 2: predict `log10(kappa_2)` directly.
- The same two schemes for `kappa_1`, using either an exact dense inverse or
  SciPy's Hager--Higham estimator to create labels.
- Benchmarks against dense `torch.linalg.cond`, SciPy/Torch Hager--Higham, and
  short Torch Lanczos/LOBPCG runs, exported as CSV and JSON.
- The paper's LRE metric plus interpretable multiplicative-error metrics.

## Reproduction boundaries

The preprint does not report the GCN layer count, hidden width, prediction-head
widths, random seeds, exact train/test random draws, or full implementation. The
`configs/paper.yaml` choices produce approximately 150,000 parameters, matching
the stated model size. Input standardization is included for stable training but
is not specified in the paper.

Consequently, this is a faithful reimplementation of the published equations
and protocol, not a bit-for-bit reproduction of the authors' unpublished code.
The paper used four A100 GPUs and dense PyTorch condition-number labels.
`configs/paper.yaml` therefore uses dense Torch labels and is the strict,
expensive path. `configs/paper_cpu.yaml` replaces those labels with sparse
ARPACK and Hager--Higham estimates; it validates the pipeline but its 1-norm
accuracy numbers are not directly comparable with the paper.

The paper defines edge magnitude features but equation (13) does not use them.
This implementation follows equation (13): graph connectivity affects message
passing, while edge magnitudes are stored but not consumed by the baseline GCN.

## Install

```bash
cd /workspace/condition_gnn_2026
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
source .venv/bin/activate
```

## Quick smoke run

```bash
condition-gnn all --config configs/smoke.yaml --norm 1 --scheme 1
condition-gnn train --config configs/smoke.yaml --norm 2 --scheme 2
condition-gnn benchmark --config configs/smoke.yaml --norm 2 --scheme 2
```

## Paper-scale run

Generating exact labels is the expensive offline stage. It requires dense
eigendecompositions and inversions for 1,200 matrices.

```bash
condition-gnn reproduce --config configs/paper.yaml
```

For a CPU-only approximation of the full protocol:

```bash
condition-gnn reproduce --config configs/paper_cpu.yaml
```

`reproduce` generates the 1,400 matrices once, trains all four combinations
(`norm in {1,2}` x `scheme in {1,2}`), and writes one benchmark table per
combination. To run a single experiment:

```bash
condition-gnn generate --config configs/paper.yaml
condition-gnn train --config configs/paper.yaml --norm 2 --scheme 1
condition-gnn benchmark --config configs/paper.yaml --norm 2 --scheme 1
condition-gnn plot --config configs/paper.yaml
```

Outputs are written under `artifacts/`:

- `norm_N_scheme_S.pt`: trained checkpoint.
- `norm_N_scheme_S_metrics.json`: GNN test metrics.
- `norm_N_scheme_S_history.json`: learning curves.
- `norm_N_scheme_S_benchmark.csv`: paper-style timing/error table.
- `training_curves.png`: train/validation loss curves.
- `training_metrics_summary.png`: GNN accuracy/error summary after training.
- `benchmark_summary.png`: timing, LRE, and accuracy comparison figures.

## Predict a condition number for a new SPD matrix

Save the matrix as SciPy sparse NPZ, NumPy NPY, or Matrix Market format, then
run:

```bash
condition-gnn predict \
  --matrix path/to/A.npz \
  --checkpoint artifacts/paper/results/norm_2_scheme_1.pt
```

The command returns `condition_number_estimate`, its base-10 logarithm, and the
norm stored in the checkpoint as JSON.
Scheme 1 additionally computes `lambda_max(A)` because its reconstruction is
`kappa2_hat = lambda_max * 10**model_output`. Scheme 2 needs no eigenvalue
calculation at inference.

The Python API is:

```python
import scipy.sparse as sp
from condition_gnn.inference import predict_condition_number

A = sp.load_npz("A.npz")
result = predict_condition_number(A, "artifacts/paper/results/norm_2_scheme_1.pt")
print(result["condition_number_estimate"])
```

## Accuracy metrics

The paper's metric is:

\[
\mathrm{LRE}=\frac{|\log_{10}\hat\kappa-\log_{10}\kappa|}
{|\log_{10}\kappa|}.
\]

Because it shrinks as `kappa` grows, this project also reports
`factor_error_median`, `factor_error_p95`, and the percentages within factors 2
and 10. These are easier to interpret than LRE alone.

The project also reports a threshold-based accuracy score:

\[
\mathrm{relative\ error}=\frac{|\hat\kappa-\kappa|}{\kappa}.
\]

`accuracy_mean_percent` is the percentage of test matrices whose relative
error is below `0.5`. For example, if 270 out of 300 test matrices satisfy
`abs(expected - predicted) / expected < 0.5`, then
`accuracy_mean_percent = 90.0`.

The same value is also stored with the more explicit name
`relative_error_below_0_5_percent`. The raw count is stored as
`accuracy_count_below_0_5`.

## Model architecture

The default model is now edge-aware. Earlier GCN layers used only sparse
connectivity:

```python
messages = x[src] * normalization
```

The current `EdgeGCNLayer` also uses `edge_attr`, which stores
`log10(abs(A_ij))`, so message passing can react to the actual matrix values:

```python
message_input = torch.cat((x[src], edge_attr), dim=1)
messages = message_mlp(message_input)
```

The model section of each YAML config controls this behavior:

```yaml
model:
  hidden_dim: 64
  gcn_layers: 2
  head_dims: [128, 64]
  dropout: 0.1
  use_edge_attr: true
  edge_hidden_dim: 64
  residual: true
```

Existing checkpoints keep using the architecture stored inside their checkpoint
metadata. To benefit from the edge-aware model, regenerate or reuse the data and
train a new checkpoint.

## Small-size sweep experiment

To train on matrices up to `30 x 30`, then test the same model on sizes
`20 x 20` through `100 x 100` and plot accuracy versus matrix size:

```bash
condition-gnn generate --config configs/small30_sweep.yaml
condition-gnn train --config configs/small30_sweep.yaml --norm 2 --scheme 2
condition-gnn size-sweep --config configs/small30_sweep.yaml --norm 2 --scheme 2
```

The sweep writes:

- `norm_2_scheme_2_size_sweep.csv`
- `norm_2_scheme_2_size_sweep.json`
- `norm_2_scheme_2_size_sweep.png`

under `artifacts/small30_sweep/results/`.

The plot uses matrix size on the x-axis and
`relative_error_below_0_5_percent` on the y-axis. The dashed vertical line
marks the largest training size, `n=30`, so sizes greater than 30 show
out-of-distribution behavior.

## GNN vs Gershgorin comparison

The project can now compare the trained GNN and a Gershgorin bound estimator on the same generated test matrices.
For each matrix, the reported accuracy is

```text
abs(true_kappa2 - predicted_kappa2) / true_kappa2 < 0.5
```

The denominator is the full test set. If Gershgorin returns `inf` because the lower bound is non-positive, that sample is counted as a failed prediction, not removed.

Run the normal benchmark on the generated test split:

```bash
condition-gnn benchmark \
  --config configs/small30_sweep.yaml \
  --norm 2 \
  --scheme 2
```

The benchmark CSV includes both `GNN` and `Gershgorin` rows:

```text
artifacts/small30_sweep/results/norm_2_scheme_2_benchmark.csv
```

Run the size sweep comparison from `20x20` to `100x100`:

```bash
condition-gnn size-sweep \
  --config configs/small30_sweep.yaml \
  --norm 2 \
  --scheme 2
```

Outputs:

```text
artifacts/small30_sweep/results/norm_2_scheme_2_size_sweep.csv
artifacts/small30_sweep/results/norm_2_scheme_2_size_sweep.json
artifacts/small30_sweep/results/norm_2_scheme_2_size_sweep.png
```

The PNG plots two curves on the same axes: `GNN` and `Gershgorin`.

## Two-dataset comparison: SPD vs general matrices

This version supports two separate test distributions for kappa_2(A):

1. `configs/small30_spd_sweep.yaml`
   - `matrix_type: spd`
   - families: `random_spd`, `tridiagonal`
   - ground truth: eigenvalue ratio, `lambda_max / lambda_min`
   - Gershgorin baseline: direct symmetric/SPD Gershgorin bound

2. `configs/small30_general_sweep.yaml`
   - `matrix_type: general`
   - families: `general_diagonal_dominant`, `general_scaled_diagonal_dominant`, `general_sparse_random`
   - ground truth: singular-value ratio, `sigma_max / sigma_min`
   - Gershgorin baseline: Gershgorin bound on `A.T @ A`, then square root

Accuracy is computed on the full test set:

```text
relative_error = abs(Expected - Predict) / Expected
accuracy = 100 * number_of_samples(relative_error < 0.5) / total_samples
```

Run both experiments:

```bash
pip install -e .
./scripts/run_two_dataset_comparison.sh
```

Or run them manually:

```bash
condition-gnn generate --config configs/small30_spd_sweep.yaml
condition-gnn train --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2
condition-gnn benchmark --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2
condition-gnn size-sweep --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2

condition-gnn generate --config configs/small30_general_sweep.yaml
condition-gnn train --config configs/small30_general_sweep.yaml --norm 2 --scheme 2
condition-gnn benchmark --config configs/small30_general_sweep.yaml --norm 2 --scheme 2
condition-gnn size-sweep --config configs/small30_general_sweep.yaml --norm 2 --scheme 2
```

Outputs:

```text
artifacts/small30_spd_sweep/results/norm_2_scheme_2_benchmark.csv
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep.png
artifacts/small30_general_sweep/results/norm_2_scheme_2_benchmark.csv
artifacts/small30_general_sweep/results/norm_2_scheme_2_size_sweep.png
```

## SPD family-separated size sweep

To avoid hiding family-specific behavior in a mixed average, run a size sweep where each test size is evaluated on two separate 100% family datasets:

- 300 matrices from `random_spd`
- 300 matrices from `tridiagonal`

Train the mixed SPD model once:

```bash
condition-gnn generate --config configs/small30_spd_sweep.yaml
condition-gnn train --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2
```

Then run the family-separated test:

```bash
condition-gnn size-sweep --config configs/small30_spd_family_sweep.yaml --norm 2 --scheme 2
```

Outputs:

```text
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep_family_split.csv
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep_family_split.json
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep_family_split.png
```

The CSV includes `family_group`, so you can compare `GNN - random_spd`, `Gershgorin - random_spd`, `GNN - tridiagonal`, and `Gershgorin - tridiagonal` directly.


## Family-separated plot outputs

The outputs are written to:

```text
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep_family_split.csv
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep_family_split.json
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep_family_split.png
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep_family_split_random_spd.png
artifacts/small30_spd_sweep/results/norm_2_scheme_2_size_sweep_family_split_tridiagonal.png
```

The first PNG is the combined view. The last two PNG files are the separate plots for the 100% `random_spd` test dataset and the 100% `tridiagonal` test dataset. The CSV includes `family_group`, so you can also filter the numeric results directly.
