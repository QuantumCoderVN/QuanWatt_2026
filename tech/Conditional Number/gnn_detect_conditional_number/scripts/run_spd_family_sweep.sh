#!/usr/bin/env bash
set -euo pipefail

# Train one mixed SPD model on random_spd + tridiagonal, then test separately
# on 100% random_spd and 100% tridiagonal datasets.

condition-gnn generate --config configs/small30_spd_sweep.yaml
condition-gnn train --config configs/small30_spd_sweep.yaml --norm 2 --scheme 2
condition-gnn size-sweep --config configs/small30_spd_family_sweep.yaml --norm 2 --scheme 2
