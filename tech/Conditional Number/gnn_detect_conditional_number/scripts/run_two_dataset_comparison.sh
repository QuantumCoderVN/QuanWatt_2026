#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/small30_spd_sweep.yaml"
  "configs/small30_general_sweep.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "========== Running $CONFIG =========="
  condition-gnn generate --config "$CONFIG"
  condition-gnn train --config "$CONFIG" --norm 2 --scheme 2
  condition-gnn benchmark --config "$CONFIG" --norm 2 --scheme 2
  condition-gnn size-sweep --config "$CONFIG" --norm 2 --scheme 2
done

echo "Done. Check artifacts/small30_spd_sweep/results and artifacts/small30_general_sweep/results."
