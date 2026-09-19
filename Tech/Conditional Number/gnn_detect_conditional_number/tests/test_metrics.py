from __future__ import annotations

import numpy as np

from condition_gnn.metrics import regression_metrics


def test_accuracy_counts_predictions_below_half_relative_error() -> None:
    true_values = np.array([100.0, 200.0, 100.0])
    predicted_values = np.array([120.0, 350.0, 40.0])

    metrics = regression_metrics(np.log10(true_values), np.log10(predicted_values))

    expected_relative_error = np.array([0.2, 0.75, 0.6])
    assert np.isclose(metrics["relative_error_mean"], expected_relative_error.mean())
    assert np.isclose(metrics["relative_error_below_0_5_percent"], 100.0 / 3.0)
    assert np.isclose(metrics["accuracy_count_below_0_5"], 1.0)
    assert np.isclose(metrics["sample_count"], 3.0)
    assert np.isclose(metrics["accuracy_mean"], 1.0 / 3.0)
    assert np.isclose(metrics["accuracy_mean_percent"], 100.0 / 3.0)
