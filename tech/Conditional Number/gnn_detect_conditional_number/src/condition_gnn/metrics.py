from __future__ import annotations

import numpy as np


def regression_metrics(true_log_kappa: np.ndarray, predicted_log_kappa: np.ndarray) -> dict[str, float]:
    true_log = np.asarray(true_log_kappa, dtype=np.float64)
    predicted_log = np.asarray(predicted_log_kappa, dtype=np.float64)
    log_error = np.abs(predicted_log - true_log)
    lre = log_error / np.maximum(np.abs(true_log), 1e-12)
    factor = np.power(10.0, np.minimum(log_error, 300.0))
    true_value = np.power(10.0, np.clip(true_log, -300.0, 300.0))
    predicted_value = np.power(10.0, np.clip(predicted_log, -300.0, 300.0))
    relative_error = np.abs(predicted_value - true_value) / np.maximum(
        np.abs(true_value), 1e-300
    )
    accurate = relative_error < 0.5
    return {
        "sample_count": float(true_value.size),
        "paper_lre_mean_percent": float(100.0 * lre.mean()),
        "paper_lre_max_percent": float(100.0 * lre.max()),
        "log10_error_mean": float(log_error.mean()),
        "log10_error_median": float(np.median(log_error)),
        "log10_error_p95": float(np.quantile(log_error, 0.95)),
        "relative_error_mean": float(relative_error.mean()),
        "relative_error_median": float(np.median(relative_error)),
        "relative_error_p95": float(np.quantile(relative_error, 0.95)),
        "relative_error_max": float(relative_error.max()),
        "relative_error_below_0_5_percent": float(100.0 * np.mean(accurate)),
        "accuracy_count_below_0_5": float(np.sum(accurate)),
        "accuracy_mean": float(np.mean(accurate)),
        "accuracy_median": float(np.median(accurate)),
        "accuracy_min": float(accurate.min()),
        "accuracy_mean_percent": float(100.0 * np.mean(accurate)),
        "factor_error_median": float(np.median(factor)),
        "factor_error_p95": float(np.quantile(factor, 0.95)),
        "within_factor_2_percent": float(100.0 * np.mean(factor <= 2.0)),
        "within_factor_10_percent": float(100.0 * np.mean(factor <= 10.0)),
        "paper_lre_below_0_5_percent": float(100.0 * np.mean(lre < 0.5)),
        "paper_lre_below_1_percent": float(100.0 * np.mean(lre < 1.0)),
    }



def _nan_metrics(sample_count: int) -> dict[str, float]:
    return {
        "sample_count": float(sample_count),
        "finite_prediction_count": 0.0,
        "finite_prediction_percent": 0.0,
        "paper_lre_mean_percent": float("nan"),
        "paper_lre_max_percent": float("nan"),
        "log10_error_mean": float("nan"),
        "log10_error_median": float("nan"),
        "log10_error_p95": float("nan"),
        "relative_error_mean": float("nan"),
        "relative_error_median": float("nan"),
        "relative_error_p95": float("nan"),
        "relative_error_max": float("nan"),
        "relative_error_below_0_5_percent": 0.0,
        "accuracy_count_below_0_5": 0.0,
        "accuracy_mean": 0.0,
        "accuracy_median": 0.0,
        "accuracy_min": 0.0,
        "accuracy_mean_percent": 0.0,
        "factor_error_median": float("nan"),
        "factor_error_p95": float("nan"),
        "within_factor_2_percent": 0.0,
        "within_factor_10_percent": 0.0,
        "paper_lre_below_0_5_percent": 0.0,
        "paper_lre_below_1_percent": 0.0,
    }


def condition_number_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    relative_error_threshold: float = 0.5,
) -> dict[str, float]:
    """Metrics computed in the original condition-number scale.

    Accuracy is the percentage of the whole test set satisfying

        abs(true - predicted) / true < relative_error_threshold.

    Infinite or non-positive predictions are counted as failures, not removed from
    the denominator. This is important for bound-based methods such as Gershgorin,
    which may return infinity when the lower eigenvalue bound is non-positive.
    """
    true_value = np.asarray(true_values, dtype=np.float64)
    predicted_value = np.asarray(predicted_values, dtype=np.float64)
    if true_value.shape != predicted_value.shape:
        raise ValueError("true_values and predicted_values must have the same shape")

    sample_count = int(true_value.size)
    if sample_count == 0:
        return _nan_metrics(0)

    valid_true = np.isfinite(true_value) & (true_value > 0.0)
    finite_prediction = np.isfinite(predicted_value) & (predicted_value > 0.0)
    valid = valid_true & finite_prediction

    relative_error = np.full(sample_count, np.inf, dtype=np.float64)
    relative_error[valid_true] = np.abs(
        predicted_value[valid_true] - true_value[valid_true]
    ) / np.maximum(np.abs(true_value[valid_true]), 1e-300)

    accurate = relative_error < relative_error_threshold
    finite_relative_error = relative_error[valid]

    if not np.any(valid):
        metrics = _nan_metrics(sample_count)
        metrics.update(
            {
                "finite_prediction_count": float(np.sum(finite_prediction)),
                "finite_prediction_percent": float(100.0 * np.mean(finite_prediction)),
                "relative_error_below_0_5_percent": float(100.0 * np.mean(accurate)),
                "accuracy_count_below_0_5": float(np.sum(accurate)),
                "accuracy_mean": float(np.mean(accurate)),
                "accuracy_median": float(np.median(accurate)),
                "accuracy_min": float(np.min(accurate)),
                "accuracy_mean_percent": float(100.0 * np.mean(accurate)),
            }
        )
        return metrics

    true_log = np.log10(true_value[valid])
    predicted_log = np.log10(predicted_value[valid])
    log_error = np.abs(predicted_log - true_log)
    lre = log_error / np.maximum(np.abs(true_log), 1e-12)
    factor = np.power(10.0, np.minimum(log_error, 300.0))

    return {
        "sample_count": float(sample_count),
        "finite_prediction_count": float(np.sum(finite_prediction)),
        "finite_prediction_percent": float(100.0 * np.mean(finite_prediction)),
        "paper_lre_mean_percent": float(100.0 * lre.mean()),
        "paper_lre_max_percent": float(100.0 * lre.max()),
        "log10_error_mean": float(log_error.mean()),
        "log10_error_median": float(np.median(log_error)),
        "log10_error_p95": float(np.quantile(log_error, 0.95)),
        "relative_error_mean": float(finite_relative_error.mean()),
        "relative_error_median": float(np.median(finite_relative_error)),
        "relative_error_p95": float(np.quantile(finite_relative_error, 0.95)),
        "relative_error_max": float(finite_relative_error.max()),
        "relative_error_below_0_5_percent": float(100.0 * np.mean(accurate)),
        "accuracy_count_below_0_5": float(np.sum(accurate)),
        "accuracy_mean": float(np.mean(accurate)),
        "accuracy_median": float(np.median(accurate)),
        "accuracy_min": float(accurate.min()),
        "accuracy_mean_percent": float(100.0 * np.mean(accurate)),
        "factor_error_median": float(np.median(factor)),
        "factor_error_p95": float(np.quantile(factor, 0.95)),
        "within_factor_2_percent": float(100.0 * np.mean(factor <= 2.0)),
        "within_factor_10_percent": float(100.0 * np.mean(factor <= 10.0)),
        "paper_lre_below_0_5_percent": float(100.0 * np.mean(lre < 0.5)),
        "paper_lre_below_1_percent": float(100.0 * np.mean(lre < 1.0)),
    }
