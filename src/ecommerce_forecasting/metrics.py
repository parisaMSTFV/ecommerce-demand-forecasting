"""Forecast evaluation metrics with explicit business interpretations."""

from __future__ import annotations

import numpy as np


def _arrays(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    if actual_array.shape != forecast_array.shape:
        raise ValueError("actual and forecast must have the same shape")
    if actual_array.size == 0:
        raise ValueError("metrics require at least one observation")
    return actual_array, forecast_array


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted absolute percentage error."""

    actual_array, forecast_array = _arrays(actual, forecast)
    denominator = np.abs(actual_array).sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined when total actual demand is zero")
    return float(np.abs(actual_array - forecast_array).sum() / denominator)


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error in order units."""

    actual_array, forecast_array = _arrays(actual, forecast)
    return float(np.abs(actual_array - forecast_array).mean())


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Signed error divided by total actual demand; positive means over-forecast."""

    actual_array, forecast_array = _arrays(actual, forecast)
    denominator = np.abs(actual_array).sum()
    if denominator == 0:
        raise ValueError("Bias is undefined when total actual demand is zero")
    return float((forecast_array - actual_array).sum() / denominator)


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actual values inside the prediction interval."""

    actual_array = np.asarray(actual, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if not (actual_array.shape == lower_array.shape == upper_array.shape):
        raise ValueError("actual, lower and upper must have the same shape")
    if np.any(lower_array > upper_array):
        raise ValueError("lower interval cannot exceed upper interval")
    return float(((actual_array >= lower_array) & (actual_array <= upper_array)).mean())
