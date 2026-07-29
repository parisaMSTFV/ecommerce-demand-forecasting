import numpy as np
import pytest

from ecommerce_forecasting.metrics import bias, interval_coverage, mae, wape


def test_point_metrics_have_expected_values() -> None:
    actual = np.array([100.0, 200.0])
    forecast = np.array([110.0, 180.0])
    assert wape(actual, forecast) == pytest.approx(0.10)
    assert mae(actual, forecast) == pytest.approx(15.0)
    assert bias(actual, forecast) == pytest.approx(-10 / 300)


def test_interval_coverage_counts_inclusive_bounds() -> None:
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([10.0, 18.0, 31.0, 35.0])
    upper = np.array([12.0, 22.0, 35.0, 39.0])
    assert interval_coverage(actual, lower, upper) == pytest.approx(0.5)


def test_wape_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="undefined"):
        wape(np.zeros(2), np.ones(2))
