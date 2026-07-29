import numpy as np
import pandas as pd

from ecommerce_forecasting.data import generate_synthetic_history, prepare_history
from ecommerce_forecasting.features import make_training_matrix
from ecommerce_forecasting.models import driver_ridge_forecast


def test_lag_features_use_only_prior_targets() -> None:
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    history = pd.DataFrame(
        {
            "date": dates,
            "demand": np.arange(100, dtype=float),
            "campaign": 0,
            "discount_rate": 0.02,
            "special_event": 0,
        }
    )
    features, target = make_training_matrix(history)
    first_target = target.iloc[0]
    assert features.iloc[0]["lag_1"] == first_target - 1
    assert features.iloc[0]["lag_56"] == first_target - 56


def test_future_target_values_cannot_change_prediction() -> None:
    raw = generate_synthetic_history("2024-01-01", "2024-07-31", ("Beauty",), seed=9)
    history = prepare_history(raw)
    train = history.iloc[:-14].copy()
    future = history.iloc[-14:].drop(columns=["demand"]).copy()
    first = driver_ridge_forecast(train, future)
    future["demand"] = 99_999_999
    second = driver_ridge_forecast(train, future)
    np.testing.assert_allclose(first, second)
