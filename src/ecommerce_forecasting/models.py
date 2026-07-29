"""Forecasting models used in rolling-origin evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecommerce_forecasting.features import make_recursive_row, make_training_matrix


def _check_inputs(train: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_train = train.sort_values("date").reset_index(drop=True)
    ordered_future = future.sort_values("date").reset_index(drop=True)
    if len(ordered_train) < 84:
        raise ValueError("Forecasting requires at least 84 days of history")
    if ordered_future.empty:
        raise ValueError("Future driver frame cannot be empty")
    expected_start = ordered_train["date"].max() + pd.Timedelta(days=1)
    if ordered_future["date"].min() != expected_start:
        raise ValueError("Future dates must start one day after training history")
    return ordered_train, ordered_future


def seasonal_naive_forecast(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Repeat the demand observed seven days earlier."""

    ordered_train, ordered_future = _check_inputs(train, future)
    demand_history = ordered_train["demand"].astype(float).tolist()
    predictions: list[float] = []
    for _ in ordered_future.itertuples(index=False):
        prediction = max(0.0, demand_history[-7])
        predictions.append(prediction)
        demand_history.append(prediction)
    return np.asarray(predictions)


def seasonal_average_forecast(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Average demand from the same weekday in the previous four weeks."""

    ordered_train, ordered_future = _check_inputs(train, future)
    demand_history = ordered_train["demand"].astype(float).tolist()
    predictions: list[float] = []
    for _ in ordered_future.itertuples(index=False):
        prediction = float(np.mean([demand_history[-lag] for lag in (7, 14, 21, 28)]))
        predictions.append(max(0.0, prediction))
        demand_history.append(max(0.0, prediction))
    return np.asarray(predictions)


def driver_ridge_forecast(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Recursive ridge model using lags, calendar and known commercial drivers."""

    ordered_train, ordered_future = _check_inputs(train, future)
    features, target = make_training_matrix(ordered_train)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=12.0)),
        ]
    )
    model.fit(features, target)

    demand_history = ordered_train["demand"].astype(float).tolist()
    predictions: list[float] = []
    for row in ordered_future.itertuples(index=False):
        driver_row = pd.Series(row._asdict())
        feature_row = make_recursive_row(row.date, driver_row, demand_history)
        prediction = max(0.0, float(model.predict(feature_row)[0]))
        predictions.append(prediction)
        demand_history.append(prediction)
    return np.asarray(predictions)


def forecast(model_name: str, train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Dispatch a model by its stable public name."""

    models = {
        "seasonal_naive": seasonal_naive_forecast,
        "seasonal_average": seasonal_average_forecast,
        "driver_ridge": driver_ridge_forecast,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name](train, future)
