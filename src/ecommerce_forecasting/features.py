"""Time-safe feature construction for recursive forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecommerce_forecasting.data import ANCHOR_DATE

LAGS: tuple[int, ...] = (1, 7, 14, 28, 56)
ROLLING_WINDOWS: tuple[int, ...] = (7, 28)

FEATURE_COLUMNS: tuple[str, ...] = (
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
    "trend_days",
    "campaign",
    "discount_rate",
    "special_event",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "rolling_mean_7",
    "rolling_mean_28",
)


def calendar_values(date: pd.Timestamp) -> dict[str, float]:
    """Return deterministic calendar features known at forecast time."""

    timestamp = pd.Timestamp(date)
    day_of_week = timestamp.dayofweek
    day_of_year = timestamp.dayofyear
    return {
        "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
        "year_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "year_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
        "trend_days": float((timestamp - ANCHOR_DATE).days),
    }


def make_training_matrix(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build lagged features using only values strictly before each target date."""

    ordered = history.sort_values("date").reset_index(drop=True).copy()
    required = {"date", "demand", "campaign", "discount_rate", "special_event"}
    missing = required - set(ordered.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    calendar = pd.DataFrame([calendar_values(value) for value in ordered["date"]])
    for column in calendar.columns:
        ordered[column] = calendar[column]
    for lag in LAGS:
        ordered[f"lag_{lag}"] = ordered["demand"].shift(lag)
    prior_demand = ordered["demand"].shift(1)
    for window in ROLLING_WINDOWS:
        ordered[f"rolling_mean_{window}"] = prior_demand.rolling(window).mean()

    supervised = ordered.dropna(subset=list(FEATURE_COLUMNS) + ["demand"])
    if supervised.empty:
        raise ValueError("Not enough history to create lag features")
    return supervised.loc[:, FEATURE_COLUMNS], supervised["demand"]


def make_recursive_row(
    date: pd.Timestamp,
    driver_row: pd.Series,
    demand_history: list[float],
) -> pd.DataFrame:
    """Build one future feature row from known drivers and prior demand only."""

    if len(demand_history) < max(LAGS):
        raise ValueError(f"At least {max(LAGS)} demand values are required")
    values = calendar_values(date)
    values.update(
        {
            "campaign": float(driver_row["campaign"]),
            "discount_rate": float(driver_row["discount_rate"]),
            "special_event": float(driver_row["special_event"]),
        }
    )
    for lag in LAGS:
        values[f"lag_{lag}"] = float(demand_history[-lag])
    for window in ROLLING_WINDOWS:
        values[f"rolling_mean_{window}"] = float(np.mean(demand_history[-window:]))
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)
