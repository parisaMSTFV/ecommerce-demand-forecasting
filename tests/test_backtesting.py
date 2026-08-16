import pandas as pd

from ecommerce_forecasting.backtesting import (
    aggregate_daily,
    calibrate_intervals,
    evaluation_dates,
    select_models,
)


def test_evaluation_windows_do_not_overlap_holdout() -> None:
    validation_starts, holdout_start = evaluation_dates(pd.Timestamp("2025-12-31"), 56, 3)
    assert validation_starts == [
        pd.Timestamp("2025-05-22"),
        pd.Timestamp("2025-07-17"),
        pd.Timestamp("2025-09-11"),
    ]
    assert holdout_start == pd.Timestamp("2025-11-06")
    assert validation_starts[-1] + pd.offsets.Day(56) == holdout_start


def test_model_selection_uses_lowest_category_wape() -> None:
    metrics = pd.DataFrame(
        {
            "category": ["Beauty", "Beauty", "Overall"],
            "model": ["seasonal_naive", "driver_ridge", "driver_ridge"],
            "wape": [0.10, 0.06, 0.07],
            "mae": [10, 6, 7],
            "bias": [0.01, 0.00, 0.00],
            "observations": [20, 20, 20],
        }
    )
    selected = select_models(metrics)
    assert selected.loc[0, "selected_model"] == "driver_ridge"


def test_interval_calibration_uses_selected_model_only() -> None:
    predictions = pd.DataFrame(
        {
            "category": ["Beauty"] * 4,
            "model": ["driver_ridge", "driver_ridge", "seasonal_naive", "seasonal_naive"],
            "horizon_week": [1, 1, 1, 1],
            "actual": [100, 100, 100, 100],
            "forecast": [90, 120, 0, 0],
        }
    )
    selections = pd.DataFrame(
        {
            "category": ["Beauty"],
            "selected_model": ["driver_ridge"],
        }
    )
    widths = calibrate_intervals(predictions, selections, 0.8)
    assert widths.loc[0, "interval_half_width"] == 20


def test_category_forecasts_reconcile_to_total() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "category": ["Beauty", "Home"],
            "forecast": [100.0, 200.0],
            "lower": [80.0, 170.0],
            "upper": [120.0, 240.0],
        }
    )
    total = aggregate_daily(predictions)
    assert total.loc[0, "forecast"] == 300
    assert total.loc[0, "lower"] == 250
    assert total.loc[0, "upper"] == 360
