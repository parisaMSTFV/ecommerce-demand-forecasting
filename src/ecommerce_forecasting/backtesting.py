"""Rolling-origin validation, model selection and holdout evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ecommerce_forecasting.metrics import bias, interval_coverage, mae, wape
from ecommerce_forecasting.models import forecast


def evaluation_dates(
    last_date: pd.Timestamp,
    horizon: int,
    validation_folds: int,
) -> tuple[list[pd.Timestamp], pd.Timestamp]:
    """Return non-overlapping validation starts and the final holdout start."""

    if horizon < 1 or validation_folds < 1:
        raise ValueError("horizon and validation_folds must be positive")
    holdout_start = pd.Timestamp(last_date) - pd.offsets.Day(horizon - 1)
    validation_starts = [
        holdout_start - pd.offsets.Day(horizon * fold_back)
        for fold_back in range(validation_folds, 0, -1)
    ]
    return validation_starts, holdout_start


def rolling_validation(
    history: pd.DataFrame,
    model_names: Iterable[str],
    horizon: int,
    validation_folds: int,
) -> pd.DataFrame:
    """Generate out-of-sample predictions for non-overlapping rolling folds."""

    last_date = history["date"].max()
    validation_starts, _ = evaluation_dates(last_date, horizon, validation_folds)
    rows: list[pd.DataFrame] = []

    for category, category_history in history.groupby("category", sort=True):
        ordered = category_history.sort_values("date").reset_index(drop=True)
        for fold_id, start_date in enumerate(validation_starts, start=1):
            end_date = start_date + pd.offsets.Day(horizon - 1)
            train = ordered.loc[ordered["date"] < start_date].copy()
            actual = ordered.loc[ordered["date"].between(start_date, end_date)].copy()
            if len(actual) != horizon:
                raise ValueError(f"Incomplete validation fold {fold_id} for {category}")
            future = actual.drop(columns=["demand"], errors="ignore")

            for model_name in model_names:
                predicted = forecast(model_name, train, future)
                fold = pd.DataFrame(
                    {
                        "date": actual["date"].to_numpy(),
                        "category": category,
                        "fold": fold_id,
                        "horizon_day": np.arange(1, horizon + 1),
                        "horizon_week": (np.arange(horizon) // 7) + 1,
                        "model": model_name,
                        "actual": actual["demand"].to_numpy(),
                        "forecast": predicted,
                    }
                )
                rows.append(fold)

    return pd.concat(rows, ignore_index=True)


def validation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise validation error by model and category."""

    records: list[dict[str, object]] = []
    for (category, model_name), group in predictions.groupby(["category", "model"], sort=True):
        records.append(
            {
                "category": category,
                "model": model_name,
                "wape": wape(group["actual"].to_numpy(), group["forecast"].to_numpy()),
                "mae": mae(group["actual"].to_numpy(), group["forecast"].to_numpy()),
                "bias": bias(group["actual"].to_numpy(), group["forecast"].to_numpy()),
                "observations": len(group),
            }
        )

    for model_name, group in predictions.groupby("model", sort=True):
        records.append(
            {
                "category": "Overall",
                "model": model_name,
                "wape": wape(group["actual"].to_numpy(), group["forecast"].to_numpy()),
                "mae": mae(group["actual"].to_numpy(), group["forecast"].to_numpy()),
                "bias": bias(group["actual"].to_numpy(), group["forecast"].to_numpy()),
                "observations": len(group),
            }
        )
    return pd.DataFrame(records).sort_values(["category", "wape"]).reset_index(drop=True)


def select_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest-WAPE model independently for each category."""

    category_metrics = metrics.loc[metrics["category"] != "Overall"].copy()
    selected = category_metrics.loc[category_metrics.groupby("category")["wape"].idxmin()].copy()
    selected = selected.rename(
        columns={
            "model": "selected_model",
            "wape": "validation_wape",
            "mae": "validation_mae",
            "bias": "validation_bias",
        }
    )
    return selected[
        [
            "category",
            "selected_model",
            "validation_wape",
            "validation_mae",
            "validation_bias",
        ]
    ].sort_values("category", ignore_index=True)


def calibrate_intervals(
    validation_predictions: pd.DataFrame,
    selections: pd.DataFrame,
    coverage: float,
) -> pd.DataFrame:
    """Calibrate absolute-error widths from selected-model validation residuals."""

    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    selected_predictions = validation_predictions.merge(
        selections[["category", "selected_model"]],
        left_on=["category", "model"],
        right_on=["category", "selected_model"],
        how="inner",
    )
    selected_predictions["absolute_error"] = np.abs(
        selected_predictions["actual"] - selected_predictions["forecast"]
    )
    widths = (
        selected_predictions.groupby(["category", "horizon_week"], as_index=False)[
            "absolute_error"
        ]
        .quantile(coverage, interpolation="higher")
        .rename(columns={"absolute_error": "interval_half_width"})
    )
    widths["target_coverage"] = coverage
    return widths


def selected_forecast(
    train_history: pd.DataFrame,
    future: pd.DataFrame,
    selections: pd.DataFrame,
    interval_widths: pd.DataFrame,
    actual_column: str | None = None,
) -> pd.DataFrame:
    """Forecast each category with its validation-selected model."""

    rows: list[pd.DataFrame] = []
    selection_map = selections.set_index("category")["selected_model"].to_dict()

    for category, category_future in future.groupby("category", sort=True):
        category_train = train_history.loc[train_history["category"] == category].copy()
        ordered_future = category_future.sort_values("date").reset_index(drop=True)
        model_name = str(selection_map[category])
        predicted = forecast(model_name, category_train, ordered_future)
        result = pd.DataFrame(
            {
                "date": ordered_future["date"],
                "category": category,
                "selected_model": model_name,
                "horizon_day": np.arange(1, len(ordered_future) + 1),
                "horizon_week": (np.arange(len(ordered_future)) // 7) + 1,
                "forecast": predicted,
            }
        )
        category_widths = interval_widths.loc[
            interval_widths["category"] == category,
            ["horizon_week", "interval_half_width"],
        ]
        result = result.merge(
            category_widths,
            on="horizon_week",
            how="left",
            validate="many_to_one",
        )
        if result["interval_half_width"].isna().any():
            raise ValueError(f"Missing interval width for {category}")
        result["lower"] = (result["forecast"] - result["interval_half_width"]).clip(lower=0)
        result["upper"] = result["forecast"] + result["interval_half_width"]
        if actual_column:
            result["actual"] = ordered_future[actual_column].to_numpy()
        rows.append(result)

    return pd.concat(rows, ignore_index=True).sort_values(["category", "date"]).reset_index(
        drop=True
    )


def holdout_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate selected models on the untouched final period."""

    if "actual" not in predictions:
        raise ValueError("Holdout predictions require actual values")
    records: list[dict[str, object]] = []
    for category, group in predictions.groupby("category", sort=True):
        records.append(_metric_record(category, group))
    records.append(_metric_record("Overall", predictions))
    return pd.DataFrame(records)


def _metric_record(label: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "category": label,
        "wape": wape(group["actual"].to_numpy(), group["forecast"].to_numpy()),
        "mae": mae(group["actual"].to_numpy(), group["forecast"].to_numpy()),
        "bias": bias(group["actual"].to_numpy(), group["forecast"].to_numpy()),
        "interval_coverage": interval_coverage(
            group["actual"].to_numpy(),
            group["lower"].to_numpy(),
            group["upper"].to_numpy(),
        ),
        "observations": len(group),
    }


def aggregate_daily(predictions: pd.DataFrame) -> pd.DataFrame:
    """Sum category forecasts into an exactly reconciled daily total."""

    columns = ["forecast", "lower", "upper"]
    if "actual" in predictions:
        columns.append("actual")
    aggregated = predictions.groupby("date", as_index=False)[columns].sum()
    aggregated["category"] = "Total"
    return aggregated
