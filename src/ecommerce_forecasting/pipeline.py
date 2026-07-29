"""End-to-end case-study pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from ecommerce_forecasting.backtesting import (
    aggregate_daily,
    calibrate_intervals,
    evaluation_dates,
    holdout_metrics,
    rolling_validation,
    select_models,
    selected_forecast,
    validation_metrics,
)
from ecommerce_forecasting.config import MODEL_NAMES, ForecastConfig
from ecommerce_forecasting.data import (
    generate_future_drivers,
    generate_synthetic_history,
    prepare_history,
)
from ecommerce_forecasting.reporting import (
    create_charts,
    write_decision_memo,
    write_summary_json,
)


def run_pipeline(config: ForecastConfig, write_artifacts: bool = True) -> dict[str, object]:
    """Run data generation, evaluation, forecasting and reporting."""

    raw_history = generate_synthetic_history(
        config.history_start,
        config.history_end,
        config.categories,
        config.random_seed,
    )
    history = prepare_history(raw_history)
    validation_predictions = rolling_validation(
        history,
        MODEL_NAMES,
        config.horizon,
        config.validation_folds,
    )
    model_metrics = validation_metrics(validation_predictions)
    selections = select_models(model_metrics)
    interval_widths = calibrate_intervals(
        validation_predictions,
        selections,
        config.interval_coverage,
    )

    _, holdout_start = evaluation_dates(
        history["date"].max(),
        config.horizon,
        config.validation_folds,
    )
    holdout_train = history.loc[history["date"] < holdout_start].copy()
    holdout_actual = history.loc[history["date"] >= holdout_start].copy()
    holdout_predictions = selected_forecast(
        holdout_train,
        holdout_actual,
        selections,
        interval_widths,
        actual_column="demand",
    )
    final_metrics = holdout_metrics(holdout_predictions)

    last_history_date = history["date"].max()
    base_drivers = generate_future_drivers(
        last_history_date,
        config.horizon,
        config.categories,
        scenario="base",
    )
    campaign_drivers = generate_future_drivers(
        last_history_date,
        config.horizon,
        config.categories,
        scenario="campaign",
    )
    base_forecast = selected_forecast(
        history,
        base_drivers,
        selections,
        interval_widths,
    )
    campaign_forecast = selected_forecast(
        history,
        campaign_drivers,
        selections,
        interval_widths,
    )

    scenario_impact = _scenario_impact(base_forecast, campaign_forecast)
    capacity_plan = aggregate_daily(base_forecast)
    capacity_plan["risk_buffer"] = capacity_plan["upper"] - capacity_plan["forecast"]
    capacity_plan["day_name"] = capacity_plan["date"].dt.day_name()

    overall = final_metrics.loc[final_metrics["category"] == "Overall"].iloc[0]
    summary: dict[str, object] = {
        "data_is_synthetic": True,
        "history_rows": int(len(history)),
        "history_start": str(history["date"].min().date()),
        "history_end": str(history["date"].max().date()),
        "forecast_horizon_days": config.horizon,
        "validation_folds": config.validation_folds,
        "holdout_wape": round(float(overall["wape"]), 6),
        "holdout_bias": round(float(overall["bias"]), 6),
        "holdout_interval_coverage": round(float(overall["interval_coverage"]), 6),
        "target_interval_coverage": config.interval_coverage,
        "scenario_incremental_orders": round(float(scenario_impact["incremental_orders"].sum()), 2),
        "selected_models": dict(
            zip(selections["category"], selections["selected_model"], strict=True)
        ),
    }

    if write_artifacts:
        _write_artifacts(
            config,
            raw_history,
            history,
            validation_predictions,
            model_metrics,
            selections,
            interval_widths,
            holdout_predictions,
            final_metrics,
            base_forecast,
            campaign_forecast,
            scenario_impact,
            capacity_plan,
            summary,
        )
    return summary


def run_smoke_pipeline(output_dir: Path) -> dict[str, object]:
    """Run a smaller configuration for CI without changing production settings."""

    config = ForecastConfig(
        history_start="2024-01-01",
        history_end="2025-02-28",
        horizon=14,
        validation_folds=2,
        categories=("Beauty", "Grocery"),
        output_dir=output_dir,
        generated_data_dir=output_dir / "generated",
    )
    return run_pipeline(config, write_artifacts=False)


def _scenario_impact(base: pd.DataFrame, campaign: pd.DataFrame) -> pd.DataFrame:
    base_totals = base.groupby("category", as_index=False)["forecast"].sum().rename(
        columns={"forecast": "base_orders"}
    )
    campaign_totals = campaign.groupby("category", as_index=False)["forecast"].sum().rename(
        columns={"forecast": "campaign_orders"}
    )
    impact = base_totals.merge(campaign_totals, on="category", validate="one_to_one")
    impact["incremental_orders"] = impact["campaign_orders"] - impact["base_orders"]
    impact["incremental_percent"] = impact["incremental_orders"] / impact["base_orders"]
    return impact


def _write_artifacts(
    config: ForecastConfig,
    raw_history: pd.DataFrame,
    history: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    model_metrics: pd.DataFrame,
    selections: pd.DataFrame,
    interval_widths: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    final_metrics: pd.DataFrame,
    base_forecast: pd.DataFrame,
    campaign_forecast: pd.DataFrame,
    scenario_impact: pd.DataFrame,
    capacity_plan: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.generated_data_dir.mkdir(parents=True, exist_ok=True)
    raw_history.to_csv(config.generated_data_dir / "synthetic_history.csv", index=False)

    tables = config.output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    validation_predictions.to_csv(tables / "validation_predictions.csv", index=False)
    model_metrics.to_csv(tables / "validation_metrics.csv", index=False)
    selections.to_csv(tables / "model_selection.csv", index=False)
    interval_widths.to_csv(tables / "interval_calibration.csv", index=False)
    holdout_predictions.to_csv(tables / "holdout_predictions.csv", index=False)
    final_metrics.to_csv(tables / "holdout_metrics.csv", index=False)
    base_forecast.to_csv(tables / "future_base_forecast.csv", index=False)
    campaign_forecast.to_csv(tables / "future_campaign_forecast.csv", index=False)
    scenario_impact.to_csv(tables / "scenario_impact.csv", index=False)
    capacity_plan.to_csv(tables / "capacity_plan.csv", index=False)

    create_charts(
        history,
        model_metrics,
        holdout_predictions,
        base_forecast,
        scenario_impact,
        config.output_dir,
    )
    write_decision_memo(
        final_metrics,
        selections,
        scenario_impact,
        capacity_plan,
        config.output_dir,
    )
    write_summary_json(summary, config.output_dir)


def with_output(config: ForecastConfig, output_dir: Path) -> ForecastConfig:
    """Return a config whose generated and report files share one root."""

    return replace(
        config,
        output_dir=output_dir,
        generated_data_dir=output_dir / "generated",
    )
