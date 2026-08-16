"""End-to-end case-study pipeline."""

from __future__ import annotations

import json
import tempfile
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
from ecommerce_forecasting.external_inputs import load_supplied_inputs
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
    return _run_with_inputs(
        config,
        raw_history,
        base_drivers,
        campaign_drivers=campaign_drivers,
        data_source_mode="generated_synthetic",
        data_is_synthetic=True,
        input_audit=None,
        write_artifacts=write_artifacts,
    )


def run_supplied_pipeline(
    config: ForecastConfig,
    history_path: Path,
    future_drivers_path: Path,
    provenance_path: Path,
    *,
    write_artifacts: bool = True,
) -> dict[str, object]:
    """Run the same evaluation on validated, user-supplied aggregate inputs."""

    if write_artifacts and config.output_dir.exists():
        if not config.output_dir.is_dir() or any(config.output_dir.iterdir()):
            raise ValueError("supplied-mode output directory must be a new or empty directory")
    raw_history, future_drivers, input_audit = load_supplied_inputs(
        history_path,
        future_drivers_path,
        provenance_path,
        horizon=config.horizon,
        validation_folds=config.validation_folds,
    )
    categories = tuple(sorted(raw_history["category"].unique()))
    supplied_config = replace(config, categories=categories)
    return _run_with_inputs(
        supplied_config,
        raw_history,
        future_drivers,
        campaign_drivers=None,
        data_source_mode="supplied",
        data_is_synthetic=bool(input_audit["declared_data_is_synthetic"]),
        input_audit=input_audit,
        write_artifacts=write_artifacts,
    )


def _run_with_inputs(
    config: ForecastConfig,
    raw_history: pd.DataFrame,
    base_drivers: pd.DataFrame,
    *,
    campaign_drivers: pd.DataFrame | None,
    data_source_mode: str,
    data_is_synthetic: bool,
    input_audit: dict[str, object] | None,
    write_artifacts: bool,
) -> dict[str, object]:
    """Evaluate history and forecast supplied future drivers under one contract."""

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

    base_forecast = selected_forecast(
        history,
        base_drivers,
        selections,
        interval_widths,
    )
    campaign_forecast = None
    scenario_impact = None
    if campaign_drivers is not None:
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
        "data_source_mode": data_source_mode,
        "data_is_synthetic": data_is_synthetic,
        "provenance_is_user_supplied": data_source_mode == "supplied",
        "history_rows": int(len(history)),
        "history_start": str(history["date"].min().date()),
        "history_end": str(history["date"].max().date()),
        "forecast_horizon_days": config.horizon,
        "validation_folds": config.validation_folds,
        "holdout_wape": round(float(overall["wape"]), 6),
        "holdout_bias": round(float(overall["bias"]), 6),
        "holdout_interval_coverage": round(float(overall["interval_coverage"]), 6),
        "target_interval_coverage": config.interval_coverage,
        "scenario_incremental_orders": (
            round(float(scenario_impact["incremental_orders"].sum()), 2)
            if scenario_impact is not None
            else None
        ),
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
            data_source_mode,
            input_audit,
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


def run_supplied_smoke_pipeline(output_dir: Path) -> dict[str, object]:
    """Exercise the supplied-input contract with an explicitly synthetic fixture."""

    config = ForecastConfig(
        horizon=14,
        validation_folds=2,
        categories=("Beauty", "Grocery"),
        output_dir=output_dir,
        generated_data_dir=output_dir / "generated",
    )
    raw_history = generate_synthetic_history(
        "2024-01-01",
        "2025-02-28",
        config.categories,
        seed=17,
    )
    future_drivers = generate_future_drivers(
        raw_history["date"].max(),
        config.horizon,
        config.categories,
        scenario="base",
    )
    provenance = {
        "schema_version": "1.0",
        "dataset_name": "CI supplied-input fixture",
        "source_description": "Generated in memory by the repository smoke test",
        "extraction_date": "2025-02-28",
        "usage_permission": "Repository test fixture",
        "data_is_synthetic": True,
        "contains_personal_data": False,
    }
    with tempfile.TemporaryDirectory(prefix="forecast-supplied-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        history_path = temp_path / "history.csv"
        future_path = temp_path / "future.csv"
        provenance_path = temp_path / "provenance.json"
        raw_history.to_csv(history_path, index=False)
        future_drivers.to_csv(future_path, index=False)
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        return run_supplied_pipeline(
            config,
            history_path,
            future_path,
            provenance_path,
            write_artifacts=False,
        )


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
    campaign_forecast: pd.DataFrame | None,
    scenario_impact: pd.DataFrame | None,
    capacity_plan: pd.DataFrame,
    summary: dict[str, object],
    data_source_mode: str,
    input_audit: dict[str, object] | None,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if data_source_mode == "generated_synthetic":
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
    if campaign_forecast is not None and scenario_impact is not None:
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
        history_label=(
            "Synthetic daily demand"
            if data_source_mode == "generated_synthetic"
            else "Supplied daily demand"
        ),
    )
    write_decision_memo(
        final_metrics,
        selections,
        scenario_impact,
        capacity_plan,
        config.output_dir,
        data_source_mode=data_source_mode,
    )
    write_summary_json(summary, config.output_dir)
    if input_audit is not None:
        write_summary_json(input_audit, config.output_dir, filename="input_provenance.json")


def with_output(config: ForecastConfig, output_dir: Path) -> ForecastConfig:
    """Return a config whose generated and report files share one root."""

    return replace(
        config,
        output_dir=output_dir,
        generated_data_dir=output_dir / "generated",
    )
