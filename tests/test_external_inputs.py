import json
from pathlib import Path

import pandas as pd
import pytest

from ecommerce_forecasting.config import ForecastConfig
from ecommerce_forecasting.data import generate_future_drivers, generate_synthetic_history
from ecommerce_forecasting.external_inputs import load_supplied_inputs
from ecommerce_forecasting.pipeline import run_supplied_pipeline


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    categories = ("Beauty", "Grocery")
    history = generate_synthetic_history("2024-01-01", "2024-06-30", categories, seed=13)
    future = generate_future_drivers(history["date"].max(), 14, categories, scenario="base")
    provenance = {
        "schema_version": "1.0",
        "dataset_name": "Contract test fixture",
        "source_description": "Synthetic rows created by the test suite",
        "extraction_date": "2024-06-30",
        "usage_permission": "Repository test fixture",
        "data_is_synthetic": True,
        "contains_personal_data": False,
    }
    history_path = root / "history.csv"
    future_path = root / "future.csv"
    provenance_path = root / "provenance.json"
    history.to_csv(history_path, index=False)
    future.to_csv(future_path, index=False)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return history_path, future_path, provenance_path


def test_supplied_pipeline_records_provenance_without_copying_raw_inputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    history_path, future_path, provenance_path = _write_inputs(input_dir)
    output_dir = tmp_path / "outputs"
    config = ForecastConfig(
        horizon=14,
        validation_folds=2,
        output_dir=output_dir,
        generated_data_dir=output_dir / "generated",
    )

    summary = run_supplied_pipeline(
        config,
        history_path,
        future_path,
        provenance_path,
    )

    audit = json.loads((output_dir / "input_provenance.json").read_text(encoding="utf-8"))
    assert summary["data_source_mode"] == "supplied"
    assert summary["data_is_synthetic"] is True
    assert summary["scenario_incremental_orders"] is None
    assert audit["raw_inputs_copied_to_output"] is False
    assert len(audit["history_sha256"]) == 64
    assert not (output_dir / "generated").exists()
    assert not (output_dir / "tables" / "future_campaign_forecast.csv").exists()


def test_supplied_contract_rejects_personal_data_declaration(tmp_path: Path) -> None:
    history_path, future_path, provenance_path = _write_inputs(tmp_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["contains_personal_data"] = True
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate category-day"):
        load_supplied_inputs(
            history_path,
            future_path,
            provenance_path,
            horizon=14,
            validation_folds=2,
        )


def test_supplied_contract_rejects_a_missing_history_day(tmp_path: Path) -> None:
    history_path, future_path, provenance_path = _write_inputs(tmp_path)
    lines = history_path.read_text(encoding="utf-8").splitlines()
    history_path.write_text("\n".join([lines[0], *lines[2:]]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="same date coverage"):
        load_supplied_inputs(
            history_path,
            future_path,
            provenance_path,
            horizon=14,
            validation_folds=2,
        )


def test_supplied_contract_rejects_infinite_numeric_value(tmp_path: Path) -> None:
    history_path, future_path, provenance_path = _write_inputs(tmp_path)
    history = pd.read_csv(history_path)
    history["observed_orders"] = history["observed_orders"].astype(float)
    history.loc[0, "observed_orders"] = float("inf")
    history.to_csv(history_path, index=False)

    with pytest.raises(ValueError, match="observed_orders must be finite"):
        load_supplied_inputs(
            history_path,
            future_path,
            provenance_path,
            horizon=14,
            validation_folds=2,
        )


def test_supplied_contract_rejects_formula_category(tmp_path: Path) -> None:
    history_path, future_path, provenance_path = _write_inputs(tmp_path)
    history = pd.read_csv(history_path)
    history.loc[history["category"] == "Beauty", "category"] = "=1+1"
    history.to_csv(history_path, index=False)

    with pytest.raises(ValueError, match="spreadsheet-formula prefix"):
        load_supplied_inputs(
            history_path,
            future_path,
            provenance_path,
            horizon=14,
            validation_folds=2,
        )
