from pathlib import Path

from ecommerce_forecasting.cli import build_parser
from ecommerce_forecasting.config import ForecastConfig
from ecommerce_forecasting.pipeline import run_smoke_pipeline, run_supplied_smoke_pipeline


def test_default_run_keeps_local_outputs_outside_committed_evidence() -> None:
    config = ForecastConfig()
    args = build_parser().parse_args(["run"])

    assert config.output_dir == Path("local-runs/latest")
    assert config.generated_data_dir == Path("local-runs/latest/generated")
    assert args.output_dir == Path("local-runs/latest")


def test_smoke_pipeline_is_reproducible(tmp_path: Path) -> None:
    first = run_smoke_pipeline(tmp_path / "first")
    second = run_smoke_pipeline(tmp_path / "second")
    assert first == second
    assert first["data_is_synthetic"] is True
    assert first["data_source_mode"] == "generated_synthetic"
    assert 0 <= first["holdout_wape"] < 1


def test_supplied_smoke_pipeline_uses_the_external_contract(tmp_path: Path) -> None:
    result = run_supplied_smoke_pipeline(tmp_path / "supplied")
    assert result["data_source_mode"] == "supplied"
    assert result["data_is_synthetic"] is True
    assert result["provenance_is_user_supplied"] is True
    assert result["scenario_incremental_orders"] is None
