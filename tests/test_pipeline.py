from pathlib import Path

from ecommerce_forecasting.pipeline import run_smoke_pipeline


def test_smoke_pipeline_is_reproducible(tmp_path: Path) -> None:
    first = run_smoke_pipeline(tmp_path / "first")
    second = run_smoke_pipeline(tmp_path / "second")
    assert first == second
    assert first["data_is_synthetic"] is True
    assert 0 <= first["holdout_wape"] < 1
