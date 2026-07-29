"""Shared project configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for the reproducible portfolio pipeline."""

    history_start: str = "2023-01-01"
    history_end: str = "2025-12-31"
    horizon: int = 56
    validation_folds: int = 3
    interval_coverage: float = 0.80
    random_seed: int = 42
    categories: tuple[str, ...] = ("Beauty", "Electronics", "Grocery", "Home")
    output_dir: Path = Path("reports")
    generated_data_dir: Path = Path("data/generated")


MODEL_NAMES: tuple[str, ...] = (
    "seasonal_naive",
    "seasonal_average",
    "driver_ridge",
)
