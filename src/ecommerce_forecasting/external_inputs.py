"""Validation and provenance controls for user-supplied forecast inputs."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_COLUMNS: tuple[str, ...] = (
    "date",
    "category",
    "observed_orders",
    "availability_rate",
    "campaign",
    "discount_rate",
    "special_event",
)
FUTURE_DRIVER_COLUMNS: tuple[str, ...] = (
    "date",
    "category",
    "campaign",
    "discount_rate",
    "special_event",
)
PROVENANCE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "dataset_name",
    "source_description",
    "extraction_date",
    "usage_permission",
    "data_is_synthetic",
    "contains_personal_data",
)


def load_supplied_inputs(
    history_path: Path,
    future_drivers_path: Path,
    provenance_path: Path,
    *,
    horizon: int,
    validation_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load and validate aggregate inputs without copying the source files."""

    history = _read_csv(history_path, "history")
    future = _read_csv(future_drivers_path, "future drivers")
    provenance = _read_provenance(provenance_path)

    minimum_days = 84 + horizon * (validation_folds + 1)
    history = _validate_frame(history, HISTORY_COLUMNS, "history")
    future = _validate_frame(future, FUTURE_DRIVER_COLUMNS, "future drivers")
    _validate_values(history, "history")
    _validate_values(future, "future drivers")
    history_dates = _validate_balanced_daily_panel(history, "history")
    future_dates = _validate_balanced_daily_panel(future, "future drivers")

    if len(history_dates) < minimum_days:
        raise ValueError(
            "history requires at least "
            f"{minimum_days} consecutive days for {validation_folds} validation folds, "
            f"a {horizon}-day holdout and the minimum training window"
        )
    if len(future_dates) != horizon:
        raise ValueError(f"future drivers must contain exactly {horizon} days per category")
    expected_future_start = history_dates.max() + pd.offsets.Day(1)
    if future_dates.min() != expected_future_start:
        raise ValueError("future drivers must start one day after the history ends")

    history_categories = set(history["category"])
    future_categories = set(future["category"])
    if future_categories != history_categories:
        missing = sorted(history_categories - future_categories)
        extra = sorted(future_categories - history_categories)
        raise ValueError(
            f"future-driver categories must match history; missing={missing}, extra={extra}"
        )

    audit: dict[str, object] = {
        "contract_version": "1.0",
        "data_source_mode": "supplied",
        "provenance_is_user_supplied": True,
        "provenance_not_independently_verified": True,
        "dataset_name": provenance["dataset_name"],
        "source_description": provenance["source_description"],
        "extraction_date": provenance["extraction_date"],
        "usage_permission": provenance["usage_permission"],
        "declared_data_is_synthetic": provenance["data_is_synthetic"],
        "declared_contains_personal_data": provenance["contains_personal_data"],
        "history_sha256": _sha256(history_path),
        "future_drivers_sha256": _sha256(future_drivers_path),
        "history_rows": len(history),
        "future_driver_rows": len(future),
        "categories": sorted(history_categories),
        "history_start": str(history_dates.min().date()),
        "history_end": str(history_dates.max().date()),
        "future_start": str(future_dates.min().date()),
        "future_end": str(future_dates.max().date()),
        "raw_inputs_copied_to_output": False,
    }
    return history, future, audit


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"could not read {label} CSV: {error}") from error


def _read_provenance(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"provenance file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read provenance JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("provenance JSON must be an object")
    missing = sorted(set(PROVENANCE_FIELDS) - set(value))
    if missing:
        raise ValueError(f"missing provenance fields: {missing}")
    if value["schema_version"] != "1.0":
        raise ValueError("provenance schema_version must be '1.0'")
    for field in ("dataset_name", "source_description", "usage_permission"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"provenance {field} must be a non-empty string")
    try:
        if not isinstance(value["extraction_date"], str):
            raise TypeError
        date.fromisoformat(value["extraction_date"])
    except (TypeError, ValueError) as error:
        raise ValueError("provenance extraction_date must be an ISO-formatted date") from error
    for field in ("data_is_synthetic", "contains_personal_data"):
        if not isinstance(value[field], bool):
            raise ValueError(f"provenance {field} must be a JSON boolean")
    if value["contains_personal_data"]:
        raise ValueError(
            "this contract accepts aggregate category-day inputs only; "
            "contains_personal_data must be false"
        )
    return value


def _validate_frame(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")
    validated = frame.loc[:, required_columns].copy()
    if validated.empty:
        raise ValueError(f"{label} cannot be empty")
    if validated.isna().any().any():
        null_columns = sorted(validated.columns[validated.isna().any()].tolist())
        raise ValueError(f"{label} contains nulls in: {null_columns}")
    try:
        validated["date"] = pd.to_datetime(validated["date"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} dates must use YYYY-MM-DD") from error
    if (validated["date"] != validated["date"].dt.normalize()).any():
        raise ValueError(f"{label} dates must not include a time component")
    validated["category"] = validated["category"].astype(str).str.strip()
    if validated["category"].eq("").any():
        raise ValueError(f"{label} category values must be non-empty")
    if validated["category"].str.startswith(("=", "+", "-", "@")).any():
        raise ValueError(f"{label} category contains a spreadsheet-formula prefix")
    numeric_columns = set(required_columns) - {"date", "category"}
    for column in numeric_columns:
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} {column} must be numeric") from error
        if not np.isfinite(validated[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{label} {column} must be finite")
    duplicates = validated.duplicated(["category", "date"], keep=False)
    if duplicates.any():
        raise ValueError(f"{label} has duplicate category-date rows")
    return validated.sort_values(["category", "date"]).reset_index(drop=True)


def _validate_values(frame: pd.DataFrame, label: str) -> None:
    if "observed_orders" in frame and frame["observed_orders"].lt(0).any():
        raise ValueError("history observed_orders must be non-negative")
    if "availability_rate" in frame and (
        frame["availability_rate"].le(0).any() or frame["availability_rate"].gt(1).any()
    ):
        raise ValueError("history availability_rate must be in (0, 1]")
    if frame["discount_rate"].lt(0).any() or frame["discount_rate"].gt(1).any():
        raise ValueError(f"{label} discount_rate must be in [0, 1]")
    for column in ("campaign", "special_event"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{label} {column} must contain only 0 or 1")


def _validate_balanced_daily_panel(frame: pd.DataFrame, label: str) -> pd.DatetimeIndex:
    date_sets = {
        category: tuple(group["date"].sort_values())
        for category, group in frame.groupby("category", sort=True)
    }
    reference = next(iter(date_sets.values()))
    if any(dates != reference for dates in date_sets.values()):
        raise ValueError(f"{label} must have the same date coverage for every category")
    dates = pd.DatetimeIndex(reference)
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    if not dates.equals(expected):
        raise ValueError(f"{label} must contain one uninterrupted daily row per category")
    return dates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
