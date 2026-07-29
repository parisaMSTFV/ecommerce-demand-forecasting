"""Deterministic synthetic data for the forecasting case study."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

ANCHOR_DATE = pd.Timestamp("2023-01-01")

CATEGORY_PARAMETERS: dict[str, dict[str, object]] = {
    "Beauty": {
        "base": 1_250,
        "daily_trend": 0.00045,
        "annual_amplitude": 0.09,
        "campaign_lift": 0.20,
        "discount_elasticity": 1.10,
        "noise": 0.055,
        "weekday": (0.98, 0.99, 1.00, 1.02, 1.08, 1.12, 0.95),
    },
    "Electronics": {
        "base": 1_850,
        "daily_trend": 0.00025,
        "annual_amplitude": 0.13,
        "campaign_lift": 0.28,
        "discount_elasticity": 1.45,
        "noise": 0.070,
        "weekday": (0.93, 0.96, 0.99, 1.03, 1.12, 1.16, 0.91),
    },
    "Grocery": {
        "base": 2_900,
        "daily_trend": 0.00060,
        "annual_amplitude": 0.05,
        "campaign_lift": 0.10,
        "discount_elasticity": 0.65,
        "noise": 0.040,
        "weekday": (0.98, 0.99, 1.00, 1.01, 1.05, 1.10, 0.97),
    },
    "Home": {
        "base": 1_500,
        "daily_trend": 0.00035,
        "annual_amplitude": 0.11,
        "campaign_lift": 0.18,
        "discount_elasticity": 0.95,
        "noise": 0.060,
        "weekday": (0.95, 0.97, 1.00, 1.03, 1.09, 1.14, 0.93),
    },
}


def _validate_categories(categories: Iterable[str]) -> tuple[str, ...]:
    chosen = tuple(categories)
    unknown = sorted(set(chosen) - set(CATEGORY_PARAMETERS))
    if unknown:
        raise ValueError(f"Unknown synthetic categories: {unknown}")
    return chosen


def _historical_drivers(dates: pd.DatetimeIndex, category: str) -> pd.DataFrame:
    """Create drivers that would have been known before each forecast date."""

    day_index = (dates - ANCHOR_DATE).days.to_numpy()
    cycle_position = day_index % 91
    category_offset = {"Beauty": 0, "Electronics": 2, "Grocery": 4, "Home": 1}[category]
    campaign_window = (cycle_position >= category_offset) & (
        cycle_position < category_offset + 6
    )
    campaign = campaign_window.astype(int)
    day_of_year = dates.dayofyear.to_numpy()
    special_event = (
        ((day_of_year >= 170) & (day_of_year <= 174))
        | ((day_of_year >= 320) & (day_of_year <= 326))
    ).astype(int)
    discount_rate = 0.02 + (0.10 * campaign) + (0.05 * special_event)

    return pd.DataFrame(
        {
            "date": dates,
            "category": category,
            "campaign": campaign,
            "discount_rate": discount_rate,
            "special_event": special_event,
        }
    )


def generate_synthetic_history(
    start: str,
    end: str,
    categories: Iterable[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate fictional observed orders and availability signals."""

    chosen = _validate_categories(categories)
    dates = pd.date_range(start, end, freq="D")
    if len(dates) < 120:
        raise ValueError("At least 120 daily observations are required.")

    frames: list[pd.DataFrame] = []
    seed_sequence = np.random.SeedSequence(seed).spawn(len(chosen))

    for category, category_seed in zip(chosen, seed_sequence, strict=True):
        params = CATEGORY_PARAMETERS[category]
        rng = np.random.default_rng(category_seed)
        frame = _historical_drivers(dates, category)

        day_index = (dates - ANCHOR_DATE).days.to_numpy()
        weekday_factor = np.take(np.asarray(params["weekday"], dtype=float), dates.dayofweek)
        annual = 1 + float(params["annual_amplitude"]) * np.sin(
            (2 * np.pi * (dates.dayofyear.to_numpy() - 25)) / 365.25
        )
        trend = 1 + float(params["daily_trend"]) * day_index
        promotion = (
            1
            + float(params["campaign_lift"]) * frame["campaign"].to_numpy()
            + float(params["discount_elasticity"]) * frame["discount_rate"].to_numpy()
            + 0.08 * frame["special_event"].to_numpy()
        )
        sigma = float(params["noise"])
        noise = rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma, size=len(dates))
        unconstrained_demand = (
            float(params["base"]) * weekday_factor * annual * trend * promotion * noise
        )

        is_stockout = rng.random(len(dates)) < 0.018
        availability_rate = np.ones(len(dates))
        availability_rate[is_stockout] = rng.uniform(0.55, 0.85, size=is_stockout.sum())
        observed_orders = np.rint(unconstrained_demand * availability_rate).astype(int)

        frame["observed_orders"] = observed_orders
        frame["availability_rate"] = availability_rate.round(4)
        frame["is_stockout"] = is_stockout.astype(int)
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["category", "date"]).reset_index(drop=True)


def generate_future_drivers(
    last_history_date: pd.Timestamp,
    horizon: int,
    categories: Iterable[str],
    scenario: str = "base",
) -> pd.DataFrame:
    """Create known-future drivers for the baseline or proposed campaign plan."""

    if scenario not in {"base", "campaign"}:
        raise ValueError("scenario must be 'base' or 'campaign'")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    chosen = _validate_categories(categories)
    dates = pd.date_range(last_history_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    frames: list[pd.DataFrame] = []

    for category in chosen:
        frame = _historical_drivers(dates, category)
        if scenario == "campaign" and category in {"Beauty", "Home"}:
            proposed_window = np.arange(horizon)
            proposed = ((proposed_window >= 14) & (proposed_window < 28)).astype(int)
            frame["campaign"] = np.maximum(frame["campaign"].to_numpy(), proposed)
            frame["discount_rate"] = np.where(
                proposed == 1,
                np.maximum(frame["discount_rate"].to_numpy(), 0.12),
                frame["discount_rate"].to_numpy(),
            )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).sort_values(["category", "date"]).reset_index(
        drop=True
    )


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    """Estimate unconstrained demand from observed orders and availability."""

    required = {
        "date",
        "category",
        "campaign",
        "discount_rate",
        "special_event",
        "observed_orders",
        "availability_rate",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")
    if history["availability_rate"].le(0).any() or history["availability_rate"].gt(1).any():
        raise ValueError("availability_rate must be in (0, 1]")

    prepared = history.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["demand"] = prepared["observed_orders"] / prepared["availability_rate"]
    prepared["demand"] = prepared["demand"].clip(lower=0)
    return prepared.sort_values(["category", "date"]).reset_index(drop=True)
