import pandas as pd
from pandas.testing import assert_frame_equal

from ecommerce_forecasting.data import (
    generate_future_drivers,
    generate_synthetic_history,
    prepare_history,
)


def test_synthetic_history_is_reproducible() -> None:
    first = generate_synthetic_history("2024-01-01", "2024-05-31", ("Beauty",), seed=7)
    second = generate_synthetic_history("2024-01-01", "2024-05-31", ("Beauty",), seed=7)
    assert_frame_equal(first, second)


def test_availability_adjustment_recovers_demand_signal() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "category": ["Beauty"],
            "campaign": [0],
            "discount_rate": [0.02],
            "special_event": [0],
            "observed_orders": [600],
            "availability_rate": [0.75],
        }
    )
    prepared = prepare_history(frame)
    assert prepared.loc[0, "demand"] == 800


def test_campaign_scenario_changes_only_target_categories() -> None:
    categories = ("Beauty", "Electronics", "Grocery", "Home")
    base = generate_future_drivers(pd.Timestamp("2025-12-31"), 56, categories, "base")
    campaign = generate_future_drivers(pd.Timestamp("2025-12-31"), 56, categories, "campaign")
    merged = base.merge(
        campaign,
        on=["date", "category"],
        suffixes=("_base", "_campaign"),
        validate="one_to_one",
    )
    changed = merged.loc[merged["discount_rate_campaign"] != merged["discount_rate_base"]]
    assert set(changed["category"]) == {"Beauty", "Home"}
    unchanged = merged.loc[merged["category"].isin(["Electronics", "Grocery"])]
    assert (unchanged["discount_rate_campaign"] == unchanged["discount_rate_base"]).all()
