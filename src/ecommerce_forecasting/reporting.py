"""Create portfolio-ready charts and the decision memo."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACCENT = "#2563EB"
SECONDARY = "#0F766E"
WARNING = "#D97706"
INK = "#172033"
MUTED = "#667085"
GRID = "#E4E7EC"


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED)
    axis.title.set_color(INK)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)


def create_charts(
    history: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    future_predictions: pd.DataFrame,
    scenario_impact: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write all charts used by the README and reports."""

    chart_dir = output_dir / "figures"
    chart_dir.mkdir(parents=True, exist_ok=True)

    history_total = history.groupby("date", as_index=False)["demand"].sum().tail(240)
    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(history_total["date"], history_total["demand"], color=ACCENT, linewidth=1.8)
    axis.set(title="Synthetic daily demand — last 240 days", ylabel="Adjusted orders")
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(chart_dir / "demand_history.png", dpi=160)
    plt.close(fig)

    overall = validation_metrics.loc[
        validation_metrics["category"] == "Overall"
    ].sort_values("wape")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(
        overall["model"],
        overall["wape"] * 100,
        color=[ACCENT, SECONDARY, WARNING][: len(overall)],
    )
    axis.bar_label(bars, fmt="%.1f%%", padding=4, color=INK)
    axis.set(title="Rolling-validation model comparison", ylabel="WAPE")
    axis.set_ylim(0, max(overall["wape"] * 100) * 1.25)
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(chart_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    holdout_columns = ["actual", "forecast", "lower", "upper"]
    holdout_total = (
        holdout_predictions.groupby("date", as_index=False)[holdout_columns]
        .sum()
        .sort_values("date")
    )
    x_values = np.arange(len(holdout_total))
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.fill_between(
        x_values,
        holdout_total["lower"].to_numpy(),
        holdout_total["upper"].to_numpy(),
        color=ACCENT,
        alpha=0.14,
        label="80% interval",
    )
    axis.plot(x_values, holdout_total["actual"], color=INK, linewidth=1.8, label="Actual")
    axis.plot(x_values, holdout_total["forecast"], color=ACCENT, linewidth=1.8, label="Forecast")
    tick_positions = np.arange(0, len(holdout_total), 7)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        holdout_total.iloc[tick_positions]["date"].dt.strftime("%b %d"),
        rotation=35,
        ha="right",
    )
    axis.set(title="Untouched 56-day holdout", ylabel="Orders")
    axis.legend(frameon=False, ncol=3)
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(chart_dir / "holdout_forecast.png", dpi=160)
    plt.close(fig)

    horizon_rows: list[dict[str, float]] = []
    for week, group in holdout_predictions.groupby("horizon_week"):
        denominator = group["actual"].abs().sum()
        horizon_rows.append(
            {
                "week": float(week),
                "wape": float((group["actual"] - group["forecast"]).abs().sum() / denominator),
            }
        )
    horizon = pd.DataFrame(horizon_rows)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(horizon["week"], horizon["wape"] * 100, marker="o", color=SECONDARY, linewidth=2)
    axis.set(
        title="Holdout error by forecast week",
        xlabel="Forecast horizon (week)",
        ylabel="WAPE",
        xticks=horizon["week"],
    )
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(chart_dir / "error_by_horizon.png", dpi=160)
    plt.close(fig)

    future_total = (
        future_predictions.groupby("date", as_index=False)[["forecast", "lower", "upper"]]
        .sum()
        .sort_values("date")
    )
    x_values = np.arange(len(future_total))
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.fill_between(
        x_values,
        future_total["lower"].to_numpy(),
        future_total["upper"].to_numpy(),
        color=SECONDARY,
        alpha=0.15,
        label="80% interval",
    )
    axis.plot(x_values, future_total["forecast"], color=SECONDARY, linewidth=2, label="Base plan")
    tick_positions = np.arange(0, len(future_total), 7)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        future_total.iloc[tick_positions]["date"].dt.strftime("%b %d"),
        rotation=35,
        ha="right",
    )
    axis.set(title="Next 56 days: reconciled capacity signal", ylabel="Forecast orders")
    axis.legend(frameon=False)
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(chart_dir / "future_capacity_forecast.png", dpi=160)
    plt.close(fig)

    ordered_impact = scenario_impact.sort_values("incremental_orders")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.barh(
        ordered_impact["category"],
        ordered_impact["incremental_orders"],
        color=SECONDARY,
    )
    axis.bar_label(bars, fmt="%+.0f", padding=4, color=INK)
    axis.set(title="Proposed campaign scenario", xlabel="Incremental forecast orders")
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(chart_dir / "campaign_scenario.png", dpi=160)
    plt.close(fig)


def write_decision_memo(
    holdout_metrics: pd.DataFrame,
    selections: pd.DataFrame,
    scenario_impact: pd.DataFrame,
    capacity_plan: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create a concise business-facing interpretation from validated outputs."""

    overall = holdout_metrics.loc[holdout_metrics["category"] == "Overall"].iloc[0]
    total_increment = scenario_impact["incremental_orders"].sum()
    peak = capacity_plan.loc[capacity_plan["upper"].idxmax()]
    selected_lines = "\n".join(
        f"- {row.category}: `{row.selected_model}`"
        for row in selections.itertuples(index=False)
    )
    content = f"""# Forecast decision note

## Recommendation

Use the category-specific model selection for the next 56-day capacity plan. The
selected system achieved **{overall.wape:.1%} WAPE** on the untouched holdout,
with **{overall.bias:+.1%} bias**. The empirical 80% intervals covered
**{overall.interval_coverage:.1%}** of category-day outcomes.

The upper planning bound peaks at **{peak.upper:,.0f} orders** on
**{peak.date:%Y-%m-%d}**. That upper bound is the safer capacity input when the
cost of under-capacity is higher than the cost of a short-lived buffer.

## Selected model by category

{selected_lines}

## Proposed campaign scenario

The proposed two-week campaign for Beauty and Home changes the modelled
56-day demand by **{total_increment:,.0f} orders** versus the base plan. This is
a demand scenario, not a causal lift estimate. Finance and operations should
apply their own margin and fulfilment constraints before approval.

## Guardrails

- Re-run backtesting when demand regime, assortment or campaign mechanics change.
- Monitor WAPE and signed bias by category and horizon every forecast cycle.
- Do not interpret the summed category intervals as a statistically exact total interval.
- Treat stockout adjustment as an approximation that depends on availability quality.
"""
    (output_dir / "decision_note.md").write_text(content, encoding="utf-8")


def write_summary_json(summary: dict[str, object], output_dir: Path) -> None:
    """Write machine-readable headline results."""

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
