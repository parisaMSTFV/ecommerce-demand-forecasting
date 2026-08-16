# Forecast decision note

## Evidence boundary

Results use generated synthetic data and are a demonstration, not evidence of performance on an external business dataset.

## Recommendation

Use the category-specific model selection for the next 56-day capacity plan. The
selected system achieved **4.9% WAPE** on the untouched holdout,
with **+0.5% bias**. The empirical 80% intervals covered
**80.8%** of category-day outcomes.

The upper planning bound peaks at **14,942 orders** on
**2026-01-02**. That upper bound is the safer capacity input when the
cost of under-capacity is higher than the cost of a short-lived buffer.

## Selected model by category

- Beauty: `driver_ridge`
- Electronics: `driver_ridge`
- Grocery: `driver_ridge`
- Home: `driver_ridge`

## Proposed campaign scenario

The proposed two-week campaign for Beauty and Home changes the modelled
56-day demand by **17,879 orders** versus the base plan. This is
a demand scenario, not a causal lift estimate. Finance and operations should
apply their own margin and fulfilment constraints before approval.


## Guardrails

- Re-run backtesting when demand regime, assortment or campaign mechanics change.
- Monitor WAPE and signed bias by category and horizon every forecast cycle.
- Do not interpret the summed category intervals as a statistically exact total interval.
- Treat stockout adjustment as an approximation that depends on availability quality.
