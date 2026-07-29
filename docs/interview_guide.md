# Interview guide

## One-minute explanation

This project forecasts daily ecommerce orders by product category for an
eight-week capacity plan. I generated a fully synthetic dataset with seasonality,
trend, campaigns, discounts and availability constraints. I compared two
seasonal baselines with a driver-aware model using rolling-origin validation,
selected the model per category, then evaluated the whole system once on an
untouched 56-day holdout. I also calibrated empirical prediction intervals and
translated the forecast into a risk-buffered capacity plan.

## Decisions to defend

**Why not use a random train-test split?**
Random splitting would let future regimes influence past predictions. Each fold
trains only on dates before its forecast window.

**Why select a model per category?**
Demand patterns and response to commercial drivers differ. A single global
winner can hide category-level failure.

**Why keep simple baselines?**
A complex model has no business value unless it improves on a credible low-cost
forecast.

**Why report bias separately from WAPE?**
Under-forecasting and over-forecasting can have different capacity costs even
when their absolute errors are equal.

**Is the campaign difference incremental lift?**
No. It is a conditional demand scenario from a predictive model. Causal lift
requires an experiment or another valid causal design.

## Improvements for production

- Replace the availability correction with a censored-demand model.
- Use richer event, price, assortment and inventory signals.
- Calibrate intervals with more folds and model cross-category dependence.
- Add drift monitoring and automated retraining gates.
- Optimise capacity against explicit under- and over-planning costs.
