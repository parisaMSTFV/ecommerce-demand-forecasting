# Modeling notes

## Baselines

`seasonal_naive` repeats demand from seven days earlier.

`seasonal_average` averages the same weekday across the previous four weeks.
Both baselines forecast recursively, so they remain valid for all 56 future days.

## Driver-aware model

`driver_ridge` combines:

- target lags at 1, 7, 14, 28 and 56 days;
- rolling demand means over 7 and 28 prior days;
- day-of-week and annual cyclical features;
- linear trend;
- planned campaign, discount and special-event inputs.

Ridge regression is intentionally chosen over a more complex model. The dataset
is synthetic and the portfolio goal is to make leakage control, model comparison
and decision logic easy to inspect.

## Prediction intervals

For the selected model in each category, absolute validation errors are grouped
by forecast week. The empirical 80th percentile becomes the interval half-width
for that category and horizon week.

This is a practical residual-based interval, not a guarantee of conditional
coverage. Category bounds are summed for capacity planning, but the resulting
total is not a statistically exact aggregate interval because cross-category
error dependence is not modelled.
