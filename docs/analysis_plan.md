# Analysis plan

## Decision

Forecast daily unconstrained orders for four fictional product categories over
the next 56 days. The result supports capacity planning and a proposed campaign
scenario.

## Unit of analysis

One row represents one category-day. The model target is adjusted demand:

`observed orders / availability rate`

This correction is deliberately simple and is treated as a limitation, not as
ground truth.

## Evaluation design

1. Reserve the final 56 days as an untouched holdout.
2. Before the holdout, run three non-overlapping 56-day rolling-origin folds.
3. Compare seasonal naive, four-week seasonal average and a driver-aware ridge
   model.
4. Select the lowest-WAPE model separately for each category.
5. Calibrate 80% interval widths from validation residuals by category and
   forecast week.
6. Evaluate the selected system once on the holdout.
7. Refit on all history and forecast the next 56 days.

No future target enters feature construction. Calendar, discount, campaign and
event variables are treated as known planning inputs.

## Success criteria

The project is complete when it:

- runs from data generation to reports with one command;
- beats or honestly reports failure against simple baselines;
- publishes holdout WAPE, MAE, bias and interval coverage;
- reconciles category forecasts exactly to the daily total;
- turns uncertainty into a capacity signal;
- states that scenario output is predictive, not causal.
