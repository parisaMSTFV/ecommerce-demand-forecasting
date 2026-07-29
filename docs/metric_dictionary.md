# Metric dictionary

| Metric | Definition | Interpretation |
|---|---|---|
| Adjusted demand | Observed orders divided by availability rate | Approximation of unconstrained demand |
| WAPE | Sum of absolute errors divided by sum of actual demand | Portfolio-level forecast error; lower is better |
| MAE | Mean absolute error | Typical category-day error in orders |
| Bias | Sum of forecast minus actual, divided by sum of actual | Positive means systematic over-forecast |
| Interval coverage | Share of actual values between lower and upper bounds | Calibration check against the 80% target |
| Risk buffer | Upper interval minus point forecast | Extra capacity implied by uncertainty |
| Scenario increment | Campaign-scenario forecast minus base forecast | Predictive planning difference, not causal lift |

WAPE is the primary model-selection metric because categories have different
volumes. Bias is reported separately because equal absolute error can imply very
different operational risk.
