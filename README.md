# Ecommerce Demand Forecasting

[![CI](https://github.com/parisaMSTFV/ecommerce-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/parisaMSTFV/ecommerce-demand-forecasting/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB)](https://www.python.org/)
[![Data](https://img.shields.io/badge/data-100%25%20synthetic-0F766E)](DATA_PROVENANCE.md)

A reproducible forecasting case study that converts category-level ecommerce
demand into an eight-week capacity plan. The project emphasises leakage-safe
evaluation, honest baseline comparison, forecast uncertainty and business
decision support.

> All records, volumes, categories and commercial drivers are synthetic. No
> employer data, code, schema or business rule is used.

## Business question

How many daily orders should an ecommerce operation prepare for during the next
56 days, and how should the plan change under a proposed campaign?

The pipeline:

- estimates unconstrained demand from orders and availability;
- compares three models through rolling-origin validation;
- selects a model independently for each category;
- evaluates the selected system on an untouched holdout;
- calibrates 80% empirical prediction intervals;
- reconciles category forecasts exactly to a daily total;
- translates uncertainty into a capacity risk buffer;
- separates a predictive campaign scenario from causal lift.

## Validated results

The complete pipeline was run on 4,384 synthetic category-day records from
2023-01-01 through 2025-12-31.

| Result | Value |
|---|---:|
| Rolling-validation WAPE — driver-aware model | **5.0%** |
| Rolling-validation WAPE — seasonal naive | 7.6% |
| Relative WAPE improvement vs seasonal naive | **33.6%** |
| Untouched 56-day holdout WAPE | **4.9%** |
| Holdout signed bias | **+0.5%** |
| 80% interval coverage on holdout | **80.8%** |
| Proposed campaign scenario vs base | **+17,879 orders** |

The driver-aware ridge model won for all four categories in validation. That
outcome was produced by the run rather than imposed by the pipeline.

| Category | Holdout WAPE | Bias | Interval coverage |
|---|---:|---:|---:|
| Beauty | 4.6% | +0.1% | 82.1% |
| Electronics | 6.7% | +1.7% | 75.0% |
| Grocery | 3.9% | +0.1% | 83.9% |
| Home | 5.6% | +0.7% | 82.1% |

![Model comparison](reports/figures/model_comparison.png)

![Holdout actual versus forecast](reports/figures/holdout_forecast.png)

The proposed two-week campaign is applied only to Beauty and Home. The model
predicts approximately 9,029 additional Beauty orders and 8,850 additional Home
orders during the 56-day plan. This is a conditional forecast scenario, not a
causal incrementality estimate.

![Campaign scenario](reports/figures/campaign_scenario.png)

The daily upper planning bound peaks at 14,942 orders on 2026-01-02. The
decision note explains when the point forecast or upper bound is the more
appropriate capacity input.

## Evaluation design

```mermaid
flowchart TD
    A["Synthetic history"] --> B["3 rolling validation folds"]
    B --> C["Model selection by category"]
    B --> D["Interval calibration"]
    C --> E["Untouched 56-day holdout"]
    D --> E
    E --> F["Future capacity plan"]
    F --> G["Base vs campaign scenario"]
```

The holdout is never used for model selection or interval calibration. Every lag
and rolling feature uses target values strictly before the prediction date.

## Models

| Model | Purpose |
|---|---|
| `seasonal_naive` | Repeat demand from seven days earlier |
| `seasonal_average` | Average the same weekday over four prior weeks |
| `driver_ridge` | Combine lagged demand, calendar, trend and known commercial drivers |

WAPE is the selection metric. MAE, signed bias and interval coverage are reported
as separate controls.

## Repository structure

```text
src/ecommerce_forecasting/   data, features, models, backtesting and reporting
tests/                       leakage, metrics, reconciliation and pipeline tests
docs/                        analysis plan, metric definitions and interview guide
reports/                     validated tables, charts and decision note
scripts/                     sensitive-content check
.github/workflows/           CI for Python 3.11 and 3.12
```

Generated row-level data is stored under `data/generated/` and excluded from Git.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make run
make check
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The full run writes reproducible outputs to `reports/`.

## Documentation

- [Analysis plan](docs/analysis_plan.md)
- [Metric dictionary](docs/metric_dictionary.md)
- [Modeling notes](docs/modeling_notes.md)
- [Interview guide](docs/interview_guide.md)
- [Data provenance](DATA_PROVENANCE.md)
- [Decision note](reports/decision_note.md)

## Limitations

- Availability adjustment is a simplified approximation of censored demand.
- Planned campaign and discount inputs must be known before the forecast is made.
- Residual intervals are empirical and do not guarantee conditional coverage.
- Summed category bounds are a planning range, not an exact total-demand interval.
- The campaign scenario is predictive; it does not estimate causal incrementality.

## License

MIT
