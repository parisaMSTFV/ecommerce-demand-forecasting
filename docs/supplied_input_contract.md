# Supplied-input contract

## Purpose and evidence boundary

The `run-supplied` path applies the repository's existing rolling-origin model
comparison, untouched holdout and interval calibration to a user's aggregate
category-day data. It does not claim the synthetic demonstration results will
transfer to another dataset. Each supplied run produces its own metrics.

The command forecasts one provided future driver plan. It intentionally does
not create the synthetic Beauty/Home campaign scenario or report causal lift.

## Command

```bash
demand-forecast run-supplied \
  --history-csv /secure/history.csv \
  --future-drivers-csv /secure/future_drivers.csv \
  --provenance-json /secure/provenance.json \
  --output-dir /secure/new-forecast-output
```

The output directory must be new or empty. Raw input files are never copied.
Derived predictions and metrics are written to the requested output, so that
location must follow the source data's access policy.

## History CSV

One row per category and calendar day is required.

| Column | Type and rule | Forecast-time meaning |
|---|---|---|
| `date` | `YYYY-MM-DD`; consecutive and unique within category | Observation date |
| `category` | Non-empty string | Aggregate forecast series |
| `observed_orders` | Numeric, at least zero | Fulfilled/observed orders |
| `availability_rate` | Numeric in `(0, 1]` | Fraction of demand opportunity available |
| `campaign` | `0` or `1` | Campaign state known on that date |
| `discount_rate` | Numeric in `[0, 1]` | Discount fraction known on that date |
| `special_event` | `0` or `1` | Pre-declared event state |

All categories must have the same uninterrupted daily coverage. With the
default 56-day horizon and three validation folds, at least 308 days are
required: 84 training days plus three validation windows and one untouched
holdout. The availability adjustment remains a simplifying assumption, not a
recovery of unobserved causal demand.

## Future-driver CSV

The file requires `date`, `category`, `campaign`, `discount_rate`, and
`special_event` under the same type and range rules. Every historical category
must have exactly 56 consecutive rows, beginning one day after history ends.
These values must be genuinely known or fixed at forecast time; otherwise the
backtest-to-production comparison is not decision-valid.

## Provenance JSON

```json
{
  "schema_version": "1.0",
  "dataset_name": "Approved aggregate demand export",
  "source_description": "Category-day export from the planning warehouse",
  "extraction_date": "2026-08-14",
  "usage_permission": "Approved for local forecasting analysis",
  "data_is_synthetic": false,
  "contains_personal_data": false
}
```

All fields are required. `contains_personal_data` must be `false`; this project
does not accept customer-level inputs. Provenance values are user declarations,
not independently verified facts.

## Validation and audit output

The run fails before modelling on missing/null columns, non-finite numbers,
invalid dates or ranges, spreadsheet-formula category prefixes, duplicate
category-days, gaps, unbalanced categories, insufficient history, category
mismatch, incorrect future horizon, personal-data declaration, or invalid
provenance.

`input_provenance.json` records the contract version, declared metadata,
coverage, row counts and SHA-256 hashes of both CSVs. It explicitly records that
the raw inputs were not copied and that provenance was not independently
verified.
