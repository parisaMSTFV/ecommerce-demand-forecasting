"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecommerce_forecasting.config import ForecastConfig
from ecommerce_forecasting.pipeline import (
    run_pipeline,
    run_smoke_pipeline,
    run_supplied_pipeline,
    run_supplied_smoke_pipeline,
    with_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ecommerce demand forecast case study.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the complete reproducible analysis.")
    run_parser.add_argument("--output-dir", type=Path, default=Path("local-runs/latest"))
    supplied_parser = subparsers.add_parser(
        "run-supplied",
        help="Run on validated aggregate history and known-future drivers.",
    )
    supplied_parser.add_argument("--history-csv", type=Path, required=True)
    supplied_parser.add_argument("--future-drivers-csv", type=Path, required=True)
    supplied_parser.add_argument("--provenance-json", type=Path, required=True)
    supplied_parser.add_argument("--output-dir", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke", help="Run a compact CI pipeline.")
    smoke_parser.add_argument("--output-dir", type=Path, default=Path("/tmp/forecast-smoke"))
    supplied_smoke_parser = subparsers.add_parser(
        "smoke-supplied",
        help="Exercise the supplied-input contract with a synthetic fixture.",
    )
    supplied_smoke_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/forecast-supplied-smoke"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = with_output(ForecastConfig(), args.output_dir)
        summary = run_pipeline(config)
    elif args.command == "run-supplied":
        config = with_output(ForecastConfig(), args.output_dir)
        summary = run_supplied_pipeline(
            config,
            args.history_csv,
            args.future_drivers_csv,
            args.provenance_json,
        )
    elif args.command == "smoke":
        summary = run_smoke_pipeline(args.output_dir)
    else:
        summary = run_supplied_smoke_pipeline(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
