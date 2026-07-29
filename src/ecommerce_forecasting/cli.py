"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecommerce_forecasting.config import ForecastConfig
from ecommerce_forecasting.pipeline import run_pipeline, run_smoke_pipeline, with_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ecommerce demand forecast case study.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the complete reproducible analysis.")
    run_parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    smoke_parser = subparsers.add_parser("smoke", help="Run a compact CI pipeline.")
    smoke_parser.add_argument("--output-dir", type=Path, default=Path("/tmp/forecast-smoke"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = with_output(ForecastConfig(), args.output_dir)
        summary = run_pipeline(config)
    else:
        summary = run_smoke_pipeline(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
