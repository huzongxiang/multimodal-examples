#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tld.reporting import plot_training_metrics, write_training_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate polished training plots from metrics.jsonl.")
    parser.add_argument("--metrics-path", type=Path, required=True, help="Path to metrics.jsonl.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots into. Defaults to the metrics file's parent directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.metrics_path.parent
    summary = write_training_summary(args.metrics_path, output_dir)
    plot_training_metrics(args.metrics_path, output_dir)
    print(f"Saved plots to: {output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
