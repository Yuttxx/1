#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_metrics(name: str, metrics: dict, section: str | None = None) -> dict:
    row = {"model": name}
    if section is not None:
        row["section"] = section
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue
        row[k] = v
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge baseline and model metrics into a CSV table.")
    parser.add_argument("--lpr_metrics", required=True)
    parser.add_argument("--baseline_metrics", default=None)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.lpr_metrics, "r", encoding="utf-8") as f:
        lpr = json.load(f)
    rows.append(flatten_metrics("proposed_or_variant", lpr.get("test", {}), section="test"))
    rows.append(flatten_metrics("proposed_or_variant", lpr.get("validation", {}), section="validation"))

    if args.baseline_metrics:
        with open(args.baseline_metrics, "r", encoding="utf-8") as f:
            baselines = json.load(f)
        for name, metrics in baselines.items():
            if isinstance(metrics, dict) and "history" in metrics:
                continue
            rows.append(flatten_metrics(name, metrics, section="test"))

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(df)


if __name__ == "__main__":
    main()
