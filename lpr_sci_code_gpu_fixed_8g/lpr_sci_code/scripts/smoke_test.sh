#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
rm -rf ./toy_processed ./outputs/toy
python examples/make_toy_data.py --output_dir ./toy_processed
python scripts/train_kt.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --output_dir ./outputs/toy
python scripts/train_lpr.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --kt_ckpt ./outputs/toy/kt_best.pt --output_dir ./outputs/toy
python scripts/run_baselines.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --kt_ckpt ./outputs/toy/kt_best.pt --output_dir ./outputs/toy/baselines --include_gru4rec
python scripts/summarize_results.py --lpr_metrics ./outputs/toy/lpr_metrics.json --baseline_metrics ./outputs/toy/baselines/baseline_metrics.json --output_csv ./outputs/toy/summary.csv
