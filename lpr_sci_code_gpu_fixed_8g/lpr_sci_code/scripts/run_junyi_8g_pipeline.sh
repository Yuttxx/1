#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

CONFIG="configs/junyi_rgcn_kt_hier_8g.yaml"
RAW_DIR="${1:-./data/raw/junyi}"
PROC_DIR="${2:-./data/processed/junyi}"
OUT_DIR="${3:-./outputs/junyi_rgcn_kt_hier_8g}"

python scripts/preprocess_junyi.py \
  --raw_dir "$RAW_DIR" \
  --output_dir "$PROC_DIR" \
  --history_len 20 \
  --path_len 10 \
  --stride 5 \
  --min_user_interactions 25

python scripts/check_dataset.py --dataset_dir "$PROC_DIR"

python scripts/train_kt.py \
  --config "$CONFIG" \
  --dataset_dir "$PROC_DIR" \
  --output_dir "$OUT_DIR"

python scripts/train_lpr.py \
  --config "$CONFIG" \
  --dataset_dir "$PROC_DIR" \
  --kt_ckpt "$OUT_DIR/kt_best.pt" \
  --output_dir "$OUT_DIR"

python scripts/evaluate_lpr.py \
  --config "$CONFIG" \
  --dataset_dir "$PROC_DIR" \
  --kt_ckpt "$OUT_DIR/kt_best.pt" \
  --lpr_ckpt "$OUT_DIR/lpr_final.pt" \
  --output_dir "$OUT_DIR/eval"
