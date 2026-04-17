#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lpr.common import ensure_dir, get_device, get_project_root, load_config, load_model_checkpoint, resolve_path, set_seed
from lpr.data import load_standard_dataset
from lpr.models import KnowledgeTracer
from lpr.trainers import evaluate_kt, make_kt_loaders, train_knowledge_tracer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the KT simulator used by the learning-path RL environment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config_path = resolve_path(args.config, base_dir=Path.cwd(), must_exist=True)
    cfg = load_config(config_path)
    project_root = get_project_root()

    set_seed(cfg.seed)
    dataset_dir_value = args.dataset_dir or cfg.data.dataset_dir
    dataset_dir = resolve_path(
        dataset_dir_value,
        base_dir=Path.cwd() if args.dataset_dir else project_root,
        must_exist=True,
    )
    output_dir_value = args.output_dir or cfg.output.output_dir
    output_dir = ensure_dir(
        resolve_path(
            output_dir_value,
            base_dir=Path.cwd() if args.output_dir else project_root,
        )
    )
    device = get_device(getattr(cfg.train, "device", None))

    dataset = load_standard_dataset(dataset_dir)
    train_loader, val_loader, test_loader = make_kt_loaders(
        dataset,
        batch_size=cfg.train.kt_batch_size,
        max_seq_len=cfg.data.max_seq_len,
        device=device,
        num_workers=getattr(cfg.train, "num_workers", 0),
        pin_memory=getattr(cfg.train, "pin_memory", None),
        train_window_stride=getattr(cfg.data, "kt_window_stride", 1),
        eval_window_stride=getattr(cfg.data, "kt_eval_window_stride", 1),
        max_train_samples=getattr(cfg.data, "kt_max_train_samples", None),
        max_eval_samples=getattr(cfg.data, "kt_max_eval_samples", None),
    )
    print(f"[INFO] Training KT on device={device}. cuda_available={torch.cuda.is_available()} num_nodes={dataset.num_nodes}")
    model = KnowledgeTracer(num_nodes=dataset.num_nodes, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
    ckpt_path = str(output_dir / "kt_best.pt")
    report = train_knowledge_tracer(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=cfg.train.kt_epochs,
        lr=cfg.train.kt_lr,
        ckpt_path=ckpt_path,
    )
    load_model_checkpoint(model, ckpt_path, map_location=device, strict=False, resize_mismatched=True)
    test_metrics = evaluate_kt(model.to(device), test_loader, device)
    metrics = {"best_val_auc": report.best_metric, "test": test_metrics, "history": report.history}
    with open(output_dir / "kt_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
