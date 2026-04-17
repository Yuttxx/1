#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lpr.common import get_device, load_config, load_model_checkpoint, write_jsonl
from lpr.data import load_standard_dataset
from lpr.models import KnowledgeBackgroundEncoder, KnowledgeTracer, LPRModel, PolicyValueNet, RGCNEncoder, TimeAwarePreferenceEncoder, TransEEncoder
from lpr.rl import RewardWeights
from lpr.trainers import LPRTrainer, make_task_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved LPR checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--kt_ckpt", required=True)
    parser.add_argument("--lpr_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(getattr(cfg.train, "device", None))
    dataset = load_standard_dataset(args.dataset_dir or cfg.data.dataset_dir)
    _, _, test_loader = make_task_loaders(
        dataset,
        batch_size=cfg.train.lpr_batch_size,
        device=device,
        num_workers=getattr(cfg.train, "num_workers", 0),
        pin_memory=getattr(cfg.train, "pin_memory", None),
        eval_task_stride=getattr(cfg.data, "lpr_eval_task_stride", 1),
        max_eval_tasks=getattr(cfg.data, "lpr_max_eval_tasks", None),
    )

    kt_model = KnowledgeTracer(num_nodes=dataset.num_nodes, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
    load_model_checkpoint(kt_model, args.kt_ckpt, map_location=device, strict=False, resize_mismatched=True)

    if cfg.variant.graph_type.lower() == "rgcn":
        graph_encoder = RGCNEncoder(dataset.num_nodes, dataset.graph.num_relations, cfg.model.hidden_dim, cfg.model.num_gnn_layers, cfg.model.dropout)
    else:
        graph_encoder = TransEEncoder(dataset.num_nodes, dataset.graph.num_relations, cfg.model.hidden_dim)
    preference_encoder = TimeAwarePreferenceEncoder(hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
    kb_encoder = KnowledgeBackgroundEncoder(hidden_dim=cfg.model.hidden_dim, mode=cfg.variant.kb_mode, dropout=cfg.model.dropout)
    policy = PolicyValueNet(hidden_dim=cfg.model.hidden_dim, num_nodes=dataset.num_nodes, dropout=cfg.model.dropout)
    model = LPRModel(graph_encoder, preference_encoder, kb_encoder, policy, hidden_dim=cfg.model.hidden_dim)
    load_model_checkpoint(model, args.lpr_ckpt, map_location=device, strict=False, resize_mismatched=True)

    trainer = LPRTrainer(
        model=model,
        graph=dataset.graph,
        kt_model=kt_model,
        device=device,
        reward_mode=cfg.variant.reward_mode,
        reward_weights=RewardWeights(**cfg.reward.to_dict()),
        candidate_mode=cfg.variant.candidate_mode,
        path_len=cfg.data.path_len,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
    )
    metrics = trainer.evaluate(test_loader, greedy=True)
    preds = trainer.generate_paths(test_loader, greedy=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_jsonl(preds[: min(len(preds), 200)], out_dir / "predictions_sample.jsonl")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
