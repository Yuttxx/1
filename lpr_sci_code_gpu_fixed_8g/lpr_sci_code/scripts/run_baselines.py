#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from lpr.baselines import GRU4RecBaseline, PopularityBaseline, RandomBaseline, SeqKNNBaseline
from lpr.common import ensure_dir, get_device, get_project_root, load_config, load_model_checkpoint, resolve_path, set_seed
from lpr.data import GRU4RecDataset, collate_gru4rec, load_standard_dataset
from lpr.metrics import (
    aggregate,
    difficulty_smoothness,
    hit_rate,
    mastery_gain_metric,
    mrr,
    ndcg_at_k,
    prerequisite_violation_rate,
    review_coverage,
)
from lpr.rl import CandidateGenerator
from lpr.trainers import history_mastery_from_feedback, train_gru4rec


def evaluate_paths(dataset, paths, kt_model, device):
    rows = []
    for row in paths:
        rows.append(
            {
                "mastery_gain": mastery_gain_metric(kt_model, row["history"], row["history_correct"], row["history_deltas"], row["path"], row["target"], device),
                "hit_rate": hit_rate(row["path"], row["target"]),
                "ndcg@10": ndcg_at_k(row["path"], row["future"], 10),
                "mrr": mrr(row["path"], row["future"]),
                "prereq_violation": prerequisite_violation_rate(dataset.graph, row["history"], row["path"]),
                "difficulty_smoothness": difficulty_smoothness(dataset.graph, row["history"], row["path"]),
                "review_coverage": review_coverage(dataset.graph, row["target"], row["path"]),
            }
        )
    return aggregate(rows)


def rollout_nonparametric(name, recommender, dataset, path_len):
    generator = CandidateGenerator(dataset.graph, mode="review_augmented")
    results = []
    for task in dataset.test_tasks:
        history = list(task["history"])
        correct = list(task.get("history_correct", [1.0] * len(history)))
        deltas = list(task.get("history_deltas", [0.0] * len(history)))
        path = []
        for _ in range(path_len):
            mastery = history_mastery_from_feedback(history, correct)
            candidates = generator.candidates(history[-1], task["target"], history, mastery)
            if name == "random":
                rec = recommender.recommend(candidates, k=1)
            elif name == "popularity":
                rec = recommender.recommend(candidates, k=1)
            else:
                rec = recommender.recommend(history, candidates, k=1)
            action = int(rec[0]) if rec else int(candidates[0])
            path.append(action)
            history.append(action)
            correct.append(1.0)
            deltas.append(deltas[-1] if deltas else 0.0)
        results.append(
            {
                "user_id": task["user_id"],
                "history": task["history"],
                "history_correct": task.get("history_correct", [1.0] * len(task["history"])),
                "history_deltas": task.get("history_deltas", [0.0] * len(task["history"])),
                "target": task["target"],
                "future": task.get("future", []),
                "path": path,
            }
        )
    return results


def rollout_gru4rec(model, dataset, device, path_len):
    generator = CandidateGenerator(dataset.graph, mode="review_augmented")
    results = []
    model.eval()
    for task in dataset.test_tasks:
        history = list(task["history"])
        correct = list(task.get("history_correct", [1.0] * len(history)))
        deltas = list(task.get("history_deltas", [0.0] * len(history)))
        path = []
        for _ in range(path_len):
            mastery = history_mastery_from_feedback(history, correct)
            candidates = generator.candidates(history[-1], task["target"], history, mastery)
            hist = torch.tensor([history], dtype=torch.long, device=device)
            corr = torch.tensor([correct], dtype=torch.float32, device=device)
            delt = torch.tensor([deltas], dtype=torch.float32, device=device)
            lens = torch.tensor([len(history)], dtype=torch.long, device=device)
            mask = torch.zeros(1, dataset.graph.num_nodes, dtype=torch.bool, device=device)
            mask[0, candidates] = True
            action = int(model.recommend(hist, corr, delt, lens, mask, k=1)[0, 0].item())
            path.append(action)
            history.append(action)
            correct.append(1.0)
            deltas.append(deltas[-1] if deltas else 0.0)
        results.append(
            {
                "user_id": task["user_id"],
                "history": task["history"],
                "history_correct": task.get("history_correct", [1.0] * len(task["history"])),
                "history_deltas": task.get("history_deltas", [0.0] * len(task["history"])),
                "target": task["target"],
                "future": task.get("future", []),
                "path": path,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple baselines on the unified LPR data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--kt_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_gru4rec", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config, base_dir=Path.cwd(), must_exist=True)
    cfg = load_config(config_path)
    project_root = get_project_root()

    set_seed(cfg.seed)
    device = get_device(getattr(cfg.train, "device", None))
    dataset_dir_value = args.dataset_dir or cfg.data.dataset_dir
    dataset = load_standard_dataset(
        resolve_path(
            dataset_dir_value,
            base_dir=Path.cwd() if args.dataset_dir else project_root,
            must_exist=True,
        )
    )
    out_dir = ensure_dir(resolve_path(args.output_dir, base_dir=Path.cwd()))

    from lpr.models import KnowledgeTracer

    kt_ckpt = resolve_path(args.kt_ckpt, base_dir=Path.cwd(), must_exist=True)
    kt_model = KnowledgeTracer(num_nodes=dataset.num_nodes, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
    load_model_checkpoint(kt_model, kt_ckpt, map_location=device, strict=False, resize_mismatched=True)
    kt_model.to(device).eval()

    results: Dict[str, Dict] = {}
    pop = PopularityBaseline().fit(dataset.train_sequences)
    paths = rollout_nonparametric("popularity", pop, dataset, cfg.data.path_len)
    results["popularity"] = evaluate_paths(dataset, paths, kt_model, device)

    rnd = RandomBaseline()
    paths = rollout_nonparametric("random", rnd, dataset, cfg.data.path_len)
    results["random"] = evaluate_paths(dataset, paths, kt_model, device)

    knn = SeqKNNBaseline().fit(dataset.train_tasks)
    paths = rollout_nonparametric("seqknn", knn, dataset, cfg.data.path_len)
    results["seqknn"] = evaluate_paths(dataset, paths, kt_model, device)

    if args.include_gru4rec:
        train_ds = GRU4RecDataset(dataset.train_tasks)
        val_ds = GRU4RecDataset(dataset.val_tasks)
        train_loader = DataLoader(train_ds, batch_size=cfg.train.gru4rec_batch_size, shuffle=True, collate_fn=collate_gru4rec)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.gru4rec_batch_size, shuffle=False, collate_fn=collate_gru4rec)
        model = GRU4RecBaseline(num_nodes=dataset.num_nodes, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
        ckpt = out_dir / "gru4rec_best.pt"
        report = train_gru4rec(model, train_loader, val_loader, device, epochs=cfg.train.gru4rec_epochs, lr=cfg.train.gru4rec_lr, ckpt_path=str(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=device))
        paths = rollout_gru4rec(model.to(device), dataset, device, cfg.data.path_len)
        results["gru4rec"] = evaluate_paths(dataset, paths, kt_model, device)
        results["gru4rec_train"] = {"best": report.best_metric, "history": report.history}

    with open(out_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
