#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from lpr.data import SequenceBuildConfig, build_tasks_from_sequences, normalize_time_deltas, save_standard_dataset, split_by_user
from lpr.graph_utils import build_graph_store, build_similarity_edges_from_annotations, infer_column, parse_list_like, estimate_difficulty_from_logs


def connect_group(items: List[int]) -> List[tuple[int, int, float]]:
    edges = []
    uniq = list(dict.fromkeys(items))
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            edges.append((uniq[i], uniq[j], 1.0))
            edges.append((uniq[j], uniq[i], 1.0))
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess JunYi into the unified LPR format.")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--path_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min_user_interactions", type=int, default=25)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--similarity_threshold", type=float, default=6.0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    log_path = raw_dir / "junyi_ProblemLog_original.csv"
    ex_path = raw_dir / "junyi_Exercise_table.csv"
    rel_train_path = raw_dir / "relationship_annotation_training.csv"
    rel_test_path = raw_dir / "relationship_annotation_testing.csv"
    if not log_path.exists() or not ex_path.exists():
        raise FileNotFoundError("JunYi raw_dir must contain junyi_ProblemLog_original.csv and junyi_Exercise_table.csv")

    ex_df = pd.read_csv(ex_path)
    ex_cols = ex_df.columns.tolist()
    name_col = infer_column(ex_cols, ["name", "exercise", "exercise_name"])
    pre_col = infer_column(ex_cols, ["prerequisites", "prerequisite"])
    topic_col = infer_column(ex_cols, ["topic"])
    area_col = infer_column(ex_cols, ["area"])
    fast_col = infer_column(ex_cols, ["seconds_per_fast_problem", "seconds_fast"])
    pretty_col = infer_column(ex_cols, ["pretty_display_name", "short_display_name", "display_name"])
    if not name_col:
        raise ValueError("Cannot find the exercise name column in junyi_Exercise_table.csv")

    concept_names = list(dict.fromkeys(ex_df[name_col].astype(str).tolist()))
    concept2id = {name: idx + 1 for idx, name in enumerate(concept_names)}
    concept_meta: Dict[str, Dict] = {}
    for _, row in ex_df.iterrows():
        name = str(row[name_col])
        cid = concept2id[name]
        concept_meta[str(cid)] = {
            "raw_name": name,
            "topic": str(row[topic_col]) if topic_col else "unknown",
            "area": str(row[area_col]) if area_col else "unknown",
            "display_name": str(row[pretty_col]) if pretty_col else name,
        }

    preview_cols = pd.read_csv(log_path, nrows=1).columns.tolist()
    user_col = infer_column(preview_cols, ["user_id", "student_id", "uid"])
    item_col = infer_column(preview_cols, ["exercise", "item_id", "problem_id", "question_id"])
    correct_col = infer_column(preview_cols, ["correct", "is_correct", "label"])
    time_col = infer_column(preview_cols, ["time_done", "timestamp", "time"])
    usecols = [c for c in [user_col, item_col, correct_col, time_col] if c]
    log_df = pd.read_csv(log_path, usecols=usecols, nrows=args.max_rows)
    log_cols = log_df.columns.tolist()
    user_col = infer_column(log_cols, ["user_id", "student_id", "uid"])
    item_col = infer_column(log_cols, ["exercise", "item_id", "problem_id", "question_id"])
    correct_col = infer_column(log_cols, ["correct", "is_correct", "label"])
    time_col = infer_column(log_cols, ["time_done", "timestamp", "time"])
    if not all([user_col, item_col, correct_col, time_col]):
        raise ValueError(f"Unexpected JunYi log columns: {log_cols}")

    log_df = log_df.dropna(subset=[user_col, item_col, correct_col, time_col]).copy()
    log_df[item_col] = log_df[item_col].astype(str)
    log_df = log_df[log_df[item_col].isin(concept2id.keys())]
    if args.max_users:
        top_users = log_df[user_col].value_counts().head(args.max_users).index
        log_df = log_df[log_df[user_col].isin(top_users)].copy()
    if log_df[correct_col].dtype == object:
        log_df[correct_col] = log_df[correct_col].astype(str).str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    log_df[correct_col] = log_df[correct_col].astype(float)
    log_df = log_df.sort_values([user_col, time_col])

    user_counts = log_df[user_col].value_counts()
    active_users = set(user_counts[user_counts >= args.min_user_interactions].index.tolist())
    log_df = log_df[log_df[user_col].isin(active_users)].copy()

    relations: Dict[str, List[tuple[int, int, float]]] = defaultdict(list)
    if pre_col:
        for _, row in ex_df[[name_col, pre_col]].fillna("").iterrows():
            dst = concept2id[str(row[name_col])]
            for src_name in parse_list_like(row[pre_col]):
                if src_name in concept2id:
                    relations["prerequisite"].append((concept2id[src_name], dst, 1.0))

    if topic_col:
        for _, group in ex_df.groupby(topic_col):
            ids = [concept2id[str(x)] for x in group[name_col].astype(str).tolist() if str(x) in concept2id]
            relations["same_topic"].extend(connect_group(ids))
    if area_col:
        for _, group in ex_df.groupby(area_col):
            ids = [concept2id[str(x)] for x in group[name_col].astype(str).tolist() if str(x) in concept2id]
            relations["same_area"].extend(connect_group(ids))

    if rel_train_path.exists() or rel_test_path.exists():
        rel_dfs = []
        if rel_train_path.exists():
            rel_dfs.append(pd.read_csv(rel_train_path))
        if rel_test_path.exists():
            rel_dfs.append(pd.read_csv(rel_test_path))
        ann = pd.concat(rel_dfs, ignore_index=True)
        ann_rel = build_similarity_edges_from_annotations(ann, concept2id, threshold=args.similarity_threshold)
        for k, v in ann_rel.items():
            relations[k].extend(v)

    log_df["concept_id"] = log_df[item_col].map(concept2id)
    difficulty = estimate_difficulty_from_logs(log_df["concept_id"].tolist(), log_df[correct_col].tolist())
    if fast_col:
        fast_stats = ex_df[[name_col, fast_col]].dropna()
        vals = fast_stats[fast_col].astype(float)
        lo, hi = float(vals.min()), float(vals.max())
        for _, row in fast_stats.iterrows():
            name = str(row[name_col])
            cid = concept2id[name]
            if cid not in difficulty:
                norm = 0.5 if hi <= lo else (float(row[fast_col]) - lo) / (hi - lo)
                difficulty[cid] = float(norm)
    median_diff = float(np.median(list(difficulty.values()))) if difficulty else 0.5
    for cid in concept2id.values():
        difficulty.setdefault(cid, median_diff)

    sequences = []
    for user_id, group in log_df.groupby(user_col):
        group = group.sort_values(time_col)
        concepts = group["concept_id"].astype(int).tolist()
        correct = group[correct_col].astype(float).tolist()
        timestamps = group[time_col].astype(float).tolist()
        sequences.append(
            {
                "user_id": str(user_id),
                "concepts": concepts,
                "correct": correct,
                "timestamps": timestamps,
                "deltas": normalize_time_deltas(timestamps),
            }
        )

    train_seq, val_seq, test_seq = split_by_user(sequences, seed=args.seed)
    task_cfg = SequenceBuildConfig(
        min_user_interactions=args.min_user_interactions,
        history_len=args.history_len,
        path_len=args.path_len,
        stride=args.stride,
    )
    train_tasks = build_tasks_from_sequences(train_seq, task_cfg)
    val_tasks = build_tasks_from_sequences(val_seq, task_cfg)
    test_tasks = build_tasks_from_sequences(test_seq, task_cfg)
    graph = build_graph_store(max(concept2id.values(), default=0) + 1, relations, difficulty, concept_meta)
    save_standard_dataset(args.output_dir, concept2id, graph, train_seq, val_seq, test_seq, train_tasks, val_tasks, test_tasks)
    print(f"[OK] JunYi processed. users={len(sequences)} train_tasks={len(train_tasks)} val_tasks={len(val_tasks)} test_tasks={len(test_tasks)}")


if __name__ == "__main__":
    main()
