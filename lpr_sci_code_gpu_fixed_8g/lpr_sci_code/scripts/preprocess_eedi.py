#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from lpr.data import SequenceBuildConfig, build_tasks_from_sequences, normalize_time_deltas, save_standard_dataset, split_by_user
from lpr.graph_utils import build_graph_store, infer_column, parse_list_like


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Eedi Task 3/4 into the unified LPR format.")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--path_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min_user_interactions", type=int, default=20)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    train_path = raw_dir / "train_task_3_4.csv"
    qmeta_path = raw_dir / "question_metadata_task_3_4.csv"
    subject_path = raw_dir / "subject_metadata.csv"
    if not train_path.exists() or not qmeta_path.exists() or not subject_path.exists():
        raise FileNotFoundError("Eedi raw_dir must contain train_task_3_4.csv, question_metadata_task_3_4.csv, subject_metadata.csv")

    qmeta = pd.read_csv(qmeta_path)
    q_cols = qmeta.columns.tolist()
    qid_col = infer_column(q_cols, ["QuestionId", "question_id", "item_id"])
    if not qid_col:
        raise ValueError(f"Cannot find question id column in {qmeta_path}")
    subject_cols = [c for c in q_cols if "subject" in c.lower() and c != qid_col]
    if not subject_cols:
        subject_cols = [c for c in q_cols if c != qid_col]

    smeta = pd.read_csv(subject_path)
    s_cols = smeta.columns.tolist()
    sid_col = infer_column(s_cols, ["SubjectId", "subject_id", "skill_id", "concept_id"])
    parent_col = infer_column(s_cols, ["ParentId", "parent_id", "parent"])
    name_col = infer_column(s_cols, ["Name", "name", "subject_name", "concept_name"])
    if not sid_col:
        raise ValueError(f"Cannot find subject id column in {subject_path}")

    subject_ids = smeta[sid_col].dropna().astype(int).tolist()
    concept2id = {str(sid): idx + 1 for idx, sid in enumerate(sorted(set(subject_ids)))}
    id2raw = {idx: int(raw) for raw, idx in [(k, v) for k, v in concept2id.items()]}
    raw_parent: Dict[int, int] = {}
    concept_meta: Dict[str, Dict] = {}
    for _, row in smeta.iterrows():
        sid = int(row[sid_col])
        cid = concept2id[str(sid)]
        parent = -1
        if parent_col and not pd.isna(row[parent_col]):
            try:
                parent = int(row[parent_col])
            except Exception:
                parent = -1
        raw_parent[sid] = parent
        concept_meta[str(cid)] = {
            "raw_name": str(sid),
            "display_name": str(row[name_col]) if name_col else str(sid),
        }

    relations: Dict[str, List[tuple[int, int, float]]] = defaultdict(list)
    for sid, parent in raw_parent.items():
        if parent != -1 and str(parent) in concept2id and str(sid) in concept2id:
            relations["parent_child"].append((concept2id[str(parent)], concept2id[str(sid)], 1.0))

    q_to_subjects: Dict[str, List[int]] = {}
    co_subject_count: Dict[tuple[int, int], int] = defaultdict(int)
    for _, row in qmeta.iterrows():
        qid = str(row[qid_col])
        subjects: List[int] = []
        for col in subject_cols:
            value = row[col]
            if pd.isna(value):
                continue
            if isinstance(value, str):
                parts = parse_list_like(value)
                if parts:
                    for p in parts:
                        if str(p) in concept2id:
                            subjects.append(concept2id[str(p)])
                else:
                    if str(value) in concept2id:
                        subjects.append(concept2id[str(value)])
            else:
                try:
                    key = str(int(value))
                    if key in concept2id:
                        subjects.append(concept2id[key])
                except Exception:
                    pass
        subjects = list(dict.fromkeys(subjects))
        if not subjects:
            continue
        q_to_subjects[qid] = subjects
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                a, b = subjects[i], subjects[j]
                co_subject_count[(a, b)] += 1
                co_subject_count[(b, a)] += 1
    max_co = max(co_subject_count.values()) if co_subject_count else 1
    for (a, b), c in co_subject_count.items():
        relations["co_subject"].append((a, b, c / max_co))

    parent_id_by_raw = raw_parent
    depth_cache: Dict[int, int] = {}

    def raw_depth(node_raw: int) -> int:
        if node_raw in depth_cache:
            return depth_cache[node_raw]
        parent = parent_id_by_raw.get(node_raw, -1)
        if parent in {-1, node_raw}:
            depth_cache[node_raw] = 0
        else:
            depth_cache[node_raw] = 1 + raw_depth(parent)
        return depth_cache[node_raw]

    question_primary: Dict[str, int] = {}
    for qid, subjects in q_to_subjects.items():
        raw_subjects = [id2raw[s] for s in subjects]
        best_raw = max(raw_subjects, key=raw_depth)
        question_primary[qid] = concept2id[str(best_raw)]

    preview_cols = pd.read_csv(train_path, nrows=1).columns.tolist()
    user_col = infer_column(preview_cols, ["UserId", "user_id", "student_id"])
    q_col = infer_column(preview_cols, ["QuestionId", "question_id", "item_id"])
    correct_col = infer_column(preview_cols, ["IsCorrect", "is_correct", "Correct", "correct", "label"])
    time_col = infer_column(preview_cols, ["Timestamp", "timestamp", "AnsweredAt", "answer_timestamp", "time"])
    usecols = [c for c in [user_col, q_col, correct_col, time_col] if c]
    inter = pd.read_csv(train_path, usecols=usecols, nrows=args.max_rows)
    if not user_col or not q_col or not correct_col:
        raise ValueError(f"Unexpected Eedi interaction columns: {preview_cols}")

    inter = inter.dropna(subset=[user_col, q_col, correct_col]).copy()
    inter[q_col] = inter[q_col].astype(str)
    inter = inter[inter[q_col].isin(question_primary.keys())].copy()
    if args.max_users:
        top_users = inter[user_col].value_counts().head(args.max_users).index
        inter = inter[inter[user_col].isin(top_users)].copy()
    inter[correct_col] = inter[correct_col].astype(float)
    if time_col and time_col in inter.columns:
        inter[time_col] = inter[time_col].astype(float)
        inter = inter.sort_values([user_col, time_col])
    else:
        inter["_order"] = np.arange(len(inter))
        time_col = "_order"
        inter = inter.sort_values([user_col, time_col])

    inter["concept_id"] = inter[q_col].map(question_primary)
    # question correctness -> concept difficulty
    q_acc = inter.groupby(q_col)[correct_col].mean().to_dict()
    concept_scores: Dict[int, List[float]] = defaultdict(list)
    for qid, cid in question_primary.items():
        if qid in q_acc:
            concept_scores[cid].append(1.0 - float(q_acc[qid]))
    difficulty = {cid: float(np.mean(vals)) for cid, vals in concept_scores.items()}
    median_diff = float(np.median(list(difficulty.values()))) if difficulty else 0.5
    for cid in concept2id.values():
        difficulty.setdefault(cid, median_diff)

    user_counts = inter[user_col].value_counts()
    active_users = set(user_counts[user_counts >= args.min_user_interactions].index.tolist())
    inter = inter[inter[user_col].isin(active_users)].copy()

    sequences = []
    for user_id, group in inter.groupby(user_col):
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
    print(f"[OK] Eedi processed. users={len(sequences)} train_tasks={len(train_tasks)} val_tasks={len(val_tasks)} test_tasks={len(test_tasks)}")


if __name__ == "__main__":
    main()
