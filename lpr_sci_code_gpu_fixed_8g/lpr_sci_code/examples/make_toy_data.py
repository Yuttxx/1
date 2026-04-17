#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from lpr.data import SequenceBuildConfig, build_tasks_from_sequences, save_standard_dataset, split_by_user
from lpr.graph_utils import build_graph_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a small synthetic dataset for smoke tests.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    concept2id = {f"c{i}": i + 1 for i in range(12)}
    relations = defaultdict(list)
    # chain prerequisites c0 -> c1 -> ... -> c11
    for i in range(11):
        relations["prerequisite"].append((i + 1, i + 2, 1.0))
    # topic-like similarity blocks
    for block in [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]:
        block = list(block)
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                relations["same_topic"].append((block[i] + 1, block[j] + 1, 1.0))
                relations["same_topic"].append((block[j] + 1, block[i] + 1, 1.0))
    difficulty = {i + 1: i / 11 for i in range(12)}
    concept_meta = {str(i + 1): {"raw_name": f"c{i}", "display_name": f"Concept-{i}"} for i in range(12)}
    graph = build_graph_store(max(concept2id.values()) + 1, relations, difficulty, concept_meta)

    sequences = []
    for u in range(20):
        start = int(rng.integers(1, 4))
        seq = [start]
        correct = [1.0]
        timestamps = [0.0]
        for t in range(1, 24):
            if rng.random() < 0.7 and seq[-1] < 12:
                nxt = seq[-1] + 1
            else:
                cand = [max(1, seq[-1] - 1), min(12, seq[-1] + 1), seq[-1]]
                nxt = int(rng.choice(cand))
            seq.append(nxt)
            correct.append(float(rng.random() > difficulty[nxt] * 0.6))
            timestamps.append(float(t))
        sequences.append(
            {
                "user_id": f"u{u}",
                "concepts": seq,
                "correct": correct,
                "timestamps": timestamps,
                "deltas": [0.0] + [1.0] * (len(seq) - 1),
            }
        )

    train_seq, val_seq, test_seq = split_by_user(sequences, seed=args.seed)
    cfg = SequenceBuildConfig(history_len=10, path_len=5, stride=5)
    train_tasks = build_tasks_from_sequences(train_seq, cfg)
    val_tasks = build_tasks_from_sequences(val_seq, cfg)
    test_tasks = build_tasks_from_sequences(test_seq, cfg)
    save_standard_dataset(args.output_dir, concept2id, graph, train_seq, val_seq, test_seq, train_tasks, val_tasks, test_tasks)
    print(f"[OK] toy dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
