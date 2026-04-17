#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from lpr.data import load_standard_dataset


def collect_ids(dataset):
    ids = set()
    for split in [dataset.train_sequences, dataset.val_sequences, dataset.test_sequences]:
        for row in split:
            ids.update(int(x) for x in row.get('concepts', []))
    for split in [dataset.train_tasks, dataset.val_tasks, dataset.test_tasks]:
        for row in split:
            ids.update(int(x) for x in row.get('history', []))
            ids.update(int(x) for x in row.get('future', []))
            if 'target' in row:
                ids.add(int(row['target']))
    for edges in dataset.graph.edges_by_rel.values():
        for s, d, _ in edges:
            ids.add(int(s))
            ids.add(int(d))
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description='Check whether a processed dataset is ID-safe for training.')
    parser.add_argument('--dataset_dir', required=True)
    args = parser.parse_args()

    dataset = load_standard_dataset(args.dataset_dir)
    ids = collect_ids(dataset)
    issues = []
    if not dataset.id2concept or dataset.id2concept[0] is not None:
        issues.append('padding_id_0_is_not_reserved')
    if 0 in dataset.concept2id.values():
        issues.append('real_concept_uses_padding_id_0')
    if ids and (min(ids) < 1):
        issues.append('real_ids_should_start_from_1')
    if ids and max(ids) >= dataset.num_nodes:
        issues.append('observed_id_out_of_range')

    payload = {
        'num_nodes': dataset.num_nodes,
        'num_real_concepts': len(dataset.concept2id),
        'train_sequences': len(dataset.train_sequences),
        'val_sequences': len(dataset.val_sequences),
        'test_sequences': len(dataset.test_sequences),
        'train_tasks': len(dataset.train_tasks),
        'val_tasks': len(dataset.val_tasks),
        'test_tasks': len(dataset.test_tasks),
        'min_observed_id': min(ids) if ids else None,
        'max_observed_id': max(ids) if ids else None,
        'padding_reserved': bool(dataset.id2concept and dataset.id2concept[0] is None),
        'issues': issues,
        'status': 'ok' if not issues else 'warning',
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
