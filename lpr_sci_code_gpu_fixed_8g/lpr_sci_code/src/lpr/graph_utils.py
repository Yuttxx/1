from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import GraphStore


def infer_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]
    for cand in candidates:
        for col in columns:
            if cand.lower() == col.lower().replace(" ", "_"):
                return col
    for cand in candidates:
        for col in columns:
            if cand.lower() in col.lower():
                return col
    return None


def parse_list_like(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-1"}:
        return []
    text = text.strip("[]")
    parts = re.split(r"[;,| ]+", text)
    return [p for p in parts if p]



def deduplicate_edges(edges: Iterable[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    best: Dict[Tuple[int, int], float] = {}
    for s, d, w in edges:
        key = (int(s), int(d))
        best[key] = max(float(w), best.get(key, 0.0))
    return [(s, d, w) for (s, d), w in best.items()]



def build_graph_store(
    num_nodes: int,
    relations: Dict[str, List[Tuple[int, int, float]]],
    difficulty: Dict[int, float],
    concept_meta: Dict[str, Any],
) -> GraphStore:
    relations = {rel: deduplicate_edges(edges) for rel, edges in relations.items()}
    prerequisites: Dict[int, List[int]] = defaultdict(list)
    successors: Dict[int, List[int]] = defaultdict(list)
    for rel_name in relations:
        if "prereq" in rel_name or rel_name == "parent_child":
            for src, dst, _ in relations[rel_name]:
                if src not in prerequisites[dst]:
                    prerequisites[dst].append(src)
                if dst not in successors[src]:
                    successors[src].append(dst)
    similarity: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for rel_name in relations:
        if "similar" in rel_name or rel_name in {"same_topic", "same_area", "co_subject"}:
            for src, dst, w in relations[rel_name]:
                similarity[src].append((dst, w))
                similarity[dst].append((src, w))
    for key in similarity:
        similarity[key] = sorted(similarity[key], key=lambda x: x[1], reverse=True)[:50]
    return GraphStore(
        num_nodes=num_nodes,
        relations=list(relations.keys()),
        edges_by_rel=relations,
        prerequisites=dict(prerequisites),
        successors=dict(successors),
        similarity=dict(similarity),
        difficulty=difficulty,
        concept_meta=concept_meta,
    )



def estimate_difficulty_from_logs(
    concept_ids: Sequence[int],
    correct: Sequence[float],
) -> Dict[int, float]:
    total = defaultdict(float)
    count = defaultdict(float)
    for cid, c in zip(concept_ids, correct):
        total[int(cid)] += float(c)
        count[int(cid)] += 1.0
    difficulties: Dict[int, float] = {}
    for cid in count:
        acc = total[cid] / max(count[cid], 1.0)
        difficulties[cid] = float(np.clip(1.0 - acc, 0.0, 1.0))
    return difficulties



def topological_distance(successors: Dict[int, List[int]], src: int, dst: int, max_depth: int = 6) -> Optional[int]:
    if src == dst:
        return 0
    queue: deque[Tuple[int, int]] = deque([(src, 0)])
    visited = {src}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt in successors.get(node, []):
            if nxt == dst:
                return depth + 1
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1))
    return None



def build_similarity_edges_from_annotations(
    annotations: pd.DataFrame,
    concept2id: Dict[str, int],
    threshold: float = 6.0,
) -> Dict[str, List[Tuple[int, int, float]]]:
    cols = annotations.columns.tolist()
    a_col = infer_column(cols, ["Exercise_A", "exercise_a", "A", "question_a"])
    b_col = infer_column(cols, ["Exercise_B", "exercise_b", "B", "question_b"])
    sim_col = infer_column(cols, ["Similarity_avg", "similarity_avg", "similarity"])
    pre_col = infer_column(cols, ["Prerequisite_avg", "prequesite_avg", "prerequisite_avg", "prerequisite"])
    relations: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
    if a_col and b_col and sim_col:
        for _, row in annotations.iterrows():
            a, b = str(row[a_col]), str(row[b_col])
            if a in concept2id and b in concept2id and float(row[sim_col]) >= threshold:
                ai, bi = concept2id[a], concept2id[b]
                w = float(row[sim_col]) / 10.0
                relations["similarity"].append((ai, bi, w))
                relations["similarity"].append((bi, ai, w))
    if a_col and b_col and pre_col:
        for _, row in annotations.iterrows():
            a, b = str(row[a_col]), str(row[b_col])
            if a in concept2id and b in concept2id and float(row[pre_col]) >= threshold:
                ai, bi = concept2id[a], concept2id[b]
                w = float(row[pre_col]) / 10.0
                relations["annotation_prerequisite"].append((ai, bi, w))
    return relations



def infer_primary_subject(question_subjects: List[List[int]], parent: Dict[int, int]) -> Dict[int, int]:
    depth_cache: Dict[int, int] = {}

    def depth(node: int) -> int:
        if node in depth_cache:
            return depth_cache[node]
        if node not in parent or parent[node] in {-1, node, None}:
            depth_cache[node] = 0
            return 0
        depth_cache[node] = 1 + depth(parent[node])
        return depth_cache[node]

    mapping: Dict[int, int] = {}
    for qid, subjects in enumerate(question_subjects):
        if not subjects:
            continue
        mapping[qid] = max(subjects, key=depth)
    return mapping
