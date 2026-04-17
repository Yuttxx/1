from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from .data import GraphStore
from .models import KnowledgeTracer
from .rl import prerequisite_satisfied


@torch.no_grad()
def mastery_gain_metric(
    kt_model: KnowledgeTracer,
    history: List[int],
    correct: List[float],
    deltas: List[float],
    path: List[int],
    target: int,
    device: torch.device,
) -> float:
    if not history:
        return 0.0
    concepts = torch.tensor([history], dtype=torch.long, device=device)
    correct_t = torch.tensor([correct], dtype=torch.float32, device=device)
    deltas_t = torch.tensor([deltas], dtype=torch.float32, device=device)
    lengths = torch.tensor([len(history)], dtype=torch.long, device=device)
    target_t = torch.tensor([target], dtype=torch.long, device=device)
    prev = float(kt_model.mastery(concepts, correct_t, deltas_t, lengths, target_t).item())
    # optimistic simulation: repeated concepts in path are treated as correct if new
    next_hist = history + path
    next_correct = correct + [1.0 for _ in path]
    next_delta = deltas + ([deltas[-1] if deltas else 0.0] * len(path))
    concepts2 = torch.tensor([next_hist], dtype=torch.long, device=device)
    correct2 = torch.tensor([next_correct], dtype=torch.float32, device=device)
    deltas2 = torch.tensor([next_delta], dtype=torch.float32, device=device)
    lengths2 = torch.tensor([len(next_hist)], dtype=torch.long, device=device)
    new = float(kt_model.mastery(concepts2, correct2, deltas2, lengths2, target_t).item())
    denom = max(1.0 - prev, 1e-6)
    return (new - prev) / denom



def hit_rate(path: Sequence[int], target: int) -> float:
    return 1.0 if target in path else 0.0



def ndcg_at_k(path: Sequence[int], future: Sequence[int], k: int) -> float:
    if not future:
        return 0.0
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(future), k)))
    if ideal <= 0:
        return 0.0
    rel = 0.0
    future_set = set(future[:k])
    for i, item in enumerate(path[:k]):
        if item in future_set:
            rel += 1.0 / np.log2(i + 2)
    return float(rel / ideal)



def mrr(path: Sequence[int], future: Sequence[int]) -> float:
    future_set = set(future)
    for i, item in enumerate(path):
        if item in future_set:
            return 1.0 / (i + 1)
    return 0.0



def prerequisite_violation_rate(graph: GraphStore, history: List[int], path: Sequence[int]) -> float:
    cur_hist = list(history)
    violations = 0
    for item in path:
        if not prerequisite_satisfied(graph, cur_hist, item, {x: 1.0 for x in cur_hist}):
            violations += 1
        cur_hist.append(item)
    return violations / max(len(path), 1)



def difficulty_smoothness(graph: GraphStore, history: List[int], path: Sequence[int]) -> float:
    seq = history[-1:] + list(path)
    if len(seq) < 2:
        return 1.0
    diffs = []
    for a, b in zip(seq[:-1], seq[1:]):
        diffs.append(abs(graph.difficulty.get(a, 0.5) - graph.difficulty.get(b, 0.5)))
    return 1.0 - float(np.mean(diffs))



def review_coverage(graph: GraphStore, target: int, path: Sequence[int]) -> float:
    prereqs = set(graph.prerequisites.get(target, []))
    if not prereqs:
        return 0.0
    return len(prereqs.intersection(path)) / len(prereqs)



def aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(metrics[0].keys())
    out = {}
    for key in keys:
        out[key] = float(np.mean([m[key] for m in metrics]))
        out[f"{key}_std"] = float(np.std([m[key] for m in metrics]))
    return out
