from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch

from .data import GraphStore
from .models import KnowledgeTracer


@dataclass
class RewardWeights:
    mastery_gain: float = 1.0
    prereq_ok: float = 0.2
    diff_match: float = 0.2
    review_recover: float = 0.2
    repeat_penalty: float = 0.1
    target_bonus: float = 1.0
    coherence: float = 0.2
    hierarchy: float = 0.5
    path_cost: float = 0.05


class CandidateGenerator:
    def __init__(
        self,
        graph: GraphStore,
        mode: str = "review_augmented",
        max_candidates: int = 32,
        similar_topk: int = 8,
        review_topk: int = 8,
        prereq_hops: int = 3,
    ):
        self.graph = graph
        self.mode = mode
        self.max_candidates = max_candidates
        self.similar_topk = similar_topk
        self.review_topk = review_topk
        self.prereq_hops = prereq_hops

    def _target_prereq_closure(self, target: int, max_depth: int | None = None) -> List[int]:
        max_depth = self.prereq_hops if max_depth is None else max_depth
        ordered: List[int] = []
        visited = {int(target)}
        frontier = [(int(target), 0)]
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for parent in self.graph.prerequisites.get(node, []):
                parent = int(parent)
                if parent in visited:
                    continue
                visited.add(parent)
                ordered.append(parent)
                frontier.append((parent, depth + 1))
        return ordered

    def _low_mastery_concepts(self, history: List[int], mastery_estimates: Dict[int, float], topk: int) -> List[int]:
        scored = [(cid, mastery_estimates.get(cid, 0.5)) for cid in set(history)]
        scored = sorted(scored, key=lambda x: x[1])[:topk]
        return [cid for cid, _ in scored]

    def candidates(self, current: int, target: int, history: List[int], mastery_estimates: Dict[int, float]) -> List[int]:
        cand: List[int] = []
        seen: Set[int] = set()

        def add(items: Iterable[int]) -> None:
            for x in items:
                x = int(x)
                if x not in seen:
                    cand.append(x)
                    seen.add(x)
                if len(cand) >= self.max_candidates:
                    return

        forward = self.graph.successors.get(current, [])
        add(forward)
        add(self.graph.prerequisites.get(target, []))
        add([target])
        if self.mode != "classic":
            add(self._target_prereq_closure(target))
            review_pool = self._low_mastery_concepts(history, mastery_estimates, self.review_topk)
            for node in review_pool:
                add(self.graph.prerequisites.get(node, []))
                add([node])
            similar = [nid for nid, _ in self.graph.similarity.get(current, [])[: self.similar_topk]]
            add(similar)
            target_sim = [nid for nid, _ in self.graph.similarity.get(target, [])[: self.similar_topk]]
            add(target_sim)
        if not cand:
            all_nodes = list(range(self.graph.num_nodes))
            add(all_nodes)
        return cand[: self.max_candidates]

    def mask_from_histories(self, histories: List[List[int]], targets: List[int], mastery_list: List[Dict[int, float]]) -> torch.Tensor:
        mask = torch.zeros(len(histories), self.graph.num_nodes, dtype=torch.bool)
        for i, (history, target, mastery) in enumerate(zip(histories, targets, mastery_list)):
            current = history[-1]
            cand = self.candidates(current, target, history, mastery)
            mask[i, cand] = True
        return mask


@torch.no_grad()
def estimate_mastery_dict(
    kt_model: KnowledgeTracer,
    history: List[int],
    correct: List[float],
    deltas: List[float],
    candidate_ids: List[int],
    device: torch.device,
) -> Dict[int, float]:
    if not history:
        return {cid: 0.5 for cid in candidate_ids}
    concepts = torch.tensor([history], dtype=torch.long, device=device)
    correct_t = torch.tensor([correct], dtype=torch.float32, device=device)
    deltas_t = torch.tensor([deltas], dtype=torch.float32, device=device)
    lengths = torch.tensor([len(history)], dtype=torch.long, device=device)
    out: Dict[int, float] = {}
    for cid in candidate_ids:
        target = torch.tensor([cid], dtype=torch.long, device=device)
        out[int(cid)] = float(kt_model.mastery(concepts, correct_t, deltas_t, lengths, target).item())
    return out


@torch.no_grad()
def estimate_batch_mastery_dicts(
    kt_model: KnowledgeTracer,
    histories: List[List[int]],
    corrects: List[List[float]],
    deltas: List[List[float]],
    candidate_sets: List[List[int]],
    device: torch.device,
) -> List[Dict[int, float]]:
    return [
        estimate_mastery_dict(kt_model, h, c, d, cand, device)
        for h, c, d, cand in zip(histories, corrects, deltas, candidate_sets)
    ]


@torch.no_grad()
def predict_correctness(
    kt_model: KnowledgeTracer,
    history: List[int],
    correct: List[float],
    deltas: List[float],
    action: int,
    device: torch.device,
) -> float:
    concepts = torch.tensor([history], dtype=torch.long, device=device)
    correct_t = torch.tensor([correct], dtype=torch.float32, device=device)
    deltas_t = torch.tensor([deltas], dtype=torch.float32, device=device)
    lengths = torch.tensor([len(history)], dtype=torch.long, device=device)
    target = torch.tensor([action], dtype=torch.long, device=device)
    p = float(kt_model.mastery(concepts, correct_t, deltas_t, lengths, target).item())
    return p


@torch.no_grad()
def mastery_gain(
    kt_model: KnowledgeTracer,
    history: List[int],
    correct: List[float],
    deltas: List[float],
    target: int,
    action: int,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    concepts = torch.tensor([history], dtype=torch.long, device=device)
    correct_t = torch.tensor([correct], dtype=torch.float32, device=device)
    deltas_t = torch.tensor([deltas], dtype=torch.float32, device=device)
    lengths = torch.tensor([len(history)], dtype=torch.long, device=device)
    target_t = torch.tensor([target], dtype=torch.long, device=device)
    prev_m = float(kt_model.mastery(concepts, correct_t, deltas_t, lengths, target_t).item())
    p_correct = predict_correctness(kt_model, history, correct, deltas, action, device)
    next_correct = 1.0 if p_correct >= 0.5 else 0.0
    next_history = history + [action]
    next_correct_list = correct + [next_correct]
    next_deltas = deltas + [0.0 if not deltas else deltas[-1]]
    concepts2 = torch.tensor([next_history], dtype=torch.long, device=device)
    correct2 = torch.tensor([next_correct_list], dtype=torch.float32, device=device)
    deltas2 = torch.tensor([next_deltas], dtype=torch.float32, device=device)
    lengths2 = torch.tensor([len(next_history)], dtype=torch.long, device=device)
    new_m = float(kt_model.mastery(concepts2, correct2, deltas2, lengths2, target_t).item())
    return new_m - prev_m, prev_m, new_m, next_correct



def prerequisite_satisfied(graph: GraphStore, history: List[int], action: int, mastery_estimates: Dict[int, float], threshold: float = 0.5) -> bool:
    prereqs = graph.prerequisites.get(action, [])
    if not prereqs:
        return True
    history_set = set(history)
    for p in prereqs:
        if p in history_set:
            continue
        if mastery_estimates.get(p, 0.0) >= threshold:
            continue
        return False
    return True



def difficulty_match(graph: GraphStore, history: List[int], action: int, mastery_estimates: Dict[int, float]) -> float:
    history_mastery = [mastery_estimates.get(cid, 0.5) for cid in set(history)]
    ability = float(np.mean(history_mastery)) if history_mastery else 0.5
    action_diff = graph.difficulty.get(action, 0.5)
    return max(0.0, 1.0 - abs(action_diff - (1.0 - ability)))



def review_bonus(graph: GraphStore, target: int, action: int, mastery_estimates: Dict[int, float], threshold: float = 0.5) -> float:
    prereqs = graph.prerequisites.get(target, [])
    if action in prereqs and mastery_estimates.get(action, 0.0) < threshold:
        return 1.0
    return 0.0



def coherence_score(graph: GraphStore, prev_node: int, action: int) -> float:
    if action in graph.successors.get(prev_node, []):
        return 1.0
    if prev_node in graph.prerequisites.get(action, []):
        return 1.0
    if any(nid == action for nid, _ in graph.similarity.get(prev_node, [])[:10]):
        return 0.5
    return 0.0


def prerequisite_closure(graph: GraphStore, target: int, max_depth: int = 4) -> Dict[int, int]:
    depths: Dict[int, int] = {}
    frontier: List[Tuple[int, int]] = [(int(target), 0)]
    visited = {int(target)}
    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for parent in graph.prerequisites.get(node, []):
            parent = int(parent)
            if parent in visited:
                continue
            visited.add(parent)
            depths[parent] = depth + 1
            frontier.append((parent, depth + 1))
    return depths


def frontier_prerequisites(
    graph: GraphStore,
    history: List[int],
    target: int,
    mastery_estimates: Dict[int, float],
    threshold: float = 0.5,
    max_depth: int = 4,
) -> List[int]:
    closure = prerequisite_closure(graph, target, max_depth=max_depth)
    history_set = set(int(x) for x in history)
    frontier: List[int] = []
    for node in closure:
        if node in history_set:
            continue
        if mastery_estimates.get(node, 0.0) >= threshold:
            continue
        parents = graph.prerequisites.get(node, [])
        ready = all((p in history_set) or (mastery_estimates.get(p, 0.0) >= threshold) for p in parents)
        if ready:
            frontier.append(node)
    return frontier


def hierarchy_score(
    graph: GraphStore,
    history: List[int],
    action: int,
    target: int,
    mastery_estimates: Dict[int, float],
    max_depth: int = 4,
) -> float:
    closure = prerequisite_closure(graph, target, max_depth=max_depth)
    frontier = set(frontier_prerequisites(graph, history, target, mastery_estimates, max_depth=max_depth))
    score = 0.0
    if action in frontier:
        score += 1.0
    if action in closure:
        depth = closure[action]
        score += 1.0 / (depth + 1.0)
    target_ready = prerequisite_satisfied(graph, history, target, mastery_estimates)
    if action == target:
        score += 1.5 if target_ready else -1.0
    elif action in graph.prerequisites.get(target, []):
        score += 0.5
    return score



def dense_reward(
    graph: GraphStore,
    kt_model: KnowledgeTracer,
    history: List[int],
    correct: List[float],
    deltas: List[float],
    action: int,
    target: int,
    mastery_estimates: Dict[int, float],
    weights: RewardWeights,
    device: torch.device,
    path_done: bool,
) -> Tuple[float, float]:
    mg, _, new_mastery, next_correct = mastery_gain(kt_model, history, correct, deltas, target, action, device)
    pre = 1.0 if prerequisite_satisfied(graph, history, action, mastery_estimates) else -1.0
    diff = difficulty_match(graph, history, action, mastery_estimates)
    review = review_bonus(graph, target, action, mastery_estimates)
    repeat = 1.0 if action in history[-3:] else 0.0
    coh = coherence_score(graph, history[-1], action) if history else 0.0
    reward = (
        weights.mastery_gain * mg
        + weights.prereq_ok * pre
        + weights.diff_match * diff
        + weights.review_recover * review
        + weights.coherence * coh
        - weights.repeat_penalty * repeat
        - weights.path_cost
    )
    if path_done and action == target:
        reward += weights.target_bonus
    return reward, next_correct


def hierarchical_reward(
    graph: GraphStore,
    kt_model: KnowledgeTracer,
    history: List[int],
    correct: List[float],
    deltas: List[float],
    action: int,
    target: int,
    mastery_estimates: Dict[int, float],
    weights: RewardWeights,
    device: torch.device,
    path_done: bool,
) -> Tuple[float, float]:
    mg, _, _, next_correct = mastery_gain(kt_model, history, correct, deltas, target, action, device)
    pre = 1.0 if prerequisite_satisfied(graph, history, action, mastery_estimates) else -1.0
    diff = difficulty_match(graph, history, action, mastery_estimates)
    review = review_bonus(graph, target, action, mastery_estimates)
    repeat = 1.0 if action in history[-3:] else 0.0
    coh = coherence_score(graph, history[-1], action) if history else 0.0
    hier = hierarchy_score(graph, history, action, target, mastery_estimates)
    reward = (
        weights.mastery_gain * mg
        + weights.prereq_ok * pre
        + weights.diff_match * diff
        + weights.review_recover * review
        + weights.coherence * coh
        + weights.hierarchy * hier
        - weights.repeat_penalty * repeat
        - weights.path_cost
    )
    if path_done and action == target and prerequisite_satisfied(graph, history, target, mastery_estimates):
        reward += weights.target_bonus
    return reward, next_correct



def sparse_reward(
    graph: GraphStore,
    history: List[int],
    action: int,
    target: int,
    mastery_estimates: Dict[int, float],
    path_done: bool,
) -> float:
    if not path_done:
        return 0.0
    if action != target:
        return 0.0
    return 1.0 if prerequisite_satisfied(graph, history, action, mastery_estimates) else 0.0



def append_step(history: List[int], correct: List[float], deltas: List[float], action: int, next_correct: float) -> Tuple[List[int], List[float], List[float]]:
    if deltas:
        next_delta = deltas[-1]
    else:
        next_delta = 0.0
    return history + [int(action)], correct + [float(next_correct)], deltas + [float(next_delta)]



def discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    out = [0.0 for _ in rewards]
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + gamma * running
        out[idx] = running
    return out



def compute_gae(rewards: List[float], values: List[float], gamma: float, lam: float) -> Tuple[List[float], List[float]]:
    returns = discounted_returns(rewards, gamma)
    adv = [0.0 for _ in rewards]
    next_adv = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        next_adv = delta + gamma * lam * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return returns, adv
