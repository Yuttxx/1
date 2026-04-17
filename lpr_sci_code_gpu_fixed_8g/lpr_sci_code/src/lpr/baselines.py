from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import sequence_mask


class PopularityBaseline:
    def __init__(self):
        self.counter: Counter[int] = Counter()

    def fit(self, sequences: List[Dict]):
        for row in sequences:
            self.counter.update(row["concepts"])
        return self

    def recommend(self, candidates: Sequence[int], k: int = 10) -> List[int]:
        ranked = sorted(candidates, key=lambda x: self.counter.get(int(x), 0), reverse=True)
        return list(ranked[:k])


class RandomBaseline:
    def recommend(self, candidates: Sequence[int], k: int = 10) -> List[int]:
        cand = list(candidates)
        np.random.shuffle(cand)
        return cand[:k]


@dataclass
class SeqKNNExample:
    history_set: set
    next_items: List[int]


class SeqKNNBaseline:
    def __init__(self, topk_users: int = 50):
        self.examples: List[SeqKNNExample] = []
        self.topk_users = topk_users

    def fit(self, tasks: List[Dict]):
        self.examples = []
        for t in tasks:
            self.examples.append(SeqKNNExample(history_set=set(t["history"]), next_items=t.get("future", [])))
        return self

    def recommend(self, history: Sequence[int], candidates: Sequence[int], k: int = 10) -> List[int]:
        hist_set = set(history)
        scores: Dict[int, float] = {int(c): 0.0 for c in candidates}
        sims: List[Tuple[float, SeqKNNExample]] = []
        for ex in self.examples:
            inter = len(hist_set.intersection(ex.history_set))
            union = max(len(hist_set.union(ex.history_set)), 1)
            sim = inter / union
            if sim > 0:
                sims.append((sim, ex))
        sims = sorted(sims, key=lambda x: x[0], reverse=True)[: self.topk_users]
        for sim, ex in sims:
            for item in ex.next_items:
                if item in scores:
                    scores[item] += sim
        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return ranked[:k]


class GRU4RecBaseline(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_nodes, hidden_dim)
        self.correct_emb = nn.Linear(1, hidden_dim)
        self.delta_emb = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_nodes)

    def forward(self, history: torch.Tensor, correct: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(history) + self.correct_emb(correct.unsqueeze(-1).clamp(min=0.0)) + self.delta_emb(deltas.unsqueeze(-1))
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = self.dropout(hidden[-1])
        return self.out(hidden)

    @torch.no_grad()
    def recommend(
        self,
        history: torch.Tensor,
        correct: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
        action_mask: torch.Tensor,
        k: int = 10,
    ) -> torch.Tensor:
        logits = self.forward(history, correct, deltas, lengths)
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return logits.topk(k=min(k, logits.size(-1)), dim=-1).indices
