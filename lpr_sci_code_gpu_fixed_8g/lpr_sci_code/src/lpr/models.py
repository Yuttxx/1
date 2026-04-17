from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import sequence_mask, softmax_masked
from .data import GraphStore


class RGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.rel_weight = nn.Parameter(torch.randn(num_relations, hidden_dim, hidden_dim) * 0.02)
        self.self_loop = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edges_by_rel: Dict[str, List[Tuple[int, int, float]]], relations: List[str]) -> torch.Tensor:
        device = x.device
        out = self.self_loop(x)
        for r_idx, rel in enumerate(relations):
            edges = edges_by_rel.get(rel, [])
            if not edges:
                continue
            src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
            dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
            weight = torch.tensor([e[2] for e in edges], dtype=torch.float32, device=device).unsqueeze(-1)
            msg = torch.matmul(x[src], self.rel_weight[r_idx]) * weight
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, msg)
            deg = torch.zeros(x.size(0), 1, device=device)
            deg.index_add_(0, dst, weight)
            out = out + agg / deg.clamp(min=1.0)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out + x)


class RGCNEncoder(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        self.layers = nn.ModuleList([RGCNLayer(hidden_dim, num_relations, dropout=dropout) for _ in range(num_layers)])

    def forward(self, graph: GraphStore) -> torch.Tensor:
        x = self.node_embedding.weight
        for layer in self.layers:
            x = layer(x, graph.edges_by_rel, graph.relations)
        return x


class TransEEncoder(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_dim: int = 128):
        super().__init__()
        self.entity = nn.Embedding(num_nodes, hidden_dim)
        self.relation = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = F.normalize(self.entity(heads), dim=-1)
        r = F.normalize(self.relation(rels), dim=-1)
        t = F.normalize(self.entity(tails), dim=-1)
        return -(h + r - t).norm(p=2, dim=-1)

    def forward(self, graph: GraphStore) -> torch.Tensor:
        return F.normalize(self.entity.weight, dim=-1)


class TimeAwarePreferenceEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.correct_proj = nn.Linear(1, hidden_dim)
        self.delta_proj = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sim_scale = nn.Parameter(torch.tensor(1.0))
        self.time_scale = nn.Parameter(torch.tensor(1.0))
        self.out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_emb: torch.Tensor,
        concepts: torch.Tensor,
        correct: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        emb = node_emb[concepts]
        corr = self.correct_proj(correct.unsqueeze(-1).clamp(min=0.0))
        time_feat = self.delta_proj(deltas.unsqueeze(-1))
        x = emb + corr + time_feat
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=concepts.size(1))
        hidden = hidden[-1]
        q = self.q_proj(hidden).unsqueeze(1)
        k = self.k_proj(out)
        v = self.v_proj(out)
        attn = (q * k).sum(-1) / math.sqrt(k.size(-1))
        last_idx = (lengths - 1).clamp(min=0)
        last_emb = out[torch.arange(out.size(0), device=out.device), last_idx]
        sim = F.cosine_similarity(last_emb.unsqueeze(1), out, dim=-1)
        attn = attn + self.sim_scale * sim - self.time_scale * deltas
        mask = sequence_mask(lengths, max_len=concepts.size(1))
        weight = softmax_masked(attn, mask)
        ctx = torch.bmm(weight.unsqueeze(1), v).squeeze(1)
        pref = self.out(torch.cat([ctx, hidden], dim=-1))
        return self.dropout(F.relu(pref))


class KnowledgeBackgroundEncoder(nn.Module):
    def __init__(self, hidden_dim: int, mode: str = "attention", dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        self.correct_proj = nn.Linear(1, hidden_dim)
        self.goal_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pref_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.kt_hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kt_fuse = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_emb: torch.Tensor,
        concepts: torch.Tensor,
        correct: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
        preference: torch.Tensor,
        goal_ids: torch.Tensor,
        kt_hidden: Optional[torch.Tensor] = None,
        kt_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hist = node_emb[concepts]
        mask = sequence_mask(lengths, max_len=concepts.size(1)).unsqueeze(-1)
        if self.mode == "zero":
            return torch.zeros(hist.size(0), hist.size(-1), device=hist.device)
        if self.mode == "mean":
            denom = mask.float().sum(dim=1).clamp(min=1.0)
            pooled = (hist * mask.float()).sum(dim=1) / denom
            return self.dropout(self.out(pooled))
        if self.mode in {"kt", "knowledge_tracing", "kt_state"}:
            if kt_hidden is None or kt_probs is None:
                raise ValueError("kb_mode=kt requires KT hidden states and mastery probabilities.")
            hist_ids = concepts.clamp(min=0)
            hist_mastery = kt_probs.gather(1, hist_ids).unsqueeze(-1) * mask.float()
            denom = hist_mastery.sum(dim=1).clamp(min=1e-6)
            mastery_pool = (hist * hist_mastery).sum(dim=1) / denom
            goal = node_emb[goal_ids]
            target_mastery = kt_probs.gather(1, goal_ids.view(-1, 1)).squeeze(1).unsqueeze(-1)
            fused = torch.cat(
                [
                    self.kt_hidden_proj(kt_hidden),
                    mastery_pool,
                    goal,
                    goal * target_mastery,
                ],
                dim=-1,
            )
            return self.dropout(F.relu(self.kt_fuse(fused)))
        correct_feat = self.correct_proj(correct.unsqueeze(-1).clamp(min=0.0))
        goal = self.goal_proj(node_emb[goal_ids]).unsqueeze(1)
        pref = self.pref_proj(preference).unsqueeze(1)
        raw = torch.tanh(hist + correct_feat + goal + pref)
        score = self.score(raw).squeeze(-1)
        weight = softmax_masked(score, mask.squeeze(-1))
        pooled = torch.bmm(weight.unsqueeze(1), hist).squeeze(1)
        return self.dropout(F.relu(self.out(pooled)))


class KnowledgeTracer(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.correct_emb = nn.Embedding(2, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.correct_emb.weight)
        self.delta_proj = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_nodes)
        self.dropout = nn.Dropout(dropout)

    def interaction_embedding(self, concepts: torch.Tensor, correct: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        base = self.node_emb(concepts)
        right = self.correct_emb.weight[1].view(1, 1, -1)
        wrong = self.correct_emb.weight[0].view(1, 1, -1)
        correct_clamped = correct.unsqueeze(-1).clamp(min=0.0, max=1.0)
        corr_emb = correct_clamped * right + (1.0 - correct_clamped) * wrong
        return base + corr_emb + self.delta_proj(deltas.unsqueeze(-1))

    def encode_sequence(
        self,
        concepts: torch.Tensor,
        correct: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.interaction_embedding(concepts, correct, deltas)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=concepts.size(1))
        out = self.dropout(out)
        return out, self.dropout(hidden[-1])

    def forward(self, concepts: torch.Tensor, correct: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out, _ = self.encode_sequence(concepts, correct, deltas, lengths)
        return self.output(out)

    @torch.no_grad()
    def state(
        self,
        concepts: torch.Tensor,
        correct: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, last_hidden = self.encode_sequence(concepts, correct, deltas, lengths)
        last_logits = self.output(last_hidden)
        return last_hidden, torch.sigmoid(last_logits)

    @torch.no_grad()
    def mastery(
        self,
        concepts: torch.Tensor,
        correct: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(concepts, correct, deltas, lengths)
        last_idx = (lengths - 1).clamp(min=0)
        last_hidden_logits = logits[torch.arange(logits.size(0), device=logits.device), last_idx]
        probs = torch.sigmoid(last_hidden_logits)
        return probs.gather(1, target_ids.view(-1, 1)).squeeze(1)


class PolicyValueNet(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int, dropout: float = 0.1):
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.bias = nn.Parameter(torch.zeros(num_nodes))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor, node_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_h = self.state_proj(state)
        logits = self.scorer(
            state_h.unsqueeze(1).expand(-1, node_emb.size(0), -1),
            node_emb.unsqueeze(0).expand(state_h.size(0), -1, -1),
        ).squeeze(-1)
        logits = logits + self.bias.unsqueeze(0)
        value = self.critic(state_h).squeeze(-1)
        return logits, value


class LPRModel(nn.Module):
    def __init__(
        self,
        graph_encoder: nn.Module,
        preference_encoder: TimeAwarePreferenceEncoder,
        kb_encoder: KnowledgeBackgroundEncoder,
        policy: PolicyValueNet,
        hidden_dim: int,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.preference_encoder = preference_encoder
        self.kb_encoder = kb_encoder
        self.policy = policy
        self.state_norm = nn.LayerNorm(hidden_dim * 3)

    def encode_state(
        self,
        graph: GraphStore,
        history: torch.Tensor,
        history_correct: torch.Tensor,
        history_deltas: torch.Tensor,
        lengths: torch.Tensor,
        target: torch.Tensor,
        kt_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_emb = self.graph_encoder(graph)
        pref = self.preference_encoder(node_emb, history, history_correct, history_deltas, lengths)
        kb = self.kb_encoder(
            node_emb,
            history,
            history_correct,
            history_deltas,
            lengths,
            pref,
            target,
            kt_hidden=None if kt_features is None else kt_features.get("hidden"),
            kt_probs=None if kt_features is None else kt_features.get("probs"),
        )
        goal = node_emb[target]
        state = self.state_norm(torch.cat([pref, kb, goal], dim=-1))
        return state, node_emb, pref, kb

    def forward(
        self,
        graph: GraphStore,
        history: torch.Tensor,
        history_correct: torch.Tensor,
        history_deltas: torch.Tensor,
        lengths: torch.Tensor,
        target: torch.Tensor,
        kt_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, node_emb, _, _ = self.encode_state(
            graph,
            history,
            history_correct,
            history_deltas,
            lengths,
            target,
            kt_features=kt_features,
        )
        logits, value = self.policy(state, node_emb)
        return logits, value, node_emb

    @torch.no_grad()
    def act(
        self,
        graph: GraphStore,
        history: torch.Tensor,
        history_correct: torch.Tensor,
        history_deltas: torch.Tensor,
        lengths: torch.Tensor,
        target: torch.Tensor,
        action_mask: torch.Tensor,
        kt_features: Optional[Dict[str, torch.Tensor]] = None,
        greedy: bool = False,
    ) -> Dict[str, torch.Tensor]:
        logits, value, _ = self.forward(
            graph,
            history,
            history_correct,
            history_deltas,
            lengths,
            target,
            kt_features=kt_features,
        )
        mask = action_mask.to(dtype=torch.bool)
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        invalid_rows = ~mask.any(dim=-1, keepdim=True)
        if invalid_rows.any():
            mask = torch.where(invalid_rows, torch.ones_like(mask), mask)
        masked_logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        probs = torch.softmax(masked_logits, dim=-1)
        if greedy:
            action = masked_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "probs": probs,
        }
