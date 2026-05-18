"""B1 Step 2 model: ULTRA-style relation graph + entity NBFNet on row graph.

Pipeline:
  build_row_graph_b2(batch)
    -> entity graph (rows + FK edges with per-(child,parent) relation ids)
    -> relation graph (HH/HT/TH/TT edges between relations)
  RelNBFNet(relation_graph, query=query_rel_id)
    -> per-relation embedding (R, H), conditioned on the query relation
  EntityNBFNet(entity_graph, edge_rel_emb)
    -> per-row representation, query embedding injected at target row
  head(target_repr, init_at_target) -> binary logit

Same (loss, yhat_dict) interface as RT so the training script is parallel
to Step 1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rt.row_graph_b2 import (
    NUM_REL_EDGE_TYPES,
    build_row_graph_b2,
)


class RelGraphLayer(nn.Module):
    """Message passing on the relation graph with 4 (HH/HT/TH/TT) edge types."""

    def __init__(self, hidden_dim, num_edge_types=NUM_REL_EDGE_TYPES, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        nn.init.normal_(self.edge_type_emb.weight, std=0.02)
        self.message_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_type):
        h_norm = self.norm(h)
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h_norm)
        else:
            src, dst = edge_index
            edge_rel = self.edge_type_emb(edge_type).to(dtype=h_norm.dtype)
            msg = self.message_proj(h_norm[src] * edge_rel)
            agg = torch.zeros_like(h_norm)
            agg.index_add_(0, dst, msg)
            deg = torch.zeros(h_norm.size(0), device=h_norm.device, dtype=h_norm.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h_norm.dtype))
            agg = agg / deg.clamp_min(1).unsqueeze(-1)
        update = self.out_proj(F.silu(self.self_proj(h_norm) + agg))
        return h + self.dropout(update)


class RelNBFNet(nn.Module):
    """Bellman-Ford on the relation graph, seeded at the query relation.

    Returns per-relation embeddings conditioned on a query. Following
    ULTRA, the boundary is all-ones at the query relation and zero elsewhere.
    """

    def __init__(self, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [RelGraphLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm_out = nn.RMSNorm(hidden_dim)

    def forward(self, num_relations, query_rel_id, rel_edge_index, rel_edge_type, device, dtype):
        """Returns per-relation embedding: (num_relations, H)."""
        if num_relations == 0:
            return torch.zeros(0, self.hidden_dim, device=device, dtype=dtype)
        h = torch.zeros(num_relations, self.hidden_dim, device=device, dtype=dtype)
        q = query_rel_id.clamp(min=0, max=num_relations - 1)
        h[q] = h[q] + 1.0
        for layer in self.layers:
            h = layer(h, rel_edge_index, rel_edge_type)
        return self.norm_out(h)


class EntityNBFLayer(nn.Module):
    """Message passing on the entity (row) graph with per-edge relation emb."""

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim)
        self.message_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_rel_emb):
        h_norm = self.norm(h)
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h_norm)
        else:
            src, dst = edge_index
            msg = self.message_proj(h_norm[src] * edge_rel_emb)
            agg = torch.zeros_like(h_norm)
            agg.index_add_(0, dst, msg)
            deg = torch.zeros(h_norm.size(0), device=h_norm.device, dtype=h_norm.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h_norm.dtype))
            agg = agg / deg.clamp_min(1).unsqueeze(-1)
        update = self.out_proj(F.silu(self.self_proj(h_norm) + agg))
        return h + self.dropout(update)


class UltraRowB2(nn.Module):
    """Full B1 Step 2 model: ULTRA on row graph.

    Same (loss, yhat_dict) interface as RT.
    """

    def __init__(
        self,
        d_text=384,
        hidden_dim=128,
        num_layers=3,
        num_rel_layers=2,
        dropout=0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        in_dim = 2 * d_text

        self.node_proj = nn.Sequential(
            nn.RMSNorm(in_dim),
            nn.Linear(in_dim, hidden_dim, bias=False),
        )

        self.rel_nbfnet = RelNBFNet(hidden_dim, num_layers=num_rel_layers, dropout=dropout)
        self.entity_layers = nn.ModuleList(
            [EntityNBFLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.RMSNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _entity_bellmanford(
        self, node_features, edge_index, edge_rel_emb, query_emb_per_sample, target_row_global
    ):
        """Per-sample Bellman-Ford on the shared entity graph.

        node_features: (N, H) initial features (content)
        edge_rel_emb: (E, H) per-edge relation embedding (already chosen per
            sample if query-conditioning is applied; for now batch-shared)
        query_emb_per_sample: (B, H) query relation embedding injected at the
            seed of each sample
        target_row_global: (B,) seed row index per sample
        Returns: (B, H) per-sample seed representation after L layers.
        """
        N = node_features.size(0)
        device = node_features.device
        B = query_emb_per_sample.size(0)
        safe_h_index = target_row_global.clamp(min=0, max=max(N - 1, 0))

        h = node_features.unsqueeze(0).expand(B, N, self.hidden_dim).clone()
        sample_arange = torch.arange(B, device=device)
        h[sample_arange, safe_h_index] = (
            h[sample_arange, safe_h_index] + query_emb_per_sample
        )

        for layer in self.entity_layers:
            new_h = torch.empty_like(h)
            for b in range(B):
                new_h[b] = layer(h[b], edge_index, edge_rel_emb)
            h = new_h

        h = self.out_norm(h)
        return h[sample_arange, safe_h_index]

    def forward(self, batch, step=None):
        device = batch["node_idxs"].device

        graph = build_row_graph_b2(batch)

        B, S = batch["node_idxs"].shape
        param_dtype = next(self.parameters()).dtype

        if graph.node_features.size(0) == 0:
            zeros = torch.zeros(B, S, 1, device=device, dtype=param_dtype)
            loss = zeros.sum() * 0.0 + sum(p.sum() * 0.0 for p in self.parameters())
            return loss, {
                "boolean": zeros,
                "number": zeros,
                "datetime": zeros,
                "text": zeros,
            }

        node_features = self.node_proj(graph.node_features.to(param_dtype))

        # ---- Relation NBFNet ----
        # All samples in a batch share the same task (and hence the same
        # query relation in expectation). We seed RelNBFNet at the first
        # valid sample's query rel; if no sample has a valid query, we fall
        # back to seeding at relation 0.
        if graph.valid_query.any():
            qrel_scalar = graph.query_rel_id[graph.valid_query][0].view(1)
        else:
            qrel_scalar = torch.zeros(1, dtype=torch.long, device=device)

        rel_emb = self.rel_nbfnet(
            graph.num_relations,
            qrel_scalar,
            graph.rel_graph_edge_index,
            graph.rel_graph_edge_type,
            device=device,
            dtype=param_dtype,
        )  # (R, H)

        # Per-edge relation embedding for the entity NBFNet.
        if rel_emb.size(0) > 0 and graph.edge_rel_id.numel() > 0:
            edge_rel_emb = rel_emb[graph.edge_rel_id]
        else:
            edge_rel_emb = torch.zeros(
                graph.edge_index.size(1), self.hidden_dim, device=device, dtype=param_dtype
            )

        # Per-sample query embedding to inject at the target row.
        if rel_emb.size(0) > 0:
            q_clamped = graph.query_rel_id.clamp(min=0, max=rel_emb.size(0) - 1)
            query_emb_per_sample = rel_emb[q_clamped]  # (B, H)
        else:
            query_emb_per_sample = torch.zeros(B, self.hidden_dim, device=device, dtype=param_dtype)

        target_repr = self._entity_bellmanford(
            node_features,
            graph.edge_index,
            edge_rel_emb,
            query_emb_per_sample,
            graph.target_row_global,
        )

        N = node_features.size(0)
        safe_target_idx = graph.target_row_global.clamp(min=0, max=max(N - 1, 0))
        init_at_target = node_features[safe_target_idx]
        combined = torch.cat([target_repr, init_at_target], dim=-1)
        target_logits = self.head(combined).squeeze(-1)

        valid = graph.valid_targets
        labels = graph.target_labels
        if valid.any():
            loss = F.binary_cross_entropy_with_logits(
                target_logits[valid], labels[valid]
            )
        else:
            loss = target_logits.sum() * 0.0 + sum(
                p.sum() * 0.0 for p in self.parameters()
            )

        boolean = torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype)
        if valid.any():
            sample_idx = torch.arange(B, device=device)[valid]
            s_idx = graph.target_cell_positions[valid]
            boolean[sample_idx, s_idx, 0] = target_logits[valid]

        return loss, {
            "boolean": boolean,
            "number": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
            "datetime": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
            "text": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
        }
