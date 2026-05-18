"""B1 Step 3 model: ULTRA for cross-DB zero-shot transfer.

Extends UltraRowB2 to handle batches that mix samples from multiple tasks
(and therefore multiple query relations). The relation NBFNet now produces
embeddings *per unique query* in the batch, and each sample gets its own
per-edge relation embeddings via lookup.

Architecturally nothing new vs Step 2 — the inductive mechanism is the
same. Only the per-sample query handling changes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rt.row_graph_b2 import build_row_graph_b2
from rt.ultra_row_b2 import EntityNBFLayer, RelGraphLayer


class BatchedRelNBFNet(nn.Module):
    """Bellman-Ford on the relation graph with a batch of Q queries.

    Returns (Q, R, H) per-relation embeddings, one slice per unique query.
    """

    def __init__(self, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [RelGraphLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm_out = nn.RMSNorm(hidden_dim)

    def forward(
        self,
        num_relations: int,
        query_rel_ids,            # (Q,) long
        rel_edge_index,           # (2, RE)
        rel_edge_type,            # (RE,)
        device,
        dtype,
    ):
        Q = int(query_rel_ids.numel())
        if num_relations == 0 or Q == 0:
            return torch.zeros(max(Q, 1), max(num_relations, 1), self.hidden_dim, device=device, dtype=dtype)

        # Boundary: (Q, R, H), all zeros except at query positions per slice.
        h = torch.zeros(Q, num_relations, self.hidden_dim, device=device, dtype=dtype)
        arange_Q = torch.arange(Q, device=device)
        q_clamped = query_rel_ids.clamp(min=0, max=num_relations - 1)
        h[arange_Q, q_clamped] = h[arange_Q, q_clamped] + 1.0

        for layer in self.layers:
            new_h = torch.empty_like(h)
            for q in range(Q):
                new_h[q] = layer(h[q], rel_edge_index, rel_edge_type)
            h = new_h

        return self.norm_out(h)  # (Q, R, H)


class UltraRowB3(nn.Module):
    """Step 3 model: ULTRA on row graph with per-sample query relations."""

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

        self.rel_nbfnet = BatchedRelNBFNet(
            hidden_dim, num_layers=num_rel_layers, dropout=dropout
        )
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
        self,
        node_features,         # (N, H)
        edge_index,            # (2, E)
        edge_rel_emb_per_b,    # (B, E, H)
        query_emb_per_sample,  # (B, H)
        target_row_global,     # (B,)
    ):
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
                new_h[b] = layer(h[b], edge_index, edge_rel_emb_per_b[b])
            h = new_h

        h = self.out_norm(h)
        return h[sample_arange, safe_h_index]

    def forward(self, batch, step=None):
        device = batch["node_idxs"].device

        graph = build_row_graph_b2(batch)

        B, S = batch["node_idxs"].shape
        param_dtype = next(self.parameters()).dtype

        if graph.node_features.size(0) == 0 or graph.num_relations == 0:
            zeros = torch.zeros(B, S, 1, device=device, dtype=param_dtype)
            loss = zeros.sum() * 0.0 + sum(p.sum() * 0.0 for p in self.parameters())
            return loss, {
                "boolean": zeros,
                "number": zeros,
                "datetime": zeros,
                "text": zeros,
            }

        node_features = self.node_proj(graph.node_features.to(param_dtype))

        # ---- RelNBFNet over unique queries ----
        unique_qrels, qrel_inv = torch.unique(
            graph.query_rel_id, return_inverse=True
        )  # (Q,), (B,)

        rel_emb_per_query = self.rel_nbfnet(
            graph.num_relations,
            unique_qrels,
            graph.rel_graph_edge_index,
            graph.rel_graph_edge_type,
            device=device,
            dtype=param_dtype,
        )  # (Q, R, H)

        # Per-sample relation embeddings: pick the slice matching each sample.
        rel_emb_per_sample = rel_emb_per_query[qrel_inv]  # (B, R, H)

        # Per-sample, per-edge relation embedding.
        if graph.edge_index.size(1) > 0:
            # rel_emb_per_sample: (B, R, H); graph.edge_rel_id: (E,) -> gather along R.
            edge_rel_emb_per_b = rel_emb_per_sample[
                :, graph.edge_rel_id, :
            ]  # (B, E, H)
        else:
            edge_rel_emb_per_b = torch.zeros(
                B, 0, self.hidden_dim, device=device, dtype=param_dtype
            )

        # Per-sample query embedding to inject at the target row.
        q_clamped = graph.query_rel_id.clamp(min=0, max=rel_emb_per_query.size(1) - 1)
        sample_arange = torch.arange(B, device=device)
        query_emb_per_sample = rel_emb_per_sample[sample_arange, q_clamped]  # (B, H)

        target_repr = self._entity_bellmanford(
            node_features,
            graph.edge_index,
            edge_rel_emb_per_b,
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
            sidx = torch.arange(B, device=device)[valid]
            s_idx = graph.target_cell_positions[valid]
            boolean[sidx, s_idx, 0] = target_logits[valid]

        return loss, {
            "boolean": boolean,
            "number": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
            "datetime": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
            "text": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
        }
