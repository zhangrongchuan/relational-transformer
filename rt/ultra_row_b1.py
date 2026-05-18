"""B1 Step 1 model: NBFNet on the row-level FK graph + binary classifier head.

Goal: validate that row-level FK structure carries any learnable signal on
rel-trial study-outcome. We deliberately keep this minimal:

- A small MLP encodes pooled per-row text features into the hidden dim.
- A query embedding (per task; for Step 1 there is just one task) acts as
  the relational "question" — h_index seeds Bellman-Ford from the target row.
- A short NBFNet propagates query-conditioned messages along FK edges. We
  use a single relation type at Step 1; ULTRA's relation graph mechanism is
  added in Step 2.
- The target row's final feature is fed to a linear binary head.

The model exposes the same `(loss, yhat_dict)` interface as
`RelationalTransformer.forward` so we can reuse `rt.main.main` unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rt.row_graph_b1 import build_row_graph


class SimpleNBFLayer(nn.Module):
    """A single GeneralizedRelationalConv-like layer with a single relation.

    DistMult-style message: msg = h_src * relation_emb. We keep relation_emb
    learnable but tied across edges in Step 1.
    """

    def __init__(self, hidden_dim, num_relations, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim)
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        nn.init.normal_(self.relation_emb.weight, std=0.02)
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
            rel = self.relation_emb(edge_type)  # (E, H)
            msg = self.message_proj(h_norm[src] * rel)
            agg = torch.zeros_like(h_norm)
            agg.index_add_(0, dst, msg)
            deg = torch.zeros(h_norm.size(0), device=h_norm.device, dtype=h_norm.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h_norm.dtype))
            agg = agg / deg.clamp_min(1).unsqueeze(-1)
        update = self.out_proj(F.silu(self.self_proj(h_norm) + agg))
        return h + self.dropout(update)


class UltraRowB1(nn.Module):
    """B1 Step 1: single-relation NBFNet on row graph + linear head.

    Same forward signature as RelationalTransformer so it slots into rt.main.
    """

    def __init__(
        self,
        d_text=384,
        hidden_dim=128,
        num_layers=3,
        num_relations=2,  # fwd FK + rev FK (set by row_graph_b1)
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations

        in_dim = 2 * d_text  # cell_feat = concat(col_name_values, text_values)
        self.node_proj = nn.Sequential(
            nn.RMSNorm(in_dim),
            nn.Linear(in_dim, hidden_dim, bias=False),
        )
        self.query_emb = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.query_emb, std=0.02)

        self.layers = nn.ModuleList(
            [SimpleNBFLayer(hidden_dim, num_relations, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.RMSNorm(hidden_dim)
        # Head: combine target row's NBFNet output with its initial content feature.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _bellmanford(self, node_features, edge_index, edge_type, h_index):
        """Bellman-Ford with boundary at target rows.

        node_features: (N, in_dim) — already projected outside.
        h_index: (B,) global node ids that seed each sample's query. Indices
            for samples without a valid target (e.g. all-padding samples in
            the tail of an eval batch) may be == N; we clamp them here to
            keep CUDA indexing safe. Their outputs are dropped at the loss /
            yhat assembly step via `valid_targets`.
        Returns: per-sample target-row representations of shape (B, hidden).
        """
        N = node_features.size(0)
        device = node_features.device
        B = h_index.size(0)

        safe_h_index = h_index.clamp(min=0, max=max(N - 1, 0))

        # Boundary: at seed nodes, inject query embedding; elsewhere zeros.
        # We use a SINGLE shared graph but per-sample query — so the boundary
        # tensor is (B, N, H). For Step 1 (small batches, small graphs) this
        # is fine. For Step 2 we may switch to a per-sample graph.
        h = node_features.unsqueeze(0).expand(B, N, self.hidden_dim).clone()
        # Seed each sample's target row with the learned query embedding.
        sample_arange = torch.arange(B, device=device)
        h[sample_arange, safe_h_index] = h[sample_arange, safe_h_index] + self.query_emb

        for layer in self.layers:
            # Apply layer per batch slice. SimpleNBFLayer operates on (N, H).
            new_h = torch.empty_like(h)
            for b in range(B):
                new_h[b] = layer(h[b], edge_index, edge_type)
            h = new_h

        h = self.out_norm(h)
        target_repr = h[sample_arange, safe_h_index]  # (B, H)
        return target_repr

    def forward(self, batch, step=None):
        device = batch["node_idxs"].device

        graph = build_row_graph(batch, num_relations=self.num_relations)
        node_features = self.node_proj(graph.node_features.to(next(self.parameters()).dtype))
        # If the graph is empty (no valid targets), return a no-op loss.
        if node_features.size(0) == 0 or graph.target_row_global.numel() == 0:
            B, S = batch["node_idxs"].shape
            zeros = torch.zeros(B, S, 1, device=device, dtype=torch.float32)
            loss = zeros.sum() * 0.0 + sum(p.sum() * 0.0 for p in self.parameters())
            return loss, {
                "boolean": zeros,
                "number": zeros,
                "datetime": zeros,
                "text": zeros,
            }

        target_repr = self._bellmanford(
            node_features,
            graph.edge_index,
            graph.edge_type,
            graph.target_row_global,
        )  # (B, H)

        # Combine with raw content feature at the target row. Same clamp as
        # in _bellmanford — invalid-target samples are filtered later.
        N = node_features.size(0)
        safe_target_idx = graph.target_row_global.clamp(min=0, max=max(N - 1, 0))
        init_at_target = node_features[safe_target_idx]
        combined = torch.cat([target_repr, init_at_target], dim=-1)
        target_logits = self.head(combined).squeeze(-1)  # (B,)

        # Compute loss on samples that actually had a labeled target cell.
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

        # Build the (B, S, 1) `boolean` prediction tensor expected by rt.main:
        # at each sample's target cell position, place the predicted logit.
        B, S = batch["node_idxs"].shape
        boolean = torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype)
        if valid.any():
            sample_idx = torch.arange(B, device=device)[valid]
            s_idx = graph.target_cell_positions[valid]
            boolean[sample_idx, s_idx, 0] = target_logits[valid]

        return loss, {
            "boolean": boolean,
            # Provide zero tensors for the other heads so eval code's
            # task_type dispatch never breaks. We never train on these.
            "number": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
            "datetime": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
            "text": torch.zeros(B, S, 1, device=device, dtype=target_logits.dtype),
        }
