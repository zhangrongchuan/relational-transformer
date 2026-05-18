"""B2 Hybrid model: RT (frozen cell encoder) + ULTRA (row-graph reasoner).

The pure-ULTRA approach (B1/B2/B3) fails on rel-trial zero-shot. Diagnosis:
the task-stub row that we read out at only has 2 placeholder cells (timestamp +
masked target), so the pooled row feature is structurally identical across
all studies, and the only signal flows through one FK edge to the studies
parent — leaving a thin signal pipe.

This hybrid lets RT's 4-way attention enrich each cell's representation with
its surrounding row/column/FK context BEFORE we pool by node. The pooled
row vector then contains a wider neighborhood semantic than mean-of-MiniLM.
ULTRA on top reasons over the row graph using these enriched features.

Key choices:
- RT is loaded from pretrain_rel-trial_study-outcome.pt (RT pretrained on
  6 other DBs, never seeing rel-trial). Frozen.
- We mirror RT.forward up to norm_out via `_rt_encode_cells`, copying the
  encoding logic so we never run RT's decoder.
- ULTRA part (RelNBFNet + EntityNBFLayer) is identical to UltraRowB3, only
  the `node_proj` input dim changes from 2*d_text=768 to RT's d_model=256.
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from rt.model import RelationalBlock, RelationalTransformer, _make_block_mask
from rt.row_graph_b2 import build_row_graph_b2
from rt.ultra_row_b2 import EntityNBFLayer, RelGraphLayer
from rt.ultra_row_b3 import BatchedRelNBFNet


class UltraRowHybrid(nn.Module):
    def __init__(
        self,
        # RT config — must match the loaded ckpt.
        rt_num_blocks=12,
        rt_d_model=256,
        rt_d_text=384,
        rt_num_heads=8,
        rt_d_ff=1024,
        rt_ckpt_path=None,
        freeze_rt=True,
        # ULTRA config
        hidden_dim=128,
        num_layers=3,
        num_rel_layers=2,
        dropout=0.05,
    ):
        super().__init__()
        self.rt_d_model = rt_d_model
        self.hidden_dim = hidden_dim
        self.freeze_rt = freeze_rt

        self.rt = RelationalTransformer(
            num_blocks=rt_num_blocks,
            d_model=rt_d_model,
            d_text=rt_d_text,
            num_heads=rt_num_heads,
            d_ff=rt_d_ff,
        )
        if rt_ckpt_path is not None:
            state = torch.load(rt_ckpt_path, map_location="cpu")
            missing, unexpected = self.rt.load_state_dict(state, strict=False)
            print(
                f"RT ckpt loaded from {rt_ckpt_path}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

        # Cast RT to bf16 to match the batch dtype (col_name_values, text_values,
        # number_values, datetime_values, boolean_values come from the loader
        # in bfloat16). The original RT training also runs in bf16 — see
        # `rt/main.py` `net = net.to(torch.bfloat16)`.
        self.rt = self.rt.to(torch.bfloat16)

        if freeze_rt:
            for p in self.rt.parameters():
                p.requires_grad_(False)

        # ULTRA-side modules.
        self.node_proj = nn.Sequential(
            nn.RMSNorm(rt_d_model),
            nn.Linear(rt_d_model, hidden_dim, bias=False),
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

    # -------- RT cell encoder (mirror of RelationalTransformer.forward) --------
    def _rt_encode_cells(self, batch):
        """Run RT's encoder up to norm_out. Returns (B, S, d_model)."""
        rt = self.rt
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]

        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device

        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
        kv_in_f2p = (
            node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]
        ).any(-1)
        q_in_f2p = (
            node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]
        ).any(-1)
        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )

        attn_masks = {
            "feat": ((same_node | kv_in_f2p) & pad).contiguous(),
            "nbr": (q_in_f2p & pad).contiguous(),
            "col": (same_col_table & pad).contiguous(),
            "full": pad.contiguous(),
        }

        if device.type == "cuda":
            make_block_mask = partial(
                _make_block_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            block_masks = {
                l: make_block_mask(m) for l, m in attn_masks.items()
            }
        else:
            block_masks = attn_masks

        x = 0
        x = x + (
            rt.norm_dict["col_name"](rt.enc_dict["col_name"](batch["col_name_values"]))
            * (~is_padding)[..., None]
        )
        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            x = x + (
                rt.norm_dict[t](rt.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~batch["masks"] & ~is_padding)[..., None]
            )
            x = x + (
                rt.mask_embs[t]
                * ((batch["sem_types"] == i) & batch["masks"] & ~is_padding)[..., None]
            )

        for block in rt.blocks:
            x = block(x, block_masks)

        x = rt.norm_out(x)
        return x

    def _entity_bellmanford(
        self,
        node_features,
        edge_index,
        edge_rel_emb_per_b,
        query_emb_per_sample,
        target_row_global,
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
        B, S = batch["node_idxs"].shape
        param_dtype = next(self.head.parameters()).dtype

        # Cast batch tensors to bfloat16 only for the RT path (RT was trained
        # in bf16). For our ULTRA-side modules we use fp32.
        if self.freeze_rt:
            self.rt.eval()
            with torch.no_grad():
                cell_feat = self._rt_encode_cells(batch)  # (B, S, d_model)
        else:
            cell_feat = self._rt_encode_cells(batch)

        # Build row graph using RT-encoded features.
        graph = build_row_graph_b2(batch, cell_feat=cell_feat.float())

        if graph.node_features.size(0) == 0 or graph.num_relations == 0:
            zeros = torch.zeros(B, S, 1, device=device, dtype=param_dtype)
            loss = zeros.sum() * 0.0 + sum(p.sum() * 0.0 for p in self.parameters())
            return loss, {
                "boolean": zeros,
                "number": zeros,
                "datetime": zeros,
                "text": zeros,
            }

        # Target-row readout: replace the row-mean for the target row with the
        # target cell's RT output directly. RT's 4-way attention has already
        # enriched the target cell with FK-neighbor signal (this is exactly the
        # readout RT itself uses). Mean-pooling the target row dilutes this
        # with timestamp/id cells, which costs ~0.1 AUC on stream-heavy schemas
        # (rel-avito) while being neutral on flat schemas.
        if graph.valid_targets.any():
            sample_arange = torch.arange(B, device=device)
            target_cell_feat = cell_feat[sample_arange, graph.target_cell_positions].float()
            node_feat = graph.node_features.clone()
            valid_rows = graph.target_row_global[graph.valid_targets]
            node_feat[valid_rows] = target_cell_feat[graph.valid_targets]
            node_features = self.node_proj(node_feat.to(param_dtype))
        else:
            node_features = self.node_proj(graph.node_features.to(param_dtype))

        unique_qrels, qrel_inv = torch.unique(
            graph.query_rel_id, return_inverse=True
        )
        rel_emb_per_query = self.rel_nbfnet(
            graph.num_relations,
            unique_qrels,
            graph.rel_graph_edge_index,
            graph.rel_graph_edge_type,
            device=device,
            dtype=param_dtype,
        )  # (Q, R, H)
        rel_emb_per_sample = rel_emb_per_query[qrel_inv]  # (B, R, H)

        if graph.edge_index.size(1) > 0:
            edge_rel_emb_per_b = rel_emb_per_sample[:, graph.edge_rel_id, :]
        else:
            edge_rel_emb_per_b = torch.zeros(
                B, 0, self.hidden_dim, device=device, dtype=param_dtype
            )

        q_clamped = graph.query_rel_id.clamp(min=0, max=rel_emb_per_query.size(1) - 1)
        sample_arange = torch.arange(B, device=device)
        query_emb_per_sample = rel_emb_per_sample[sample_arange, q_clamped]

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
