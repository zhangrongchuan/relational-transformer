"""Row-level graph builder for B1 Step 1.

The Rust sampler returns cell-level batches with shape (B, S). Each cell
belongs to a row identified by `node_idxs[b, s]`. FK edges are stored at the
row level: `f2p_nbr_idxs[b, s]` lists the parent row IDs of the row that cell
(b, s) belongs to (every cell in the same row holds the same list).

For Step 1 we build a single row graph per batch:
  - nodes  = unique rows across all B samples
  - edges  = directed (child_row -> parent_row), one per (row, parent) pair
  - edge_type = 0 (single relation type, deliberately simple for Step 1)
  - per-row feature = mean of `col_name_values + text_values` cell vectors

The target row per sample is the row containing the `is_targets` cell. We
also expose `target_row_global` so the model can read out predictions only
at those rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RowGraph:
    node_features: torch.Tensor       # (N_total_rows, d_feat)
    edge_index: torch.Tensor          # (2, E) long
    edge_type: torch.Tensor           # (E,) long
    num_relations: int
    target_row_global: torch.Tensor   # (B,) long, indices into nodes
    target_labels: torch.Tensor       # (B,) float (0/1)
    row_to_sample: torch.Tensor       # (N_total_rows,) long, which b each row came from
    num_rows_per_sample: torch.Tensor # (B,) long
    target_cell_positions: torch.Tensor  # (B,) long, the s of the target cell within sample b
    valid_targets: torch.Tensor       # (B,) bool, samples that actually have a target cell


def build_row_graph(batch, num_relations: int = 1) -> RowGraph:
    """Construct a single row graph spanning all B samples of the batch."""
    node_idxs = batch["node_idxs"]            # (B, S) int32
    f2p_nbr_idxs = batch["f2p_nbr_idxs"]      # (B, S, 5) int32
    is_padding = batch["is_padding"].bool()   # (B, S)
    is_targets = batch["is_targets"].bool()   # (B, S)
    col_emb = batch["col_name_values"]        # (B, S, d_text)
    txt_emb = batch["text_values"]            # (B, S, d_text)
    boolean_values = batch["boolean_values"]  # (B, S, 1)

    B, S = node_idxs.shape
    device = node_idxs.device

    cell_feat = torch.cat([col_emb, txt_emb], dim=-1).float()  # (B, S, 2*d_text)
    d_feat = cell_feat.shape[-1]

    node_feature_chunks = []
    row_to_sample_chunks = []
    edge_src_chunks = []
    edge_dst_chunks = []
    target_row_global = torch.zeros(B, dtype=torch.long, device=device)
    target_labels = torch.zeros(B, dtype=torch.float32, device=device)
    num_rows_per_sample = torch.zeros(B, dtype=torch.long, device=device)
    target_cell_positions = torch.zeros(B, dtype=torch.long, device=device)
    valid_targets = torch.zeros(B, dtype=torch.bool, device=device)

    cursor = 0  # running offset into the global node id space
    for b in range(B):
        valid_mask = ~is_padding[b]            # (S,)
        valid_positions = torch.nonzero(valid_mask, as_tuple=True)[0]  # (Sv,)
        Sv = valid_positions.numel()
        if Sv == 0:
            num_rows_per_sample[b] = 0
            target_row_global[b] = cursor
            continue

        rows_b = node_idxs[b, valid_positions]   # (Sv,) raw row ids
        feats_b = cell_feat[b, valid_positions]  # (Sv, d_feat)

        uniq_rows, inv = torch.unique(rows_b, return_inverse=True)  # (R,), (Sv,)
        R = uniq_rows.numel()

        # Mean-pool cell features per local row.
        pooled = torch.zeros(R, d_feat, device=device, dtype=feats_b.dtype)
        pooled.index_add_(0, inv, feats_b)
        counts = torch.zeros(R, device=device, dtype=feats_b.dtype)
        counts.index_add_(0, inv, torch.ones_like(inv, dtype=feats_b.dtype))
        pooled = pooled / counts.clamp_min(1).unsqueeze(-1)

        node_feature_chunks.append(pooled)
        row_to_sample_chunks.append(
            torch.full((R,), b, dtype=torch.long, device=device)
        )
        num_rows_per_sample[b] = R

        # Pick one representative cell per row: the first occurrence in
        # valid_positions, found via a scatter on inv.
        first_cell_position = torch.full(
            (R,), Sv, dtype=torch.long, device=device
        )
        rng = torch.arange(Sv, device=device)
        first_cell_position.scatter_reduce_(
            0, inv, rng, reduce="amin", include_self=True
        )
        # first_cell_position[r] is now the position-in-valid_positions of the
        # first cell that belongs to local row r.
        rep_cell_s = valid_positions[first_cell_position]  # (R,) cell indices in S

        # f2p_nbr_idxs[b, rep_cell_s] has shape (R, MAX_F2P); valid parents are >= 0.
        rep_f2p = f2p_nbr_idxs[b, rep_cell_s]  # (R, MAX_F2P)
        # For each (local_row r, slot k), check if rep_f2p[r, k] is a parent we
        # actually have a node for (i.e. it appears in uniq_rows).
        flat_parents = rep_f2p.reshape(-1)         # (R * MAX_F2P,)
        src_local = torch.arange(R, device=device).unsqueeze(-1).expand_as(rep_f2p).reshape(-1)
        # Mark valid parents.
        candidate_mask = flat_parents >= 0
        if candidate_mask.any():
            # Map raw parent ids to local ids using searchsorted on sorted uniq_rows.
            sorted_uniq, sort_idx = torch.sort(uniq_rows)
            # searchsorted requires same dtype on both sides
            parent_clamped = flat_parents.to(torch.long).clamp_min(0)
            pos = torch.searchsorted(sorted_uniq.to(torch.long), parent_clamped)
            pos = pos.clamp_max(R - 1)
            mapped = sort_idx[pos]
            hit = (sorted_uniq[pos] == parent_clamped) & candidate_mask
            if hit.any():
                edge_src_chunks.append(src_local[hit] + cursor)
                edge_dst_chunks.append(mapped[hit] + cursor)

        # Target row + label.
        tgt_positions = torch.nonzero(is_targets[b] & valid_mask, as_tuple=True)[0]
        if tgt_positions.numel() > 0:
            tgt_s = tgt_positions[0]
            target_cell_positions[b] = tgt_s
            tgt_row_raw = node_idxs[b, tgt_s].to(torch.long)
            sorted_uniq, sort_idx = torch.sort(uniq_rows)
            pos = torch.searchsorted(sorted_uniq.to(torch.long), tgt_row_raw.unsqueeze(0))
            pos = pos.clamp_max(R - 1)
            if sorted_uniq[pos[0]] == tgt_row_raw:
                target_row_global[b] = sort_idx[pos[0]] + cursor
                target_labels[b] = (boolean_values[b, tgt_s, 0].float() > 0.0).float()
                valid_targets[b] = True
            else:
                target_row_global[b] = cursor
        else:
            target_row_global[b] = cursor

        cursor += R

    node_features = (
        torch.cat(node_feature_chunks, dim=0)
        if node_feature_chunks
        else torch.zeros(0, d_feat, device=device)
    )
    row_to_sample = (
        torch.cat(row_to_sample_chunks, dim=0)
        if row_to_sample_chunks
        else torch.zeros(0, dtype=torch.long, device=device)
    )
    if edge_src_chunks:
        fwd_src = torch.cat(edge_src_chunks)
        fwd_dst = torch.cat(edge_dst_chunks)
        # Add reverse edges with a separate relation type so messages flow
        # back from FK parents to children. Without this, "leaf" rows (such
        # as the synthetic task-stub row for forecast/classification tasks)
        # never receive aggregated messages, leaving their NBFNet output
        # identical across samples.
        edge_index = torch.stack(
            [torch.cat([fwd_src, fwd_dst]), torch.cat([fwd_dst, fwd_src])],
            dim=0,
        )
        edge_type = torch.cat(
            [
                torch.zeros(fwd_src.numel(), dtype=torch.long, device=device),
                torch.ones(fwd_dst.numel(), dtype=torch.long, device=device),
            ]
        )
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_type = torch.zeros(0, dtype=torch.long, device=device)
    num_relations = max(num_relations, 2)

    return RowGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_type=edge_type,
        num_relations=num_relations,
        target_row_global=target_row_global,
        target_labels=target_labels,
        row_to_sample=row_to_sample,
        num_rows_per_sample=num_rows_per_sample,
        target_cell_positions=target_cell_positions,
        valid_targets=valid_targets,
    )
