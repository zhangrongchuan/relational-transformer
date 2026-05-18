"""Row graph builder for B1 Step 2.

Extends Step 1 with:
- Per-edge relation id keyed on (child_table_idx, parent_table_idx). Each
  unique (child_table, parent_table) pair gets a batch-local relation id;
  fwd and rev FK directions get distinct relations.
- A relation graph: nodes are the batch's relation ids, edges connect
  relations that share an endpoint (HH/HT/TH/TT). This is consumed by the
  RelNBFNet to produce inductive relation embeddings.
- The "query relation" per batch: the relation id of the first FK edge
  outgoing from any target row. For a forecast/clf task this is the FK
  from the task-stub row to the entity row (e.g. study-outcome -> studies).

For zero-shot transfer at Step 3, the per-(child,parent) relation ids will
be new at test time. ULTRA's claim is that the relation graph topology
gives them meaningful embeddings inductively — that is the whole point of
adding this graph at Step 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


# Relation-graph edge types
REL_EDGE_HH = 0
REL_EDGE_HT = 1
REL_EDGE_TH = 2
REL_EDGE_TT = 3
NUM_REL_EDGE_TYPES = 4


@dataclass
class RowGraphB2:
    node_features: torch.Tensor          # (N, d_feat)
    edge_index: torch.Tensor             # (2, E)
    edge_rel_id: torch.Tensor            # (E,) batch-local relation id, in [0, num_relations)
    num_relations: int

    rel_graph_edge_index: torch.Tensor   # (2, RE) edges among relations
    rel_graph_edge_type: torch.Tensor    # (RE,) values in {0,1,2,3}

    query_rel_id: torch.Tensor           # (B,) batch-local relation id of each sample's "task" FK
    valid_query: torch.Tensor            # (B,) whether the query rel was identified

    target_row_global: torch.Tensor      # (B,)
    target_cell_positions: torch.Tensor  # (B,)
    target_labels: torch.Tensor          # (B,)
    valid_targets: torch.Tensor          # (B,)

    num_rows_per_sample: torch.Tensor    # (B,)


def _build_rel_graph_edges(entity_edge_index, edge_rel_id, num_relations, device):
    """Build the (HH/HT/TH/TT) relation graph from the entity graph.

    Two relations r1, r2 are connected at a node n if both have an edge
    incident at n. The 4 types differentiate which side (head/src vs tail/dst)
    each relation has at n.
    """
    if entity_edge_index.numel() == 0 or num_relations == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    src, dst = entity_edge_index
    head_by_node: Dict[int, set] = {}
    tail_by_node: Dict[int, set] = {}
    s_list = src.tolist()
    d_list = dst.tolist()
    r_list = edge_rel_id.tolist()
    for s, d, r in zip(s_list, d_list, r_list):
        head_by_node.setdefault(s, set()).add(r)
        tail_by_node.setdefault(d, set()).add(r)

    rel_edge_set = set()  # set of (r1, r2, edge_type)
    all_nodes = set(head_by_node.keys()) | set(tail_by_node.keys())
    for n in all_nodes:
        H = head_by_node.get(n, set())
        T = tail_by_node.get(n, set())
        for r1 in H:
            for r2 in H:
                if r1 != r2:
                    rel_edge_set.add((r1, r2, REL_EDGE_HH))
            for r2 in T:
                rel_edge_set.add((r1, r2, REL_EDGE_HT))
        for r1 in T:
            for r2 in H:
                rel_edge_set.add((r1, r2, REL_EDGE_TH))
            for r2 in T:
                if r1 != r2:
                    rel_edge_set.add((r1, r2, REL_EDGE_TT))

    if not rel_edge_set:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    pairs = sorted(rel_edge_set)
    src_arr = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    dst_arr = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    type_arr = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=device)
    return torch.stack([src_arr, dst_arr], dim=0), type_arr


def build_row_graph_b2(batch, cell_feat=None) -> RowGraphB2:
    """If `cell_feat` is provided ((B, S, D)), it's used as the per-cell feature
    to pool into row representations. Otherwise we fall back to the simple
    `concat(col_name_values, text_values)` used by Steps 1/2/3."""
    node_idxs = batch["node_idxs"]            # (B, S) int32
    f2p_nbr_idxs = batch["f2p_nbr_idxs"]      # (B, S, 5) int32
    table_name_idxs = batch["table_name_idxs"]  # (B, S) int32
    is_padding = batch["is_padding"].bool()
    is_targets = batch["is_targets"].bool()
    col_emb = batch["col_name_values"]
    txt_emb = batch["text_values"]
    boolean_values = batch["boolean_values"]

    B, S = node_idxs.shape
    device = node_idxs.device
    if cell_feat is None:
        cell_feat = torch.cat([col_emb, txt_emb], dim=-1).float()
    else:
        cell_feat = cell_feat.float() if cell_feat.dtype != torch.float32 else cell_feat
    d_feat = cell_feat.shape[-1]

    node_feature_chunks = []

    # Edge accumulators: separate src/dst lists and the (child_table, parent_table)
    # key per fwd edge. We will assign batch-local relation ids only after we
    # see all edges.
    fwd_src_chunks = []
    fwd_dst_chunks = []
    fwd_child_tbl_chunks = []
    fwd_parent_tbl_chunks = []

    # Map row global id -> its table_idx (one per row). Built incrementally as
    # we add rows. We pick the first cell's table_name_idx as the row's table.
    row_table_idx_chunks = []

    target_row_global = torch.zeros(B, dtype=torch.long, device=device)
    target_labels = torch.zeros(B, dtype=torch.float32, device=device)
    target_cell_positions = torch.zeros(B, dtype=torch.long, device=device)
    valid_targets = torch.zeros(B, dtype=torch.bool, device=device)
    num_rows_per_sample = torch.zeros(B, dtype=torch.long, device=device)

    cursor = 0
    for b in range(B):
        valid_mask = ~is_padding[b]
        valid_positions = torch.nonzero(valid_mask, as_tuple=True)[0]
        Sv = valid_positions.numel()
        if Sv == 0:
            target_row_global[b] = cursor
            continue

        rows_b = node_idxs[b, valid_positions]
        feats_b = cell_feat[b, valid_positions]
        table_b = table_name_idxs[b, valid_positions].long()

        uniq_rows, inv = torch.unique(rows_b, return_inverse=True)
        R = uniq_rows.numel()

        pooled = torch.zeros(R, d_feat, device=device, dtype=feats_b.dtype)
        pooled.index_add_(0, inv, feats_b)
        counts = torch.zeros(R, device=device, dtype=feats_b.dtype)
        counts.index_add_(0, inv, torch.ones_like(inv, dtype=feats_b.dtype))
        pooled = pooled / counts.clamp_min(1).unsqueeze(-1)

        node_feature_chunks.append(pooled)
        num_rows_per_sample[b] = R

        # For each local row r, find the first cell position and the table idx.
        first_cell_position = torch.full((R,), Sv, dtype=torch.long, device=device)
        rng = torch.arange(Sv, device=device)
        first_cell_position.scatter_reduce_(0, inv, rng, reduce="amin", include_self=True)
        rep_cell_s_in_valid = first_cell_position
        rep_table_per_row = table_b[rep_cell_s_in_valid]  # (R,) the table_idx of each local row
        row_table_idx_chunks.append(rep_table_per_row)

        rep_cell_s = valid_positions[first_cell_position]
        rep_f2p = f2p_nbr_idxs[b, rep_cell_s]  # (R, MAX_F2P)
        flat_parents = rep_f2p.reshape(-1)
        src_local = torch.arange(R, device=device).unsqueeze(-1).expand_as(rep_f2p).reshape(-1)
        candidate_mask = flat_parents >= 0

        if candidate_mask.any():
            sorted_uniq, sort_idx = torch.sort(uniq_rows)
            parent_clamped = flat_parents.to(torch.long).clamp_min(0)
            pos = torch.searchsorted(sorted_uniq.to(torch.long), parent_clamped)
            pos = pos.clamp_max(R - 1)
            mapped = sort_idx[pos]
            hit = (sorted_uniq[pos] == parent_clamped) & candidate_mask
            if hit.any():
                src_hit = src_local[hit] + cursor   # child row global id
                dst_hit = mapped[hit] + cursor      # parent row global id
                child_table_hit = rep_table_per_row[src_local[hit]]
                parent_table_hit = rep_table_per_row[mapped[hit]]
                fwd_src_chunks.append(src_hit)
                fwd_dst_chunks.append(dst_hit)
                fwd_child_tbl_chunks.append(child_table_hit)
                fwd_parent_tbl_chunks.append(parent_table_hit)

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
    row_table_idx = (
        torch.cat(row_table_idx_chunks, dim=0)
        if row_table_idx_chunks
        else torch.zeros(0, dtype=torch.long, device=device)
    )

    # Now assemble the relation ids. For each fwd edge we have
    # (child_table, parent_table). Encode it and a separate (rev) pair so
    # forward and reverse FK get distinct relations. Then take unique ->
    # batch-local relation ids.
    if fwd_src_chunks:
        fwd_src = torch.cat(fwd_src_chunks)
        fwd_dst = torch.cat(fwd_dst_chunks)
        fwd_child_tbl = torch.cat(fwd_child_tbl_chunks)
        fwd_parent_tbl = torch.cat(fwd_parent_tbl_chunks)

        # Encode (direction, child_table, parent_table) as one int64 key.
        max_tbl = int(max(fwd_child_tbl.max().item(), fwd_parent_tbl.max().item())) + 1
        # fwd direction (0): key = 0 * 2 * max_tbl^2 + child * max_tbl + parent
        # rev direction (1): key = 1 * 2 * max_tbl^2 + parent * max_tbl + child
        # We multiply by 2 to keep fwd/rev disjoint cleanly.
        base = max_tbl * max_tbl
        fwd_keys = fwd_child_tbl * max_tbl + fwd_parent_tbl
        rev_keys = fwd_parent_tbl * max_tbl + fwd_child_tbl + base
        all_keys = torch.cat([fwd_keys, rev_keys])
        unique_keys, edge_rel_id = torch.unique(all_keys, return_inverse=True)
        num_relations = int(unique_keys.numel())

        edge_index = torch.stack(
            [torch.cat([fwd_src, fwd_dst]), torch.cat([fwd_dst, fwd_src])], dim=0
        )
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_rel_id = torch.zeros(0, dtype=torch.long, device=device)
        num_relations = 0

    rel_graph_edge_index, rel_graph_edge_type = _build_rel_graph_edges(
        edge_index, edge_rel_id, num_relations, device
    )

    # Identify the query relation per sample: the first outgoing edge of the
    # target row in the entity graph. For forecast/clf this is "task_row ->
    # entity_row" — a unique relation per task type.
    query_rel_id = torch.zeros(B, dtype=torch.long, device=device)
    valid_query = torch.zeros(B, dtype=torch.bool, device=device)
    if edge_index.size(1) > 0:
        src_all = edge_index[0]
        for b in range(B):
            if not valid_targets[b]:
                continue
            t = target_row_global[b]
            matches = (src_all == t).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                query_rel_id[b] = edge_rel_id[matches[0]]
                valid_query[b] = True

    return RowGraphB2(
        node_features=node_features,
        edge_index=edge_index,
        edge_rel_id=edge_rel_id,
        num_relations=num_relations,
        rel_graph_edge_index=rel_graph_edge_index,
        rel_graph_edge_type=rel_graph_edge_type,
        query_rel_id=query_rel_id,
        valid_query=valid_query,
        target_row_global=target_row_global,
        target_cell_positions=target_cell_positions,
        target_labels=target_labels,
        valid_targets=valid_targets,
        num_rows_per_sample=num_rows_per_sample,
    )
