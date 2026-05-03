import torch


SAME_ROW = 0
SAME_COL = 1
PKFK = 2
NUM_TOKEN_RELATIONS = 3


def _load_data_cls():
    try:
        from torch_geometric.data import Data
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_geometric is required to build ULTRA token graphs. "
            "Run `pixi install` after adding torch-geometric to pyproject.toml."
        ) from exc
    return Data


def _append_bidirectional_edges(src, dst, rel, edge_src, edge_dst, edge_type):
    if src.numel() == 0:
        return
    edge_src.extend([src, dst])
    edge_dst.extend([dst, src])
    edge_type.extend(
        [
            torch.full_like(src, rel, dtype=torch.long),
            torch.full_like(src, rel, dtype=torch.long),
        ]
    )


def _append_group_edges(
    positions,
    keys,
    rel,
    edge_src,
    edge_dst,
    edge_type,
    max_neighbors=None,
):
    if positions.numel() < 2:
        return

    for key in torch.unique(keys):
        group = positions[keys == key].sort().values
        num_nodes = group.numel()
        if num_nodes < 2:
            continue

        if max_neighbors is None or num_nodes <= max_neighbors + 1:
            left, right = torch.triu_indices(
                num_nodes, num_nodes, offset=1, device=group.device
            )
            _append_bidirectional_edges(
                group[left], group[right], rel, edge_src, edge_dst, edge_type
            )
            continue

        if max_neighbors <= 0:
            continue

        # Connect a bounded circular neighborhood. Edges are stored in both
        # directions with the same relation id, so half as many shifts gives
        # each token at most max_neighbors same-column neighbors.
        num_shifts = max(1, max_neighbors // 2)
        base = torch.arange(num_nodes, device=group.device)
        for shift in range(1, num_shifts + 1):
            _append_bidirectional_edges(
                group,
                group[(base + shift) % num_nodes],
                rel,
                edge_src,
                edge_dst,
                edge_type,
            )


def _deduplicate_edges(edge_src, edge_dst, edge_type, num_nodes, device):
    if not edge_src:
        empty_index = torch.empty(2, 0, dtype=torch.long, device=device)
        empty_type = torch.empty(0, dtype=torch.long, device=device)
        return empty_index, empty_type

    src = torch.cat(edge_src).long()
    dst = torch.cat(edge_dst).long()
    typ = torch.cat(edge_type).long()

    key = (src * num_nodes + dst) * NUM_TOKEN_RELATIONS + typ
    order = key.argsort()
    sorted_key = key[order]
    keep = torch.ones_like(sorted_key, dtype=torch.bool)
    keep[1:] = sorted_key[1:] != sorted_key[:-1]
    index = order[keep]

    edge_index = torch.stack([src[index], dst[index]], dim=0)
    edge_type = typ[index]
    return edge_index, edge_type


def build_ultra_token_graph(batch, same_col_max_neighbors=32):
    """Build a PyG token graph from an RT batch.

    The Rust sampler remains the only sampler. This adapter treats every RT
    cell token as one graph node. It stores semantically undirected edges in
    both directions, using three schema-invariant relation ids:

    0 same-row, 1 same-column, 2 primary/foreign-key row link.
    """
    Data = _load_data_cls()

    node_idxs = batch["node_idxs"]
    table_name_idxs = batch["table_name_idxs"]
    col_name_idxs = batch["col_name_idxs"]
    f2p_nbr_idxs = batch["f2p_nbr_idxs"]
    masks = batch["masks"].bool()
    is_padding = batch["is_padding"].bool()

    if node_idxs.ndim != 2:
        raise ValueError(
            f"Expected node_idxs to have shape [B, S], got {node_idxs.shape}."
        )
    if f2p_nbr_idxs.ndim != 3:
        raise ValueError(
            f"Expected f2p_nbr_idxs to have shape [B, S, K], got {f2p_nbr_idxs.shape}."
        )

    batch_size, seq_len = node_idxs.shape
    num_graph_nodes = batch_size * seq_len
    device = node_idxs.device

    edge_src = []
    edge_dst = []
    edge_type = []

    token_offsets = torch.arange(batch_size, device=device, dtype=torch.long) * seq_len
    local_positions = torch.arange(seq_len, device=device, dtype=torch.long)

    for batch_idx in range(batch_size):
        valid = ~is_padding[batch_idx]
        positions = local_positions[valid]
        if positions.numel() == 0:
            continue

        offset = token_offsets[batch_idx]
        global_positions = positions + offset

        row_keys = node_idxs[batch_idx, positions].long()
        _append_group_edges(
            global_positions,
            row_keys,
            SAME_ROW,
            edge_src,
            edge_dst,
            edge_type,
        )

        table_keys = table_name_idxs[batch_idx, positions].long()
        col_keys = col_name_idxs[batch_idx, positions].long()
        col_key_base = col_keys.max().clamp_min(0) + 1
        column_keys = table_keys * col_key_base + col_keys
        _append_group_edges(
            global_positions,
            column_keys,
            SAME_COL,
            edge_src,
            edge_dst,
            edge_type,
            max_neighbors=same_col_max_neighbors,
        )

        nbrs = f2p_nbr_idxs[batch_idx].long()
        node_ids = node_idxs[batch_idx].long()
        pkfk_mask = (
            (node_ids[None, :, None] == nbrs[:, None, :])
            & (nbrs[:, None, :] >= 0)
        ).any(-1)
        pkfk_mask = pkfk_mask & valid[:, None] & valid[None, :]
        pkfk_mask = pkfk_mask & ~torch.eye(seq_len, dtype=torch.bool, device=device)

        src_local, dst_local = pkfk_mask.nonzero(as_tuple=True)
        _append_bidirectional_edges(
            src_local.long() + offset,
            dst_local.long() + offset,
            PKFK,
            edge_src,
            edge_dst,
            edge_type,
        )

    edge_index, edge_type = _deduplicate_edges(
        edge_src, edge_dst, edge_type, num_graph_nodes, device
    )
    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_graph_nodes)
    data.num_relations = NUM_TOKEN_RELATIONS
    data.target_token_ids = masks.flatten().nonzero(as_tuple=False).flatten().long()
    data.batch_size = batch_size
    data.seq_len = seq_len
    data.edge_counts = torch.stack(
        [(edge_type == relation_id).sum() for relation_id in range(NUM_TOKEN_RELATIONS)]
    )

    return data
