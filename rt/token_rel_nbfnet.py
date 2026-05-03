from pathlib import Path
import sys

import torch
from torch import nn

from rt.ultra_graph import NUM_TOKEN_RELATIONS


ULTRA_ROOT = Path(__file__).resolve().parents[1] / "ultra"
if ULTRA_ROOT.exists() and str(ULTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRA_ROOT))

from ultra.models import RelNBFNet  # noqa: E402


class TokenRelNBFNetBranch(nn.Module):
    """Run ULTRA's RelNBFNet directly on the sampled token graph.

    In this branch, RelNBFNet nodes are RT cell tokens, and its relation types
    are the three token graph edges: same-row, same-column, and pkfk. The first
    implementation intentionally supports only B=1 with one masked query token.
    """

    def __init__(
        self,
        output_dim,
        hidden_dim=256,
        num_layers=4,
        num_relations=NUM_TOKEN_RELATIONS,
        message_func="distmult",
        aggregate_func="sum",
        short_cut=True,
        layer_norm=True,
        use_rspmm=False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.use_rspmm = use_rspmm

        self.rel_nbfnet = RelNBFNet(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim] * num_layers,
            num_relation=num_relations,
            message_func=message_func,
            aggregate_func=aggregate_func,
            short_cut=short_cut,
            layer_norm=layer_norm,
            concat_hidden=False,
        )
        if hidden_dim == output_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        batch_size = int(data.batch_size)
        seq_len = int(data.seq_len)
        target_token_ids = data.target_token_ids

        if batch_size != 1:
            raise ValueError(
                "TokenRelNBFNetBranch currently supports only batch_size=1. "
                f"Got batch_size={batch_size}."
            )
        if target_token_ids.numel() != 1:
            raise ValueError(
                "TokenRelNBFNetBranch currently supports exactly one masked "
                f"query token. Got {target_token_ids.numel()} query tokens."
            )

        if data.edge_index.numel() == 0:
            return self._empty_output(seq_len, target_token_ids.device)

        query = target_token_ids.to(device=data.edge_index.device, dtype=torch.long)
        if self.use_rspmm:
            token_features = self.rel_nbfnet(data, query=query)
        else:
            token_features = self._bellmanford_pyg(data, h_index=query)["node_feature"]

        if token_features.shape[:2] != (1, data.num_nodes):
            raise ValueError(
                "Unexpected RelNBFNet output shape. Expected "
                f"[1, {data.num_nodes}, hidden], got {tuple(token_features.shape)}."
            )

        token_features = self.output_proj(token_features.squeeze(0))
        return token_features.view(batch_size, seq_len, self.output_dim)

    def _bellmanford_pyg(self, data, h_index):
        batch_size = len(h_index)
        param = next(self.rel_nbfnet.parameters())
        query = torch.ones(
            batch_size,
            self.rel_nbfnet.dims[0],
            device=h_index.device,
            dtype=param.dtype,
        )
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(
            batch_size,
            data.num_nodes,
            self.rel_nbfnet.dims[0],
            device=h_index.device,
            dtype=param.dtype,
        )
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(
            data.num_edges,
            device=h_index.device,
            dtype=param.dtype,
        )

        hiddens = []
        edge_weights = []
        layer_input = boundary
        for layer in self.rel_nbfnet.layers:
            # Requiring gradients on edge weights forces ULTRA's layer to use
            # standard PyG message passing instead of compiling the rspmm
            # extension. The edge weights themselves are not learned.
            layer_edge_weight = edge_weight.clone().requires_grad_()
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                layer_edge_weight,
            )
            if self.rel_nbfnet.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(layer_edge_weight)
            layer_input = hidden

        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.rel_nbfnet.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.rel_nbfnet.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def _empty_output(self, seq_len, device):
        param = next(self.parameters())
        return torch.zeros(
            1,
            seq_len,
            self.output_dim,
            device=device,
            dtype=param.dtype,
        )
