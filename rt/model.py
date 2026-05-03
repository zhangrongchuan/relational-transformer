import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from ml_dtypes import bfloat16
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from rt.ultra_graph import build_ultra_token_graph

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)


class MaskedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, block_mask):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if block_mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        else:
            x = flex_attention(q, k, v, block_mask=block_mask)

        x = rearrange(x, "b h s d -> b s (h d)")
        x = self.wo(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
    ):
        super().__init__()

        self.norms = nn.ModuleDict(
            {l: nn.RMSNorm(d_model) for l in ["feat", "nbr", "col", "full", "ffn"]}
        )
        self.attns = nn.ModuleDict(
            {
                l: MaskedAttention(d_model, num_heads)
                for l in ["feat", "nbr", "col", "full"]
            }
        )
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, block_masks):
        for l in ["col", "feat", "nbr", "full"]:
            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])
        x = x + self.ffn(self.norms["ffn"](x))
        return x


def _make_block_mask(mask, batch_size, seq_len, device):
    def _mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod=_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )


class RelationalTransformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        d_text,
        num_heads,
        d_ff,
        use_token_rel=False,
        token_rel_num_layers=4,
        token_rel_hidden_dim=256,
        same_col_max_neighbors=32,
        token_rel_aux_weight=0.0,
    ):
        super().__init__()

        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, bias=True),
                "text": nn.Linear(d_text, d_model, bias=True),
                "datetime": nn.Linear(1, d_model, bias=True),
                "col_name": nn.Linear(d_text, d_model, bias=True),
                "boolean": nn.Linear(1, d_model, bias=True),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, bias=True),
                "text": nn.Linear(d_model, d_text, bias=True),
                "datetime": nn.Linear(d_model, 1, bias=True),
                "boolean": nn.Linear(d_model, 1, bias=True),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {
                "number": nn.RMSNorm(d_model),
                "text": nn.RMSNorm(d_model),
                "datetime": nn.RMSNorm(d_model),
                "col_name": nn.RMSNorm(d_model),
                "boolean": nn.RMSNorm(d_model),
            }
        )
        self.mask_embs = nn.ParameterDict(
            {
                t: nn.Parameter(torch.randn(d_model))
                for t in ["number", "text", "datetime", "boolean"]
            }
        )
        self.blocks = nn.ModuleList(
            [RelationalBlock(d_model, num_heads, d_ff) for i in range(num_blocks)]
        )
        self.norm_out = nn.RMSNorm(d_model)
        self.d_model = d_model
        self.use_token_rel = use_token_rel
        self.same_col_max_neighbors = same_col_max_neighbors
        self.token_rel_aux_weight = token_rel_aux_weight

        if use_token_rel:
            from rt.token_rel_nbfnet import TokenRelNBFNetBranch

            self.token_rel_branch = TokenRelNBFNetBranch(
                output_dim=d_model,
                hidden_dim=token_rel_hidden_dim,
                num_layers=token_rel_num_layers,
            )
            self.token_rel_gate = nn.Linear(d_model, 1)
            nn.init.zeros_(self.token_rel_gate.weight)
            nn.init.constant_(self.token_rel_gate.bias, -4.0)
        else:
            self.token_rel_branch = None
            self.token_rel_gate = None

    def keep_token_rel_float(self):
        if self.token_rel_branch is not None:
            self.token_rel_branch.float()
        return self

    def freeze_rt_parameters(self):
        if self.token_rel_branch is None:
            raise ValueError("freeze_rt_parameters requires use_token_rel=True.")
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.token_rel_branch.parameters():
            param.requires_grad_(True)
        for param in self.token_rel_gate.parameters():
            param.requires_grad_(True)
        return self

    def _apply_token_rel_branch(self, batch, x):
        token_graph = build_ultra_token_graph(
            batch,
            same_col_max_neighbors=self.same_col_max_neighbors,
        )
        h_struct = self.token_rel_branch(token_graph).to(dtype=x.dtype)
        gate = torch.sigmoid(self.token_rel_gate(x))
        return x + gate * h_struct, h_struct, gate

    def _prediction_loss(self, x, batch, return_yhat):
        loss_out = x.new_zeros(())
        yhat_out = (
            {"number": None, "text": None, "datetime": None, "boolean": None}
            if return_yhat
            else None
        )

        sem_types = batch["sem_types"]  # (B,S) ints 0..3
        masks = batch["masks"].bool()  # (B,S) where to train

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            yhat = self.dec_dict[t](x)  # (B,S, D_t)
            y = batch[f"{t}_values"]  # (B,S, D_y)
            sem_type_mask = (sem_types == i) & masks  # (B,S) mask for this type

            if not sem_type_mask.any():
                if return_yhat:
                    # still touch the param to avoid unused param error
                    loss_out = loss_out + (yhat.sum() * 0.0)
                    yhat_out[t] = yhat
                continue

            if t in ("number", "datetime"):
                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)
            elif t == "boolean":
                loss_t = F.binary_cross_entropy_with_logits(
                    yhat, (y > 0).float(), reduction="none"
                ).mean(-1)
            elif t == "text":
                raise ValueError("masking text not supported")

            loss_out = loss_out + (loss_t * sem_type_mask).sum()

            if return_yhat:
                yhat_out[t] = yhat

        loss_out = loss_out / masks.sum()
        return loss_out, yhat_out

    def forward(self, batch):
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]

        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device

        # Padding mask for attention pairs (allow only non-pad -> non-pad)
        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])  # (B, S, S)

        # cells in the same node
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]  # (B, S, S)

        # kv index is among q's foreign -> primary neighbors
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(
            -1
        )  # (B, S, S)

        # q index is among kv's primary -> foreign neighbors (reverse relation)
        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(
            -1
        )  # (B, S, S)

        # Same column AND same table
        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )  # (B, S, S)

        # Final boolean masks (apply padding once here)
        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr": q_in_f2p & pad,
            "col": same_col_table & pad,
            "full": pad,
        }

        # Make them contiguous for better kernel performance
        for l in attn_masks:
            attn_masks[l] = attn_masks[l].contiguous()

        # Convert to block masks
        make_block_mask = partial(
            _make_block_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        block_masks = {
            l: make_block_mask(attn_mask) for l, attn_mask in attn_masks.items()
        }

        x = 0
        x = x + (
            self.norm_dict["col_name"](
                self.enc_dict["col_name"](batch["col_name_values"])
            )
            * (~is_padding)[..., None]
        )

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            x = x + (
                self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~batch["masks"] & ~is_padding)[..., None]
            )
            x = x + (
                self.mask_embs[t]
                * ((batch["sem_types"] == i) & batch["masks"] & ~is_padding)[..., None]
            )

        for i, block in enumerate(self.blocks):
            x = block(x, block_masks)

        x = self.norm_out(x)
        h_struct = None
        gate = None
        if self.use_token_rel:
            x, h_struct, gate = self._apply_token_rel_branch(batch, x)

        loss_out, yhat_out = self._prediction_loss(x, batch, return_yhat=True)
        if gate is not None:
            yhat_out["_token_rel_gate_mean"] = gate.detach().mean()

        if (
            self.training
            and h_struct is not None
            and self.token_rel_aux_weight > 0.0
        ):
            main_loss = loss_out
            aux_loss, _ = self._prediction_loss(
                h_struct,
                batch,
                return_yhat=False,
            )
            loss_out = main_loss + self.token_rel_aux_weight * aux_loss
            yhat_out["_loss_main"] = main_loss.detach()
            yhat_out["_loss_aux"] = aux_loss.detach()

        return loss_out, yhat_out
