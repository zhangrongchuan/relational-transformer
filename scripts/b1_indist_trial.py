"""B1 Step 1: in-distribution sanity check on rel-trial study-outcome.

Train UltraRowB1 (single-relation NBFNet on row graph) on rel-trial
study-outcome train split. Eval on val and test.

Pass criterion (see analysis.md §7): val AUC > 0.60. If it passes we move on
to Step 2 (full ULTRA relation graph). If it doesn't, the row-graph
structural signal hypothesis is wrong and we revisit the whole angle.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import RelationalDataset
from rt.ultra_row_b1 import UltraRowB1


def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(task, split, batch_size, seq_len, num_workers, embedding_model, d_text, seed):
    db, table, target, drop = task
    ds = RelationalDataset(
        tasks=[(db, table, target, split, drop)],
        batch_size=batch_size,
        seq_len=seq_len,
        rank=0,
        world_size=1,
        max_bfs_width=256,
        embedding_model=embedding_model,
        d_text=d_text,
        seed=seed,
    )
    ds.sampler.shuffle_py(0)
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        in_order=True,
    )


def evaluate(net, loader, device, max_eval_steps):
    net.eval()
    preds, labels = [], []
    pbar = tqdm(
        total=(min(max_eval_steps, len(loader)) if max_eval_steps > 0 else len(loader)),
        desc="eval",
    )
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            true_batch_size = batch.pop("true_batch_size")
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            batch["masks"][true_batch_size:, :] = False
            batch["is_targets"][true_batch_size:, :] = False
            batch["is_padding"][true_batch_size:, :] = True

            loss, yhat = net(batch)
            tgt = batch["is_targets"]
            p = yhat["boolean"][tgt].flatten().float().cpu().numpy()
            y = batch["boolean_values"][tgt].flatten().float().cpu().numpy()
            preds.append(p)
            labels.append(y)
            pbar.update(1)
            if max_eval_steps > 0 and (i + 1) >= max_eval_steps:
                break
    pbar.close()
    preds = np.concatenate(preds)
    labels = (np.concatenate(labels) > 0.0).astype(int)  # threshold 0 (not 0.5) — labels are standardized; see b_hybrid_zeroshot.py for why
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, preds)


def main():
    seed = 0
    seed_everything(seed)

    task = ("rel-trial", "study-outcome", "outcome", [])
    embedding_model = "all-MiniLM-L12-v2"
    d_text = 384
    seq_len = 1024
    batch_size = 32
    eval_batch_size = 32
    num_workers = 2

    hidden_dim = 128
    num_layers = 3
    dropout = 0.05

    lr = 3e-4
    wd = 0.01
    max_steps = 6000
    eval_freq = 300
    max_eval_steps = -1  # full val/test for trustworthy AUC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project="b1-indist-trial", config=locals())
    wandb.define_metric("auc/*", summary="max")
    wandb.define_metric("loss", summary="min")
    print(f"run: {run.name}")

    train_loader = make_loader(task, "train", batch_size, seq_len, num_workers, embedding_model, d_text, seed)
    val_loader   = make_loader(task, "val",   eval_batch_size, seq_len, num_workers, embedding_model, d_text, seed)
    test_loader  = make_loader(task, "test",  eval_batch_size, seq_len, num_workers, embedding_model, d_text, seed)

    net = UltraRowB1(
        d_text=d_text,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"param_count={n_params:_}")
    wandb.log({"param_count": n_params}, step=0)

    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    pbar = tqdm(total=max_steps, desc="train")
    best_val = -float("inf")
    best_test_at_best_val = -float("inf")
    step = 0

    def do_eval(step):
        nonlocal best_val, best_test_at_best_val
        val_auc = evaluate(net, val_loader, device, max_eval_steps)
        test_auc = evaluate(net, test_loader, device, max_eval_steps)
        print(f"step={step}  val_auc={val_auc:.4f}  test_auc={test_auc:.4f}")
        wandb.log(
            {"auc/val": val_auc, "auc/test": test_auc, "step": step},
            step=step,
        )
        if val_auc > best_val:
            best_val = val_auc
            best_test_at_best_val = test_auc
            wandb.log({"auc/best_val": best_val, "auc/test_at_best_val": best_test_at_best_val}, step=step)
        net.train()

    # Eval at step 0 (random init baseline).
    do_eval(0)

    while step < max_steps:
        train_loader.dataset.sampler.shuffle_py(int(step / len(train_loader)))
        for batch in train_loader:
            if step >= max_steps:
                break
            true_batch_size = batch.pop("true_batch_size")
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            batch["masks"][true_batch_size:, :] = False
            batch["is_targets"][true_batch_size:, :] = False
            batch["is_padding"][true_batch_size:, :] = True

            net.train()
            loss, _ = net(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % 20 == 0:
                wandb.log(
                    {"loss": loss.item(), "lr": sched.get_last_lr()[0]},
                    step=step,
                )
            step += 1
            pbar.update(1)

            if step % eval_freq == 0:
                do_eval(step)

    do_eval(step)
    pbar.close()

    print(f"\nDONE. best_val_auc={best_val:.4f}  test_at_best_val={best_test_at_best_val:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
