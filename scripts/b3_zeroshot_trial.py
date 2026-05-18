"""B1 Step 3: zero-shot transfer to rel-trial study-outcome.

The critical thesis experiment. Train UltraRowB3 on the 11 classification
tasks from the other 6 DBs (everything except rel-trial). Evaluate zero-shot
on rel-trial study-outcome.

Pass criterion (analysis.md §7): test AUC > 0.55 (the published RT baseline
on rel-trial study-outcome is 51.8 ≈ random; >0.55 is "meaningful gain").
Stretch: > 0.60.

NOTE: This is the actual thesis-defining experiment. Step 1 / Step 2 only
verified that the row-graph + ULTRA mechanism does not have implementation
bugs and can fit rel-trial *in-distribution*. Step 3 is the only experiment
whose result we cannot predict from Steps 1-2.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import RelationalDataset
from rt.tasks import forecast_clf_tasks
from rt.ultra_row_b3 import UltraRowB3


def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(tasks, split, batch_size, seq_len, num_workers, embedding_model, d_text, seed):
    """tasks: list of (db, table, target, drop) tuples."""
    ds = RelationalDataset(
        tasks=[(db, table, target, split, drop) for (db, table, target, drop) in tasks],
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

            _, yhat = net(batch)
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

    # Training: all clf tasks except rel-trial.
    train_tasks = [t for t in forecast_clf_tasks if t[0] != "rel-trial"]
    # Eval: rel-trial study-outcome only.
    eval_tasks = [t for t in forecast_clf_tasks if t == ("rel-trial", "study-outcome", "outcome", [])]

    print(f"#train_tasks = {len(train_tasks)}")
    for t in train_tasks:
        print(f"  train: {t}")
    print(f"#eval_tasks = {len(eval_tasks)}")
    for t in eval_tasks:
        print(f"  eval: {t}")

    embedding_model = "all-MiniLM-L12-v2"
    d_text = 384
    seq_len = 1024
    batch_size = 32
    eval_batch_size = 32
    num_workers = 2

    hidden_dim = 128
    num_layers = 3
    num_rel_layers = 2
    dropout = 0.05

    lr = 3e-4
    wd = 0.01
    max_steps = 20000  # more steps because more data
    eval_freq = 500
    max_eval_steps = -1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(project="b3-zeroshot-trial", config=locals())
    wandb.define_metric("auc/*", summary="max")
    wandb.define_metric("loss", summary="min")
    print(f"run: {run.name}")

    train_loader = make_loader(train_tasks, "train", batch_size, seq_len, num_workers, embedding_model, d_text, seed)
    # rel-trial study-outcome val + test for zero-shot eval.
    val_loader = make_loader(eval_tasks, "val", eval_batch_size, seq_len, num_workers, embedding_model, d_text, seed)
    test_loader = make_loader(eval_tasks, "test", eval_batch_size, seq_len, num_workers, embedding_model, d_text, seed)

    net = UltraRowB3(
        d_text=d_text,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_rel_layers=num_rel_layers,
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
        print(f"step={step}  zs_val_auc={val_auc:.4f}  zs_test_auc={test_auc:.4f}", flush=True)
        wandb.log({"auc/val": val_auc, "auc/test": test_auc, "step": step}, step=step)
        if val_auc > best_val:
            best_val = val_auc
            best_test_at_best_val = test_auc
            wandb.log({"auc/best_val": best_val, "auc/test_at_best_val": best_test_at_best_val}, step=step)
        net.train()

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

            if step % 50 == 0:
                wandb.log({"loss": loss.item(), "lr": sched.get_last_lr()[0]}, step=step)
            step += 1
            pbar.update(1)

            if step % eval_freq == 0:
                do_eval(step)

    do_eval(step)
    pbar.close()
    print(f"\nDONE. best_zs_val_auc={best_val:.4f}  zs_test_at_best_val={best_test_at_best_val:.4f}")
    print(f"RT zero-shot baseline (from paper Table 1): 51.8")
    wandb.finish()


if __name__ == "__main__":
    main()
