"""Training entrypoint for the hybrid (frozen RT + ULTRA-side) zero-shot
transfer experiment.

Mirrors the structure of `rt/main.py`: this module owns all training and
evaluation logic; script files (`scripts/b_hybrid_zeroshot.py`) just
collect config and call `hybrid_main(...)`.

The hybrid pretrain protocol:
  - Train UltraRowHybrid (frozen RT cell encoder + 313K-param ULTRA-side)
    on `train_tasks` (a list of clf tasks from non-held-out DBs).
  - Continuously evaluate on `eval_tasks` (the held-out DB's clf tasks)
    every `eval_freq` steps as the zero-shot test.
  - Track best-val checkpoint per (db, table) pair and report
    test-at-best-val.

Matches the RT paper's Table 1 "No" column protocol: target DB is never
seen during training (the loaded RT ckpt is also leave-DB-out).
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
from rt.ultra_row_hybrid import UltraRowHybrid


def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(
    tasks, split, batch_size, seq_len, num_workers,
    embedding_model, d_text, max_bfs_width, seed,
):
    ds = RelationalDataset(
        tasks=[(db, table, target, split, drop)
               for (db, table, target, drop) in tasks],
        batch_size=batch_size,
        seq_len=seq_len,
        rank=0,
        world_size=1,
        max_bfs_width=max_bfs_width,
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


def _evaluate(net, loader, device, max_eval_steps, desc):
    net.eval()
    preds, labels = [], []
    total = (
        min(max_eval_steps, len(loader))
        if max_eval_steps > 0
        else len(loader)
    )
    pbar = tqdm(total=total, desc=desc, leave=False)
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            true_bs = batch.pop("true_batch_size")
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            batch["masks"][true_bs:, :] = False
            batch["is_targets"][true_bs:, :] = False
            batch["is_padding"][true_bs:, :] = True
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
    # The Rust sampler stores boolean labels in standardized form:
    #   stored = (raw_label - pos_rate) / sqrt(p*(1-p))
    # So class 0 maps to a negative value and class 1 to a positive value
    # for any pos_rate. The class boundary is 0, not 0.5. RT's own eval
    # code does the same (rt/main.py: `np.array([int(x > 0) for x in labels])`).
    labels = (np.concatenate(labels) > 0.0).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, preds)


def _train_step(net, opt, sched, batch, device, true_bs, max_grad_norm):
    for k in batch:
        batch[k] = batch[k].to(device, non_blocking=True)
    batch["masks"][true_bs:, :] = False
    batch["is_targets"][true_bs:, :] = False
    batch["is_padding"][true_bs:, :] = True
    net.train()
    loss, _ = net(batch)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in net.parameters() if p.requires_grad], max_grad_norm,
    )
    opt.step()
    sched.step()
    return loss.item()


def hybrid_main(
    *,
    # misc
    project,
    run_name,
    seed=0,
    # data
    train_tasks,
    eval_tasks,
    batch_size,
    num_workers=2,
    max_bfs_width=256,
    embedding_model="all-MiniLM-L12-v2",
    d_text=384,
    seq_len=1024,
    # RT (frozen, loaded from ckpt)
    rt_ckpt_path,
    rt_num_blocks=12,
    rt_d_model=256,
    rt_num_heads=8,
    rt_d_ff=1024,
    # ULTRA-side
    hidden_dim=128,
    num_layers=3,
    num_rel_layers=2,
    dropout=0.05,
    # optimization
    lr=3e-4,
    wd=0.01,
    max_grad_norm=1.0,
    max_steps=2000,
    # eval
    eval_freq=200,
    max_eval_steps=160,
    loss_log_freq=50,
):
    _seed_everything(seed)

    print(f"#train_tasks = {len(train_tasks)}")
    for t in train_tasks:
        print(f"  train: {t}")
    print(f"#eval_tasks  = {len(eval_tasks)}")
    for t in eval_tasks:
        print(f"  eval : {t}")
    print(f"RT ckpt: {rt_ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "seed": seed,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "lr": lr,
            "wd": wd,
            "eval_freq": eval_freq,
            "max_eval_steps": max_eval_steps,
            "rt_num_blocks": rt_num_blocks,
            "rt_d_model": rt_d_model,
            "rt_num_heads": rt_num_heads,
            "rt_d_ff": rt_d_ff,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_rel_layers": num_rel_layers,
            "dropout": dropout,
            "embedding_model": embedding_model,
            "d_text": d_text,
            "seq_len": seq_len,
            "rt_ckpt_path": rt_ckpt_path,
        },
    )
    wandb.define_metric("loss/*", summary="min")
    wandb.define_metric("auc/*", summary="max")
    print(f"run: {run.name}")

    train_loader = _make_loader(
        train_tasks, "train", batch_size, seq_len, num_workers,
        embedding_model, d_text, max_bfs_width, seed,
    )
    eval_loaders = {}
    for (db, table, target, drop) in eval_tasks:
        for split in ("val", "test"):
            eval_loaders[(db, table, split)] = _make_loader(
                [(db, table, target, drop)], split, batch_size, seq_len,
                num_workers, embedding_model, d_text, max_bfs_width, seed,
            )

    net = UltraRowHybrid(
        rt_num_blocks=rt_num_blocks,
        rt_d_model=rt_d_model,
        rt_d_text=d_text,
        rt_num_heads=rt_num_heads,
        rt_d_ff=rt_d_ff,
        rt_ckpt_path=rt_ckpt_path,
        freeze_rt=True,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_rel_layers=num_rel_layers,
        dropout=dropout,
    ).to(device)
    n_total = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"param_count_total={n_total:_}  trainable={n_trainable:_}")
    wandb.log(
        {"param_count_total": n_total, "param_count_trainable": n_trainable},
        step=0,
    )

    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = optim.AdamW(trainable, lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    best_by_task = {}  # (db, table) -> (best_val, test_at_best_val, step)

    def do_eval(step):
        net.eval()
        metrics = {}
        for (db, table, split), loader in eval_loaders.items():
            metrics[(db, table, split)] = _evaluate(
                net, loader, device, max_eval_steps,
                desc=f"eval {db}/{table}/{split}",
            )
        tasks_seen = sorted({(db, t) for (db, t, _) in metrics})
        for (db, t) in tasks_seen:
            val = metrics.get((db, t, "val"), float("nan"))
            test = metrics.get((db, t, "test"), float("nan"))
            print(
                f"  step={step}  {db}/{t}  zs_val={val:.4f}  zs_test={test:.4f}",
                flush=True,
            )
            wandb.log(
                {f"auc/{db}_{t}_val": val, f"auc/{db}_{t}_test": test},
                step=step,
            )
            prev = best_by_task.get((db, t), (-float("inf"), -float("inf"), -1))
            if val > prev[0]:
                best_by_task[(db, t)] = (val, test, step)
                wandb.log(
                    {
                        f"auc/{db}_{t}_best_val": val,
                        f"auc/{db}_{t}_test_at_best_val": test,
                    },
                    step=step,
                )

    print("\n" + "=" * 70)
    print(f"ZERO-SHOT PRETRAIN: {run_name}, {max_steps} steps")
    print("=" * 70)

    pbar = tqdm(total=max_steps, desc=run_name)
    step = 0
    do_eval(0)

    while step < max_steps:
        train_loader.dataset.sampler.shuffle_py(
            int(step / len(train_loader))
        )
        for batch in train_loader:
            if step >= max_steps:
                break
            true_bs = batch.pop("true_batch_size")
            loss_val = _train_step(
                net, opt, sched, batch, device, true_bs, max_grad_norm,
            )
            if step % loss_log_freq == 0:
                wandb.log(
                    {"loss/pretrain": loss_val, "lr/pretrain": sched.get_last_lr()[0]},
                    step=step,
                )
            step += 1
            pbar.update(1)
            if step % eval_freq == 0:
                do_eval(step)
    pbar.close()
    do_eval(step)

    print("\n" + "=" * 70)
    print(f"SUMMARY for {run_name}")
    print("=" * 70)
    for (db, t), (vbest, ttest, vstep) in best_by_task.items():
        print(
            f"  {db}/{t}  best_zs_val={vbest:.4f}  "
            f"zs_test_at_best={ttest:.4f}  (step {vstep})"
        )
    wandb.finish()
