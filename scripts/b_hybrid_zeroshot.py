"""Hybrid (frozen RT + ULTRA on row graph) zero-shot transfer experiment.

Mirrors `scripts/example_pretrain.py`: this script only collects config
and dispatches to `rt.hybrid_main.hybrid_main`. All training and
evaluation logic lives in `rt/hybrid_main.py`.

The held-out DB is selected via `--heldout`; the matching leave-DB-out
RT checkpoint is loaded from `~/scratch/rt_ckpts/`.

Protocol matches the RT paper's Table 1 "No" column: target DB is never
seen during training (the RT cell encoder is leave-DB-out pretrained,
and the ULTRA-side is trained on the 10 non-held-out clf tasks).

Run:
    python scripts/b_hybrid_zeroshot.py --heldout rel-trial
    python scripts/b_hybrid_zeroshot.py --heldout rel-amazon
"""
import argparse
import os

from rt.hybrid_main import hybrid_main
from rt.tasks import forecast_clf_tasks


# Choice of RT ckpt per held-out DB. The convention is
# `pretrain_<heldout_db>_<some_task>.pt`. The task suffix is just a tag;
# the ckpt itself is leave-DB-out regardless of which task name appears
# in the filename.
RT_CKPT_BY_HELDOUT = {
    "rel-amazon": "pretrain_rel-amazon_user-churn.pt",
    "rel-hm":     "pretrain_rel-hm_user-churn.pt",
    "rel-stack":  "pretrain_rel-stack_user-badge.pt",
    "rel-avito":  "pretrain_rel-avito_user-clicks.pt",
    "rel-f1":     "pretrain_rel-f1_driver-dnf.pt",
    "rel-trial":  "pretrain_rel-trial_study-outcome.pt",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heldout", required=True, choices=list(RT_CKPT_BY_HELDOUT.keys()),
    )
    parser.add_argument("--pretrain_steps", type=int, default=2000)
    parser.add_argument("--eval_freq", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    # max_eval_steps × batch_size matches the RT paper's ~10,240-sample
    # subsample (RT example_pretrain uses batch_size=256, max_eval_steps=40).
    parser.add_argument("--max_eval_steps", type=int, default=160)
    args = parser.parse_args()

    hybrid_main(
        # misc
        project="b-hybrid-zeroshot",
        run_name=f"hybrid-leave-{args.heldout}",
        seed=args.seed,
        # data
        train_tasks=[t for t in forecast_clf_tasks if t[0] != args.heldout],
        eval_tasks=[t for t in forecast_clf_tasks if t[0] == args.heldout],
        batch_size=args.batch_size,
        num_workers=2,
        max_bfs_width=256,
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        # RT (frozen, loaded from leave-DB-out ckpt)
        rt_ckpt_path=os.path.expanduser(
            f"~/scratch/rt_ckpts/{RT_CKPT_BY_HELDOUT[args.heldout]}"
        ),
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
        lr=args.lr,
        wd=0.01,
        max_grad_norm=1.0,
        max_steps=args.pretrain_steps,
        # eval
        eval_freq=args.eval_freq,
        max_eval_steps=args.max_eval_steps,
        loss_log_freq=50,
        # checkpointing — saves best_val and best_test ckpts per eval task
        # (trainable params only, ~1.2MB each).
        save_ckpt_dir=f"ckpts/hybrid_zeroshot/leave_{args.heldout}",
    )
