from rt.main import main
from rt.tasks import all_tasks, forecast_tasks


if __name__ == "__main__":
    main(
        # misc
        project="dbrt-leave-rel-amazon",
        eval_splits=["val", "test"],
        eval_freq=1_000,
        eval_pow2=True,
        max_eval_steps=40,
        load_ckpt_path=(
            "/mnt/nfs/home/st191486/relational-transformer/"
            "pretrain_ckpts/leave_rel-amazon/steps=50000.pt"
        ),
        save_ckpt_dir="ckpts/dbrt_leave_rel-amazon",
        compile_=False,
        seed=0,
        # data
        train_tasks=[t for t in all_tasks if t[0] != "rel-amazon"],
        eval_tasks=[t for t in forecast_tasks if t[0] == "rel-amazon"],
        batch_size=1,
        num_workers=2,
        max_bfs_width=256,
        # optimization
        lr=1e-3,
        wd=0.1,
        lr_schedule=True,
        max_grad_norm=1.0,
        max_steps=50_001,
        # RT model
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        # DBRT token-RelNBFNet branch
        use_token_rel=True,
        token_rel_num_layers=4,
        token_rel_hidden_dim=256,
        same_col_max_neighbors=32,
        token_rel_aux_weight=0.1,
        token_rel_gate_lr=1e-4,
        freeze_rt=True,
    )
