我先给你一个总的判断：这个项目的核心不是“训练一个普通表格模型”，而是把关系型数据库变成“图上的序列建模问题”。最重要的心智模型是这句:

`一行(row)是一个 node，一格(cell)是一个 token，表关系/同行/同列 变成 attention 约束。`

训练入口在 [scripts/example_pretrain.py](/mnt/nfs/home/st191486/relational-transformer/scripts/example_pretrain.py:5)，真正的训练主循环在 [rt/main.py](/mnt/nfs/home/st191486/relational-transformer/rt/main.py:66)，模型在 [rt/model.py](/mnt/nfs/home/st191486/relational-transformer/rt/model.py:104)，数据预处理和动态采样分别在 [rustler/src/pre.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/pre.rs:67) 和 [rustler/src/fly.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/fly.rs:223)。

**整体怎么工作**
1. 先下载 Relbench 数据，并把原始 parquet 放到 `~/scratch/relbench`，README 里的流程在 [README.md](/mnt/nfs/home/st191486/relational-transformer/README.md:35)。
2. Rust 预处理器读每张表和 parquet metadata，识别主键、外键、时间列，把每一行转成 `Node` 结构，并建立 `f2p`/`p2f` 图边，见 [rustler/src/common.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/common.rs:48) 和 [rustler/src/pre.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/pre.rs:399)。
3. 预处理还会产出 `text.json`、`column_index.json`、`table_info.json`、`nodes.rkyv`、`offsets.rkyv`、`p2f_adj.rkyv`，见 [rustler/src/pre.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/pre.rs:622)。
4. `rt.embed` 再把所有文本和值域里的字符串做 SentenceTransformer embedding，写成 `text_emb_*.bin`，见 [rt/embed.py](/mnt/nfs/home/st191486/relational-transformer/rt/embed.py:34)。
5. Python 侧的 `RelationalDataset` 只是个薄封装，真正的 batch 采样在 Rust `Sampler` 里完成，见 [rt/data.py](/mnt/nfs/home/st191486/relational-transformer/rt/data.py:46)。
6. `Sampler` 会从某个目标 row 出发，围绕它按关系图采样一个固定长度序列；序列里的每个位置是一格 cell，不是一整行，见 [rustler/src/fly.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/fly.rs:357)。
7. 模型把 `列名 embedding + 值 embedding/掩码 embedding` 合成 cell 表示，再跑 4 种注意力:
   - `feat`: 同一行，或本 cell 指向的主表邻居，见 [rt/model.py](/mnt/nfs/home/st191486/relational-transformer/rt/model.py:187)
   - `nbr`: 反向邻居关系，见 [rt/model.py](/mnt/nfs/home/st191486/relational-transformer/rt/model.py:188)
   - `col`: 同表同列，见 [rt/model.py](/mnt/nfs/home/st191486/relational-transformer/rt/model.py:189)
   - `full`: 全局非 padding，见 [rt/model.py](/mnt/nfs/home/st191486/relational-transformer/rt/model.py:190)
8. 最后只在被 mask 的目标 cell 上算 loss。这个项目目前只支持数值/布尔/时间目标，不支持 text masking，见 [rt/model.py](/mnt/nfs/home/st191486/relational-transformer/rt/model.py:250)。

**你应该按什么顺序读**
1. 先看 README 和 3 个实验脚本，理解“预训练、继续预训练、微调”三种运行方式，见 [README.md](/mnt/nfs/home/st191486/relational-transformer/README.md:121)、[scripts/example_pretrain.py](/mnt/nfs/home/st191486/relational-transformer/scripts/example_pretrain.py:5)、[scripts/example_contd_pretrain.py](/mnt/nfs/home/st191486/relational-transformer/scripts/example_contd_pretrain.py:5)、[scripts/example_finetune.py](/mnt/nfs/home/st191486/relational-transformer/scripts/example_finetune.py:5)。
2. 再看任务定义 [rt/tasks.py](/mnt/nfs/home/st191486/relational-transformer/rt/tasks.py:1)。这里能看出哪些任务是 forecast，哪些是 autocomplete，哪些列需要 drop 防止泄漏。
3. 再看训练入口 [rt/main.py](/mnt/nfs/home/st191486/relational-transformer/rt/main.py:66)。你会知道模型实际吃什么、什么时候 eval、怎么存 ckpt。
4. 再看数据包装 [rt/data.py](/mnt/nfs/home/st191486/relational-transformer/rt/data.py:46)。这里最关键的是它把任务描述翻译成 Rust sampler 需要的 `dataset_tuples / target_columns / columns_to_drop`。
5. 然后看 Rust 采样器 [rustler/src/fly.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/fly.rs:223)。这是最核心的“数据长什么样，为什么长这样”。
6. 最后看模型 [rt/model.py](/mnt/nfs/home/st191486/relational-transformer/rt/model.py:104)。这时你会发现它其实很干净，因为最复杂的逻辑已经提前在 sampler 里做完了。

**最值得抓住的几个关键点**
- `Node` 不是单个 cell，而是“整行的稀疏 cell 集合”，见 [rustler/src/common.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/common.rs:48)。
- 训练样本的监督信号通常只有一个目标 cell，`masks` 和 `is_targets` 都是围绕这个 cell 标出来的，见 [rustler/src/fly.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/fly.rs:492)。
- `columns_to_drop` 是为了防 leakage，尤其 autocomplete 任务会把同一时刻或同一行里不该看到的列扔掉，见 [rustler/src/fly.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/fly.rs:457)。
- sampler 最后一批会“补齐并回绕”，训练时接受这种重复样本，评估时用 `true_batch_size` 把假尾巴屏蔽掉，见 [rustler/src/fly.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/fly.rs:336) 和 [rt/main.py](/mnt/nfs/home/st191486/relational-transformer/rt/main.py:273)。
- `class_value_idxs` 目前被采出来了，但模型里没有真正用到；这是你读代码时要能识别出来的“历史遗留/未接上线的字段”。

**怎么 debug 才能彻底懂**
1. 先别上多卡，先别开 compile。调试时优先把 `compile_=False`、`num_workers=0`、`batch_size=1`、`seq_len=128`。否则 `torch.compile` 和 DataLoader 多进程会把问题藏起来。
2. 第一层只看预处理结果。盯住 `~/scratch/pre/<db>/table_info.json`、`column_index.json`、`text.json`，确认“列名索引、表 offset、文本词表”是否合理。
3. 第二层只看 sampler。最好的入口是 [scripts/visualize_rustler.py](/mnt/nfs/home/st191486/relational-transformer/scripts/visualize_rustler.py:20)，它会把一个样本的图上下文画出来，你能直接看见采样到了哪些 node、哪些边。
4. 第三层只看 Python batch。调 `RelationalDataset(...)[0]`，打印每个 key 的 shape，重点看 `node_idxs / sem_types / masks / is_targets / is_padding / f2p_nbr_idxs`。
5. 第四层只看一次 forward。把一个 batch 喂给 `RelationalTransformer.forward`，确认 4 个 attention mask 的 shape 和 `loss` 是否正常。
6. 第五层再看一次 train step。只跑 1 step，观察 `loss.backward()` 后哪些参数有梯度，尤其是 `enc_dict`、`blocks`、`dec_dict`。
7. 第六层再看 eval。重点盯 `true_batch_size`、`preds`、`labels` 的长度对不对，见 [rt/main.py](/mnt/nfs/home/st191486/relational-transformer/rt/main.py:283)。

**推荐的断点位置**
- [rt/data.py](/mnt/nfs/home/st191486/relational-transformer/rt/data.py:116)：看 sampler 返回的原始 batch 长什么样。
- [rustler/src/fly.rs](/mnt/nfs/home/st191486/relational-transformer/rustler/src/fly.rs:357)：看 seed node 是怎么扩展成序列的。
- [rt/model.py](/mnt/nfs/home/st191486/relational-transformer/rt/model.py:153)：看注意力 mask 和输入表示怎么拼起来。
- [rt/main.py](/mnt/nfs/home/st191486/relational-transformer/rt/main.py:396)：看一次完整训练 step。
- [rt/main.py](/mnt/nfs/home/st191486/relational-transformer/rt/main.py:261)：看一次 eval batch 的处理。

我顺手确认了当前环境：仓库代码本身我已经读通了，但这台机器现在没有 `pixi`，系统 Python 里也没有 `torch/numpy`，而且 `~/scratch/pre` 还没准备好，所以我这次是“静态读代码 + 环境探测”，不是把训练真正跑起来。如果你愿意，我下一步可以直接继续帮你做一份“最小单步调试脚本”，把 `dataset -> sampler -> model -> one train step` 串起来，你就能按断点一步一步看。