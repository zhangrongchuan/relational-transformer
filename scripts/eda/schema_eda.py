"""EDA for Table 1 tasks: dump schema, FK graph, label table, entity→signal paths."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("RELBENCH_DATASET_DIR", os.path.expanduser("~/.cache/relbench"))

from collections import defaultdict, deque
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# Table 1 tasks with their published RT zero-shot AUROC
TASKS = [
    ("rel-amazon", "item-churn",   70.9),
    ("rel-amazon", "user-churn",   64.0),
    ("rel-avito",  "user-clicks",  59.5),
    ("rel-avito",  "user-visits",  61.8),
    ("rel-f1",     "driver-dnf",   81.2),
    ("rel-f1",     "driver-top3",  89.3),
    ("rel-hm",     "user-churn",   62.8),
    ("rel-stack",  "user-badge",   80.1),
    ("rel-stack",  "user-engagement", 75.7),
    ("rel-trial",  "study-outcome", 51.8),
]

def build_fk_graph(db):
    """Undirected schema graph: table -> list[(neighbor_table, fk_col, direction)]"""
    g = defaultdict(list)
    for name, tbl in db.table_dict.items():
        for fk_col, parent in tbl.fkey_col_to_pkey_table.items():
            g[name].append((parent, fk_col, "fwd"))
            g[parent].append((name, fk_col, "rev"))
    return g

def bfs_distances(g, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v, _, _ in g[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def schema_stats(db):
    n_tables = len(db.table_dict)
    n_fks = sum(len(t.fkey_col_to_pkey_table) for t in db.table_dict.values())
    n_rows = {n: len(t.df) for n, t in db.table_dict.items()}
    cols_per_table = {n: len(t.df.columns) for n, t in db.table_dict.items()}
    return n_tables, n_fks, n_rows, cols_per_table

# Group tasks by dataset to avoid reloading
by_ds = defaultdict(list)
for ds, t, auc in TASKS:
    by_ds[ds].append((t, auc))

result = {}
for ds_name, tasks in by_ds.items():
    print(f"\n========== {ds_name} ==========", flush=True)
    ds = get_dataset(ds_name, download=False)
    db = ds.get_db(upto_test_timestamp=False)
    n_tables, n_fks, n_rows, cols = schema_stats(db)
    g = build_fk_graph(db)
    
    print(f"#tables={n_tables}  #FK_edges(directed)={n_fks}")
    print(f"Tables (rows, #cols):")
    for nm in db.table_dict:
        print(f"   {nm:<32} rows={n_rows[nm]:>10}  cols={cols[nm]}")
    print(f"Schema FK graph:")
    seen = set()
    for u in db.table_dict:
        for v, fk, dirn in g[u]:
            key = tuple(sorted([u, v])) + (fk,)
            if dirn == "fwd" and key not in seen:
                print(f"   {u} --[{fk}]--> {v}")
                seen.add(key)

    ds_info = {
        "n_tables": n_tables, "n_fks": n_fks,
        "tables": {nm: {"rows": n_rows[nm], "cols": cols[nm]} for nm in db.table_dict},
    }
    task_infos = []
    for task_name, auc in tasks:
        try:
            t = get_task(ds_name, task_name)
            entity = t.entity_table
            dist = bfs_distances(g, entity)
            diameter = max(dist.values()) if dist else 0
            unreachable = [tn for tn in db.table_dict if tn not in dist]
            train_table = t.get_table("train", mask_input_cols=False)
            train_df = train_table.df
            target_col = t.target_col
            n_train = len(train_df)
            pos_rate = float(train_df[target_col].mean()) if target_col in train_df.columns else None
            # distinct entities sampled
            n_entities = train_df[t.entity_col].nunique()
            
            print(f"\n--- task: {task_name}  (RT zero-shot AUROC={auc}) ---")
            print(f"  entity_table = {entity}   target_col={target_col}")
            print(f"  train rows = {n_train}   pos_rate = {pos_rate:.3f}" if pos_rate is not None else "")
            print(f"  unique entities in train = {n_entities}")
            print(f"  BFS distances from {entity}:")
            for tn, d in sorted(dist.items(), key=lambda x: (x[1], x[0])):
                print(f"     hop={d}  {tn}  (rows={n_rows[tn]})")
            if unreachable:
                print(f"  UNREACHABLE tables (no FK path): {unreachable}")
            print(f"  schema diameter from entity = {diameter}")
            
            task_infos.append({
                "task": task_name, "auc": auc, "entity": entity,
                "target_col": target_col, "n_train": n_train,
                "pos_rate": pos_rate, "n_entities": n_entities,
                "distances": dist, "diameter": diameter,
                "unreachable": unreachable,
            })
        except Exception as e:
            print(f"ERR for {task_name}: {e}")
            task_infos.append({"task": task_name, "auc": auc, "error": str(e)})
    
    ds_info["tasks"] = task_infos
    result[ds_name] = ds_info

# Save summary
out_path = "/tmp/eda/schema_summary.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n\nSaved JSON to {out_path}")
