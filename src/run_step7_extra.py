"""Step 7 extras: Gini by development snapshot and Gini on currently-performing (cur_dq = 0) loans only.
Run after run_step7.py. Appends to results.md."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from metrics import gini

d = pd.read_parquet("data/interim/dev_scored.parquet"); o = pd.read_parquet("data/interim/oot_scored.parquet")
cd = pd.concat([pd.read_parquet(f"data/interim/{n}_clean.parquet", columns=["loan_id", "snap", "cur_dq"]) for n in ["train", "holdout"]])
co = pd.read_parquet("data/interim/oot_clean.parquet", columns=["loan_id", "snap", "cur_dq"])
d = d.merge(cd.assign(snap=pd.to_datetime(cd.snap)), on=["loan_id", "snap"]); o = o.merge(co.assign(snap=pd.to_datetime(co.snap)), on=["loan_id", "snap"])
lines = ["\n**Gini by sample (champion, calibrated)**\n", "| Sample | Loans | Gini | Gini, cur_dq = 0 only |", "|---|---|---|---|"]
for name, g in [(f"Dev {str(s)[:7]}", g) for s, g in d.groupby("snap")] + [("Dev pooled", d), ("OOT 2018-12", o)]:
    n0 = g[g.cur_dq == 0]
    lines.append(f"| {name} | {len(g):,} | {gini(g.default_12m, g.pd):.4f} | {gini(n0.default_12m, n0.pd):.4f} |")
lines.append(f"\nShare of loans with cur_dq >= 1: dev {(d.cur_dq >= 1).mean():.4%}, OOT {(o.cur_dq >= 1).mean():.4%}.\n")
t = open("results.md").read()
if "\n**Gini by sample (champion" in t: t = t[:t.index("\n**Gini by sample (champion")]
open("results.md", "w").write(t.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
