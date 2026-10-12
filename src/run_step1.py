"""Run Step 1 end to end. Usage: python -I src/run_step1.py  (from pd-mortgage/)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from load import load_orig, load_perf
from snapshots import add_default_event, build_snapshot

orig = load_orig()
print("orig", orig.shape)
perf = load_perf()
print("perf", perf.shape, perf["period"].min(), perf["period"].max())
perf = add_default_event(perf)

out = []
for y in range(2015, 2020):
    s, n_all = build_snapshot(perf, f"{y}-12-01", orig)
    print(f"snap {y}-12: active-at-snap rows {n_all:,}; kept {len(s):,}; defaults {s.default_12m.sum():,}; "
          f"rate {s.default_12m.mean():.4%}")
    out.append(s)
snaps = pd.concat(out, ignore_index=True)
snaps.to_parquet("data/interim/snapshots.parquet")
print(snaps.groupby("snap")["default_12m"].agg(loans="size", defaults="sum", rate="mean"))

# ---- results.md section (replaces any earlier Step 1 section)
g = snaps.groupby("snap")["default_12m"].agg(loans="size", defaults="sum", rate="mean")
sec = ("\n## Step 1: loans, defaults and default rate per snapshot\n\n(Output of `python -I src/run_step1.py`. Snapshot 2019-12 outcome window covers 2020, the COVID period.)\n\n"
       "| Snapshot | Loans | Defaults | Default rate |\n|---|---|---|---|\n"
       + "".join(f"| {k:%Y-%m} | {int(r.loans):,} | {int(r.defaults):,} | {r.rate:.4%} |\n" for k, r in g.iterrows())
       + f"\nPerformance file: {len(perf):,} rows, {perf['period'].min():%Y-%m} to {perf['period'].max():%Y-%m}. Origination: {len(orig):,} loans, {orig['vintage'].nunique()} vintages (2012-2018).\n\n")
t = open("results.md").read() if os.path.exists("results.md") else "# Results\n\nMetrics recorded exactly as printed by code.\n"
if "\n## Step 1:" in t:
    i = t.index("\n## Step 1:"); j = t.find("\n## Step 2:", i); t = t[:i] + (t[j:] if j > 0 else "")
k = t.find("\n## Step 2:")
t = (t[:k] + sec + t[k:]) if k > 0 else t.rstrip("\n") + "\n" + sec
open("results.md", "w").write(t)
