"""Step 12d: month-end snapshots Jan 2018 - Dec 2020 (Step 1 logic, monthly instead of yearly).
Writes scoring INPUTS (no label) and, separately, LABELS (available 12 months later) per month.
Run from pd-mortgage/ with the project venv: python azure_ml/build_monthly_snapshots.py"""
import os, sys
sys.path.insert(0, "src")
import pandas as pd
from snapshots import add_default_event, build_snapshot

os.makedirs("data/interim/monthly/inputs", exist_ok=True); os.makedirs("data/interim/monthly/labels", exist_ok=True)
orig = pd.read_parquet("data/interim/orig.parquet")
perf = add_default_event(pd.read_parquet("data/interim/perf.parquet"))
for snap in pd.date_range("2018-01-01", "2020-12-01", freq="MS"):
    s, _ = build_snapshot(perf, snap, orig)
    tag = f"{snap:%Y-%m}"
    s[["loan_id", "snap", "default_12m"]].to_parquet(f"data/interim/monthly/labels/{tag}.parquet")
    s.drop(columns=["default_12m"]).to_parquet(f"data/interim/monthly/inputs/{tag}.parquet")
    print(tag, len(s), int(s.default_12m.sum()), flush=True)
