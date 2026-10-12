"""Step 2: exclusions waterfall and development / OOT / monitoring split.
Usage (from pd-mortgage/): python -I src/run_step2.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from snapshots import add_default_event

SEED = 42
DEV = ["2015-12-01", "2016-12-01", "2017-12-01"]
OOT = ["2018-12-01"]
MON = ["2019-12-01"]
SNAPS = DEV + OOT + MON

waterfall = []
def log_step(df, label, removed=None):
    waterfall.append({"step": label, "loans": len(df), "removed": removed,
                      "defaults": int(df["default_12m"].sum()) if "default_12m" in df else None})
    return df

# --- Step 1 exclusions, recounted from the performance file
perf = pd.read_parquet("data/interim/perf.parquet", columns=["loan_id", "period", "dq_status", "zb_code"])
perf = add_default_event(perf)
at = perf[perf["period"].isin(pd.to_datetime(SNAPS))]
n0 = len(at)
log_step(at, "Loan records in performance file at the 5 December snapshots")
a = at[~at["dflt_event"]]
log_step(a, "Excluded: already in default at snapshot (90+ DPD, REO, credit-event termination)", n0 - len(a))
b = a[a["zb_code"].isna()]
log_step(b, "Excluded: terminated at snapshot without default (e.g. prepaid)", len(a) - len(b))
c = b[b["dq_num"].notna()]
log_step(c, "Excluded: delinquency status unknown (XX)", len(b) - len(c))
del perf, at, a, b, c

snaps = pd.read_parquet("data/interim/snapshots.parquet")
assert waterfall[-1]["loans"] == len(snaps)
log_step(snaps, "Eligible population with 12-month outcome (Step 1 output)")
# --- Product exclusions
io = snaps["io_flag"] == "Y"
snaps = snaps[~io]
log_step(snaps, "Excluded: interest-only loans", int(io.sum()))
arm = snaps["amort_type"] != "FRM"
snaps = snaps[~arm]
log_step(snaps, "Excluded: non-FRM (ARM) loans", int(arm.sum()))
# Kept with flag (not removed): HARP, and invalid key fields (decided in Step 3)
memo = {
    "HARP refinances (kept, flagged)": int((snaps.harp == "Y").sum()),
    "FICO = 9999 not available (Step 3)": int((snaps.fico == 9999).sum()),
    "DTI = 999 not available (Step 3)": int((snaps.dti == 999).sum()),
    "LTV/CLTV = 999 not available (Step 3)": int(((snaps.ltv == 999) | (snaps.cltv == 999)).sum()),
    "Units = 99 not available (Step 3)": int((snaps.units == 99).sum()),
}

snaps["snap"] = pd.to_datetime(snaps["snap"])
dev = snaps[snaps.snap.isin(pd.to_datetime(DEV))].reset_index(drop=True)
oot = snaps[snaps.snap.isin(pd.to_datetime(OOT))].reset_index(drop=True)
mon = snaps[snaps.snap.isin(pd.to_datetime(MON))].reset_index(drop=True)

# --- 70/30 split of development sample by loan_id
gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEED)
tr_idx, ho_idx = next(gss.split(dev, groups=dev["loan_id"]))
train, hold = dev.iloc[tr_idx], dev.iloc[ho_idx]
assert set(train.loan_id).isdisjoint(hold.loan_id)

for name, d in [("train", train), ("holdout", hold), ("oot", oot), ("monitoring", mon)]:
    d.reset_index(drop=True).to_parquet(f"data/interim/{name}.parquet")

samples = {"Development (Dec 2015-2017)": dev, "  Train (70% of loans)": train,
           "  In-time holdout (30% of loans)": hold, "OOT (Dec 2018)": oot,
           "Monitoring (Dec 2019)": mon}
rows = [{"sample": k, "loans": len(d), "unique_loans": d.loan_id.nunique(),
         "defaults": int(d.default_12m.sum()), "default_rate": d.default_12m.mean()}
        for k, d in samples.items()]
wf = pd.DataFrame(waterfall)
sm = pd.DataFrame(rows)
overlap = oot.loan_id.isin(dev.loan_id).mean()
print(wf.to_string()); print(); print(sm.to_string()); print(memo)
print(f"OOT loans also in development sample: {overlap:.2%}")

with open("results.md", "a") as f:
    f.write("\n## Step 2: exclusions waterfall (pooled over the 5 December snapshots)\n\n")
    f.write("| Step | Loans remaining | Removed | Defaults |\n|---|---|---|---|\n")
    for r in waterfall:
        rem = "" if r["removed"] is None else f"{r['removed']:,}"
        d = "" if r["defaults"] is None else f"{r['defaults']:,}"
        f.write(f"| {r['step']} | {r['loans']:,} | {rem} | {d} |\n")
    f.write("\nKept but flagged / deferred to Step 3 (loan-snapshot rows):\n\n")
    for k, v in memo.items():
        f.write(f"- {k}: {v:,}\n")
    f.write("\n## Step 2: samples\n\n| Sample | Loan-snapshots | Unique loans | Defaults | Default rate |\n|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['sample']} | {r['loans']:,} | {r['unique_loans']:,} | {r['defaults']:,} | {r['default_rate']:.4%} |\n")
    f.write(f"\nSplit: GroupShuffleSplit on loan_id, train_size 0.7, random_state {SEED}; train and holdout loan_ids are disjoint (asserted).\n")
    f.write(f"OOT loans that also appear in the development sample: {overlap:.2%}.\n")
