"""Step 3: data-quality checks. Usage (from pd-mortgage/): python -I src/run_step3.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dq import dq_report, clean_features

NAMES = ["train", "holdout", "oot", "monitoring"]
S = pd.concat([pd.read_parquet(f"data/interim/{n}.parquet").assign(sample=n) for n in NAMES],
              ignore_index=True)
issues = []
def log(issue, check, n, treatment, reason, dropped=""):
    issues.append(dict(issue=issue, check=check, records=int(n), treatment=treatment,
                       reason=reason, dropped=dropped))

# 1 Sentinels
for c, code in {"fico": 9999, "ltv": 999, "cltv": 999, "dti": 999, "units": 99}.items():
    n = (S[c] == code).sum()
    if n: log(f"{c} = {code} (not available)", "Sentinel", n, "Set to missing; own WoE bin",
              "Guide: sentinel code for not available")
n = (S.first_time_buyer == "9").sum()
if n: log("first_time_buyer = 9 (not available)", "Sentinel", n, "Set to missing", "Guide sentinel")
# DTI sentinel is structural: compare with HARP
dti_miss = S.dti == 999
harp = S.harp == "Y"
print("DTI sentinel vs HARP: P(dti=999|harp)=%.4f P(harp|dti=999)=%.4f" %
      (dti_miss[harp].mean(), harp[dti_miss].mean()))
dti_harp_overlap = (dti_miss & harp).sum()

# 2 Range
n = ((S.fico < 300) | ((S.fico > 850) & (S.fico != 9999))).sum()
log("fico outside 300-850 (excluding 9999)", "Range", n, "Set to missing", "Guide range")
n = ((S.ltv > 105) & (S.ltv != 999)).sum()
n_h = ((S.ltv > 105) & (S.ltv != 999) & ~harp).sum()
log("ltv > 105 (excluding 999)", "Range", n, "Kept (HARP allows LTV > 105); binned in WoE",
    f"{n - n_h:,} are HARP loans; {n_h:,} non-HARP")
n = ((S.cltv > 200) & (S.cltv != 999)).sum()
log("cltv > 200 (excluding 999)", "Range", n, "Set to missing", "Outside guide range 6-200")
n = ((S.dti > 65) & (S.dti != 999)).sum() + (S.dti <= 0).sum()
log("dti outside 1-65 (excluding 999)", "Range", n, "None needed", "No outliers found")
log("orig_rate <= 0", "Range", (S.orig_rate <= 0).sum(), "None needed", "No outliers found")

# 3 Consistency
log("cltv < ltv", "Consistency", ((S.cltv < S.ltv) & (S.cltv != 999) & (S.ltv != 999)).sum(),
    "None needed", "CLTV >= LTV holds everywhere")
n = (S.upb_ratio > 1).sum()
log("current UPB > original UPB", "Consistency", n, "Flag upb_gt_orig=1; cap upb_ratio at 1; rows kept",
    f"{(S.upb_ratio > 1.01).sum():,} exceed by >1%: rounding of early balances or capitalisation after modification")

# 4 Completeness
def md(df):
    df = df.reset_index()
    h = '| ' + ' | '.join(map(str, df.columns)) + ' |\n|' + '---|' * len(df.columns) + '\n'
    return h + ''.join('| ' + ' | '.join(str(v) for v in r) + ' |\n' for r in df.itertuples(index=False))


def blank(s):  # empty strings are read as NaN
    return s.isna().mean()
comp = pd.DataFrame({c: S.groupby("snap")[c].apply(blank) for c in S.columns
                     if c not in ("sample", "snap", "default_12m")}).T
comp["max_missing"] = comp.max(axis=1)
over = comp[comp.max_missing > 0.2].index.tolist()
print("fields >20% missing (blank):", over)
log("pre_harp_id blank in 84% of rows", "Completeness", S.pre_harp_id.isna().sum(),
    "Drop field", "Blank unless loan is a HARP refinance; harp flag carries the information", "pre_harp_id")
log("program blank in 98% of rows", "Completeness", S.program.isna().sum(),
    "Blank recoded to 'N' (no special programme); keep as category",
    "Blank means no Home Possible / HFA Advantage; informative, not missing")
log("msa blank", "Completeness", S.msa.isna().sum(), "Not used (state used instead)",
    "13.7% blank, under the 20% threshold, but ~400-level code; state gives geography")
log("dti missing (999)", "Completeness", dti_miss.sum(), "Own WoE bin",
    f"16% overall; structural: {dti_harp_overlap:,} of {dti_miss.sum():,} are HARP loans (not reported for HARP)")
const = [c for c in S.columns if S[c].nunique(dropna=False) == 1]
for c in const:
    log(f"{c} constant ({S[c].iloc[0]!r})", "Completeness", len(S), "Drop field", "No information", c)

# 5 Time continuity (12 months before each snapshot) and 7 Duplicates
perf = pd.read_parquet("data/interim/perf.parquet", columns=["loan_id", "period"])
dups = int(perf.duplicated(["loan_id", "period"]).sum())
log("duplicate loan_id + period rows in performance file", "Duplicates", dups,
    "Drop duplicates" if dups else "None needed", f"{len(perf):,} rows checked")
gap_n = 0; gap_tot = 0
for snap, g in S.groupby("snap"):
    ids = set(g.loan_id)
    h = perf[(perf.period <= snap) & (perf.period > snap - pd.DateOffset(months=12))]
    h = h[h.loan_id.astype(str).isin(ids)]
    h = h.assign(loan_id=h.loan_id.astype(str))
    agg = h.groupby("loan_id")["period"].agg(["count", "min"])
    exp = (snap.year - agg["min"].dt.year) * 12 + (snap.month - agg["min"].dt.month) + 1
    gap_n += int((agg["count"] != exp).sum()); gap_tot += len(agg)
log("gaps in monthly history in the 12 months before snapshot", "Time continuity", gap_n,
    "None needed" if gap_n == 0 else "Flag; drop if material", f"{gap_tot:,} loan-snapshots checked")
del perf

# 6 Stability by snapshot
num = ["fico", "ltv", "cltv", "dti", "orig_rate", "orig_upb", "upb_ratio", "loan_age"]
C = clean_features(S)
stab = C.groupby("snap")[num].mean().round(3)
stab_na = C.groupby("snap")[["fico", "dti", "cltv", "ltv"]].apply(lambda g: g.isna().mean()).round(4)
print(stab); print(stab_na)
drift = (stab.max() / stab.min() - 1).round(3).drop(["loan_age", "upb_ratio"])  # both rise with seasoning by construction
print("max/min-1 across snapshots:\n", drift)
big = drift[drift > 0.15].index.tolist()
log("field mean shifts >15% across snapshots", "Stability", 0,
    "Note for validation report", "Fields: " + ", ".join(big) if big else "none")
issues[-1]["records"] = len(big)

# 7 Target sanity
by_snap = S.groupby("snap")["default_12m"].agg(["size", "sum", "mean"])
by_vint = S[S["sample"] != "monitoring"].groupby("vintage")["default_12m"].agg(["size", "sum", "mean"])
print(by_snap); print(by_vint)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
colour = "#2a6f97"
for a, d, t, lab in [(ax[0], by_snap, "By snapshot (12-month outcome window follows)", [f"{x:%b %Y}" for x in by_snap.index]),
                     (ax[1], by_vint, "By vintage (Dec 2015-2018 snapshots, ex-COVID)", [str(x) for x in by_vint.index])]:
    a.bar(range(len(d)), d["mean"] * 100, color=colour, width=0.55)
    for i, v in enumerate(d["mean"] * 100):
        a.text(i, v + 0.05, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    a.set_xticks(range(len(d))); a.set_xticklabels(lab, fontsize=8)
    a.set_title(t, fontsize=10, loc="left"); a.grid(axis="y", color="#dddddd", lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_ylabel("12-month default rate (%)")
plt.tight_layout(); os.makedirs("reports", exist_ok=True)
plt.savefig("reports/dq_target_by_snapshot.png", dpi=150)

# Apply treatments and save cleaned samples
for n in NAMES:
    clean_features(pd.read_parquet(f"data/interim/{n}.parquet")).to_parquet(f"data/interim/{n}_clean.parquet")

log_df = pd.DataFrame(issues)
_t = open("results.md").read()
if "\n## Step 3: data-quality log" in _t:   # idempotent re-runs
    open("results.md", "w").write(_t[:_t.index("\n## Step 3: data-quality log")])
dropped = [x for x in log_df.dropped if x]
with open("results.md", "a") as f:
    f.write(f"\n## Step 3: data-quality log\n\nIssues logged: {len(log_df)}. Fields dropped: {', '.join(dropped)}.\n\n")
    f.write("| # | Check | Issue | Records affected | Treatment | Reason |\n|---|---|---|---|---|---|\n")
    for i, r in enumerate(log_df.itertuples(), 1):
        f.write(f"| {i} | {r.check} | {r.issue} | {r.records:,} | {r.treatment} | {r.reason} |\n")
    f.write("\nFields over 20% missing (blank) in at least one snapshot: " + ", ".join(over) + "\n")
    f.write("\n### Field means by snapshot (after sentinel treatment)\n\n" + md(stab) + "\n")
    f.write("\n### Missing share by snapshot\n\n" + md(stab_na) + "\n")
    f.write("\n### Target sanity: 12-month default rate\n\n" + md(by_snap.rename(columns={"size": "loans", "sum": "defaults", "mean": "rate"})) + "\n\n"
            + md(by_vint.rename(columns={"size": "loans", "sum": "defaults", "mean": "rate"})) + "\n\nChart: reports/dq_target_by_snapshot.png\n")
print(log_df[["check", "issue", "records"]].to_string())
