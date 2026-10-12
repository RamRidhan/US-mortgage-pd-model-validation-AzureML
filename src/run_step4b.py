"""Step 4b: variable selection on the TRAIN split. Usage: python -I src/run_step4b.py"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from woe import *

IV_MIN, IV_SUSPECT, RHO_MIN, CORR_MAX, VIF_MAX = 0.02, 0.5, 0.8, 0.7, 5.0
train = pd.read_parquet("data/interim/train_clean.parquet")
train["snap"] = pd.to_datetime(train["snap"])
bins = pickle.load(open("data/interim/bins.pkl", "rb"))
ivt = pd.read_parquet("data/interim/iv_table.parquet").set_index("variable")
md = lambda df: ("| " + " | ".join(map(str, df.reset_index().columns)) + " |\n|" + "---|" * (df.shape[1] + 1) + "\n"
                 + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.reset_index().itertuples(index=False)))

drops, flags = {}, {}
alive = list(ivt.sort_values("iv", ascending=False).index)

# Rule 1: Information Value
for v in list(alive):
    if ivt.loc[v, "iv"] < IV_MIN:
        drops[v] = f"IV {ivt.loc[v, 'iv']:.4f} < {IV_MIN}"; alive.remove(v)
    elif ivt.loc[v, "iv"] > IV_SUSPECT:
        flags[v] = f"IV {ivt.loc[v, 'iv']:.2f} > {IV_SUSPECT}: leakage check"
# Rule 2: direction of risk
for v in list(alive):
    exp = EXPECTED.get(v)
    got = ivt.loc[v, "trend_of_risk"]
    if exp is not None and not np.isnan(got) and exp != got:
        drops[v] = f"direction of risk ({got:+.0f}) opposite to expectation ({exp:+d})"; alive.remove(v)
# Rule 3: stability across the development snapshots
for v in list(alive):
    if ivt.loc[v, "min_snap_rho"] < RHO_MIN:
        drops[v] = f"unstable bin ordering across snapshots (min Spearman {ivt.loc[v, 'min_snap_rho']:.2f} < {RHO_MIN})"
        alive.remove(v)
print("after IV/direction/stability:", alive)

# Rule 4: pairwise correlation of WoE values, keep the stronger (higher IV)
W = pd.DataFrame({v: woe_transform(bins[v], train[v]) for v in ivt.index})
corr_all = W[alive].corr()
kept = []
for v in alive:                                   # alive is sorted by IV, descending
    clash = [k for k in kept if abs(corr_all.loc[v, k]) > CORR_MAX]
    if clash:
        drops[v] = f"abs(r) = {abs(corr_all.loc[v, clash[0]]):.2f} with {clash[0]} (higher IV {ivt.loc[clash[0], 'iv']:.2f} vs {ivt.loc[v, 'iv']:.2f})"
    else:
        kept.append(v)
# Rule 5: VIF
sample = W[kept].sample(100_000, random_state=42)
while True:
    vif = vif_table(sample[kept])
    if vif.max() < VIF_MAX or len(kept) <= 8:
        break
    worst = vif.idxmax(); drops[worst] = f"VIF {vif.max():.1f} >= {VIF_MAX}"; kept.remove(worst)
final = kept
print("FINAL", final); print(vif.round(2).to_dict())
assert 8 <= len(final) <= 12, len(final)
json.dump(final, open("data/interim/selected_vars.json", "w"))

# --- IV table with outcome per variable
status = ivt[["type", "n_bins", "iv", "trend_of_risk", "min_snap_rho", "iv_min_snap", "iv_max_snap"]].copy()
status["outcome"] = ["KEEP" if v in final else "DROP: " + drops.get(v, "") for v in status.index]
status["note"] = [flags.get(v, "") for v in status.index]
status = status.round(4)
cm = W[final].corr().round(2)
bt = {v: bins[v].binning_table.build()[["Bin", "Count", "Count (%)", "Event rate", "WoE", "IV"]].round(4) for v in final}
with open("results.md", "r") as f: t = f.read()
if "\n## Step 4:" in t:
    t = t[:t.index("\n## Step 4:")]
with open("results.md", "w") as f:
    f.write(t)
    f.write(f"\n## Step 4: WoE binning and variable selection (train split only, {len(train):,} loan-snapshots, {int(train[TARGET].sum()):,} defaults)\n\n")
    f.write("Binning: optbinning, max 8 bins, min 4 bins where feasible, monotonic trend for numerics, missing in its own bin.\n\n")
    f.write("### IV table, all 20 candidates (with outcome of every selection rule)\n\n" + md(status))
    f.write(f"\nFinal variables ({len(final)}): {', '.join(final)}\n\n")
    f.write("### Drop reasons\n\n" + "".join(f"- **{k}**: {v}\n" for k, v in drops.items()))
    f.write("\n### WoE correlation matrix, final variables\n\n" + md(cm))
    f.write("\n### VIF, final variables\n\n" + md(vif.round(2).to_frame("VIF")))
    f.write("\n### Bin tables, final variables\n")
    for v in final:
        f.write(f"\n**{v}**\n\n" + md(bt[v]))
print(status[["iv", "outcome", "note"]].to_string())
print(cm)
