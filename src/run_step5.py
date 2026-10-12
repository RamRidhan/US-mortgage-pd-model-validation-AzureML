"""Step 5: champion WoE logistic scorecard. Uses train and in-time holdout only (OOT untouched)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from woe import TARGET
from scorecard import *
from metrics import gini, ks_curve

final = json.load(open("data/interim/selected_vars.json"))
bins = pickle.load(open("data/interim/bins.pkl", "rb"))
tr = pd.read_parquet("data/interim/train_clean.parquet")
ho = pd.read_parquet("data/interim/holdout_clean.parquet")
Xtr, Xho = risk_woe(tr, bins, final), risk_woe(ho, bins, final)

model, cols, dropped = fit_logit(Xtr, tr[TARGET])
print(model.summary()); print("dropped for sign:", dropped)
params = model.params
z_tr, p_tr = predict_logit(params, Xtr); z_ho, p_ho = predict_logit(params, Xho)
g_tr, g_ho = gini(tr[TARGET], p_tr), gini(ho[TARGET], p_ho)
s_ho = score_from_logit(z_ho)
_, _, ks_ho = ks_curve(ho[TARGET], p_ho); _, _, ks_tr = ks_curve(tr[TARGET], p_tr)
print(f"Gini train {g_tr:.4f} holdout {g_ho:.4f} gap {g_tr - g_ho:.4f}; KS train {ks_tr:.4f} holdout {ks_ho:.4f}")
print("mean PD train %.5f vs observed %.5f" % (p_tr.mean(), tr[TARGET].mean()))

pts = points_table(params, bins)
# check: table points reproduce the score for a sample of loans (look up each loan's bin)
from woe import bin_index
smp = ho.sample(2000, random_state=1)
tot = np.zeros(len(smp))
for v in cols:
    t = bins[v].binning_table.build().iloc[:-1]
    tt = t[t["Count"] > 0]
    idx = bin_index(bins[v], smp[v])
    full = bins[v].binning_table.build().iloc[:-1]
    p_v = pts[pts.variable == v]["points"].values
    # map table row index -> points (rows with Count == 0 are not in pts)
    rowpts = pd.Series(index=full.index, dtype=float); rowpts.loc[tt.index] = p_v
    tot += rowpts.loc[idx].values
direct = score_from_logit(predict_logit(params, risk_woe(smp, bins, final))[0])
print("max |table score - direct score| =", np.abs(tot - direct).max())
assert np.abs(tot - direct).max() < 1e-6

# plots
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
good, bad, ks = ks_curve(ho[TARGET], p_ho)
x = np.arange(1, len(good) + 1) / len(good)
c1, c2 = "#2a6f97", "#d1660f"
ax[0].plot(x, bad, color=c2, lw=2); ax[0].plot(x, good, color=c1, lw=2, ls="--")
k = np.argmax(np.abs(bad - good)); ax[0].vlines(x[k], good[k], bad[k], color="#555555", lw=1.2)
ax[0].text(x[k] + 0.02, (good[k] + bad[k]) / 2, f"KS = {ks:.2f}", fontsize=9)
ax[0].text(0.55, 0.93, "Bads (defaults)", color=c2, fontsize=9); ax[0].text(0.55, 0.42, "Goods", color=c1, fontsize=9)
ax[0].set_xlabel("Share of loans, riskiest first"); ax[0].set_ylabel("Cumulative share of group")
ax[0].set_title("KS curve, in-time holdout", fontsize=10, loc="left")
edges = np.linspace(np.floor(s_ho.min() / 20) * 20, np.ceil(s_ho.max() / 20) * 20, 40)
yh = ho[TARGET].values
ax[1].hist(s_ho[yh == 0], bins=edges, density=True, histtype="step", color=c1, lw=1.8, ls="--")
ax[1].hist(s_ho[yh == 1], bins=edges, density=True, histtype="step", color=c2, lw=1.8)
ax[1].set_xlabel("Score (points)"); ax[1].set_ylabel("Density"); ax[1].set_title("Score distribution, goods vs bads (holdout)", fontsize=10, loc="left")
ax[1].text(0.03, 0.88, "Bads", color=c2, transform=ax[1].transAxes, fontsize=9); ax[1].text(0.03, 0.78, "Goods", color=c1, transform=ax[1].transAxes, fontsize=9)
for a in ax:
    a.grid(color="#dddddd", lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("reports/champion_ks_score.png", dpi=150)

json.dump(dict(vars=cols, params=params.to_dict(), dropped_for_sign=dropped), open("data/interim/champion.json", "w"))
pts.to_csv("reports/scorecard_points_step5.csv", index=False)

md = lambda df: ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * df.shape[1] + "\n"
                 + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False)))
coef = pd.DataFrame({"variable": params.index, "coefficient": params.round(4).values,
                     "std_err": model.bse.round(4).values, "z": model.tvalues.round(2).values,
                     "p_value": [f"{x:.2e}" for x in model.pvalues]})
t = open("results.md").read()
if "\n## Step 5:" in t: t = t[:t.index("\n## Step 5:")]
with open("results.md", "w") as f:
    f.write(t + f"\n## Step 5: champion WoE logistic scorecard\n\nFitted on train ({len(tr):,} loan-snapshots, {int(tr[TARGET].sum()):,} defaults). Fitted on risk-WoE (= -WoE); all coefficients positive. Variables dropped for a wrong sign: {dropped if dropped else 'none'}. Final model variables ({len(cols)}): {', '.join(cols)}.\n\n")
    f.write("### Coefficients and p-values\n\n" + md(coef))
    f.write(f"\nGini train: {g_tr:.4f}\n\nGini in-time holdout: {g_ho:.4f}\n\nGini gap (train - holdout): {g_tr - g_ho:.4f}\n\nKS train / holdout: {ks_tr:.4f} / {ks_ho:.4f}\n\nMean PD on train {p_tr.mean():.5f} vs observed {tr[TARGET].mean():.5f}\n\n")
    f.write("Scaling: Factor = 20/ln2 = %.4f, Offset = 600 - Factor*ln(50) = %.4f (600 points at 50:1 good:bad, 20 points to double the odds).\n\n" % (FACTOR, OFFSET))
    f.write("Charts: reports/champion_ks_score.png\n\n### Points table (fitted intercept; Step 7 shifts the constant after calibration)\n\n")
    out = pts.copy(); out["event_rate"] = out["event_rate"].map("{:.4%}".format); out["woe"] = out["woe"].round(3); out["points"] = out["points"].round(1)
    f.write(md(out[["variable", "bin", "count", "event_rate", "woe", "points"]]))
