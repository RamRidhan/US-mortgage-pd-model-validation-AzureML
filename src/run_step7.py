"""Step 7: calibration, master scale, OOT scoring (once) and grade back-test.
Usage: python -I src/run_step7.py [--dry]   (--dry scores the in-time holdout instead of OOT, writes nothing)"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import brentq
from scipy.stats import binom, beta
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from woe import TARGET
from scorecard import *
from metrics import gini, ks_curve
from run_step6 import raw_X

DRY = "--dry" in sys.argv
EDGES = [0.00035, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.04]   # 9 grades, fixed on development data
ch = json.load(open("data/interim/champion.json")); bins = pickle.load(open("data/interim/bins.pkl", "rb"))
P = pd.Series(ch["params"]); V = ch["vars"]
dev = pd.concat([pd.read_parquet(f"data/interim/{n}_clean.parquet") for n in ["train", "holdout"]], ignore_index=True)

# 7a calibrate intercept: mean PD on development = observed development default rate
X = risk_woe(dev, bins, V)
rate = dev[TARGET].mean()
shift = brentq(lambda s: predict_logit(P, X, s)[1].mean() - rate, -2, 2)
z, p = predict_logit(P, X, shift)
assert abs(p.mean() - rate) < 1e-9
dev["pd"] = p; dev["score"] = score_from_logit(z)
dev["grade"] = np.digitize(p, EDGES) + 1

# 7b master scale
def ms(df):
    g = df.groupby("grade").agg(loans=("pd", "size"), defaults=(TARGET, "sum"), pd_min=("pd", "min"), pd_max=("pd", "max"),
                                grade_pd=("pd", "mean"), score_min=("score", "min"), score_max=("score", "max"))
    g["share"] = g.loans / g.loans.sum(); g["observed"] = g.defaults / g.loans
    return g
scale = ms(dev)
assert len(scale) == 9 and (scale.share >= 0.0095).all() and (scale.defaults > 0).all(), scale
grade_pd = scale["grade_pd"]
print(scale.round(5).to_string())

# 7c score the out-of-time sample (Dec 2018) once
oot = pd.read_parquet("data/interim/holdout_clean.parquet" if DRY else "data/interim/oot_clean.parquet")
zo, po = predict_logit(P, risk_woe(oot, bins, V), shift)
oot["pd"] = po; oot["score"] = score_from_logit(zo); oot["grade"] = np.digitize(po, EDGES) + 1
g_oot = gini(oot[TARGET], po); _, _, ks_oot = ks_curve(oot[TARGET], po)
ho = pd.read_parquet("data/interim/holdout_clean.parquet")
g_ho = gini(ho[TARGET], predict_logit(P, risk_woe(ho, bins, V), shift)[1])
act, exp = oot[TARGET].mean(), po.mean()
print(f"OOT Gini {g_oot:.4f} KS {ks_oot:.4f}; holdout Gini {g_ho:.4f}; actual {act:.5%} expected {exp:.5%} ratio {act/exp:.3f}")

# challenger on the same sample, for the comparison table only
meta = json.load(open("data/interim/challenger_meta.json"))
bst = lgb.Booster(model_file="data/interim/challenger.txt")
Xc = raw_X(oot, meta["cats"]); pc = bst.predict(Xc)
Xch = raw_X(ho, meta["cats"]); pch = bst.predict(Xch)
gc_oot, gc_ho = gini(oot[TARGET], pc), gini(ho[TARGET], pch)
print(f"Challenger OOT Gini {gc_oot:.4f} (holdout {gc_ho:.4f}); mean PD {pc.mean():.5%}")

# 7d back-test by grade
rows = []
for g, grp in oot.groupby("grade"):
    n, d, pg = len(grp), int(grp[TARGET].sum()), grade_pd[g]
    p_b = 1 - binom.cdf(d - 1, n, pg); p_j = beta.cdf(pg, d + 0.5, n - d + 0.5)
    p_low = binom.cdf(d, n, pg)                       # chance of this few defaults if PD were right
    light = "Green" if p_b >= 0.05 else ("Amber" if p_b >= 0.01 else "Red")
    rows.append(dict(grade=g, loans=n, defaults=d, grade_pd=pg, observed=d / n, ratio=(d / n) / pg,
                     p_binomial=p_b, p_jeffreys=p_j, light=light, p_conservative=p_low,
                     conservative="Yes" if p_low < 0.05 else ""))
bt = pd.DataFrame(rows)
print(bt.round(5).to_string())
if DRY: sys.exit()

# ---- outputs
dev_scale = scale
pts = points_table(P, bins, shift)
pts.to_csv("reports/scorecard_points.csv", index=False)
dev_scale.to_csv("reports/master_scale_dev.csv"); bt.to_csv("reports/backtest_oot.csv", index=False)
json.dump(dict(shift=shift, edges=EDGES, grade_pd=grade_pd.to_dict(), g_oot=g_oot, g_ho=g_ho,
               gc_oot=gc_oot, gc_ho=gc_ho, act=act, exp=exp), open("data/interim/step7.json", "w"))
oot[["loan_id", "snap", "default_12m", "pd", "score", "grade"]].to_parquet("data/interim/oot_scored.parquet")
dev[["loan_id", "snap", "default_12m", "pd", "score", "grade"]].to_parquet("data/interim/dev_scored.parquet")

fig, ax = plt.subplots(figsize=(5.2, 4.2))
c1, c2 = "#2a6f97", "#d1660f"
ax.plot([1e-4, 0.5], [1e-4, 0.5], color="#999999", lw=1, ls=":")
ax.scatter(bt.grade_pd, bt.observed.clip(lower=1e-5), s=40, color=[{"Green": "#3a8f4f", "Amber": "#d9a400", "Red": "#c0392b"}[l] for l in bt.light], zorder=3)
for r in bt.itertuples(): ax.annotate(str(r.grade), (r.grade_pd, max(r.observed, 1e-5)), xytext=(5, 4), textcoords="offset points", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Grade PD (development, calibrated)"); ax.set_ylabel("Observed default rate, OOT Dec 2018")
ax.set_title("PD back-test by grade (colour = traffic light)", fontsize=10, loc="left"); ax.grid(color="#dddddd", lw=0.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("reports/backtest_by_grade.png", dpi=150)

md = lambda df: ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * df.shape[1] + "\n"
                 + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False)))
sc = dev_scale.reset_index()
sc_out = pd.DataFrame({"grade": sc.grade, "PD range": [f"{a:.4%} to {b:.4%}" for a, b in zip(sc.pd_min, sc.pd_max)],
                       "grade PD": sc.grade_pd.map("{:.4%}".format), "loans": sc.loans.map("{:,}".format), "share": sc.share.map("{:.2%}".format),
                       "defaults": sc.defaults, "observed rate": sc.observed.map("{:.4%}".format)})
bt_out = pd.DataFrame({"grade": bt.grade, "loans": bt.loans.map("{:,}".format), "defaults": bt.defaults,
                       "expected defaults": (bt.loans * bt.grade_pd).round(1), "grade PD": bt.grade_pd.map("{:.4%}".format),
                       "observed": bt.observed.map("{:.4%}".format), "obs/PD": bt.ratio.round(2),
                       "p binomial": bt.p_binomial.map("{:.4f}".format), "p Jeffreys": bt.p_jeffreys.map("{:.4f}".format),
                       "light": bt.light, "p (too few)": bt.p_conservative.map("{:.4f}".format), "too conservative": bt.conservative})
t = open("results.md").read()
if "\n## Step 7:" in t: t = t[:t.index("\n## Step 7:")]
t = t.replace("| Gini, out-of-time (after Step 7) | see Step 7 | see Step 7 |", f"| Gini, out-of-time (after Step 7) | {g_oot:.4f} | {gc_oot:.4f} |")
t = t.replace("| Gini drop, holdout to OOT | see Step 7 | see Step 7 |", f"| Gini drop, holdout to OOT | {g_ho - g_oot:+.4f} | {gc_ho - gc_oot:+.4f} |")
t = t.replace("| Calibration by grade (Step 7) | see Step 7 | not calibrated |", "| Calibration by grade (Step 7) | see Step 7 back-test | not calibrated; mean PD OOT %.4f%% vs %.4f%% actual |" % (pc.mean() * 100, act * 100))
shift_pts = -FACTOR * shift
with open("results.md", "w") as f:
    f.write(t + "\n## Step 7: calibration, master scale and OOT back-test\n\n")
    f.write(f"**7a Calibration.** Intercept shift {shift:+.6f} (log-odds) so that mean PD on the development sample ({len(dev):,} loan-snapshots, Dec 2015-2017) = observed default rate {rate:.4%} (mean PD before shift {predict_logit(P, X)[1].mean():.4%}). "
            f"Scores rise by {shift_pts:+.2f} points in total after the shift (constant spread over the 9 variables). Final points table: reports/scorecard_points.csv. "
            "Limitation: the anchor covers 2016 to 2018 only, a benign period with no downturn; a regulatory long-run average would need a full cycle.\n\n")
    f.write("**7b Master scale** (edges fixed on development; grade 1 = lowest risk)\n\n" + md(sc_out))
    f.write(f"\n**7c Out-of-time (Dec 2018, {len(oot):,} loans, {int(oot[TARGET].sum()):,} defaults)**\n\n")
    f.write(f"- OOT Gini: {g_oot:.4f} (in-time holdout {g_ho:.4f}; drop {g_ho - g_oot:+.4f}); OOT KS: {ks_oot:.4f}\n")
    f.write(f"- OOT actual default rate {act:.4%} vs expected (mean PD) {exp:.4%}; actual/expected = {act / exp:.3f}\n\n")
    f.write("**7d Back-test by grade** (traffic light on the one-sided binomial p-value: Green >= 0.05, Amber 0.01 to 0.05, Red < 0.01)\n\n" + md(bt_out))
    f.write("\nChart: reports/backtest_by_grade.png\n")
print("written")
