"""Step 8: monitoring MI pack on OOT (Dec 2018) and monitoring (Dec 2019) vs development.
Usage: python -I src/run_step8.py"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from woe import TARGET, bin_index
from scorecard import *
from metrics import gini, psi, rag, grade_backtest

ch = json.load(open("data/interim/champion.json")); bins = pickle.load(open("data/interim/bins.pkl", "rb"))
P = pd.Series(ch["params"]); V = ch["vars"]; s7 = json.load(open("data/interim/step7.json"))
SHIFT, EDGES = s7["shift"], s7["edges"]; grade_pd = pd.Series({int(k): v for k, v in s7["grade_pd"].items()})
dev = pd.read_parquet("data/interim/dev_scored.parquet"); oot = pd.read_parquet("data/interim/oot_scored.parquet")
dev_c = pd.concat([pd.read_parquet(f"data/interim/{n}_clean.parquet") for n in ["train", "holdout"]], ignore_index=True)
oot_c = pd.read_parquet("data/interim/oot_clean.parquet"); mon_c = pd.read_parquet("data/interim/monitoring_clean.parquet")
zm, pm = predict_logit(P, risk_woe(mon_c, bins, V), SHIFT)
mon = mon_c[["loan_id", "snap", TARGET]].copy(); mon["pd"] = pm; mon["score"] = score_from_logit(zm); mon["grade"] = np.digitize(pm, EDGES) + 1
mon.to_parquet("data/interim/mon_scored.parquet")
SAMPLES = {"OOT Dec 2018": oot, "Dec 2019": mon}
CLEAN = {"OOT Dec 2018": oot_c, "Dec 2019": mon_c}

# ---- PSI on grade mix and on score deciles (cut points from development)
dev_share = dev.grade.value_counts(normalize=True).sort_index()
cuts = np.unique(np.quantile(dev.score, np.linspace(0, 1, 11)[1:-1]))
sbin = lambda s: np.digitize(s, cuts)
dev_sc = pd.Series(sbin(dev.score)).value_counts(normalize=True).sort_index()
res = {}
for k, d in SAMPLES.items():
    sh = d.grade.value_counts(normalize=True).reindex(dev_share.index, fill_value=0)
    ss = pd.Series(sbin(d.score)).value_counts(normalize=True).reindex(dev_sc.index, fill_value=0)
    r = dict(psi_grade=psi(dev_share, sh), psi_score=psi(dev_sc, ss))
    gdev = gini(dev[TARGET], dev.pd); g = gini(d[TARGET], d.pd)
    r.update(gini=g, gini_dev=gdev, gini_drop=(gdev - g) / gdev)
    bt = grade_backtest(d, grade_pd); r["bt"] = bt
    r["bt_counts"] = bt.light.value_counts().to_dict()
    r["bt_worst"] = "Red" if (bt.light == "Red").any() else ("Amber" if (bt.light == "Amber").any() else "Green")
    r["actual"], r["expected"] = d[TARGET].mean(), d.pd.mean(); r["ae"] = r["actual"] / r["expected"]
    r["n"], r["defaults"] = len(d), int(d[TARGET].sum())
    r["share"] = sh
    # CSI per variable on WoE bins
    csi = {}
    for v in V:
        e = pd.Series(bin_index(bins[v], dev_c[v])).value_counts(normalize=True)
        a = pd.Series(bin_index(bins[v], CLEAN[k][v])).value_counts(normalize=True)
        idx = e.index.union(a.index); csi[v] = psi(e.reindex(idx, fill_value=0), a.reindex(idx, fill_value=0))
    r["csi"] = pd.Series(csi).sort_values(ascending=False)
    res[k] = r
# RAG
for k, r in res.items():
    r["rag_psi_grade"] = rag(r["psi_grade"], 0.10, 0.25); r["rag_psi_score"] = rag(r["psi_score"], 0.10, 0.25)
    r["rag_gini"] = rag(r["gini_drop"], 0.10, 0.20)
    ae = r["ae"]; r["rag_ae"] = "Green" if 0.8 <= ae <= 1.2 else ("Amber" if (0.7 <= ae < 0.8 or 1.2 < ae <= 1.5) else "Red")
    r["rag_csi_max"] = rag(r["csi"].max(), 0.10, 0.25)
    print(k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in r.items() if x not in ("bt", "csi", "share")})
    print(r["csi"].round(4).to_dict()); print(r["bt"].round(5).to_string())

# ---- Gini by snapshot
gs = []
for s, g in dev.groupby("snap"): gs.append((f"Dev {str(s)[:7]}", gini(g[TARGET], g.pd), len(g)))
gs.append(("OOT 2018-12", res["OOT Dec 2018"]["gini"], len(oot))); gs.append(("Dec 2019", res["Dec 2019"]["gini"], len(mon)))
print(gs)

# ---- COVID / forbearance check (Dec 2019 snapshot; also Dec 2018 as baseline)
pe = pd.read_parquet("data/interim/perf_ext.parquet")
pe["dq"] = pd.to_numeric(pe.dq_status.astype(str), errors="coerce")
pe["forb"] = pe.assist.isin(["F", "T", "R"]) | pe.defer_flag.notna() | pe.disaster.notna()
w = pe[(pe.period >= "2020-04-01") & (pe.period <= "2020-12-01")]
w = w.assign(g=np.where(w.assist == "F", "Forbearance plan (F)", np.where(w.forb, "Other assistance / deferral / disaster", "No flag")))
loanmonth = w.groupby("g").agg(rows=("dq", "size"), dq1=("dq", lambda x: (x >= 1).mean()), dq3=("dq", lambda x: (x >= 3).mean())).reset_index()
print(loanmonth)
CRED = ["02", "03", "09", "15"]

def classify(d, label):
    snap = pd.Timestamp(d.snap.iloc[0]); bad = d[d[TARGET] == 1].loan_id.astype(str)
    x = pe[pe.loan_id.astype(str).isin(set(bad)) & (pe.period > snap)].copy()
    x["loan_id"] = x.loan_id.astype(str)
    ev = x[(x.dq >= 3) | (x.dq_status.astype(str) == "RA") | x.zb_code.astype(str).isin(CRED)]
    first = ev.groupby("loan_id").period.min()
    x["first_def"] = x.loan_id.map(first)
    forb = x[(x.period <= x.first_def) & (x.period <= snap + pd.DateOffset(months=12)) & x.forb].groupby("loan_id").size() > 0
    cred = x[x.zb_code.astype(str).isin(CRED) | (x.dq_status.astype(str) == "RA")].groupby("loan_id").size() > 0
    last = x.sort_values("period").groupby("loan_id").tail(1).set_index("loan_id")
    cured = ((last.dq == 0) | (last.zb_code.astype(str) == "01")) & ~last.index.isin(cred[cred].index)
    out = pd.DataFrame({"forborne": forb.reindex(first.index, fill_value=False), "credit_loss_event": cred.reindex(first.index, fill_value=False),
                        "cured_or_paid_off": cured.reindex(first.index, fill_value=False)})
    out["outcome"] = np.where(out.credit_loss_event, "Credit-event termination (REO/short sale/charge-off/note sale)",
                      np.where(out.cured_or_paid_off, "Cured to current or paid off", "Still delinquent / modified at end of data"))
    tab = pd.crosstab(out.outcome, out.forborne.map({True: "Forborne", False: "Not forborne"}), margins=True)
    return out, tab
cov = {}
for k, d in SAMPLES.items():
    o, tab = classify(d, k); cov[k] = (o, tab); print(k); print(tab)
o19 = cov["Dec 2019"][0]
excl = int((o19.forborne & o19.cured_or_paid_off & ~o19.credit_loss_event).sum())
adj_rate = (res["Dec 2019"]["defaults"] - excl) / res["Dec 2019"]["n"]
res["Dec 2019"]["adj_rate"], res["Dec 2019"]["adj_excl"] = adj_rate, excl
res["Dec 2019"]["adj_ae"] = adj_rate / res["Dec 2019"]["expected"]
ex_ids = set(o19.index[o19.forborne & o19.cured_or_paid_off & ~o19.credit_loss_event])
m2 = mon.copy(); m2["default_adj"] = ((m2[TARGET] == 1) & ~m2.loan_id.astype(str).isin(ex_ids)).astype(int)
res["Dec 2019"]["gini_adj"] = gini(m2.default_adj, m2.pd)
res["Dec 2019"]["bt_adj"] = grade_backtest(m2.assign(**{TARGET: m2.default_adj}), grade_pd)
res["Dec 2019"]["gini_adj_nocur"] = gini(m2[mon_c.cur_dq.values == 0].default_adj, m2[mon_c.cur_dq.values == 0].pd)
print("adjusted Gini", res["Dec 2019"]["gini_adj"]); print(res["Dec 2019"]["bt_adj"].round(5).to_string())
o18 = cov["OOT Dec 2018"][0]; excl18 = int((o18.forborne & o18.cured_or_paid_off & ~o18.credit_loss_event).sum())
print("adjusted Dec 2019 rate", adj_rate, "A/E", res["Dec 2019"]["adj_ae"], "excluded", excl)
pickle.dump(dict(res=res, gs=gs, loanmonth=loanmonth, cov={k: v[1] for k, v in cov.items()}, excl18=excl18), open("data/interim/step8.pkl", "wb"))
