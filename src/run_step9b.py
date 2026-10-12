"""Step 9b-e: staging, lifetime PD, ECL under three scenarios, sensitivities and PMA sizing (Dec 2019 portfolio).
Run after run_step8.py and run_step9a.py. Usage: python -I src/run_step9b.py"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.special import expit, logit

LGD = 0.10; TMAX = 30
SCEN = {"Upside": (-0.5, 0.20), "Base": (0.0, 0.50), "Downside": (3.0, 0.30)}      # illustrative assumption
beta = json.load(open("data/interim/step9_beta.json")); BETA = beta["spec"]["beta"]; BETA_X = beta["ex_covid"]["beta"]
s8 = pickle.load(open("data/interim/step8.pkl", "rb"))["res"]; M, O = s8["Dec 2019"], s8["OOT Dec 2018"]

mon = pd.read_parquet("data/interim/mon_scored.parquet"); mc = pd.read_parquet("data/interim/monitoring_clean.parquet")
mon = mon.merge(mc[["loan_id", "cur_dq", "orig_rate", "maturity_dt", "loan_age"]], on="loan_id", validate="1:1")
cu = pd.read_parquet("data/interim/perf.parquet", columns=["loan_id", "period", "cur_upb"])
cu = cu[cu.period == "2019-12-01"].assign(loan_id=lambda d: d.loan_id.astype(str))[["loan_id", "cur_upb"]]
mon = mon.merge(cu, on="loan_id", validate="1:1"); mon["cur_upb"] = mon.cur_upb.astype(float)
N = len(mon); print("loans", N, "balance", mon.cur_upb.sum())

# ---- 9a staging: PD at first observation = earliest scored snapshot (Dec 2015-2018)
prev = pd.concat([pd.read_parquet("data/interim/dev_scored.parquet"), pd.read_parquet("data/interim/oot_scored.parquet")])
prev = prev.sort_values("snap").groupby("loan_id").first()["pd"].rename("pd_first")
mon = mon.merge(prev, left_on="loan_id", right_index=True, how="left")
print("no earlier observation:", mon.pd_first.isna().sum())
mon["pd_first"] = mon.pd_first.fillna(mon.pd)
mon["sicr_rel"] = (mon.pd >= 2.5 * mon.pd_first) & (mon.pd > 0.003)
mon["stage"] = np.where((mon.cur_dq >= 1) | mon.sicr_rel, 2, 1)
st = mon.groupby("stage").agg(loans=("pd", "size"), balance=("cur_upb", "sum"), mean_pd=("pd", "mean"))
st["share_balance"] = st.balance / st.balance.sum()
print(st); print("stage 2 from dq:", (mon.cur_dq >= 1).sum(), " from relative PD only:", ((mon.cur_dq < 1) & mon.sicr_rel).sum())

# ---- 9b lifetime PD machinery
hb_tab = pd.read_csv("reports/base_hazard_by_age.csv").set_index("k")["mean"]
kmax = int(hb_tab.index.max()); hb = np.r_[hb_tab.values, hb_tab.values[-1]]        # flat beyond last observed age
a = np.minimum((mon.loan_age.values // 12), kmax + 1).astype(int)
tt = np.arange(1, TMAX + 1)
kidx = np.minimum(a[:, None] + tt[None, :] - 1, kmax + 1)                             # year t uses age a + t - 1 (year 1 = own 12m PD)
R = hb[kidx] / hb[np.minimum(a, kmax + 1)][:, None]
mat = mon.maturity_dt.astype(int); rem = np.maximum((mat // 100) * 12 + mat % 100 - (2019 * 12 + 12), 1).values
T = np.minimum(TMAX, np.ceil(rem / 12)).astype(int)
i_m = mon.orig_rate.values / 1200; n = rem.astype(float)
k = 12 * (tt[None, :] - 1)
g = (1 + i_m[:, None]) ** n[:, None]
bal = mon.cur_upb.values[:, None] * (g - (1 + i_m[:, None]) ** k) / (g - 1)
bal = np.where(k < n[:, None], bal, 0.0) * (tt[None, :] <= T[:, None])
disc = (1 + mon.orig_rate.values[:, None] / 100) ** (-tt[None, :])
EADd = bal * disc
stage2 = (mon.stage.values == 2)

def ecl_vec(pd12, lgd=LGD):
    h = np.minimum(1.0, R * pd12[:, None]); S_prev = np.hstack([np.ones((len(h), 1)), np.cumprod(1 - h, axis=1)[:, :-1]])
    marg = S_prev * h * (tt[None, :] <= T[:, None])
    e12 = marg[:, 0] * lgd * EADd[:, 0]
    full = (marg * EADd).sum(axis=1) * lgd
    return np.where(stage2, full, e12), marg

pd0 = mon.pd.values.copy()
_, marg0 = ecl_vec(pd0)
lifetime_pd = marg0.sum(axis=1)
print("mean 12m PD %.4f%%  mean lifetime PD %.3f%%; stage1 %.3f%% stage2 %.3f%%" % (pd0.mean() * 100, lifetime_pd.mean() * 100, lifetime_pd[~stage2].mean() * 100, lifetime_pd[stage2].mean() * 100))

def scenario_ecl(beta, lgd=LGD, mult=None):
    out = {}
    for name, (dU, w) in SCEN.items():
        p = expit(logit(pd0) + beta * dU)
        if mult is not None: p = np.minimum(1 - 1e-9, p * mult)
        e, _ = ecl_vec(p, lgd); out[name] = e
    return out

E = scenario_ecl(BETA)
bal_tot = mon.cur_upb.sum()
tab = []
for name, (dU, w) in SCEN.items():
    e = E[name]; tab.append(dict(scenario=name, dU_pp=dU, weight=w, ecl=e.sum(), stage1=e[~stage2].sum(), stage2=e[stage2].sum(), coverage=e.sum() / bal_tot))
weighted = sum(SCEN[k][1] * E[k] for k in SCEN)
tab.append(dict(scenario="Probability-weighted", dU_pp=np.nan, weight=1.0, ecl=weighted.sum(), stage1=weighted[~stage2].sum(), stage2=weighted[stage2].sum(), coverage=weighted.sum() / bal_tot))
tab = pd.DataFrame(tab); print(tab.round(4).to_string())
# why weighted > base: skew of weights vs convexity
dU_mean = sum(v[0] * v[1] for v in SCEN.values())
E_mean = ecl_vec(expit(logit(pd0) + BETA * dU_mean))[0].sum()
print("weighted-mean dU", dU_mean, "ECL at mean dU", E_mean, "base", E["Base"].sum(), "weighted", weighted.sum())
decomp = dict(base=E["Base"].sum(), at_mean=E_mean, weighted=weighted.sum(), dU_mean=dU_mean)

# sensitivities
sens = {}
for l in (0.05, 0.10, 0.20):
    EL = scenario_ecl(BETA, lgd=l); wl = sum(SCEN[k][1] * EL[k] for k in SCEN); sens[l] = (wl.sum(), wl.sum() / bal_tot, EL["Base"].sum(), EL["Downside"].sum())
EX = scenario_ecl(BETA_X); wx = sum(SCEN[k][1] * EX[k] for k in SCEN)
print("LGD sens", sens); print("ex-COVID beta %.4f weighted ECL %.0f base %.0f down %.0f" % (BETA_X, wx.sum(), EX["Base"].sum(), EX["Downside"].sum()))
share_dq = weighted[mon.cur_dq.values >= 1].sum() / weighted.sum(); share_s2 = weighted[stage2].sum() / weighted.sum()
print("share of weighted ECL from delinquent loans %.3f, from stage 2 %.3f" % (share_dq, share_s2))

# ---- PMA sizing (illustrative)
Em = E["Base"].sum()
E_mech = ecl_vec(np.minimum(1 - 1e-9, pd0 * M["ae"]))[0].sum()          # PDs recalibrated to reported 2020 experience
E_adj = ecl_vec(np.minimum(1 - 1e-9, pd0 * M["adj_ae"]))[0].sum()       # recalibrated to forbearance-adjusted experience
ratio = O["bt"].set_index("grade")["ratio"]                              # OOT observed / grade PD, known at Dec 2019
E_slope = ecl_vec(np.minimum(1 - 1e-9, pd0 * ratio.reindex(mon.grade).values))[0].sum()
pma = dict(base=Em, mech=E_mech, adj=E_adj, slope=E_slope)
print("PMA sizing:", pma)

# ---- hurricane check on the dev-sample default spike (Dec 2016 snapshot)
pe = pd.read_parquet("data/interim/perf_ext.parquet", columns=["loan_id", "period", "disaster"])
dv = pd.concat([pd.read_parquet("data/interim/dev_scored.parquet"), pd.read_parquet("data/interim/oot_scored.parquet")])
hur = []
for sn, gg in dv.groupby("snap"):
    ids = set(gg[gg.default_12m == 1].loan_id.astype(str)); sn = pd.Timestamp(sn)
    x = pe[(pe.period > sn) & (pe.period <= sn + pd.DateOffset(months=12))]
    x = x[x.loan_id.astype(str).isin(ids)]
    flagged = x[x.disaster.notna()].loan_id.astype(str).nunique()
    hur.append((str(sn)[:7], len(ids), flagged, flagged / max(len(ids), 1)))
print(hur)

pickle.dump(dict(st=st, tab=tab, sens=sens, decomp=decomp, pma=pma, hur=hur, EX=(wx.sum(), EX["Base"].sum(), EX["Downside"].sum()),
                 life=(pd0.mean(), lifetime_pd.mean(), lifetime_pd[~stage2].mean(), lifetime_pd[stage2].mean()), share=(share_dq, share_s2),
                 bal=bal_tot, N=N, n_s2dq=int((mon.cur_dq >= 1).sum()), n_s2rel=int(((mon.cur_dq < 1) & mon.sicr_rel).sum()), beta=beta, hb=hb_tab),
            open("data/interim/step9.pkl", "wb"))
