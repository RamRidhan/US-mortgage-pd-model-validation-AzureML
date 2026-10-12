"""Step 9a: monthly portfolio default rate, unemployment regression (beta) and base hazard by loan age.
Usage: python -I src/run_step9a.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import statsmodels.api as sm

pe = pd.read_parquet("data/interim/perf_ext.parquet", columns=["loan_id", "period", "dq_status", "zb_code", "loan_age"])
pe["dq"] = pd.to_numeric(pe.dq_status.astype(str), errors="coerce")
ev = ((pe.dq >= 3) | (pe.dq_status.astype(str) == "RA") | pe.zb_code.astype(str).isin(["02", "03", "09", "15"])).values.astype(np.int32)
perf_ok = (~(ev.astype(bool)) & pe.zb_code.isna().values & pe.dq.notna().values)      # performing, same rule as Step 1
codes = pe.loan_id.cat.codes.values
per = pe.period.values; age = pe.loan_age.values
order = np.lexsort((per, codes))
codes, per, age, ev, perf_ok = codes[order], per[order], age[order], ev[order], perf_ok[order]
n = len(codes)
starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
lens = np.diff(np.r_[starts, n])
last = np.repeat(starts + lens - 1, lens)
cs = np.cumsum(ev, dtype=np.int64)
cs_loan_base = np.repeat(np.r_[0, cs[starts[1:] - 1]], lens)    # cumulative events before the loan's first row (unused offset)
idx = np.arange(n)
j = np.minimum(idx + 12, last)
fut = (cs[j] - cs[idx]) > 0                                      # any default event in the next 12 rows (months)
per_m = pd.DatetimeIndex(per)
# sanity: the monthly flag reproduces Step 1 for Dec 2018 snapshot
chk = fut[perf_ok & (per_m == "2018-12-01")]
print("Dec 2018 rows", chk.size, "default rate", chk.mean())

# ---- portfolio monthly default rate, snapshots Jan 2013 - Dec 2019
df = pd.DataFrame({"period": per_m[perf_ok], "fut": fut[perf_ok], "age": age[perf_ok]})
dr = df[(df.period >= "2013-01-01") & (df.period <= "2019-12-01")].groupby("period").fut.agg(["size", "sum", "mean"])
u = pd.read_csv("data/raw/UNRATE.csv", parse_dates=["observation_date"]).set_index("observation_date")["UNRATE"]
dr["u0"] = u.reindex(dr.index).values
dr["u12"] = u.reindex(dr.index + pd.DateOffset(months=12)).values
dr["dU"] = dr.u12 - dr.u0
dr["logit"] = np.log(dr["mean"] / (1 - dr["mean"]))
print(dr.round(4).to_string())

def fit(d, label):
    X = sm.add_constant(d["dU"]); m = sm.OLS(d["logit"], X).fit(); h = sm.OLS(d["logit"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    print(label, "n", len(d), "alpha %.4f beta %.4f se_ols %.4f se_hac %.4f R2 %.3f" % (m.params["const"], m.params["dU"], m.bse["dU"], h.bse["dU"], m.rsquared))
    return dict(label=label, n=len(d), alpha=float(m.params["const"]), beta=float(m.params["dU"]), se_ols=float(m.bse["dU"]), se_hac=float(h.bse["dU"]), r2=float(m.rsquared))
A = fit(dr, "2013-2019 snapshots (spec)")
B = fit(dr[dr.index <= "2018-12-01"], "2013-2018 snapshots (outcome windows end by Dec 2019, pre-COVID)")

# ---- base hazard by loan age (years): P(default within 12m | performing), snapshots <= Dec 2018 (no hindsight at Dec 2019)
h = df[df.period <= "2018-12-01"].assign(k=lambda x: x.age // 12).groupby("k").fut.agg(["size", "sum", "mean"])
print(h.round(5))
json.dump(dict(spec=A, ex_covid=B), open("data/interim/step9_beta.json", "w"))
dr.to_csv("reports/monthly_default_rate_unrate.csv"); h.to_csv("reports/base_hazard_by_age.csv")
