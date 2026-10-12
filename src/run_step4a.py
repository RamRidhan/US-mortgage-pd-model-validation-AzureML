"""Step 4a: fit WoE bins on the TRAIN split only and compute diagnostics. Usage: python -I src/run_step4a.py"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from woe import *

train = pd.read_parquet("data/interim/train_clean.parquet")
train["snap"] = pd.to_datetime(train["snap"])
print("train", train.shape, "default rate", train[TARGET].mean())

bins, rows, stab = {}, [], {}
for col, dt in [(c, "numerical") for c in NUMERIC] + [(c, "categorical") for c in CATEGORICAL]:
    ob = fit_binning(train, col, dt)
    bins[col] = ob
    t = ob.binning_table.build()
    n_bins = n_bins_of(ob)
    st = snapshot_stability(ob, train, col)
    stab[col] = st
    # direction: Spearman of bin order vs event rate on non-missing bins
    tt = t.iloc[:-1]
    tt = tt[~tt["Bin"].astype(str).isin(["Special", "Missing"])]
    trend = np.sign(pd.Series(range(len(tt))).corr(tt["Event rate"].reset_index(drop=True), method="spearman")) if len(tt) >= 2 else np.nan
    rows.append(dict(variable=col, type=dt, n_bins=n_bins, iv=iv_of(ob), trend_of_risk=trend,
                     min_snap_rho=st["rho"].min(), iv_min_snap=st["iv"].min(), iv_max_snap=st["iv"].max(),
                     status=ob.status))
    print(col, dt, n_bins, round(iv_of(ob), 4), ob.status)
ivt = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
pickle.dump(bins, open("data/interim/bins.pkl", "wb"))
pickle.dump(stab, open("data/interim/bin_stability.pkl", "wb"))
ivt.to_parquet("data/interim/iv_table.parquet")
print(ivt.round(4).to_string())
