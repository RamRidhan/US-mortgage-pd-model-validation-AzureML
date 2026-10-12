"""WoE binning, IV, stability, correlation and VIF helpers (optbinning wrapper)."""
import numpy as np
import pandas as pd
from optbinning import OptimalBinning

TARGET = "default_12m"
NUMERIC = ["fico", "ltv", "cltv", "dti", "orig_rate", "units", "n_borrowers", "orig_term",
           "loan_age", "cur_dq", "max_dq_12m", "months_since_dq", "upb_ratio", "ever_modified"]
CATEGORICAL = ["purpose", "occupancy", "prop_type", "first_time_buyer", "channel", "state"]
# Expected sign of the relationship with default (+1 risk rises with the variable, -1 falls)
EXPECTED = {"fico": -1, "ltv": 1, "cltv": 1, "dti": 1, "orig_rate": 1, "cur_dq": 1,
            "max_dq_12m": 1, "months_since_dq": -1, "ever_modified": 1}


# Rare but important behavioural states (e.g. 2 months past due is ~0.4% of loans): a 5% minimum
# bin size would hide them, so these use 0.1%.
SPARSE = {"cur_dq", "max_dq_12m", "months_since_dq", "ever_modified", "units"}


def n_bins_of(ob):
    t = ob.binning_table.build().iloc[:-1]
    return int((t["Count"] > 0).sum())


def fit_binning(df, col, dtype, y=TARGET, max_n_bins=8, min_n_bins=4):
    """Monotonic (numeric) or category-grouped WoE binning on the training split only.
    Asks for min_n_bins and falls back to fewer where the data cannot support them."""
    kw = dict(name=col, dtype=dtype, min_bin_size=0.001 if col in SPARSE else 0.05,
              max_n_bins=max_n_bins)
    if col in SPARSE:
        kw["min_prebin_size"] = 0.001   # CART pre-binning also defaults to 5% leaves
    if dtype == "numerical":
        kw["monotonic_trend"] = "auto_asc_desc"
    else:
        kw["cat_cutoff"] = 0.01
    x = df[col].astype(object).where(df[col].notna(), np.nan).values if dtype == "categorical" \
        else df[col].values
    best = None
    for mn in (min_n_bins, 3, 2, None):
        ob = OptimalBinning(min_n_bins=mn, **kw)
        ob.fit(x, df[y].values)
        if ob.status in ("OPTIMAL", "FEASIBLE"):
            best = best or ob
            if mn is None or n_bins_of(ob) >= mn:
                return ob
    if best is None:
        raise RuntimeError(f"binning failed for {col}")
    return best


def woe_transform(ob, s):
    x = s.astype(object).where(s.notna(), np.nan).values if ob.dtype == "categorical" else s.values
    return ob.transform(x, metric="woe", metric_missing="empirical", metric_special="empirical")


def iv_of(ob):
    t = ob.binning_table.build()
    return float(t["IV"].iloc[-1])


def bin_index(ob, s):
    """Index of the bin each row falls in (missing bin last), used for stability."""
    x = s.astype(object).where(s.notna(), np.nan).values if ob.dtype == "categorical" else s.values
    return ob.transform(x, metric="indices", metric_missing="empirical")


def iv_from_counts(g, b):
    g = np.asarray(g, float); b = np.asarray(b, float)
    pg, pb = g / g.sum(), b / b.sum()
    ok = (pg > 0) & (pb > 0)
    return float(((pg - pb)[ok] * np.log(pg[ok] / pb[ok])).sum())


def snapshot_stability(ob, df, col, y=TARGET):
    """Per development snapshot: IV with fixed bins and Spearman of bin default rate vs the
    train-wide bin default rate (rank pattern stability)."""
    idx_all = bin_index(ob, df[col])
    tab = pd.DataFrame({"bin": idx_all, "y": df[y].values, "snap": df["snap"].values})
    ref = tab.groupby("bin")["y"].mean()
    out = {}
    for s, g in tab.groupby("snap"):
        c = g.groupby("bin")["y"].agg(["size", "sum"])
        c["good"] = c["size"] - c["sum"]
        rate = c["sum"] / c["size"]
        common = ref.index.intersection(c.index)
        rho = rate.loc[common].corr(ref.loc[common], method="spearman") if len(common) >= 2 else np.nan
        out[s] = dict(iv=iv_from_counts(c["good"], c["sum"]), rho=rho)
    return pd.DataFrame(out).T


def vif_table(W):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = np.column_stack([np.ones(len(W)), W.values])
    return pd.Series([variance_inflation_factor(X, i + 1) for i in range(W.shape[1])], index=W.columns)
