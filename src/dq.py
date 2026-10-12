"""Step 3: data-quality helpers. Treatments are applied in clean_features()."""
import numpy as np
import pandas as pd

SENTINELS = {"fico": 9999, "ltv": 999, "cltv": 999, "dti": 999, "units": 99}
CLTV_MAX = 200          # guide range for Original CLTV
DROP_COLS = ["pre_harp_id", "ppm", "io_flag", "amort_type", "mi_cancel"]


def dq_report(df, cols):
    return pd.DataFrame({
        "missing_pct": df[cols].isna().mean().round(4),
        "min": df[cols].min(), "p50": df[cols].median(), "max": df[cols].max()})


def clean_features(df):
    """Apply the treatments logged in results.md. Returns a new frame."""
    d = df.copy()
    for c, code in SENTINELS.items():
        d[c] = d[c].where(d[c] != code)                 # sentinel -> NaN
    d["fico"] = d["fico"].where(d["fico"].between(300, 850))
    d["cltv"] = d["cltv"].where(d["cltv"] <= CLTV_MAX)  # out of guide range -> missing
    d["first_time_buyer"] = d["first_time_buyer"].replace("9", np.nan)
    d["upb_gt_orig"] = (d["upb_ratio"] > 1).astype(int)  # flag, rows kept
    d["upb_ratio"] = d["upb_ratio"].clip(upper=1)
    d["program"] = d["program"].fillna("N")              # blank = not a special programme
    return d.drop(columns=DROP_COLS)
