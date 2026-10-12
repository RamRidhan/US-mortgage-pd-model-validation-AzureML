"""Step 1c-1e: default event, snapshots, 12-month outcome and behavioural features."""
import numpy as np
import pandas as pd

CREDIT_ZB = ["02", "03", "09", "15"]   # third-party sale, short sale/charge-off, REO disp., note sale
NEVER = 999                            # months_since_dq when never delinquent


def add_default_event(perf):
    dq = pd.to_numeric(perf["dq_status"].astype(str), errors="coerce")
    perf["dq_num"] = dq.astype("float32")        # NaN for "RA" and "XX"
    perf["dflt_event"] = ((dq >= 3) | (perf["dq_status"].astype(str) == "RA")
                          | perf["zb_code"].astype(str).isin(CREDIT_ZB))
    return perf


def build_snapshot(perf, snap, orig):
    snap = pd.Timestamp(snap)
    upto = perf[perf["period"] <= snap]
    at = upto[upto["period"] == snap]
    n_all = len(at)
    # active (no zero-balance code), not in default, and a known 0/1/2 delinquency bucket
    at = at[~at["dflt_event"] & at["zb_code"].isna() & at["dq_num"].notna()].copy()
    at["loan_id"] = at["loan_id"].astype(str)
    ids = set(at["loan_id"])

    win = perf[(perf["period"] > snap) & (perf["period"] <= snap + pd.DateOffset(months=12))]
    bad_ids = set(win.loc[win["dflt_event"], "loan_id"].astype(str))
    at["default_12m"] = at["loan_id"].isin(bad_ids).astype(int)
    at["snap"] = snap

    # behavioural features: history up to and including snap only
    h = upto[upto["loan_id"].astype(str).isin(ids)].copy()
    h["loan_id"] = h["loan_id"].astype(str)
    last12 = h[h["period"] > snap - pd.DateOffset(months=12)]
    at["cur_dq"] = at["dq_num"].astype(int)
    at["max_dq_12m"] = at["loan_id"].map(last12.groupby("loan_id")["dq_num"].max()).fillna(0).astype(int)
    late = h[h["dq_num"] >= 1].groupby("loan_id")["period"].max()
    gap = (snap.year - late.dt.year) * 12 + (snap.month - late.dt.month)
    at["months_since_dq"] = at["loan_id"].map(gap).fillna(NEVER).astype(int)
    mod = h.loc[h["mod_flag"].notna(), "loan_id"].unique()
    at["ever_modified"] = at["loan_id"].isin(mod).astype(int)

    at = at.merge(orig, on="loan_id", how="left", validate="1:1")
    at["upb_ratio"] = at["cur_upb"] / at["orig_upb"]
    keep = ["loan_id", "snap", "default_12m", "cur_dq", "max_dq_12m", "months_since_dq",
            "upb_ratio", "ever_modified", "loan_age"] + [c for c in orig.columns if c != "loan_id"]
    return at[keep], n_all
