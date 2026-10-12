"""Re-read the performance files with the forbearance fields (positions checked against the User Guide
layout: 25 Payment Deferral Flag, 29 Delinquency Due to Disaster, 30 Borrower Assistance Plan).
Usage: python -I src/perf_ext.py"""
import glob
import pandas as pd

KEEP = {0: "loan_id", 1: "period", 3: "dq_status", 4: "loan_age", 8: "zb_code", 24: "defer_flag", 28: "disaster", 29: "assist"}

def build(raw="data/raw", out="data/interim/perf_ext.parquet"):
    parts = []
    for f in sorted(glob.glob(f"{raw}/sample_perf_*.txt")):
        for ch in pd.read_csv(f, sep="|", header=None, dtype=str, usecols=list(KEEP), chunksize=2_000_000):
            ch = ch.rename(columns=KEEP)
            ch["period"] = pd.to_datetime(ch["period"], format="%Y%m")
            ch["loan_age"] = pd.to_numeric(ch["loan_age"]).astype("int16")
            for c in ch.columns:
                if c not in ("period", "loan_age"):
                    ch[c] = ch[c].astype("category")
            parts.append(ch)
    perf = pd.concat(parts, ignore_index=True)
    for c in perf.columns:
        if c not in ("period", "loan_age"):
            perf[c] = perf[c].astype("category")
    perf.to_parquet(out)
    return perf

if __name__ == "__main__":
    p = build(); print(p.shape)
    for c in ("defer_flag", "disaster", "assist"):
        print(c, p[c].value_counts(dropna=False).to_dict())
