"""Step 1a/1b: load Freddie Mac origination and performance files to parquet.

Layout checked against the SFLLD General User Guide (Release 47, July 2026) and the
Disclosure Changes summary. The sample files here carry 31 origination fields (no
Servicer Name) and 35 performance fields (Defect Settlement Date still at position 7).
"""
import glob
import pandas as pd

ORIG_COLS = ["fico", "first_pay_dt", "first_time_buyer", "maturity_dt", "msa", "mi_pct",
             "units", "occupancy", "cltv", "dti", "orig_upb", "ltv", "orig_rate", "channel",
             "ppm", "amort_type", "state", "prop_type", "zip", "loan_id", "purpose",
             "orig_term", "n_borrowers", "seller", "super_conf", "pre_harp_id", "program",
             "harp", "valuation_method", "io_flag", "mi_cancel"]

# 0-based positions in the performance file
PERF_KEEP = {0: "loan_id", 1: "period", 2: "cur_upb", 3: "dq_status", 4: "loan_age",
             7: "mod_flag", 8: "zb_code", 9: "zb_date"}
PERF_NCOLS = 35

NUMERIC_ORIG = ["fico", "mi_pct", "units", "cltv", "dti", "orig_upb", "ltv", "orig_rate",
                "orig_term", "n_borrowers"]


def load_orig(raw="data/raw", out="data/interim/orig.parquet"):
    frames = []
    for f in sorted(glob.glob(f"{raw}/sample_orig_*.txt")):
        d = pd.read_csv(f, sep="|", header=None, dtype=str)
        assert d.shape[1] == len(ORIG_COLS), (f, d.shape[1])  # layout change fails loudly
        d.columns = ORIG_COLS
        d["vintage"] = int(f[-8:-4])
        frames.append(d)
    orig = pd.concat(frames, ignore_index=True)
    assert orig["loan_id"].is_unique
    for c in NUMERIC_ORIG:
        orig[c] = pd.to_numeric(orig[c], errors="coerce")
    orig.to_parquet(out)
    return orig


def load_perf(raw="data/raw", out="data/interim/perf.parquet", chunksize=2_000_000):
    parts = []
    for f in sorted(glob.glob(f"{raw}/sample_perf_*.txt")):
        with open(f) as fh:
            assert len(fh.readline().rstrip("\n").split("|")) == PERF_NCOLS, f
        for ch in pd.read_csv(f, sep="|", header=None, dtype=str,
                              usecols=list(PERF_KEEP), chunksize=chunksize):
            ch = ch.rename(columns=PERF_KEEP)
            ch["period"] = pd.to_datetime(ch["period"], format="%Y%m")
            ch["cur_upb"] = pd.to_numeric(ch["cur_upb"]).astype("float32")
            ch["loan_age"] = pd.to_numeric(ch["loan_age"]).astype("int16")
            for c in ("loan_id", "dq_status", "mod_flag", "zb_code"):
                ch[c] = ch[c].astype("category")
            parts.append(ch)
    perf = pd.concat(parts, ignore_index=True)   # categories union via object -> recast
    for c in ("loan_id", "dq_status", "mod_flag", "zb_code"):
        perf[c] = perf[c].astype("category")
    perf.to_parquet(out)
    return perf
