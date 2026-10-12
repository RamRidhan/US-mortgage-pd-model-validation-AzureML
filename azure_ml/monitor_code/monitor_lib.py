"""Step 12e: monitoring metrics for one scored month. Self-contained (numpy / pandas / scipy only; bins are plain JSON)
so it can run as an Azure ML command component with this folder as the code folder."""
import json
import numpy as np
import pandas as pd
from scipy.stats import binom, beta

# thresholds (my choice, as in Step 8)
PSI_AMBER, PSI_RED = 0.10, 0.25
CSI_RED = 0.25
DQ_MULTIPLE, DQ_FLOOR = 2.0, 0.005          # alert if field rate > 2x development AND above 0.5 percentage points
BAD_CODES = {"fico": [9999], "ltv": [999], "cltv": [999], "dti": [999], "units": [99]}
RANGES = {"fico": (300, 850), "cltv": (1, 200), "dti": (1, 65), "orig_rate": (0.01, 15), "upb_ratio": (0, 1.5)}
DQ_INC_AMBER, DQ_INC_RED = 1.5, 2.0          # delinquency incidence vs development: extension added after the COVID rehearsal (see decisions.md)
DEV_GINI = 0.7973


def psi(expected, actual, eps=1e-6):
    e, a = np.asarray(expected, float) + eps, np.asarray(actual, float) + eps
    return float(((a - e) * np.log(a / e)).sum())


def gini(y, p):
    """2*AUC - 1 via the rank-sum (Mann-Whitney) formula; no scikit-learn needed."""
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(np.asarray(p))
    n1 = y.sum(); n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return 2 * auc - 1


def read_scores(path):
    """Batch APPEND_ROW output is SPACE-separated with no header row (checked on the real 2018-01 file: 226,322 rows,
    'loan_id pd grade score'). Also tolerates a comma-separated file or a header row."""
    df = pd.read_csv(path, header=None, names=["loan_id", "pd", "grade", "score"], sep=r"[\s,]+", engine="python", dtype=str)
    if str(df.iloc[0, 0]).strip().lower() == "loan_id":
        df = df.iloc[1:]
    df = df.astype({"pd": float, "grade": int, "score": float})
    return df.reset_index(drop=True)


def rag_psi(x): return "Green" if x < PSI_AMBER else ("Amber" if x <= PSI_RED else "Red")


def clean(df):
    d = df.copy()
    for c, codes in BAD_CODES.items():
        if c in d: d[c] = d[c].where(~d[c].isin(codes))
    d["fico"] = d["fico"].where(d["fico"].between(300, 850))
    d["cltv"] = d["cltv"].where(d["cltv"] <= 200)
    d["upb_ratio"] = d["upb_ratio"].clip(upper=1)
    return d


def bin_index(b, x):
    """Bin number = count of split points <= x (bins are [a, b)); missing values get the extra last bin."""
    x = np.asarray(x, dtype=float)
    idx = np.searchsorted(np.asarray(b["splits"]), x, side="right")
    return np.where(np.isnan(x), len(b["woe"]), idx)


def bin_shares(b, s):
    return pd.Series(bin_index(b, s.values)).value_counts(normalize=True)


def field_quality(df, variables):
    """Per field: share missing (incl. sentinel codes) and share out of range."""
    out = {}
    for v in variables:
        s = df[v]
        bad = s.isna() | s.isin(BAD_CODES.get(v, []))
        lo, hi = RANGES.get(v, (-np.inf, np.inf))
        rng = (~bad) & ~s.between(lo, hi)
        out[v] = dict(missing=float(bad.mean()), out_of_range=float(rng.mean()))
    return pd.DataFrame(out).T


def build_baseline(dev_inputs, dev_scored, bins, variables):
    """Development reference: per-variable bin shares, grade shares, field quality rates."""
    d = clean(dev_inputs)
    return dict(   # JSON-serialisable on purpose: no pickles across pandas versions
        var_shares={v: {str(k): float(x) for k, x in bin_shares(bins[v], d[v]).items()} for v in variables},
        grade_shares={str(k): float(x) for k, x in dev_scored["grade"].value_counts(normalize=True).sort_index().items()},
        quality=field_quality(dev_inputs, variables).to_dict(orient="index"),
        dq_share=float((dev_inputs["cur_dq"] >= 1).mean()))


def month_metrics(inputs, scores, base, bins, variables, csi_exclude=()):
    """Signals available immediately after scoring: CSI per variable, grade-mix PSI, data quality."""
    d = clean(inputs); res = {}
    csi = {}
    for v in variables:
        e = pd.Series({int(k): x for k, x in base["var_shares"][v].items()}); a = bin_shares(bins[v], d[v]); idx = e.index.union(a.index)
        csi[v] = psi(e.reindex(idx, fill_value=0), a.reindex(idx, fill_value=0))
    res["csi"] = csi
    res["dq_incidence_ratio"] = float((inputs["cur_dq"] >= 1).mean() / base["dq_share"])
    gb = pd.Series({int(k): x for k, x in base["grade_shares"].items()}).sort_index()
    sh = scores["grade"].value_counts(normalize=True).reindex(gb.index, fill_value=0)
    res["psi_grade"] = psi(gb, sh)
    q = field_quality(inputs, variables); dq = {}
    for v in variables:
        for kind in ("missing", "out_of_range"):
            now, dev = q.loc[v, kind], base["quality"][v][kind]
            dq[f"{v}.{kind}"] = dict(now=now, dev=dev, flag=bool(now > DQ_MULTIPLE * dev and now > DQ_FLOOR))
    res["dq"] = dq
    csi_alert = {v: x for v, x in csi.items() if v not in csi_exclude}
    ratio = res["dq_incidence_ratio"]
    res["alerts"] = ([f"CSI {v} {x:.3f} > {CSI_RED}" for v, x in csi_alert.items() if x > CSI_RED]
                     + ([f"delinquent share {ratio:.1f}x development (> {DQ_INC_RED}x)"] if ratio > DQ_INC_RED else [])
                     + ([f"PSI grade mix {res['psi_grade']:.3f} ({rag_psi(res['psi_grade'])})"] if res["psi_grade"] > PSI_AMBER else [])
                     + [f"data quality {k} {x['now']:.3%} vs dev {x['dev']:.3%}" for k, x in dq.items() if x["flag"]])
    res["status"] = "Red" if (res["psi_grade"] > PSI_RED or any(x > CSI_RED for x in csi_alert.values()) or ratio > DQ_INC_RED or any(x["flag"] for x in dq.values())) else (
        "Amber" if (res["psi_grade"] > PSI_AMBER or ratio > DQ_INC_AMBER) else "Green")
    return res


def performance_metrics(scores, labels, grade_pd):
    """Available only after the 12-month window: Gini, actual vs expected, grade back-test."""
    d = scores.merge(labels[["loan_id", "default_12m"]], on="loan_id", validate="1:1")
    act, exp = d.default_12m.mean(), d.pd.mean()
    g = gini(d.default_12m, d.pd); drop = (DEV_GINI - g) / DEV_GINI
    lights = {}
    for grade, grp in d.groupby("grade"):
        n, k, p = len(grp), int(grp.default_12m.sum()), grade_pd[int(grade)]
        pb = 1 - binom.cdf(k - 1, n, p)
        lights[int(grade)] = "Green" if pb >= 0.05 else ("Amber" if pb >= 0.01 else "Red")
    ae = act / exp
    return dict(n=len(d), defaults=int(d.default_12m.sum()), actual=act, expected=exp, ae=ae, gini=g, gini_drop=drop, lights=lights,
                rag_gini="Green" if drop < 0.10 else ("Amber" if drop <= 0.20 else "Red"),
                rag_ae="Green" if 0.8 <= ae <= 1.2 else ("Amber" if (0.7 <= ae <= 1.5) else "Red"),
                n_red_grades=sum(v == "Red" for v in lights.values()))
