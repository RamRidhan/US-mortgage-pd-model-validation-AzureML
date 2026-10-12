"""Discrimination and calibration metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks_curve(y, p):
    """Cumulative share of goods and bads when sorted from riskiest to safest (by p)."""
    y = np.asarray(y); o = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[o]
    bad = np.cumsum(ys) / ys.sum()
    good = np.cumsum(1 - ys) / (1 - ys).sum()
    return good, bad, float(np.max(np.abs(bad - good)))


def psi(expected, actual, eps=1e-6):
    """Population Stability Index between two share vectors (same bin order)."""
    e, a = np.asarray(expected, float) + eps, np.asarray(actual, float) + eps
    return float(((a - e) * np.log(a / e)).sum())


def rag(value, green, amber, higher_is_worse=True):
    """RAG for a metric with thresholds (green_limit, amber_limit)."""
    if higher_is_worse:
        return "Green" if value < green else ("Amber" if value <= amber else "Red")
    return "Green" if value > green else ("Amber" if value >= amber else "Red")


def grade_backtest(df, grade_pd, target="default_12m"):
    """One-sided binomial and Jeffreys tests of observed defaults vs grade PD, with traffic light."""
    from scipy.stats import binom, beta
    import pandas as pd
    rows = []
    for g, grp in df.groupby("grade"):
        n, d, pg = len(grp), int(grp[target].sum()), grade_pd[g]
        p_b = 1 - binom.cdf(d - 1, n, pg); p_j = beta.cdf(pg, d + 0.5, n - d + 0.5); p_low = binom.cdf(d, n, pg)
        rows.append(dict(grade=g, loans=n, defaults=d, expected=n * pg, grade_pd=pg, observed=d / n, ratio=(d / n) / pg,
                         p_binomial=p_b, p_jeffreys=p_j, p_conservative=p_low,
                         light="Green" if p_b >= 0.05 else ("Amber" if p_b >= 0.01 else "Red")))
    return pd.DataFrame(rows)
