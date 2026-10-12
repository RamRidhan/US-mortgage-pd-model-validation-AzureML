"""WoE logistic scorecard: risk-WoE design matrix, fit, points table.

Sign convention: bin WoE = ln(%goods/%bads) (optbinning, same as the Step 4 formula), so a HIGHER
WoE means SAFER. The model is fitted on risk-WoE = -WoE, so higher means riskier and every
coefficient must be positive.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from woe import woe_transform

FACTOR = 20 / np.log(2)
OFFSET = 600 - FACTOR * np.log(50)       # 600 points at 50:1 good:bad odds, 20 points to double the odds


def risk_woe(df, bins, variables):
    return pd.DataFrame({v: -woe_transform(bins[v], df[v]) for v in variables}, index=df.index)


def fit_logit(X, y):
    """Fit; while any coefficient is not positive, drop the most negative one and refit."""
    dropped = []
    cols = list(X.columns)
    while True:
        m = sm.Logit(y, sm.add_constant(X[cols])).fit(disp=0)
        neg = m.params.drop("const")
        neg = neg[neg <= 0]
        if neg.empty:
            return m, cols, dropped
        worst = neg.idxmin(); dropped.append((worst, float(neg.min())))
        cols.remove(worst)


def predict_logit(params, X, shift=0.0):
    z = params["const"] + shift + (X[[c for c in params.index if c != "const"]] * params.drop("const")).sum(axis=1)
    return z.values, 1 / (1 + np.exp(-z.values))


def points_table(params, bins, shift=0.0):
    """Points per variable and bin. Score = OFFSET + FACTOR * ln(good:bad odds) = sum of bin points."""
    vars_ = [c for c in params.index if c != "const"]
    n = len(vars_); b0 = params["const"] + shift
    rows = []
    for v in vars_:
        t = bins[v].binning_table.build().iloc[:-1]
        for _, r in t.iterrows():
            if r["Count"] == 0:
                continue
            pts = FACTOR * (-b0 / n + params[v] * r["WoE"]) + OFFSET / n
            rows.append(dict(variable=v, bin=str(r["Bin"]), count=int(r["Count"]),
                             event_rate=r["Event rate"], woe=r["WoE"], points=pts))
    return pd.DataFrame(rows)


def score_from_logit(z):
    return OFFSET + FACTOR * (-z)
