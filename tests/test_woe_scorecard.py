"""Tests for the WoE transform, the sign convention, the scorecard scaling and the plain-JSON batch scorer."""
import numpy as np
import pandas as pd
import pytest
from woe import fit_binning, woe_transform
from scorecard import FACTOR, OFFSET, fit_logit, risk_woe, score_from_logit
import score as batch_score


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    n = 30000
    x = rng.normal(700, 60, n)
    p = 1 / (1 + np.exp((x - 640) / 25 + 3.0))            # default risk falls as x rises
    df = pd.DataFrame({"x": x, "default_12m": (rng.random(n) < p).astype(int)})
    df.loc[rng.random(n) < 0.03, "x"] = np.nan             # 3% missing
    return df


def test_higher_woe_means_safer(data):
    ob = fit_binning(data, "x", "numerical")
    t = ob.binning_table.build().iloc[:-1]
    t = t[t["Count"] > 0]
    body = t[~t["Bin"].astype(str).isin(["Special", "Missing"])]
    assert body["WoE"].is_monotonic_increasing             # higher x = safer = higher WoE
    assert body["Event rate"].is_monotonic_decreasing


def test_missing_has_its_own_woe(data):
    ob = fit_binning(data, "x", "numerical")
    woe = woe_transform(ob, data["x"])
    miss = woe[data["x"].isna().values]
    assert np.all(miss == miss[0]) and len(miss) > 0


def test_risk_woe_gives_positive_coefficient(data):
    ob = fit_binning(data, "x", "numerical")
    X = risk_woe(data, {"x": ob}, ["x"])
    model, cols, dropped = fit_logit(X, data["default_12m"])
    assert cols == ["x"] and not dropped and model.params["x"] > 0


def test_scorecard_scaling():
    # 600 points at 50:1 good:bad odds, 20 points to double the odds
    assert score_from_logit(-np.log(50)) == pytest.approx(600.0)
    assert score_from_logit(-np.log(100)) - score_from_logit(-np.log(50)) == pytest.approx(20.0)
    assert OFFSET == pytest.approx(600 - FACTOR * np.log(50))


def test_plain_json_woe_equals_optbinning(data):
    ob = fit_binning(data, "x", "numerical")
    t = ob.binning_table.build().iloc[:-1]
    plain = dict(splits=[float(s) for s in ob.splits], woe=[float(w) for w in t["WoE"].iloc[:len(ob.splits) + 1]], missing_woe=float(t["WoE"].iloc[-1]))
    ref = woe_transform(ob, data["x"])
    mine = batch_score.woe_plain(data["x"].values, plain)
    assert np.abs(ref - mine).max() < 1e-12


def test_batch_scorer_grades_follow_pd():
    cfg = {"vars": ["x"], "params": {"const": -5.0, "x": 1.0}, "shift": 0.0, "edges": [0.001, 0.01, 0.1]}
    bins = {"x": {"splits": [0.0], "woe": [1.0, -1.0], "missing_woe": 0.0}}
    df = pd.DataFrame({"loan_id": ["a", "b", "c"], "x": [-1.0, np.nan, 1.0]})
    out = batch_score.score_frame(df, cfg, bins)
    assert out["pd"].between(0, 1).all()
    assert list(out.sort_values("pd")["grade"]) == sorted(out["grade"])      # higher PD never gets a lower grade
    assert out.loc[out.loan_id == "c", "pd"].iloc[0] > out.loc[out.loan_id == "a", "pd"].iloc[0]   # high x = riskier-WoE here
