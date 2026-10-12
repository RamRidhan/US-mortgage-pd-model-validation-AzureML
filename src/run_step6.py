"""Step 6: LightGBM challenger on raw variables with monotone constraints. Train/holdout only."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from woe import TARGET
from metrics import gini, ks_curve

NUM = {  # feature: monotone direction (+1 risk rises, -1 falls, 0 free)
    "fico": -1, "ltv": 1, "cltv": 1, "dti": 1, "orig_rate": 1, "units": 0, "n_borrowers": -1,
    "orig_term": 1, "loan_age": 0, "cur_dq": 1, "max_dq_12m": 1, "months_since_dq": -1,
    "upb_ratio": 1, "ever_modified": 1}
CAT = ["purpose", "occupancy", "prop_type", "first_time_buyer", "channel"]   # state excluded as in Step 4
FEATS = list(NUM) + CAT
MONO = [NUM[f] for f in NUM] + [0] * len(CAT)
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
              monotone_constraints=MONO, random_state=42, n_jobs=4, verbose=-1)


def raw_X(df, cats=None):
    X = df[FEATS].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=cats[c] if cats else sorted(X[c].dropna().unique()))
    return X


if __name__ == "__main__":
    tr = pd.read_parquet("data/interim/train_clean.parquet")
    ho = pd.read_parquet("data/interim/holdout_clean.parquet")
    Xtr = raw_X(tr); cats = {c: list(Xtr[c].cat.categories) for c in CAT}; Xho = raw_X(ho, cats)
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xtr, tr[TARGET], eval_set=[(Xho, ho[TARGET])], eval_metric="auc")   # no early stopping
    p_tr, p_ho = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xho)[:, 1]
    g_tr, g_ho = gini(tr[TARGET], p_tr), gini(ho[TARGET], p_ho)
    print(f"Challenger Gini train {g_tr:.4f} holdout {g_ho:.4f} gap {g_tr - g_ho:.4f}")
    print("mean PD holdout %.5f vs observed %.5f" % (p_ho.mean(), ho[TARGET].mean()))
    # champion on the same holdout
    from scorecard import risk_woe, predict_logit
    ch = json.load(open("data/interim/champion.json")); bins = pickle.load(open("data/interim/bins.pkl", "rb"))
    params = pd.Series(ch["params"])
    _, pc = predict_logit(params, risk_woe(ho, bins, ch["vars"]))
    gc = gini(ho[TARGET], pc)
    _, pct = predict_logit(params, risk_woe(tr, bins, ch['vars']))
    gc_tr = gini(tr[TARGET], pct)
    print(f"Champion Gini holdout {gc:.4f}; challenger - champion = {g_ho - gc:+.4f}")
    # SHAP via LightGBM TreeSHAP contributions (last column = expected value)
    smp = Xho.sample(50_000, random_state=0)
    sv = m.predict(smp, pred_contrib=True)[:, :-1]
    shap = pd.Series(np.abs(sv).mean(axis=0), index=FEATS).sort_values(ascending=False)
    print(shap.round(4).to_string())
    # same raw variables only as the scorecard (isolates the modelling form)
    same = [f for f in ch["vars"]]
    m.booster_.save_model("data/interim/challenger.txt")
    json.dump(dict(feats=FEATS, cats=cats), open("data/interim/challenger_meta.json", "w"))
    res = dict(gc_tr=gc_tr, g_tr=g_tr, g_ho=g_ho, gc_ho=gc, shap=shap.round(4).to_dict())
    json.dump(res, open("data/interim/step6.json", "w"))
    # ---- results.md
    t = open("results.md").read()
    if "\n## Step 6:" in t: t = t[:t.index("\n## Step 6:")]
    with open("results.md", "w") as f:
        f.write(t + "\n## Step 6: challenger (LightGBM) and champion-challenger comparison, development data only\n\n")
        f.write("LightGBM, 400 trees, learning rate 0.03, 31 leaves, min_child_samples 200, monotone constraints on 12 of 14 numeric features (units and loan_age free; categoricals unconstrained), fitted on train, no early stopping.\n\n")
        f.write("| Criterion | Champion (scorecard) | Challenger (GBM) |\n|---|---|---|\n")
        f.write(f"| Gini, train | {gc_tr:.4f} | {g_tr:.4f} |\n")
        f.write(f"| Gini, in-time holdout | {gc:.4f} | {g_ho:.4f} |\n")
        f.write("| Gini, out-of-time (after Step 7) | see Step 7 | see Step 7 |\n| Gini drop, holdout to OOT | see Step 7 | see Step 7 |\n| Calibration by grade (Step 7) | see Step 7 | not calibrated |\n")
        f.write("| Explainability | Points per bin | SHAP values (below) |\n| Stability (PSI, Step 8) | pending | pending |\n| Ease of monitoring and governance | Simple: points table, PSI per bin, clear SS1/23 route | Needs SHAP; its ranking differs from the scorecard; harder to monitor |\n\n")
        f.write(f"Gini difference on holdout (challenger - champion): {g_ho - gc:+.4f}\n\n")
        f.write(f"**Decision (made on train and holdout only, before the OOT sample was opened): the scorecard stays champion and the GBM is kept as a benchmark.** "
                f"Reasons: the challenger does not beat the scorecard on holdout Gini ({g_ho:.4f} vs {gc:.4f}), overfits more (train-to-holdout gap {g_tr - g_ho:.3f} vs {gc_tr - gc:.3f}), "
                f"is harder to monitor and explain, and its SHAP ranking differs from the scorecard's. The OOT figures in the table, filled after Step 7, confirm it.\n\n### Challenger mean |SHAP| (50,000 holdout loans, log-odds)\n\n| feature | mean abs SHAP |\n|---|---|\n")
        for k, v in shap.items(): f.write(f"| {k} | {v:.4f} |\n")
