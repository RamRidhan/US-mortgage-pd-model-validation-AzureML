"""Export the champion's WoE bins and coefficients as plain JSON (no optbinning / pickle needed at scoring time) and verify
the plain implementation reproduces optbinning's WoE exactly on the development sample.  python azure_ml/export_plain_artifacts.py"""
import json, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "azure_ml/batch_code")
import score

ch = json.load(open("data/interim/champion.json")); s7 = json.load(open("data/interim/step7.json"))
bins = pickle.load(open("data/interim/bins.pkl", "rb"))
plain = {}
for v in ch["vars"]:
    ob = bins[v]; t = ob.binning_table.build()
    body = t.iloc[:-1]                                   # drop Totals
    n = len(ob.splits) + 1
    plain[v] = dict(splits=[float(x) for x in ob.splits], woe=[float(x) for x in body["WoE"].iloc[:n]], missing_woe=float(body["WoE"].iloc[-1]))
json.dump(plain, open("azure_ml/batch_model/bins.json", "w"), indent=1)
json.dump({"vars": ch["vars"], "params": ch["params"], "shift": s7["shift"], "edges": s7["edges"]}, open("azure_ml/batch_model/config.json", "w"), indent=1)

dev = pd.concat([pd.read_parquet(f"data/interim/{n}_clean.parquet") for n in ["train", "holdout"]])
for v in ch["vars"]:
    ref = bins[v].transform(dev[v].values, metric="woe", metric_missing="empirical", metric_special="empirical")
    mine = score.woe_plain(dev[v].values.astype(float), plain[v])
    assert np.abs(ref - mine).max() < 1e-12, (v, np.abs(ref - mine).max())
print("plain WoE equals optbinning WoE for all", len(ch["vars"]), "variables on", len(dev), "development rows")
