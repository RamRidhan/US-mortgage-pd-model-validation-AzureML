"""Batch scoring script for the champion scorecard (Azure ML batch endpoint, custom-model route).
Only numpy and pandas are needed. Pipeline: clean -> WoE -> risk-WoE (-WoE) -> logistic with calibration
shift -> PD -> master-scale grade (1 = lowest risk) -> scorecard points."""
import glob, json, os
import numpy as np
import pandas as pd

FACTOR = 20 / np.log(2)
OFFSET = 600 - FACTOR * np.log(50)
CFG, BINS = None, None


def clean(df):
    d = df.copy()
    if "fico" in d: d["fico"] = d["fico"].where((d["fico"] != 9999) & d["fico"].between(300, 850))
    if "cltv" in d: d["cltv"] = d["cltv"].where((d["cltv"] != 999) & (d["cltv"] <= 200))
    if "dti" in d: d["dti"] = d["dti"].where(d["dti"] != 999)
    if "upb_ratio" in d: d["upb_ratio"] = d["upb_ratio"].clip(upper=1)
    return d


def woe_plain(x, b):
    """Bin = number of split points <= x (bins are [a, b)); missing gets its own WoE."""
    x = np.asarray(x, dtype=float)
    idx = np.searchsorted(np.asarray(b["splits"]), x, side="right")
    w = np.asarray(b["woe"])[np.minimum(idx, len(b["woe"]) - 1)]
    return np.where(np.isnan(x), b["missing_woe"], w)


def score_frame(df, cfg, bins):
    d = clean(df)
    z = np.full(len(d), cfg["params"]["const"] + cfg["shift"], dtype=float)
    for v in cfg["vars"]:
        z = z + cfg["params"][v] * (-woe_plain(d[v].values, bins[v]))
    p = 1 / (1 + np.exp(-z))
    out = pd.DataFrame({"pd": p, "grade": np.digitize(p, np.array(cfg["edges"])) + 1, "score": OFFSET + FACTOR * (-z)})
    if "loan_id" in df.columns:
        out.insert(0, "loan_id", df["loan_id"].values)
    return out


def init(model_dir=None):
    global CFG, BINS
    root = model_dir or os.environ["AZUREML_MODEL_DIR"]
    cfg_path = glob.glob(os.path.join(root, "**", "config.json"), recursive=True)[0]
    CFG = json.load(open(cfg_path)); BINS = json.load(open(os.path.join(os.path.dirname(cfg_path), "bins.json")))


def run(mini_batch):
    frames = []
    for f in mini_batch:
        frames.append(score_frame(pd.read_parquet(f), CFG, BINS))
    return pd.concat(frames, ignore_index=True)
