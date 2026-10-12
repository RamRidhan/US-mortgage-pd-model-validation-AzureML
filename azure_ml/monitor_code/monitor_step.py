"""Command-component entry point for the monitoring job (Steps 12e-12f). Logs every metric with MLflow and, if any signal
is red, posts to a Teams / Logic App webhook (URL in env ALERT_WEBHOOK_URL) and marks the job failed.

  --month 2020-04 --scores <scores.csv> --inputs <folder with that month's parquet> --history <folder with earlier scores>
  --baseline <baseline.json> --bins <bins.json> --labels <folder, optional: only if the month 12m earlier has labels>
"""
import argparse, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlflow, pandas as pd
import monitor_lib as ml

ap = argparse.ArgumentParser()
for k in ("month", "scores", "inputs", "baseline", "bins", "config"): ap.add_argument(f"--{k}", required=True)
ap.add_argument("--old_scores"); ap.add_argument("--old_labels")
ap.add_argument("--csi_exclude", default="upb_ratio", help="comma-separated variables left out of CSI alerting (upb_ratio: sample artefact, no new originations)")
ap.add_argument("--red_grades_alert", type=int, default=3, help="alert when this many grades fail the back-test (grades 7-8 are standing findings, F3)")
a = ap.parse_args()
cfg = json.load(open(a.config)); bins = json.load(open(a.bins)); base = json.load(open(a.baseline))
grade_pd = {int(k): v for k, v in json.load(open(a.config))["grade_pd"].items()}
import glob
def _files(p): return sorted(glob.glob(os.path.join(p, "*.parquet"))) if os.path.isdir(p) else [p]
inputs = pd.concat([pd.read_parquet(f) for f in _files(a.inputs)], ignore_index=True); scores = ml.read_scores(a.scores)
r = ml.month_metrics(inputs, scores, base, bins, cfg["vars"], csi_exclude=tuple(x for x in a.csi_exclude.split(",") if x))
mlflow.log_metric("psi_grade", r["psi_grade"])
for v, x in r["csi"].items(): mlflow.log_metric(f"csi_{v}", x)
mlflow.log_metric("dq_flags", sum(x["flag"] for x in r["dq"].values()))
mlflow.log_param("month", a.month); mlflow.log_param("status", r["status"])
alerts = list(r["alerts"])
if a.old_scores and a.old_labels:                 # label delay: performance of the month scored 12 months ago
    p = ml.performance_metrics(ml.read_scores(a.old_scores), pd.read_parquet(a.old_labels), grade_pd)
    for k in ("gini", "gini_drop", "ae", "n_red_grades"): mlflow.log_metric(f"perf_{k}", p[k])
    if "Red" in (p["rag_gini"], p["rag_ae"]) or p["n_red_grades"] >= a.red_grades_alert: alerts.append(f"performance: Gini {p['gini']:.3f} ({p['rag_gini']}), A/E {p['ae']:.2f} ({p['rag_ae']}), red grades {p['n_red_grades']}")
if alerts:
    msg = f"PD model monitoring {a.month}: {r['status'].upper()}\n" + "\n".join(alerts[:10])
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if url:
        import requests; requests.post(url, json={"text": msg}, timeout=30)
    print(msg)
    if r["status"] == "Red" or any(s.startswith("performance") for s in alerts):
        sys.exit(1)                               # a failed run is itself visible in the portal and via run-status events
