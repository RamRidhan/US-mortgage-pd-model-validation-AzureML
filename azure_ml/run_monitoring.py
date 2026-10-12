"""Step 12e from the terminal: submit the monitoring pipeline for a range of months, one after another, and print each result.
A RED month makes the job exit with an error ON PURPOSE (that is the alert), so the script tells an alert apart from a real failure
by checking whether the job logged its metrics.

  python azure_ml/run_monitoring.py --from 2018-01 --to 2018-03        # try a few months
  python azure_ml/run_monitoring.py --from 2018-01 --to 2020-12        # everything
"""
import argparse, os, time
import pandas as pd
from azure.core.exceptions import ServiceRequestError
from azure.ai.ml import Input, command, dsl
from azure.ai.ml.entities import Environment
from common import get_client

ap = argparse.ArgumentParser(); ap.add_argument("--from", dest="start", default="2018-01"); ap.add_argument("--to", dest="end", default="2018-01")
ap.add_argument("--no-metrics", action="store_true", help="skip reading metrics back (status only)"); a = ap.parse_args()
ml = get_client()
TERMINAL = ("complet", "fail", "cancel")


def retry(fn, what, tries=60, wait=30):
    for i in range(tries):
        try:
            return fn()
        except ServiceRequestError as e:
            print(f"  network problem during {what} ({type(e).__name__}); retry {i + 1}/{tries} in {wait}s", flush=True); time.sleep(wait)
    raise RuntimeError(f"gave up on {what}")


def status(name): return str(retry(lambda: ml.jobs.get(name).status, "status check")).lower()


try:
    ml.environments.get("pd-monitor-env", label="latest")
except Exception:
    ml.environments.create_or_update(Environment(name="pd-monitor-env", image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
                                                 conda_file="azure_ml/monitor_env.yml"))
    print("created environment pd-monitor-env (first run builds it: about 10 minutes)")

monitor = command(
    name="monitor_step", display_name="PD monitoring", code="azure_ml/monitor_code", environment="pd-monitor-env@latest", compute="cpu-cluster",
    command="python monitor_step.py --month ${{inputs.month}} --scores ${{inputs.scores}} --inputs ${{inputs.snapshot}} "
            "--baseline ${{inputs.assets}}/baseline.json --bins ${{inputs.assets}}/bins.json --config ${{inputs.assets}}/config.json "
            "$[[--old_scores ${{inputs.old_scores}}]] $[[--old_labels ${{inputs.old_labels}}]]",
    inputs={"month": "", "scores": Input(type="uri_file"), "snapshot": Input(type="uri_folder"), "assets": Input(type="uri_folder"),
            "old_scores": Input(type="uri_file", optional=True), "old_labels": Input(type="uri_file", optional=True)},
    environment_variables={"ALERT_WEBHOOK_URL": os.environ.get("ALERT_WEBHOOK_URL", "")})


@dsl.pipeline(name="pd-monitoring", compute="cpu-cluster")
def pipeline(month: str, scores, snapshot, assets, old_scores=None, old_labels=None):
    monitor(month=month, scores=scores, snapshot=snapshot, assets=assets, old_scores=old_scores, old_labels=old_labels)


def scores_in(t): return Input(type="uri_file", path=f"azureml://datastores/workspaceblobstore/paths/batch-scores/champion/{t}/scores.csv")


def child_metrics(parent_name):
    """Metrics logged by the monitoring step (empty dict if the step never got that far)."""
    if a.no_metrics: return {}
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(ml.workspaces.get(ml.workspace_name).mlflow_tracking_uri)
        kids = list(ml.jobs.list(parent_job_name=parent_name))
        if not kids: return {}
        run = MlflowClient().get_run(kids[0].name)
        return dict(run.data.metrics, status=run.data.params.get("status"))
    except Exception as e:
        return {"_unavailable": type(e).__name__}


rows = []
for m in pd.date_range(a.start + "-01", a.end + "-01", freq="MS"):
    tag = f"{m:%Y-%m}"
    kw = dict(month=tag, scores=scores_in(tag), snapshot=Input(type="uri_folder", path=f"azureml:snapshot-{tag}:1"),
              assets=Input(type="uri_folder", path="azureml:pd-monitoring-assets:1"))
    old = f"{m - pd.DateOffset(months=12):%Y-%m}"
    if old >= "2018-01":                       # label delay: only now can the month scored 12 months ago be evaluated
        kw.update(old_scores=scores_in(old), old_labels=Input(type="uri_file", path=f"azureml:labels-{old}:1"))
    job = retry(lambda: ml.jobs.create_or_update(pipeline(**kw), experiment_name="pd-monitoring"), f"submit {tag}")
    print(tag, "submitted", job.name, flush=True)
    while not any(t in status(job.name) for t in TERMINAL): time.sleep(30)
    st = status(job.name); met = child_metrics(job.name)
    line = f"{tag}: {st}"
    if "psi_grade" in met:
        line += f" | status {met.get('status')} | psi_grade {met['psi_grade']:.4f} | dq_flags {met.get('dq_flags', 0):.0f}"
        if "perf_gini" in met: line += f" | perf_gini {met['perf_gini']:.3f} | perf_ae {met['perf_ae']:.2f}"
        if "fail" in st: line += "  <- RED ALERT (job ends in error on purpose)"
    elif "fail" in st:
        line += "  <- FAILED WITHOUT METRICS: probably a real error. Ask for the logs." + (f" (metrics unavailable: {met})" if met else "")
    print(line, flush=True); rows.append(dict(month=tag, state=st, **{k: v for k, v in met.items()}))
    if "fail" in st and "psi_grade" not in met and not met.get("_unavailable"):
        raise SystemExit(f"stopping at {tag}: failed before logging metrics")
new = pd.DataFrame(rows); path = "azure_ml/monitoring_runs.csv"
if os.path.exists(path):                       # merge with earlier runs; a rerun of a month replaces its row
    old = pd.read_csv(path); new = pd.concat([old[~old.month.isin(new.month)], new], ignore_index=True)
new.sort_values("month").to_csv(path, index=False)
print("saved", path, "-", len(new), "months so far")
