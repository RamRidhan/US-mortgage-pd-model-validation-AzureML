"""Step 12d: invoke the batch endpoint once per month, in order, and wait for each job. Records the run count.
Safe to rerun: months that already have output are skipped, jobs already running are waited for, and network drops are retried.
python azure_ml/run_batches.py [--deployment champion] [--start N] [--limit N]"""
import argparse, json, time
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from common import get_client

ap = argparse.ArgumentParser(); ap.add_argument("--deployment", default="champion")
ap.add_argument("--limit", type=int, default=36, help="score only N months (use 1 for a test)")
ap.add_argument("--start", type=int, default=0, help="index of the first month (0 = 2018-01)"); a = ap.parse_args()
ml = get_client(); runs = {}
TERMINAL = ("complet", "fail", "cancel")


def retry(fn, what, tries=60, wait=30):
    """Call fn(); on a network error wait and retry (a Wi-Fi or DNS blip must not kill a 2-hour run)."""
    for i in range(tries):
        try:
            return fn()
        except ServiceRequestError as e:
            print(f"  network problem during {what} ({type(e).__name__}); retry {i + 1}/{tries} in {wait}s", flush=True); time.sleep(wait)
    raise RuntimeError(f"gave up on {what} after {tries} retries")


def state(job_name):
    return str(retry(lambda: ml.jobs.get(job_name).status, "status check")).lower()


def wait(job_name):
    while not any(t in state(job_name) for t in TERMINAL):
        time.sleep(30)
    return state(job_name)


# first wait for anything already running (e.g. a job whose log stream was interrupted)
for j in retry(lambda: list(ml.batch_endpoints.list_jobs("pd-batch")), "job listing"):
    if not any(t in state(j.name) for t in TERMINAL):
        print("waiting for running job", j.name, flush=True); wait(j.name)

for tag in [f"{y}-{m:02d}" for y in (2018, 2019, 2020) for m in range(1, 13)][a.start:a.start + a.limit]:
    try:
        job = retry(lambda: ml.batch_endpoints.invoke(
            endpoint_name="pd-batch", deployment_name=a.deployment,
            inputs={"snapshot": Input(type=AssetTypes.URI_FOLDER, path=f"azureml:snapshot-{tag}:1")},
            outputs={"score": Output(type=AssetTypes.URI_FILE, path=f"azureml://datastores/workspaceblobstore/paths/batch-scores/{a.deployment}/{tag}/scores.csv")}), f"invoke {tag}")
    except HttpResponseError as e:
        if "OutputPathAlreadyExists" in str(e) or "already exists" in str(e):
            runs[tag] = "already scored (earlier run)"; print(tag, "already scored, skipping", flush=True); continue
        raise
    print(tag, "submitted", job.name, flush=True)
    final = wait(job.name)
    if "complet" not in final:
        raise SystemExit(f"{tag}: job {job.name} ended as {final}; open it in Studio (Jobs) or ask for the logs, then rerun")
    runs[tag] = job.name; print(tag, "completed", flush=True)
if a.start == 0 and a.limit == 36: json.dump(runs, open("azure_ml/batch_runs.json", "w"), indent=1)
print("months covered:", len(runs), "| new batch jobs this run:", sum(not v.startswith("already") for v in runs.values()))
