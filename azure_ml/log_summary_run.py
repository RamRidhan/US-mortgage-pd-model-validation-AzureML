"""Log all monthly monitoring results into ONE Azure ML run so Studio draws them as trend lines (Jobs > pd-monitoring-summary > Metrics).
x-axis = step (1 = 2018-01 ... 36 = 2020-12); the step-to-month mapping is in the run's parameters.
Run after run_monitoring.py has produced azure_ml/monitoring_runs.csv.   python azure_ml/log_summary_run.py"""
import pandas as pd
import mlflow
from common import get_client

ml = get_client()
mlflow.set_tracking_uri(ml.workspaces.get(ml.workspace_name).mlflow_tracking_uri)
mlflow.set_experiment("pd-monitoring-summary")
df = pd.read_csv("azure_ml/monitoring_runs.csv").sort_values("month").reset_index(drop=True)
full = pd.date_range("2018-01-01", "2020-12-01", freq="MS").strftime("%Y-%m")
df["step"] = [list(full).index(m) + 1 for m in df.month]
# delinquent-share ratio and status code are not in the csv: status is ordinal, ratio comes from the inputs, so log what the job logged
status_code = {"Green": 0, "Amber": 1, "Red": 2}
with mlflow.start_run(run_name="monitoring-summary"):
    for r in df.itertuples():
        step = int(r.step)
        for col in ("psi_grade", "csi_cur_dq", "csi_months_since_dq", "csi_fico", "csi_orig_rate", "csi_dti", "csi_cltv", "csi_upb_ratio", "dq_flags", "perf_gini", "perf_ae", "perf_n_red_grades"):
            v = getattr(r, col, None)
            if v is not None and pd.notna(v):
                mlflow.log_metric(col, float(v), step=step)
        mlflow.log_metric("status_0green_1amber_2red", status_code.get(r.status, -1), step=step)
    # artifact upload fails with mlflow 3.x + azureml-mlflow, so the step -> month mapping goes in parameters instead
    mapping = ",".join(f"{int(s)}={m}" for s, m in zip(df.step, df.month))
    mlflow.log_param("step_to_month_1", mapping[:480]); mlflow.log_param("step_to_month_2", mapping[480:960])
    mlflow.log_param("status_code", "0=Green 1=Amber 2=Red")
    print("logged", len(df), "months")
