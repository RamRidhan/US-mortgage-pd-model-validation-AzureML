"""Step 12c: register the champion (custom-model route), build a small environment and create the batch endpoint.
The auto-generated MLflow environment failed to build (pip backtracked to an unbuildable pyarrow on Python 3.12 / numpy 2.5 / pandas 3),
so the scorecard is deployed as plain JSON artifacts plus azure_ml/batch_code/score.py in a conservative environment.
python azure_ml/deploy_batch.py   (after: python azure_ml/export_plain_artifacts.py)"""
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.entities import (BatchDeployment, BatchEndpoint, BatchRetrySettings, CodeConfiguration, Environment, Model)
from common import get_client

ml = get_client()
model = ml.models.create_or_update(Model(name="mortgage_pd_champion", path="azure_ml/batch_model", type=AssetTypes.CUSTOM_MODEL,
                                         description="WoE logistic scorecard (champion): plain JSON bins and coefficients, scored by score.py"))
print("registered champion version", model.version)
env = ml.environments.create_or_update(Environment(name="pd-batch-env", image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
                                                   conda_file="azure_ml/batch_env.yml"))
print("environment version", env.version)
ml.batch_endpoints.begin_create_or_update(BatchEndpoint(name="pd-batch", description="Monthly mortgage PD scoring")).result()
for old in ("champion", "challenger"):             # remove the deployments that used the failed MLflow route
    try:
        ml.batch_deployments.begin_delete(name=old, endpoint_name="pd-batch").result(); print("deleted old deployment", old)
    except Exception as e:
        print("no old deployment", old, type(e).__name__)
dep = BatchDeployment(name="champion", endpoint_name="pd-batch", model=model, environment=env,
                      code_configuration=CodeConfiguration(code="azure_ml/batch_code", scoring_script="score.py"),
                      compute="cpu-cluster", instance_count=1, max_concurrency_per_instance=1, mini_batch_size=1,
                      output_action=BatchDeploymentOutputAction.APPEND_ROW, output_file_name="scores.csv",
                      retry_settings=BatchRetrySettings(max_retries=2, timeout=1200), error_threshold=-1, logging_level="info")
ml.batch_deployments.begin_create_or_update(dep).result()
ep = ml.batch_endpoints.get("pd-batch"); ep.defaults.deployment_name = "champion"
ml.batch_endpoints.begin_create_or_update(ep).result()
print("deployed champion; default deployment set")
