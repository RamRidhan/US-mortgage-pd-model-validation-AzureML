"""Step 12a: compute cluster that scales to zero, and upload the monthly snapshots to the PRIVATE workspace blob store.
Run once, after `az login` and creating the workspace (see azure_ml/IMPLEMENT.md).  python azure_ml/setup_workspace.py"""
import glob, os
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import Input
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from common import get_client

ml = get_client()
ml.compute.begin_create_or_update(AmlCompute(name="cpu-cluster", size="Standard_DS3_v2", min_instances=0, max_instances=2, idle_time_before_scale_down=120)).result()
print("compute ready")
# one folder-asset per month so each batch run takes a single month as input (data stays in workspace storage)
for f in sorted(glob.glob("data/interim/monthly/inputs/*.parquet")):
    tag = os.path.basename(f)[:-8]
    os.makedirs(f"data/interim/monthly/upload/{tag}", exist_ok=True)
    dst = f"data/interim/monthly/upload/{tag}/{tag}.parquet"
    if not os.path.exists(dst):
        import shutil; shutil.copy(f, dst)
    ml.data.create_or_update(Data(name=f"snapshot-{tag}", version="1", type=AssetTypes.URI_FOLDER, path=f"data/interim/monthly/upload/{tag}"))
    print("uploaded", tag)

# labels (known only 12 months later) are uploaded as separate assets; the monitoring job reads them only for the month 12 months back
for f in sorted(glob.glob("data/interim/monthly/labels/*.parquet")):
    tag = os.path.basename(f)[:-8]
    ml.data.create_or_update(Data(name=f"labels-{tag}", version="1", type=AssetTypes.URI_FILE, path=f))
    print("uploaded labels", tag)
