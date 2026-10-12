"""Build the development baseline and the config the monitoring job needs, then register them as data assets.
python azure_ml/prepare_monitoring_assets.py"""
import json, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_code"))
import pandas as pd
import monitor_lib as ml

ch = json.load(open("data/interim/champion.json")); V = ch["vars"]; s7 = json.load(open("data/interim/step7.json"))
bins = json.load(open("azure_ml/batch_model/bins.json"))
snaps = pd.read_parquet("data/interim/snapshots.parquet"); snaps["snap"] = pd.to_datetime(snaps["snap"])
base = ml.build_baseline(snaps[snaps.snap <= "2017-12-01"], pd.read_parquet("data/interim/dev_scored.parquet"), bins, V)
os.makedirs("azure_ml/monitoring_assets", exist_ok=True)
json.dump(base, open("azure_ml/monitoring_assets/baseline.json", "w")); json.dump(bins, open("azure_ml/monitoring_assets/bins.json", "w"))
json.dump({"vars": V, "grade_pd": s7["grade_pd"]}, open("azure_ml/monitoring_assets/config.json", "w"))
print("assets written; register with: az ml data create --name pd-monitoring-assets --version 1 --type uri_folder --path azure_ml/monitoring_assets")
