"""Run the batch scoring script locally exactly as Azure would (init, then run on a mini-batch of files) and compare with Step 7."""
import os, sys, glob
sys.path.insert(0, "azure_ml/batch_code")
import numpy as np, pandas as pd
import score
os.environ["AZUREML_MODEL_DIR"] = os.path.abspath("azure_ml/batch_model")
score.init()
files = [f"data/interim/monthly/inputs/2018-12.parquet"]
out = score.run(files)
ref = pd.read_parquet("data/interim/oot_scored.parquet")[["loan_id", "pd", "grade", "score"]].rename(columns=lambda c: c if c == "loan_id" else c + "_ref")
m = out.merge(ref, on="loan_id", validate="1:1")
print("rows", len(out), "max |PD diff| %.2e" % (m.pd - m.pd_ref).abs().max(), "grade mismatches", int((m.grade != m.grade_ref).sum()), "max |score diff| %.2e" % (m.score - m.score_ref).abs().max())
assert (m.pd - m.pd_ref).abs().max() < 1e-9 and (m.grade == m.grade_ref).all() and len(m) == len(ref)
print("OK")
