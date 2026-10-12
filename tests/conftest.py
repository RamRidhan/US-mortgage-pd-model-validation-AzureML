import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("src", os.path.join("azure_ml", "batch_code")):
    sys.path.insert(0, os.path.join(ROOT, p))
