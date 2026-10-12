"""Shared Azure ML client. Reads azure_ml/config.json (git-ignored; copy config.example.json)."""
import json, os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def get_client():
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")))
    return MLClient(DefaultAzureCredential(), cfg["subscription_id"], cfg["resource_group"], cfg["workspace_name"])
