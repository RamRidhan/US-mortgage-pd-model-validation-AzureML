# Azure ML deployment and monitoring: the sequence that worked

Run everything from the repository root with the project environment (`pip install -r requirements.txt -r requirements-azure.txt`).
Total cost of my run: GBP 1.42 (36 monthly batch runs, 36 monitoring runs).

## 1. Account, workspace, budget
- Azure account (Azure for Students gives a $100 credit for 12 months). In the portal, Cost Management, Budgets: create a budget with email alerts before running anything (I used $20).
- Create the workspace. If creation fails with a Key Vault `accessPolicies/write` error, create the vault first with `--enable-rbac-authorization false` and pass it to `az ml workspace create --key-vault`.
- `cp azure_ml/config.example.json azure_ml/config.json` and fill in the subscription ID (the file is git-ignored). Then `az login` on your machine.

## 2. Export, test and rehearse locally first
```
python azure_ml/export_plain_artifacts.py          # champion bins and coefficients as plain JSON (checked against optbinning)
python azure_ml/test_batch_score_local.py          # the batch scoring script matches Step 7 on all 254,118 Dec 2018 loans
python azure_ml/build_monthly_snapshots.py         # 36 month-end snapshots, inputs and labels kept apart
python azure_ml/simulate_local.py                  # whole loop without Azure: the expected result
```

## 3. Deploy
```
python azure_ml/setup_workspace.py                 # zero-scale cluster, monthly snapshot and label assets (private storage)
python azure_ml/deploy_batch.py                    # champion as a custom model, small environment, batch endpoint pd-batch
python azure_ml/run_batches.py --limit 1           # one-month test first
python azure_ml/run_batches.py                     # all 36 months (rerunnable: scored months are skipped)
```
Why a custom model and not the MLflow no-code deployment: the auto-generated environment failed to build (pip backtracked to an unbuildable pyarrow with Python 3.12, numpy 2.5 and pandas 3). The deployed scorer (`batch_code/score.py`) needs only numpy and pandas (Python 3.10 environment, `batch_env.yml`). See D59 in `decisions.md`.

## 4. Monitor
```
python azure_ml/prepare_monitoring_assets.py       # development baseline and bins as JSON
az ml data create --name pd-monitoring-assets --version 1 --type uri_folder --path azure_ml/monitoring_assets -g <resource-group> -w <workspace>
python azure_ml/run_monitoring.py --from 2018-01 --to 2018-01   # test one month (expect Green)
python azure_ml/run_monitoring.py --from 2018-01 --to 2020-12   # all months; red months end in "failed" on purpose
python azure_ml/log_summary_run.py                 # one run with all months as trend lines in Studio
```
Set `ALERT_WEBHOOK_URL` (Teams or Logic App) to post alerts; without it a red month only fails the job and prints the alert. See `runbook.md` for who is alerted and when the model is escalated.

## Things that went wrong (and the fix)
- Batch output `scores.csv` is space-separated with no header (D61).
- Alerts needed two corrections after local testing on real files (D62): `upb_ratio` excluded from CSI alerting, and three or more red grades required for a back-test alert.
- Network drops: `run_batches.py` and `run_monitoring.py` retry and are safe to rerun.

## Clean up
`az group delete -n <resource-group> --yes` (deletes the workspace, endpoint, runs and charts: save screenshots first).
