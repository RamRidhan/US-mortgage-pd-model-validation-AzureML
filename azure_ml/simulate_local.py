"""Local rehearsal of Steps 12d-12g without Azure: score each month with the packaged champion, compute the monitoring
signals, and apply label delay (performance for month m is only computed at the run of month m+12).
Run from pd-mortgage/: python azure_ml/simulate_local.py   (writes reports/azure_sim_*.csv|png and a results.md section)"""
import json, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_code"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import monitor_lib as ml

ch = json.load(open("data/interim/champion.json")); V = ch["vars"]
bins = json.load(open("azure_ml/batch_model/bins.json")); grade_pd = {int(k): v for k, v in json.load(open("data/interim/step7.json"))["grade_pd"].items()}
snaps = pd.read_parquet("data/interim/snapshots.parquet"); snaps["snap"] = pd.to_datetime(snaps["snap"])
dev_in = snaps[snaps.snap <= "2017-12-01"]
base = ml.build_baseline(dev_in, pd.read_parquet("data/interim/dev_scored.parquet"), bins, V)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_code"))
import score as batch_score          # the same scoring script that runs in the batch endpoint
CFG = json.load(open("azure_ml/batch_model/config.json"))
months = [f"{m:%Y-%m}" for m in pd.date_range("2018-01-01", "2020-12-01", freq="MS")]
scores, rows, perf = {}, [], []
for run_i, tag in enumerate(months):
    inp = pd.read_parquet(f"data/interim/monthly/inputs/{tag}.parquet")
    sc = batch_score.score_frame(inp, CFG, bins); scores[tag] = sc
    r = ml.month_metrics(inp, sc, base, bins, V, csi_exclude=("upb_ratio",))   # upb_ratio drift is a sample artefact (no new originations)
    rows.append(dict(month=tag, loans=len(inp), psi_grade=r["psi_grade"], dq_ratio=r["dq_incidence_ratio"], csi_max=max(r["csi"].values()), csi_max_var=max(r["csi"], key=r["csi"].get),
                     **{f"csi_{v}": x for v, x in r["csi"].items()}, n_dq_flags=sum(x["flag"] for x in r["dq"].values()),
                     dq_flagged=";".join(k for k, x in r["dq"].items() if x["flag"]), status=r["status"], alerts=" | ".join(r["alerts"])))
    # label delay: at this run, only the month 12 months earlier has a complete outcome window
    if run_i >= 12:
        old = months[run_i - 12]
        lab = pd.read_parquet(f"data/interim/monthly/labels/{old}.parquet")
        p = ml.performance_metrics(scores[old], lab, grade_pd)
        perf.append(dict(scored_month=old, evaluated_at_run=tag, **{k: v for k, v in p.items() if k != "lights"}, lights="".join(p["lights"][g][0] for g in sorted(p["lights"]))))
M = pd.DataFrame(rows); Pf = pd.DataFrame(perf)
M.to_csv("reports/azure_sim_monthly_signals.csv", index=False); Pf.to_csv("reports/azure_sim_performance_lagged.csv", index=False)
print(M[["month", "loans", "psi_grade", "dq_ratio", "csi_max", "csi_max_var", "n_dq_flags", "status"]].round(4).to_string())
print(Pf[["scored_month", "evaluated_at_run", "ae", "gini", "gini_drop", "rag_gini", "rag_ae", "n_red_grades"]].round(3).to_string())

first = {}
for name, cond in {"CSI > 0.25 (any variable)": M.csi_max > 0.25,
                   "CSI > 0.25 excluding upb_ratio": M[[f"csi_{v}" for v in V if v != "upb_ratio"]].max(axis=1) > 0.25,
                   "Grade PSI amber (> 0.10)": M.psi_grade > 0.10, "Delinquent share > 1.5x development (amber)": M.dq_ratio > 1.5, "Delinquent share > 2x development (red)": M.dq_ratio > 2.0, "Grade PSI red (> 0.25)": M.psi_grade > 0.25,
                   "Data quality flag": M.n_dq_flags > 0}.items():
    first[name] = M.month[cond].iloc[0] if cond.any() else "never"
first["Gini or A/E not Green (performance)"] = (Pf.evaluated_at_run[(Pf.rag_gini != "Green") | (Pf.rag_ae != "Green")].iloc[0] if ((Pf.rag_gini != "Green") | (Pf.rag_ae != "Green")).any() else "never")
first["Performance first red, scored month (A/E red)"] = Pf.scored_month[Pf.rag_ae == "Red"].iloc[0] if (Pf.rag_ae == "Red").any() else "never"
print(first)

# exhibit
x = pd.to_datetime(M.month); xp = pd.to_datetime(Pf.evaluated_at_run)
fig, ax = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True)
BLUE, ORANGE, GREY = "#2a6f97", "#d1660f", "#444444"
ax[0].plot(x, M.psi_grade.clip(lower=1e-4), color=BLUE, lw=2, ls="--"); ax[0].plot(x, M.csi_cur_dq.clip(lower=1e-4), color=ORANGE, lw=2)
ax[0].plot(x, M.csi_months_since_dq, color=ORANGE, lw=1.3, ls=":")
for t in (0.10, 0.25): ax[0].axhline(t, color="#777777", lw=0.8, ls=":")
ax[0].text(x.iloc[0], 0.27, "red 0.25", fontsize=7, color="#555555"); ax[0].text(x.iloc[0], 0.108, "amber 0.10", fontsize=7, color="#555555")
ax[0].set_yscale("log"); ax[0].set_ylim(5e-4, 0.6); ax[0].set_ylabel("PSI / CSI (log scale)")
ax[0].text(x.iloc[-1], M.psi_grade.iloc[-1], " PSI grade mix", color=BLUE, fontsize=8, va="center")
ax[0].text(x.iloc[-1], M.csi_cur_dq.iloc[-1] * 0.6, " CSI cur_dq", color=ORANGE, fontsize=8, va="center")
ax[0].text(x.iloc[-1], M.csi_months_since_dq.iloc[-1] * 1.5, " CSI months_since_dq", color=ORANGE, fontsize=8, va="center")
ax[0].set_title("Early warning 1: PSI / CSI as specified (peaks just under amber; never red)", fontsize=10, loc="left")
ax[1].plot(x, M.dq_ratio, color=GREY, lw=2)
for t, l in ((1.5, "amber 1.5x"), (2.0, "red 2x")): ax[1].axhline(t, color="#777777", lw=0.8, ls=":"); ax[1].text(x.iloc[0], t + 0.1, l, fontsize=7, color="#555555")
ax[1].set_ylabel("Share 30+ DPD / development"); ax[1].set_title("Early warning 2: delinquent share (added rule; red from Apr 2020)", fontsize=10, loc="left")
ax[2].plot(xp, Pf.gini, color=BLUE, lw=2); ax[2].set_ylim(0, 1); ax[2].set_ylabel("Gini (blue)")
a2 = ax[2].twinx(); a2.plot(xp, Pf.ae, color=ORANGE, lw=2, ls="--"); a2.set_ylabel("Actual / expected (orange, dashed)"); a2.set_yscale("log")
ax[2].set_title("Performance: the month scored 12 months earlier, shown at the month the result becomes available", fontsize=10, loc="left")
for a in ax:
    a.grid(color="#dddddd", lw=0.6); a.spines["top"].set_visible(False)
ax[0].axvline(pd.Timestamp("2020-03-15"), color="#b3261e", lw=1, alpha=0.6); ax[0].text(pd.Timestamp("2020-03-20"), 3e-3, "COVID\nMar 2020", fontsize=7, color="#b3261e")
plt.tight_layout(); plt.savefig("reports/azure_sim_exhibit.png", dpi=150)

t = open("results.md").read()
if "\n## Azure phase" in t: t = t[:t.index("\n## Azure phase")]
md = lambda df: ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * df.shape[1] + "\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False)))
with open("results.md", "w") as f:
    f.write(t + "\n## Azure phase: local rehearsal of the monthly batch-and-monitor loop (no Azure resources used)\n\n"
            "36 month-end snapshots (Jan 2018 - Dec 2020) scored with the deployed scoring script (batch_code/score.py) (reproduces Step 7 PD and grade exactly), monitoring signals computed per month, performance lagged 12 months. "
            "The Azure run (model versions, endpoint type, batch runs, alert timing in the workspace, monthly cost) is to be filled in after deployment.\n\n"
            "### First month each alert fires (local rehearsal)\n\n" + md(pd.DataFrame(list(first.items()), columns=["signal", "first month"])) +
            "\n### Monthly signals\n\n" + md(M[["month", "loans", "psi_grade", "dq_ratio", "csi_max", "csi_max_var", "n_dq_flags", "status"]].round(4)) +
            "\n### Lagged performance (month scored, run in which it becomes available)\n\n" + md(Pf[["scored_month", "evaluated_at_run", "defaults", "actual", "expected", "ae", "gini", "gini_drop", "rag_gini", "rag_ae", "n_red_grades"]].round(4)) +
            "\nChart: reports/azure_sim_exhibit.png\n")
