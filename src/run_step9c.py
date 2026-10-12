"""Step 9c: write the Step 9 section of results.md from step9.pkl. Usage: python -I src/run_step9c.py"""
import pickle, json
import pandas as pd
D = pickle.load(open("data/interim/step9.pkl", "rb")); st, tab, sens, dec, pma, beta = D["st"], D["tab"], D["sens"], D["decomp"], D["pma"], D["beta"]
s8 = pickle.load(open("data/interim/step8.pkl", "rb"))["res"]; M = s8["Dec 2019"]
md = lambda df: ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * df.shape[1] + "\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False)))
usd = lambda x: f"${x/1e6:,.2f}m"
bal = D["bal"]
S = st.reset_index()
stage_tab = pd.DataFrame({"stage": S.stage, "loans": S.loans.map("{:,}".format), "balance": S.balance.map(usd), "share of balance": S.share_balance.map("{:.2%}".format), "mean 12m PD": S.mean_pd.map("{:.3%}".format)})
scen_tab = pd.DataFrame({"scenario": tab.scenario, "dU (pp)": tab.dU_pp.map(lambda v: "" if pd.isna(v) else f"{v:+.1f}"), "weight": tab.weight.map("{:.0%}".format),
                         "ECL": tab.ecl.map(usd), "of which Stage 1": tab.stage1.map(usd), "of which Stage 2": tab.stage2.map(usd), "coverage (ECL / balance)": tab.coverage.map("{:.4%}".format)})
sens_tab = pd.DataFrame([[f"{l:.0%}", usd(v[0]), f"{v[1]:.4%}", usd(v[2]), usd(v[3])] for l, v in sens.items()], columns=["LGD", "weighted ECL", "coverage", "Base ECL", "Downside ECL"])
b, A, B = beta["spec"], beta["spec"], beta["ex_covid"]
t = open("results.md").read()
if "\n## Step 9:" in t: t = t[:t.index("\n## Step 9:")]
with open("results.md", "w") as f:
    f.write(t + "\n## Step 9: IFRS 9 extension, Dec 2019 portfolio (227,165 loans, " + usd(bal) + " balance)\n\n")
    f.write("Illustrative assumptions (mine): scenarios and weights below; LGD 10%; discount at original rate; no prepayment; Stage 3 out of scope (loans already in default are excluded from the snapshots).\n\n")
    f.write("### 9a Staging (Stage 2 if 30+ days past due, or PD >= 2.5x PD at first observation and PD > 0.3%)\n\n" + md(stage_tab))
    f.write(f"\nStage 2 triggers: {D['n_s2dq']:,} loans 30+ DPD; {D['n_s2rel']:,} on the relative-PD test only. Stage 3: out of scope.\n\n")
    f.write(f"### 9b Lifetime PD\n\nMean 12-month PD {D['life'][0]:.3%}; mean lifetime PD {D['life'][1]:.3%} (Stage 1 {D['life'][2]:.3%}, Stage 2 {D['life'][3]:.3%}). Base hazard by loan age (years): "
            + ", ".join(f"{int(k)}: {v:.3%}" for k, v in D["hb"].items()) + "; held flat beyond year 6 (no data).\n\n")
    f.write(f"### 9d Unemployment regression: logit(default rate) = alpha + beta x dU (84 month-end snapshots, Jan 2013 - Dec 2019; dU = change in UNRATE over the 12-month outcome window)\n\n")
    f.write(f"- Specification as set (2013-2019, n = {A['n']}): alpha = {A['alpha']:.4f}, **beta = {A['beta']:.4f}**, standard error {A['se_ols']:.4f} (OLS) / {A['se_hac']:.4f} (Newey-West, 12 lags), R-squared {A['r2']:.3f}\n")
    f.write(f"- Sensitivity, pre-COVID snapshots only (2013-2018, n = {B['n']}): beta = {B['beta']:.4f}, standard error {B['se_ols']:.4f} (OLS) / {B['se_hac']:.4f} (Newey-West), R-squared {B['r2']:.3f}\n")
    f.write(f"- Scenario weights (assumed): Upside -0.5 pp 20%, Base 0.0 pp 50% (the table's base value was garbled; assumed 0), Downside +3.0 pp 30%. Beta used for ECL: {A['beta']:.4f}.\n\n")
    f.write("### 9c ECL by scenario (LGD 10%)\n\n" + md(scen_tab))
    f.write(f"\nWeighted minus base: {usd(dec['weighted'] - dec['base'])}. Of this, {usd(dec['at_mean'] - dec['base'])} comes from the weights being skewed to the downside (expected dU = {dec['dU_mean']:+.1f} pp, ECL at that dU {usd(dec['at_mean'])}) and {usd(dec['weighted'] - dec['at_mean'])} from convexity.\n")
    f.write(f"Share of weighted ECL: Stage 2 {D['share'][1]:.1%}; loans 30+ DPD {D['share'][0]:.1%}.\n\nWith the pre-COVID beta ({B['beta']:.4f}): weighted ECL {usd(D['EX'][0])}, Base {usd(D['EX'][1])}, Downside {usd(D['EX'][2])}.\n\n")
    f.write("### LGD sensitivity\n\n" + md(sens_tab))
    f.write("\n### 9e Post model adjustment sizing (illustrative; base-scenario ECL = " + usd(pma['base']) + ")\n\n")
    f.write("| Reference point | ECL | Difference to model ECL |\n|---|---|---|\n")
    f.write(f"| Model as is (base scenario) | {usd(pma['base'])} | |\n")
    f.write(f"| PDs scaled by OOT actual/expected by grade (known at Dec 2019) | {usd(pma['slope'])} | {usd(pma['slope'] - pma['base'])} ({pma['slope'] / pma['base'] - 1:+.1%}) |\n")
    f.write(f"| PDs scaled to reported 2020 experience (A/E {M['ae']:.2f}) | {usd(pma['mech'])} | {usd(pma['mech'] - pma['base'])} ({pma['mech'] / pma['base'] - 1:+.0%}) |\n")
    f.write(f"| PDs scaled to forbearance-adjusted 2020 experience (A/E {M['adj_ae']:.2f}) | {usd(pma['adj'])} | {usd(pma['adj'] - pma['base'])} ({pma['adj'] / pma['base'] - 1:+.1%}) |\n")
    f.write(f"\nPMA needed to undo a mechanical recalibration to reported 2020 defaults: {usd(pma['adj'] - pma['mech'])} ({pma['adj'] / pma['mech'] - 1:+.0%} of that figure).\n")
    f.write("\n### Hurricane check on the development default spike: share of each snapshot's defaults with a disaster flag during the window\n\n| snapshot | defaults | with disaster flag | share |\n|---|---|---|---|\n")
    for s, n, fl, sh in D["hur"]: f.write(f"| {s} | {n:,} | {fl:,} | {sh:.1%} |\n")
print("done")
