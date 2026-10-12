"""Step 8b: build the MI pack (PDF + PNG charts) and results.md section from step8.pkl.
Usage: python -I src/run_step8b.py"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from metrics import rag
D = pickle.load(open("data/interim/step8.pkl", "rb")); res, gs, lm, cov = D["res"], D["gs"], D["loanmonth"], D["cov"]
O, M = res["OOT Dec 2018"], res["Dec 2019"]
STATUS = {"Green": "#2e7d4f", "Amber": "#b87800", "Red": "#b3261e"}
GREY, BLUE, ORANGE = "#8a8f98", "#2a6f97", "#d1660f"
dev_share = O["share"].index  # grades 1..9
import json; s7 = json.load(open("data/interim/step7.json"))
dev = pd.read_parquet("data/interim/dev_scored.parquet"); dshare = dev.grade.value_counts(normalize=True).sort_index()

def style(ax):
    ax.grid(axis="y", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---- charts
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (k, r) in zip(axs, res.items()):
    bt = r["bt"]; x = np.arange(len(bt))
    ax.bar(x - 0.19, bt.expected / bt.loans * 100, 0.36, color=GREY, label="Expected (grade PD)")
    ax.bar(x + 0.19, bt.observed * 100, 0.36, color=ORANGE if "2019" in k else BLUE, label="Actual")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(bt.grade); ax.set_xlabel("Grade (1 = lowest risk)")
    ax.set_title(f"{k}: actual vs expected default rate", fontsize=10, loc="left"); style(ax); ax.legend(frameon=False, fontsize=8, loc="upper left")
axs[0].set_ylabel("12-month default rate (%, log scale)"); plt.tight_layout(); plt.savefig("reports/mi_ae_by_grade.png", dpi=150); plt.close()
f_ae = "reports/mi_ae_by_grade.png"

fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(9)
for off, (lab, sh, c) in zip((-0.27, 0, 0.27), (("Development", dshare, GREY), ("OOT Dec 2018", O["share"], BLUE), ("Dec 2019", M["share"], ORANGE))):
    ax.bar(x + off, sh.reindex(range(1, 10), fill_value=0) * 100, 0.26, color=c, label=lab)
ax.set_xticks(x); ax.set_xticklabels(range(1, 10)); ax.set_xlabel("Grade"); ax.set_ylabel("Share of loans (%)"); style(ax)
ax.set_title("Grade distribution", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("reports/mi_grade_distribution.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 3.6)); labs = [g[0] for g in gs]; vals = [g[1] for g in gs]
ax.bar(range(len(gs)), vals, 0.55, color=[GREY] * 3 + [BLUE, ORANGE])
for i, v in enumerate(vals): ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(len(gs))); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 1); ax.set_ylabel("Gini"); style(ax)
ax.set_title("Gini by snapshot (champion, calibrated)", fontsize=10, loc="left"); plt.tight_layout(); plt.savefig("reports/mi_gini_by_snapshot.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 3.8)); c19 = M["csi"].sort_values()
ax.barh(c19.index, c19.values, color=ORANGE, height=0.55)
for t in (0.10, 0.25): ax.axvline(t, color="#555555", lw=0.9, ls=":")
ax.text(0.10, len(c19) - 0.35, " 0.10", fontsize=8); ax.text(0.25, len(c19) - 0.35, " 0.25", fontsize=8)
for i, v in enumerate(c19.values): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("CSI vs development (WoE bins)"); ax.grid(axis="x", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Characteristic stability by variable, Dec 2019 (sorted)", fontsize=10, loc="left"); plt.tight_layout(); plt.savefig("reports/mi_csi_dec2019.png", dpi=150); plt.close()

# ---- RAG table rows
rows = [("PSI, grade mix", "<0.10 / .10-.25 / >.25", f"{O['psi_grade']:.4f}", O["rag_psi_grade"], f"{M['psi_grade']:.4f}", M["rag_psi_grade"]),
        ("PSI, score deciles", "<0.10 / .10-.25 / >.25", f"{O['psi_score']:.4f}", O["rag_psi_score"], f"{M['psi_score']:.4f}", M["rag_psi_score"]),
        ("Gini (development %.4f)" % O["gini_dev"], "drop <10% / 10-20% / >20%", f"{O['gini']:.4f} (drop {O['gini_drop']:+.1%})", O["rag_gini"], f"{M['gini']:.4f} (drop {M['gini_drop']:+.1%})", M["rag_gini"]),
        ("Grade back-test (worst grade)", "p >=.05 / .01-.05 / <.01", "G%d A%d R%d" % tuple(O["bt_counts"].get(k, 0) for k in ("Green", "Amber", "Red")), O["bt_worst"], "G%d A%d R%d" % tuple(M["bt_counts"].get(k, 0) for k in ("Green", "Amber", "Red")), M["bt_worst"]),
        ("Overall actual / expected", "0.8-1.2 / 0.7-1.5 / outside", f"{O['ae']:.3f} ({O['actual']:.3%} vs {O['expected']:.3%})", O["rag_ae"], f"{M['ae']:.3f} ({M['actual']:.3%} vs {M['expected']:.3%})", M["rag_ae"]),
        ("CSI, worst variable", "<0.10 / .10-.25 / >.25", f"{O['csi'].max():.4f} ({O['csi'].idxmax()})", O["rag_csi_max"], f"{M['csi'].max():.4f} ({M['csi'].idxmax()})", M["rag_csi_max"])]
adj_ae = M["adj_ae"]; adj_rag = "Green" if 0.8 <= adj_ae <= 1.2 else ("Amber" if 0.7 <= adj_ae <= 1.5 else "Red")
adj_rows = [("Dec 2019 Gini, forbearance-cured defaults treated as non-default", f"{M['gini_adj']:.4f} (drop {(M['gini_dev'] - M['gini_adj']) / M['gini_dev']:+.1%})",
             rag((M['gini_dev'] - M['gini_adj']) / M['gini_dev'], 0.10, 0.20)),
            ("Dec 2019 actual / expected, same adjustment", f"{adj_ae:.3f} ({M['adj_rate']:.3%} vs {M['expected']:.3%})", adj_rag)]
ba = M["bt_adj"].light.value_counts().to_dict()
adj_rows.append(("Dec 2019 grade back-test, same adjustment", "G%d A%d R%d" % tuple(ba.get(k, 0) for k in ("Green", "Amber", "Red")), "Red" if ba.get("Red", 0) else ("Amber" if ba.get("Amber", 0) else "Green")))

# ---- PDF
def page(pdf, title, status=None, note=None):
    fig = plt.figure(figsize=(11.69, 8.27)); fig.text(0.05, 0.94, title, fontsize=16, weight="bold")
    if status:
        fig.patches.append(FancyBboxPatch((0.82, 0.925), 0.13, 0.045, boxstyle="round,pad=0.004", transform=fig.transFigure, color=STATUS[status]))
        fig.text(0.885, 0.9475, status.upper(), color="white", ha="center", va="center", fontsize=12, weight="bold")
    if note: fig.text(0.05, 0.9, textwrap.fill(note, 140), fontsize=9.5, color="#333333", va="top")
    return fig

def table(fig, cells, header, rect, colw=None, rag_col=None, fs=8.5):
    ax = fig.add_axes(rect); ax.axis("off")
    t = ax.table(cellText=cells, colLabels=header, loc="upper left", cellLoc="left", colWidths=colw)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0: cell.set_facecolor("#eeeeee"); cell.set_text_props(weight="bold")
        elif rag_col is not None and c in rag_col:
            v = cells[r - 1][c]
            if v in STATUS: cell.set_facecolor(STATUS[v]); cell.set_text_props(color="white", weight="bold")
    return ax

def img(fig, path, rect):
    ax = fig.add_axes(rect); ax.axis("off"); ax.imshow(plt.imread(path))

with PdfPages("reports/MI_pack.pdf") as pdf:
    f = page(pdf, "Model monitoring pack: 12-month behavioural PD scorecard", None,
             "Baseline: development sample (Dec 2015-2017). Tested: OOT Dec 2018 and monitoring Dec 2019 (COVID). RAG thresholds are the analyst's choice (industry rules of thumb).")
    table(f, [list(r[:3]) + [r[3], r[4], r[5]] for r in rows], ["Test", "Thresholds (G / A / R)", "OOT Dec 2018", "RAG", "Dec 2019", "RAG"], (0.05, 0.50, 0.9, 0.36), [0.2, 0.2, 0.2, 0.07, 0.2, 0.07], rag_col=[3, 5])
    f.text(0.05, 0.47, "Dec 2019 after adjusting for COVID forbearance (see page 7)", fontsize=11, weight="bold")
    table(f, [list(r) for r in adj_rows], ["Test", "Value", "RAG"], (0.05, 0.33, 0.9, 0.12), [0.55, 0.3, 0.1], rag_col=[2])
    f.text(0.05, 0.22, textwrap.fill("Reading: the Dec 2019 failure is in the outcome, not the inputs. Population (PSI) is stable, but 93% of the 9,059 reported defaults were forborne loans that cured or paid off; only 136 ended in a credit-event termination. With forbearance-cured loans treated as non-defaults, Gini is back to 0.80 and actual/expected is 0.86. The grade back-test still shows grade 8 Red and grade 7 Amber, the same slope issue seen in OOT.", 120), fontsize=10, va="top")
    pdf.savefig(f); plt.close(f)

    f = page(pdf, "Test 1: Population stability (PSI)", "Green" if "Red" not in (O["rag_psi_grade"], M["rag_psi_grade"]) and "Amber" not in (O["rag_psi_grade"], M["rag_psi_grade"]) else "Amber",
             "PSI = sum (A-E) ln(A/E); E = development share per grade / score decile, A = share now. Grade mix and score distribution are stable in both snapshots.")
    img(f, "reports/mi_grade_distribution.png", (0.05, 0.38, 0.62, 0.5))
    table(f, [["OOT Dec 2018", f"{O['psi_grade']:.4f}", f"{O['psi_score']:.4f}"], ["Dec 2019", f"{M['psi_grade']:.4f}", f"{M['psi_score']:.4f}"]], ["Sample", "PSI grade", "PSI score"], (0.05, 0.25, 0.5, 0.1))
    pdf.savefig(f); plt.close(f)

    f = page(pdf, "Test 2: Discrimination (Gini)", M["rag_gini"], "Relative drop vs development Gini %.4f. Thresholds: <10%% Green, 10-20%% Amber, >20%% Red." % O["gini_dev"])
    img(f, "reports/mi_gini_by_snapshot.png", (0.05, 0.4, 0.62, 0.5))
    table(f, [[g[0], f"{g[1]:.4f}", f"{g[2]:,}"] for g in gs] + [["Dec 2019, forbearance-cured defaults as non-default", f"{M['gini_adj']:.4f}", f"{M['n']:,}"]], ["Snapshot", "Gini", "Loans"], (0.05, 0.05, 0.6, 0.3), [0.6, 0.2, 0.2])
    pdf.savefig(f); plt.close(f)

    f = page(pdf, "Test 3: Grade back-test, actual vs expected by grade", M["bt_worst"],
             "One-sided binomial test per grade (Green p>=0.05, Amber 0.01-0.05, Red <0.01). OOT: 7 Green, 1 Amber, 1 Red. Dec 2019: all 9 grades Red, observed 1.6x to 21x expected.")
    img(f, f_ae, (0.03, 0.5, 0.94, 0.4))
    bt = M["bt"]; bo = O["bt"]
    table(f, [[g, f"{a:.4%}", f"{o:.4%}", l1, f"{d:.4%}", l2] for g, a, o, l1, d, l2 in zip(bo.grade, bo.grade_pd, bo.observed, bo.light, bt.observed, bt.light)],
          ["Grade", "Grade PD", "OOT observed", "RAG", "Dec 2019 observed", "RAG"], (0.05, 0.05, 0.9, 0.4), [0.12, 0.2, 0.2, 0.12, 0.2, 0.12], rag_col=[3, 5])
    pdf.savefig(f); plt.close(f)

    f = page(pdf, "Test 4: Overall actual vs expected default rate", M["rag_ae"], "Thresholds: 0.8-1.2 Green; 0.7-0.8 or 1.2-1.5 Amber; outside Red.")
    table(f, [["OOT Dec 2018", f"{O['actual']:.4%}", f"{O['expected']:.4%}", f"{O['ae']:.3f}", O["rag_ae"]],
              ["Dec 2019 (reported)", f"{M['actual']:.4%}", f"{M['expected']:.4%}", f"{M['ae']:.3f}", M["rag_ae"]],
              ["Dec 2019 (forbearance-cured excluded)", f"{M['adj_rate']:.4%}", f"{M['expected']:.4%}", f"{adj_ae:.3f}", adj_rag]],
          ["Sample", "Actual", "Expected", "A / E", "RAG"], (0.05, 0.6, 0.9, 0.22), [0.38, 0.14, 0.14, 0.14, 0.1], rag_col=[4])
    pdf.savefig(f); plt.close(f)

    f = page(pdf, "Test 5: Characteristic stability (CSI) per variable", M["rag_csi_max"], "CSI = PSI on each variable's WoE bins vs development. Dec 2019 flags upb_ratio only (0.49): the sample has no 2019 originations, so the book seasons and pays down.")
    img(f, "reports/mi_csi_dec2019.png", (0.05, 0.38, 0.62, 0.5))
    table(f, [[v, f"{O['csi'][v]:.4f}", f"{M['csi'][v]:.4f}"] for v in M["csi"].index], ["Variable", "CSI OOT Dec 2018", "CSI Dec 2019"], (0.05, 0.05, 0.6, 0.3), [0.4, 0.3, 0.3])
    pdf.savefig(f); plt.close(f)

    f = page(pdf, "Finding: COVID forbearance and the Dec 2019 'defaults'", None, "Loan-months Apr-Dec 2020 by assistance flag, and what happened to the 9,059 Dec 2019 defaults by end of data (Mar 2026).")
    table(f, [[r.g, f"{r.rows:,}", f"{r.dq1:.1%}", f"{r.dq3:.1%}"] for r in lm.itertuples()], ["Assistance flag", "Loan-months", "30+ DPD", "90+ DPD"], (0.05, 0.72, 0.9, 0.14), [0.4, 0.2, 0.2, 0.2])
    t19 = cov["Dec 2019"]; cols = [c for c in t19.columns]
    table(f, [[i] + [f"{t19.loc[i, c]:,}" for c in cols] for i in t19.index], ["Outcome of Dec 2019 default"] + cols, (0.05, 0.42, 0.9, 0.2), [0.5, 0.17, 0.17, 0.16])
    pdf.savefig(f); plt.close(f)

    f = page(pdf, "Methodology and thresholds", None, "")
    f.text(0.05, 0.85, "- Baseline: development sample (train + holdout, Dec 2015-2017, 585,900 loan-snapshots).\n- PSI on 9 grade shares and on development score deciles; CSI on each model variable's WoE bins.\n"
           "- Gini drop is relative to the pooled development Gini.\n- Back-test: one-sided binomial per grade against grade PD (development mean calibrated PD).\n"
           "- RAG thresholds are common industry rules of thumb, chosen by the analyst, not regulatory values.\n- Forbearance fields: Borrower Assistance Plan (F/T/R), Payment Deferral Flag (P/C), Delinquency Due to Disaster (Y).\n"
           "- Outcomes after Dec 2020 use data to Mar 2026, i.e. hindsight a monitoring team would not have had at the time.", fontsize=11, va="top")
    pdf.savefig(f); plt.close(f)

# ---- results.md
md = lambda df: ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * df.shape[1] + "\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False)))
t = open("results.md").read()
if "\n## Step 8:" in t: t = t[:t.index("\n## Step 8:")]
t = t.replace("| Stability (PSI, Step 8) | pending | pending |", f"| Stability (PSI, Step 8) | grade-mix PSI {O['psi_grade']:.4f} (OOT), {M['psi_grade']:.4f} (Dec 2019) | not computed |")
rag_df = pd.DataFrame([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows], columns=["Test", "Thresholds (G/A/R)", "OOT Dec 2018", "RAG", "Dec 2019", "RAG "])
btO = O["bt"].assign(snapshot="OOT Dec 2018"); btM = M["bt"].assign(snapshot="Dec 2019"); btA = M["bt_adj"].assign(snapshot="Dec 2019 adjusted")
bt_all = pd.concat([btO, btM, btA]); bt_all = pd.DataFrame({"snapshot": bt_all.snapshot, "grade": bt_all.grade, "loans": bt_all.loans, "defaults": bt_all.defaults, "expected": bt_all.expected.round(1),
    "observed": bt_all.observed.map("{:.4%}".format), "obs/PD": bt_all.ratio.round(2), "p binomial": bt_all.p_binomial.map("{:.4f}".format), "light": bt_all.light})
cs = pd.DataFrame({"variable": M["csi"].index, "CSI OOT Dec 2018": [round(O["csi"][v], 4) for v in M["csi"].index], "CSI Dec 2019": M["csi"].round(4).values})
with open("results.md", "w") as f:
    f.write(t + "\n## Step 8: monitoring MI pack (reports/MI_pack.pdf)\n\nThresholds are my choice (industry rules of thumb). Baseline = development sample.\n\n### RAG table\n\n" + md(rag_df))
    f.write(f"\nGini by snapshot: " + "; ".join(f"{a} {b:.4f}" for a, b, _ in gs) + "\n\n### CSI per variable\n\n" + md(cs))
    f.write("\n### Grade back-test, all snapshots\n\n" + md(bt_all))
    f.write("\n### COVID check: delinquency by assistance flag, loan-months Apr-Dec 2020\n\n" + md(lm.assign(dq1=lm.dq1.map("{:.1%}".format), dq3=lm.dq3.map("{:.1%}".format)).rename(columns={"g": "flag", "dq1": "30+ DPD", "dq3": "90+ DPD"})))
    for k in ("OOT Dec 2018", "Dec 2019"):
        tt = cov[k].reset_index(); tt.columns = [str(c) for c in tt.columns]
        f.write(f"\n### What happened to the {k} defaults (data to Mar 2026)\n\n" + md(tt))
    f.write(f"\nDec 2019 defaults treated as non-defaults (forborne and later cured or paid off, no credit-event termination): {M['adj_excl']:,} of {M['defaults']:,}. For OOT Dec 2018 the same rule would exclude {D['excl18']:,} of {O['defaults']:,}.\n")
    f.write(f"\nDec 2019 adjusted: default rate {M['adj_rate']:.4%}, actual/expected {M['adj_ae']:.3f}, Gini {M['gini_adj']:.4f} (all loans) / {M['gini_adj_nocur']:.4f} (cur_dq = 0 only); grade back-test lights {ba}.\n")
print("ok")
