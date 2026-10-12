"""Build reports/Validation_Report.pdf (SS1/23 style). Numbers are copied from results.md / decisions.md.
Usage: python -I src/make_validation_report.py"""
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

BODY = ParagraphStyle("b", fontName="Helvetica", fontSize=8.3, leading=10.4, spaceAfter=3)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=11, leading=13, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1f3b57"))
TITLE = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=2)
SUB = ParagraphStyle("s", fontName="Helvetica", fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=4)
CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=7, leading=8.4)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=9, bulletIndent=0, spaceAfter=1.5)
SEV = {"High": "#b3261e", "Medium": "#b87800", "Low": "#2e7d4f"}
P = lambda t, s=BODY: Paragraph(t, s)
def bullets(items): return [Paragraph(i, BUL, bulletText="•") for i in items]

def tbl(rows, widths, header=True, fs=7, zebra=True, align_right_from=None):
    data = [[Paragraph(str(c), CELLB if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef4")))
    t.setStyle(TableStyle(st)); return t

S = []
S.append(P("Independent Validation Report: 12-Month Behavioural PD Model, US Single-Family Mortgages", TITLE))
S.append(P("Model: WoE logistic scorecard (champion) with LightGBM challenger, Freddie Mac loan-level data, vintages 2012-2018. Report date 12 October 2026. Approach follows the structure of PRA SS1/23 model validation. "
           "<b>Independence statement:</b> the same person developed and validated this model. The review is written from the stance of a reviewer who did not build it, but it is an exercise in independence, not true independence. "
           "The report reflects the numbers in <i>results.md</i> and decisions in <i>decisions.md</i>.", SUB))

S.append(P("1. Summary and overall rating", H1))
S.append(P("<b>Overall rating: Fit for use with conditions</b> (rank-ordering and portfolio-level PD in benign conditions; not fit for regulatory capital or for use in a downturn until F1 and F2 are remediated). "
           "The scorecard rank-orders risk well: Gini 0.784 on the in-time holdout, 0.843 on the out-of-time (OOT) sample of Dec 2018 and 0.72 on loans that were current at the snapshot. It is conservative in aggregate on OOT "
           "(actual 0.449% against 0.482% expected, ratio 0.93). The scorecard was also deployed on an Azure ML batch endpoint and monitored monthly for 36 months; the deployed scores equal the local ones to 3e-16. The conditions arise from what the data cannot show: the calibration window has no downturn, two of nine grades fail back-testing, and in 2020 the 90-day default rule recorded payment holidays as defaults."))
S += bullets(["<b>F1 (High)</b> The PD is anchored to Dec 2015-2017 snapshots, when unemployment fell in every outcome window (-0.3, -0.6, -0.2 pp); the long-run average has no downturn.",
              "<b>F2 (High)</b> Dec 2019 snapshot (2020 window): reported actual/expected 7.53 and Gini 0.476, but 8,466 of 9,059 defaults were forborne and 8,477 cured or paid off; after adjustment actual/expected is 0.86 and Gini 0.805. The model cannot see forbearance.",
              "<b>F3 (Medium)</b> Grade 8 observed 4.03% against a 2.38% PD (109 defaults in 2,702 loans, binomial p &lt; 0.0001); grade 7 observed 1.32% against 1.06% (p = 0.024); grades 3 to 5 observed 0.49x to 0.57x of PD."])

S.append(P("2. Model scope and use", H1))
S.append(P("The model estimates the probability that a performing, active US conventional fixed-rate mortgage reaches 90+ days past due, REO or a credit-event termination within 12 months of a December snapshot. "
           "It is a behavioural (account-level, post-origination) model built on Freddie Mac's standard single-family dataset (350,000 sampled loans, 2012-2018 vintages). "
           "<b>It must not be used</b> for UK mortgages (different default experience, products, law and data), for application scoring (it uses payment history), for loans already in default (Stage 3 is out of scope), for adjustable-rate, interest-only or non-standard loans (none in the sample), "
           "or for regulatory capital until F1 is closed. Fixed-rate loans only; HARP refinances are included and flagged (15.8% of loan-snapshots)."))

S.append(P("3. Data", H1))
S.append(P("<b>Default definition.</b> 90+ days past due, REO acquisition, or zero-balance code 02, 03, 09 or 15 (third-party sale, short sale or charge-off, REO disposition, note sale), within 12 months after the snapshot; this mirrors the CRR 90-day backstop. "
           "Loans already in default, terminated or with unknown status at the snapshot are excluded. Layout checked against the Freddie Mac General User Guide (Release 47); the files have 31 origination fields (no Servicer Name), which a column-count assertion caught."))
S.append(tbl([["Sample", "Snapshots (outcome window)", "Loan-snapshots", "Defaults", "Default rate"],
              ["Development", "Dec 2015, 2016, 2017 (2016-2018)", "585,900 (265,421 loans)", "2,661", "0.454%"],
              ["  Train / in-time holdout (70/30 by loan)", "", "410,447 / 175,453", "1,881 / 780", "0.458% / 0.445%"],
              ["Out-of-time", "Dec 2018 (2019)", "254,118", "1,141", "0.449%"],
              ["Monitoring (COVID)", "Dec 2019 (2020)", "227,165", "9,059", "3.99%"]], [4.8, 4.6, 3.6, 2.2, 3.0]))
S.append(Spacer(1, 3))
S.append(P("<b>Exclusions waterfall</b> (five December snapshots pooled): 1,080,276 loan records, less 3,406 already in default, less 9,687 terminated without default, 0 with unknown delinquency status, 0 interest-only, 0 ARM = 1,067,183 loans, 12,861 defaults. "
           "<b>Data quality:</b> 24 issues logged, 5 fields dropped (pre_harp_id 84% blank; ppm, amort_type, io_flag, mi_cancel constant). DTI = 999 (not available) affects 168,270 rows (16%) and 99.97% are HARP loans, so the missing bin proxies HARP; "
           "all 36,053 loans with LTV above 105 are HARP; 1,753 CLTV above 200 set to missing; 431 loans with UPB above original flagged and capped. Zero duplicate loan-month rows and zero gaps in monthly history. "
           "Default rates move with the economy except for a hurricane effect: 31% of Dec 2016 snapshot defaults carry a disaster flag (Dec 2015: 2.5%; Dec 2018: 3.5%)."))

S.append(P("4. Conceptual soundness", H1))
S.append(P("A logistic regression on weight-of-evidence (WoE) bins suits a low-default portfolio (0.46% annual) where a regulator and credit officers need monotone, readable risk drivers and points per bin. "
           "Twenty candidates were binned (optbinning, monotone trends, own bin for missing) and screened by IV above 0.02, expected direction, stability of bin ordering across the three development snapshots (Spearman at least 0.8), "
           "WoE correlation below 0.7 and VIF below 5 (maximum 2.0). Ten variables passed; the sign check removed orig_term (negative coefficient, correlated 0.59 with orig_rate), leaving nine, all with positive coefficients and p &lt; 0.001: "
           "months_since_dq 0.60, cur_dq 0.29, fico 0.49, cltv 0.40, dti 0.31, orig_rate 0.23, ever_modified 0.28, n_borrowers 0.82, upb_ratio 0.50. Risk falls with FICO and rises with CLTV, DTI, rate and recent delinquency, as expected; "
           "state failed stability (Spearman 0.60) and was dropped."))
S.append(P("<b>Known weaknesses.</b> (i) The delinquency variables carry IV of 2.0 to 2.4 (a loan two payments behind reaches default 64% of the time against 0.28% for a current loan). I checked for leakage (history to the snapshot only, defaulted loans excluded) and found none, "
           "but discrimination among loans that are current is lower (Gini 0.68 development, 0.72 OOT). (ii) Several variables have two or three bins. (iii) No interactions or macro terms, and the model is static. "
           "(iv) ever_modified rests on 473 training loans. (v) Calibration is an intercept shift only (-0.0152), with no slope term."))

S.append(P("5. Outcomes analysis", H1))
S.append(tbl([["Measure", "Train", "In-time holdout", "OOT Dec 2018", "Notes"],
              ["Gini", "0.8029", "0.7837", "0.8430", "Train-to-holdout gap 0.019. Development snapshots alone: 0.821, 0.758, 0.829"],
              ["KS", "0.632", "0.604", "0.699", ""],
              ["Gini, current loans only (cur_dq = 0)", "", "0.679 (pooled dev)", "0.719", ""],
              ["Actual / expected default rate", "1.00 (by construction)", "", "0.449% / 0.482% = 0.93", "Mean PD calibrated to development rate 0.4542%"]], [4.3, 2.6, 2.9, 3.5, 4.9]))
S.append(Spacer(1, 3))
S.append(P("<b>Back-test by grade, OOT Dec 2018</b> (grade PD = development mean calibrated PD; traffic light on the one-sided binomial p-value, Green 0.05 or above, Amber 0.01 to 0.05, Red below 0.01; Jeffreys test agrees in every grade). "
           "Both tests assume independent defaults, so a Red is a prompt to investigate."))
bt = pd.read_csv("reports/backtest_oot.csv")
rows = [["Grade", "Loans", "Defaults", "Grade PD", "Observed", "Obs / PD", "p (binomial)", "p (Jeffreys)", "Light", "Too conservative (p &lt; 0.05)"]]
for r in bt.itertuples():
    pb = "&lt; 0.0001" if r.p_binomial < 1e-4 and r.p_binomial < 0.5 else f"{r.p_binomial:.4f}"
    pj = "&lt; 0.0001" if r.p_jeffreys < 1e-4 and r.p_jeffreys < 0.5 else f"{r.p_jeffreys:.4f}"
    rows.append([r.grade, f"{r.loans:,}", r.defaults, f"{r.grade_pd:.3%}", f"{r.observed:.3%}", f"{r.ratio:.2f}", pb, pj, r.light, "Yes" if r.p_conservative < 0.05 else ""])
t = tbl(rows, [1.2, 1.7, 1.5, 1.7, 1.7, 1.5, 2.0, 2.0, 1.4, 3.3])
sty = []
for i, r in enumerate(bt.itertuples(), start=1):
    sty.append(("TEXTCOLOR", (8, i), (8, i), colors.white)); sty.append(("BACKGROUND", (8, i), (8, i), colors.HexColor({"Green": "#2e7d4f", "Amber": "#b87800", "Red": "#b3261e"}[r.light])))
t.setStyle(TableStyle(sty)); S.append(t); S.append(Spacer(1, 3))
S.append(P("Seven grades are Green, one Amber (7) and one Red (8). Grades 3, 4, 5 and 9 are too conservative (observed 0.48x, 0.49x, 0.57x and 0.92x of PD; lower-tail p 0.0003, below 0.0001, below 0.0001 and 0.014). "
           "The same shape appears on development data (grades 3-4 observed 0.43x and 0.67x of PD; grades 6-8 1.33x, 1.30x and 1.36x), so the PD curve is too flat across grades 3 to 8; this is not a 2019 effect. Grades 1 and 2 hold 3 and 1 OOT defaults, so power is low."))

S.append(P("6. Stability and monitoring", H1))
S.append(tbl([["Test (threshold set by reviewer: Green / Amber / Red)", "OOT Dec 2018", "Dec 2019", "Dec 2019 adjusted for forbearance"],
              ["PSI grade mix (&lt; 0.10 / 0.10-0.25 / &gt; 0.25)", "0.0018 Green", "0.0077 Green", "n/a"],
              ["Gini relative drop vs development 0.797 (&lt; 10% / 10-20% / &gt; 20%)", "0.843, -5.7% Green", "0.476, +40.3% Red", "0.805, -0.9% Green"],
              ["Overall actual / expected (0.8-1.2 / 0.7-1.5 / outside)", "0.93 Green", "7.53 Red (3.99% vs 0.53%)", "0.86 Green (0.453%)"],
              ["Grade back-test (worst grade)", "1 Red, 1 Amber, 7 Green", "9 Red (1.6x to 21x PD)", "1 Red, 1 Amber, 7 Green"],
              ["CSI, worst variable (&lt; 0.10 / 0.10-0.25 / &gt; 0.25)", "orig_rate 0.067 Green", "upb_ratio 0.488 Red; others below 0.07", "n/a"]], [6.5, 3.4, 4.2, 4.1]))
S.append(Spacer(1, 3))
S.append(P("Dec 2019 is the COVID test. The population is stable (PSI Green); the failure is in the outcome. Loans on a forbearance plan show as delinquent (86% 30+ days past due and 50% 90+ in Apr-Dec 2020, against 0.9% and 0.2% for loans with no flag). "
           "Of the 9,059 reported defaults, 8,466 (93.5%) were forborne at or before default, 8,477 (93.6%) cured or paid off by March 2026 and 136 (1.5%) ended in a credit-event termination. "
           "The upb_ratio CSI reflects the absence of 2019 originations in the sample. Caveat: the adjusted figures use outcome data to 2026 (hindsight), and loans that cured through a payment deferral still carry deferred balances."))
S.append(Image("reports/mi_ae_by_grade.png", width=16.5 * cm, height=16.5 * cm * 3.8 / 11))
S.append(P("<b>Monthly monitoring on Azure ML (Jan 2018 to Dec 2020, 36 batch runs, 36 monitoring runs, total cost GBP 1.42).</b> Grade-mix PSI peaked at 0.090 (May 2020) and no variable CSI passed 0.25 once upb_ratio, a sample artefact, was excluded: the specified drift thresholds never alerted on COVID. "
           "A rule added after seeing the data, delinquent share above 2x the development level, went Red in April 2020 (5.2x; Amber in March at 1.5x). Performance, available 12 months after scoring, alerted next: three red grades in the May 2020 run (month scored 2019-05) and Gini 0.53 with actual/expected 4.17 in the June 2020 run. "
           "So the sequence is drift on a tail-sensitive signal first, then performance, a year after the loans were scored but only two months after the shock."))
S.append(Image("reports/azure_sim_exhibit.png", width=11.5 * cm, height=11.5 * cm * 8.6 / 10))

S.append(P("7. Benchmarking: champion against challenger", H1))
S.append(tbl([["Criterion", "Champion (scorecard)", "Challenger (LightGBM, monotone constraints)"],
              ["Gini, in-time holdout / OOT", "0.7837 / 0.8430", "0.7803 / 0.8312"],
              ["Gini gap, train to holdout", "0.019", "0.083 (train 0.864)"],
              ["Mean PD on OOT vs 0.449% actual", "0.482% (calibrated)", "0.533% (uncalibrated)"],
              ["Explainability / governance", "Points per bin; PSI per bin", "SHAP; ranking differs (FICO, borrowers, DTI first)"]], [5.0, 5.2, 8.0]))
S.append(P("The challenger does not gain on accuracy, overfits more and is harder to monitor, so the scorecard remains champion and the GBM a benchmark. The decision was taken on development data before the OOT sample was opened; OOT confirms it. "
           "An unregularised, untuned challenger is a modest test. A bank would need a clear OOT Gini gain (say 3+ points) that holds under stress, stable SHAP rankings and equal calibration before adopting a GBM."))

S.append(P("8. IFRS 9 extension (illustrative)", H1))
S.append(P("<b>Staging.</b> Stage 2 if 30+ days past due or PD at least 2.5x its PD at first observation and above 0.3%; Stage 3 is out of scope. Dec 2019: Stage 1 219,026 loans ($40.70bn); Stage 2 8,139 loans ($1.57bn, 3.7% of balance), 2,283 on the delinquency test and 5,856 on relative PD alone. "
           "<b>Lifetime PD</b> scales the observed hazard by loan age to each loan's own 12-month PD (flat beyond age 6, where data end). <b>ECL</b> uses scheduled EAD without prepayment, discounting at the original rate and LGD of 10% (assumed). "
           "<b>Scenarios</b> (my assumption): upside -0.5 pp unemployment (20%), base 0.0 pp (50%), downside +3.0 pp (30%), linked through logit(default rate) = alpha + beta x dU, beta = 0.1817 (OLS SE 0.0244; Newey-West SE 0.0623; R-squared 0.40)."))
S.append(tbl([["", "Upside", "Base", "Downside", "Weighted", "Coverage (weighted)"],
              ["ECL, LGD 10%", "$53.14m", "$55.69m", "$73.54m", "$60.54m", "0.143% of $42.26bn"],
              ["Weighted ECL at LGD 5% / 20%", "", "", "", "$30.27m / $121.07m", "0.072% / 0.286%"],
              ["Weighted ECL, pre-COVID beta 0.3049", "", "$55.69m", "$88.81m", "$64.78m", ""]], [5.2, 2.0, 2.0, 2.2, 3.6, 3.2]))
S.append(Spacer(1, 3))
S.append(P("Weighted ECL exceeds base by $4.85m: $4.31m because the scenarios are skewed to the downside (expected unemployment change +0.8 pp), $0.54m from convexity. "
           "<b>PMA view.</b> At Dec 2019, +$1.3m (+2.4%) for the grade-slope miscalibration (F3). For 2020 reporting, scaling PDs to the reported default rate would raise ECL from $55.7m to $166.0m (+198%); scaling to the forbearance-adjusted rate gives $51.0m (-8.4%). "
           "A negative PMA of about $115m would offset a mechanical recalibration; I would not release the -$4.7m against the model as it stands (the 0.86 ratio is inside the OOT calibration noise and uses hindsight). "
           "Release the forbearance overlay in line with observed cure rates and remove the grade overlay after recalibration; log owner, size, rationale and expiry."))

S.append(P("9. Limitations and assumptions", H1))
S += bullets(["Calibration anchor covers 2016-2018 only, a benign period; no downturn (F1).",
              "The 90-day default definition counts forborne loans; the adjusted view uses hindsight to March 2026 (F2).",
              "OOT loans overlap development loans (80.7%), so OOT is later in time but not independent by borrower (F4); sample size is 350,000 loans, not a whole book.",
              "Back-test p-values assume independent defaults; real defaults are correlated through the cycle, and 9 grades give about 0.5 false alarms at the 5% level.",
              "No prepayment in EAD; LGD assumed at 10%; discounting at the original rate as a proxy for the effective rate; ages above 6 years unobserved; relative risk persists for life; staging fixed across scenarios.",
              "Scenarios, weights and base shock (assumed 0.0 pp) are illustrative; the macro link is weak and influenced by hurricane and forbearance effects.",
              "First-observed PD in staging uses the first scored snapshot, not origination, because a behavioural model cannot score at origination.",
              "Monitoring thresholds are the analyst's choice; the delinquent-share rule, the three-red-grade alert and the upb_ratio exclusion were set after seeing the data (F9).",
              "Single developer and validator; no challenge from a second party."])

S.append(P("10. Findings and remediation", H1))
S.append(P("Severity: <b>High</b> = could materially misstate PD, capital or ECL, must fix before use; <b>Medium</b> = fix within an agreed period; <b>Low</b> = improvement.", CELL))
S.append(Spacer(1, 2))
F = [["ID", "Finding", "Severity", "Evidence", "Remediation", "Owner"],
 ["F1", "Calibration anchor excludes a downturn", "High", "Development outcome windows (2016-2018) all had falling unemployment (-0.3, -0.6, -0.2 pp); development default rate 0.454%. The only stress window (2020) is distorted by forbearance (F2), so no clean downturn evidence exists. Beta 0.18 (Newey-West SE 0.062).", "Add a downturn margin of conservatism; calibrate to a long-run average that includes a downturn (e.g. 2008-2012 vintages) when data allow; document the margin.", "Model owner"],
 ["F2", "Model and monitoring misread COVID forbearance", "High", "Dec 2019: actual/expected 7.53 and Gini 0.476 (drop 40.3%), all 9 grades Red. 8,466 of 9,059 defaults forborne; 8,477 cured or paid off; 136 credit-event terminations. Adjusted: actual/expected 0.86, Gini 0.805. Mechanical recalibration would raise ECL from $55.7m to $166.0m (+198%).", "Do not recalibrate on reported 2020 defaults; add forbearance status to monitoring and test a forborne-then-cured default category; apply a PMA (about -$115m against a mechanical recalibration), released as cure evidence builds.", "Model owner, Finance (ECL)"],
 ["F3", "Grades 7 and 8 under-predicted; grades 3-5 and 9 over-predicted", "Medium", "Grade 8 observed 4.03% against 2.38% PD (p &lt; 0.0001); grade 7 1.32% against 1.06% (Jeffreys p = 0.021); grades 3-5 0.48x-0.57x of PD. Same shape in development (grades 6-8 1.30x-1.36x).", "Recalibrate with intercept and slope on development data; retest on a fresh sample; interim PMA +$1.3m (+2.4%).", "Model developer"],
 ["F4", "OOT sample not independent by borrower", "Medium", "80.7% of Dec 2018 loans are development loans. OOT Gini 0.843 exceeds holdout 0.784; development snapshots range 0.758 to 0.829.", "Validate on a borrower-disjoint or newer-vintage sample (2019 onwards) before relying on the OOT Gini.", "Model developer"],
 ["F5", "Prepayment ignored in EAD", "Medium", "12.7% of loans prepaid within 12 months in both the Dec 2015 (21,313 of 167,815) and Dec 2018 (32,372 of 254,118) snapshots. ECL overstated; impact not quantified.", "Add an observed prepayment (CPR) term structure to EAD; report the ECL change.", "Finance (ECL)"],
 ["F6", "LGD assumed, not estimated", "Medium", "ECL is linear in LGD: weighted ECL $30.27m at 5%, $60.54m at 10%, $121.07m at 20%. The performance file holds loss fields; they were not used.", "Estimate LGD (and a downturn LGD) from actual losses by LTV band and compare with the 10% assumption.", "Finance (ECL)"],
 ["F7", "Lifetime PD and staging simplifications", "Medium", "Hazard flat beyond age 6; each loan keeps its relative risk for life (Stage 2 mean lifetime PD 47.4%; 52% of weighted ECL from the 1% of loans 30+ days past due). 5,856 of 8,139 Stage 2 loans staged by relative PD alone, measured from first scored snapshot; staging held fixed across scenarios.", "Add mean reversion of relative risk; derive PD at initial recognition; restage under each scenario.", "Finance (ECL)"],
 ["F8", "Macro link weak and contaminated", "Medium", "Beta 0.1817 (OLS SE 0.0244; Newey-West 0.0623; R-squared 0.40); pre-COVID beta 0.3049 moves weighted ECL from $60.5m to $64.8m. 31% of Dec 2016 snapshot defaults are hurricane-flagged; 2020 defaults are forbearance-driven.", "Re-estimate on disaster- and forbearance-adjusted defaults, add house prices; source scenarios and weights from the governance forum.", "Model owner"],
 ["F9", "Monitoring thresholds as specified missed COVID; added rules were tuned after the fact", "Medium", "Azure run, 36 months: grade-mix PSI peak 0.090 (amber line 0.10), no CSI above 0.25 excluding upb_ratio; delinquent share 5.2x development in April 2020 (red, rule added post hoc); performance alert May 2020, Gini 0.53 in June 2020. upb_ratio CSI 0.49 is red from June 2019 for a sample reason (no new originations). Grade-level back-test is red in grades 7-8 in most months.", "Pre-register tail-sensitive signals (delinquent share, forbearance share) before the next stress; test alert rules on a later sample; treat grades 7-8 as a standing finding with its own threshold.", "Model owner"],
 ["F10", "Sparse grades and thin variables", "Low", "Grade 1 has 3 OOT defaults (2 in development), grade 2 has 1 (5); grade 2 observed 0.16x PD on one default (lower-tail p = 0.015). ever_modified rests on 473 training loans; DTI missing bin (168,270 rows) is 99.97% HARP.", "Pool grades 1-2 for testing or add data; split grade 9 (PD 4%-88%); monitor HARP separately.", "Model developer"]]
rows = [[Paragraph(str(c), CELLB if r == 0 else CELL) for c in row] for r, row in enumerate(F)]
t = Table(rows, colWidths=[0.8 * cm, 2.9 * cm, 1.3 * cm, 6.6 * cm, 5.1 * cm, 1.6 * cm], repeatRows=1)
sty = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef4")),
       ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
for i, row in enumerate(F[1:], start=1):
    sty.append(("BACKGROUND", (2, i), (2, i), colors.HexColor(SEV[row[2]])))
    rows[i][2] = Paragraph(f'<font color="white"><b>{row[2]}</b></font>', CELL)
t = Table(rows, colWidths=[0.8 * cm, 2.9 * cm, 1.3 * cm, 6.6 * cm, 5.1 * cm, 1.6 * cm], repeatRows=1); t.setStyle(TableStyle(sty)); S.append(t)

# ---- results.md: findings summary (inserted after Step 9, before the Azure sections)
import os
from collections import Counter
cnt = Counter(r[2] for r in F[1:])
top = "".join(f"- {r[0]} ({r[2]}): {r[1]}\n" for r in F[1:4])
sec = (f"\n## Step 10: validation findings (reports/Validation_Report.pdf)\n\nOverall rating: Fit for use with conditions. Findings: {len(F) - 1} in total, "
       f"{cnt.get('High', 0)} High, {cnt.get('Medium', 0)} Medium, {cnt.get('Low', 0)} Low.\n\nTop three:\n{top}\n")
if os.path.exists("results.md"):
    t = open("results.md").read()
    if "\n## Step 10:" in t:
        i = t.index("\n## Step 10:"); j = t.find("\n## ", i + 5); t = t[:i] + (t[j:] if j > 0 else "")
    k = t.find("\n## Azure phase")
    t = (t[:k] + sec + t[k:]) if k > 0 else t.rstrip("\n") + "\n" + sec
    open("results.md", "w").write(t)

doc = SimpleDocTemplate("reports/Validation_Report.pdf", pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                        title="Independent Validation Report: 12-Month Behavioural PD Model", author="Ram Ridhan")
def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.HexColor("#666666"))
    c.drawString(1.4 * cm, 0.6 * cm, "Independent validation report, US mortgage behavioural PD model. Self-validated exercise."); c.drawRightString(A4[0] - 1.4 * cm, 0.6 * cm, f"Page {d.page}"); c.restoreState()
doc.build(S, onFirstPage=footer, onLaterPages=footer)
