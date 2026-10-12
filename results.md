# Results

Metrics recorded exactly as printed by code.

## Step 1: loans, defaults and default rate per snapshot

(Output of `python -I src/run_step1.py`. Snapshot 2019-12 outcome window covers 2020, the COVID period.)

| Snapshot | Loans | Defaults | Default rate |
|---|---|---|---|
| 2015-12 | 167,815 | 592 | 0.3528% |
| 2016-12 | 193,953 | 1,084 | 0.5589% |
| 2017-12 | 224,132 | 985 | 0.4395% |
| 2018-12 | 254,118 | 1,141 | 0.4490% |
| 2019-12 | 227,165 | 9,059 | 3.9879% |

Performance file: 23,809,108 rows, 2012-01 to 2026-03. Origination: 350,000 loans, 7 vintages (2012-2018).


## Step 2: exclusions waterfall (pooled over the 5 December snapshots)

| Step | Loans remaining | Removed | Defaults |
|---|---|---|---|
| Loan records in performance file at the 5 December snapshots | 1,080,276 |  |  |
| Excluded: already in default at snapshot (90+ DPD, REO, credit-event termination) | 1,076,870 | 3,406 |  |
| Excluded: terminated at snapshot without default (e.g. prepaid) | 1,067,183 | 9,687 |  |
| Excluded: delinquency status unknown (XX) | 1,067,183 | 0 |  |
| Eligible population with 12-month outcome (Step 1 output) | 1,067,183 |  | 12,861 |
| Excluded: interest-only loans | 1,067,183 | 0 | 12,861 |
| Excluded: non-FRM (ARM) loans | 1,067,183 | 0 | 12,861 |

Kept but flagged / deferred to Step 3 (loan-snapshot rows):

- HARP refinances (kept, flagged): 168,220
- FICO = 9999 not available (Step 3): 100
- DTI = 999 not available (Step 3): 168,270
- LTV/CLTV = 999 not available (Step 3): 24
- Units = 99 not available (Step 3): 148

## Step 2: samples

| Sample | Loan-snapshots | Unique loans | Defaults | Default rate |
|---|---|---|---|---|
| Development (Dec 2015-2017) | 585,900 | 265,421 | 2,661 | 0.4542% |
|   Train (70% of loans) | 410,447 | 185,794 | 1,881 | 0.4583% |
|   In-time holdout (30% of loans) | 175,453 | 79,627 | 780 | 0.4446% |
| OOT (Dec 2018) | 254,118 | 254,118 | 1,141 | 0.4490% |
| Monitoring (Dec 2019) | 227,165 | 227,165 | 9,059 | 3.9879% |

Split: GroupShuffleSplit on loan_id, train_size 0.7, random_state 42; train and holdout loan_ids are disjoint (asserted).
OOT loans that also appear in the development sample: 80.73%.

## Step 3: data-quality log

Issues logged: 24. Fields dropped: pre_harp_id, ppm, amort_type, io_flag, mi_cancel.

| # | Check | Issue | Records affected | Treatment | Reason |
|---|---|---|---|---|---|
| 1 | Sentinel | fico = 9999 (not available) | 100 | Set to missing; own WoE bin | Guide: sentinel code for not available |
| 2 | Sentinel | ltv = 999 (not available) | 24 | Set to missing; own WoE bin | Guide: sentinel code for not available |
| 3 | Sentinel | cltv = 999 (not available) | 24 | Set to missing; own WoE bin | Guide: sentinel code for not available |
| 4 | Sentinel | dti = 999 (not available) | 168,270 | Set to missing; own WoE bin | Guide: sentinel code for not available |
| 5 | Sentinel | units = 99 (not available) | 148 | Set to missing; own WoE bin | Guide: sentinel code for not available |
| 6 | Sentinel | first_time_buyer = 9 (not available) | 41 | Set to missing | Guide sentinel |
| 7 | Range | fico outside 300-850 (excluding 9999) | 0 | Set to missing | Guide range |
| 8 | Range | ltv > 105 (excluding 999) | 36,053 | Kept (HARP allows LTV > 105); binned in WoE | 36,053 are HARP loans; 0 non-HARP |
| 9 | Range | cltv > 200 (excluding 999) | 1,753 | Set to missing | Outside guide range 6-200 |
| 10 | Range | dti outside 1-65 (excluding 999) | 0 | None needed | No outliers found |
| 11 | Range | orig_rate <= 0 | 0 | None needed | No outliers found |
| 12 | Consistency | cltv < ltv | 0 | None needed | CLTV >= LTV holds everywhere |
| 13 | Consistency | current UPB > original UPB | 431 | Flag upb_gt_orig=1; cap upb_ratio at 1; rows kept | 308 exceed by >1%: rounding of early balances or capitalisation after modification |
| 14 | Completeness | pre_harp_id blank in 84% of rows | 898,963 | Drop field | Blank unless loan is a HARP refinance; harp flag carries the information |
| 15 | Completeness | program blank in 98% of rows | 1,042,864 | Blank recoded to 'N' (no special programme); keep as category | Blank means no Home Possible / HFA Advantage; informative, not missing |
| 16 | Completeness | msa blank | 146,136 | Not used (state used instead) | 13.7% blank, under the 20% threshold, but ~400-level code; state gives geography |
| 17 | Completeness | dti missing (999) | 168,270 | Own WoE bin | 16% overall; structural: 168,220 of 168,270 are HARP loans (not reported for HARP) |
| 18 | Completeness | ppm constant ('N') | 1,067,183 | Drop field | No information |
| 19 | Completeness | amort_type constant ('FRM') | 1,067,183 | Drop field | No information |
| 20 | Completeness | io_flag constant ('N') | 1,067,183 | Drop field | No information |
| 21 | Completeness | mi_cancel constant ('9999') | 1,067,183 | Drop field | No information |
| 22 | Duplicates | duplicate loan_id + period rows in performance file | 0 | None needed | 23,809,108 rows checked |
| 23 | Time continuity | gaps in monthly history in the 12 months before snapshot | 0 | None needed | 1,067,183 loan-snapshots checked |
| 24 | Stability | field mean shifts >15% across snapshots | 0 | Note for validation report | none |

Fields over 20% missing (blank) in at least one snapshot: pre_harp_id, program

### Field means by snapshot (after sentinel treatment)

| snap | fico | ltv | cltv | dti | orig_rate | orig_upb | upb_ratio | loan_age |
|---|---|---|---|---|---|---|---|---|
| 2015-12-01 00:00:00 | 750.82 | 75.397 | 76.718 | 32.83 | 3.978 | 205343.634 | 0.93 | 22.084 |
| 2016-12-01 00:00:00 | 750.165 | 74.801 | 75.93 | 33.113 | 3.923 | 207775.487 | 0.915 | 27.085 |
| 2017-12-01 00:00:00 | 749.489 | 74.696 | 75.643 | 33.564 | 3.962 | 212302.036 | 0.902 | 31.476 |
| 2018-12-01 00:00:00 | 749.135 | 74.912 | 75.705 | 34.033 | 4.091 | 216744.505 | 0.89 | 35.981 |
| 2019-12-01 00:00:00 | 749.393 | 74.719 | 75.478 | 33.964 | 4.087 | 214678.797 | 0.854 | 47.811 |


### Missing share by snapshot

| snap | fico | dti | cltv | ltv |
|---|---|---|---|---|
| 2015-12-01 00:00:00 | 0.0001 | 0.223 | 0.0025 | 0.0 |
| 2016-12-01 00:00:00 | 0.0001 | 0.1874 | 0.002 | 0.0 |
| 2017-12-01 00:00:00 | 0.0001 | 0.1537 | 0.0016 | 0.0 |
| 2018-12-01 00:00:00 | 0.0001 | 0.125 | 0.0013 | 0.0 |
| 2019-12-01 00:00:00 | 0.0001 | 0.1246 | 0.0013 | 0.0 |


### Target sanity: 12-month default rate

| snap | loans | defaults | rate |
|---|---|---|---|
| 2015-12-01 00:00:00 | 167815 | 592 | 0.003527694187051217 |
| 2016-12-01 00:00:00 | 193953 | 1084 | 0.005588982897918568 |
| 2017-12-01 00:00:00 | 224132 | 985 | 0.004394731675976656 |
| 2018-12-01 00:00:00 | 254118 | 1141 | 0.004490040060129546 |
| 2019-12-01 00:00:00 | 227165 | 9059 | 0.03987850241014241 |


| vintage | loans | defaults | rate |
|---|---|---|---|
| 2012 | 133819 | 537 | 0.0040128830734051215 |
| 2013 | 142524 | 716 | 0.005023715304089136 |
| 2014 | 138507 | 763 | 0.005508746850339694 |
| 2015 | 160607 | 684 | 0.004258843014314445 |
| 2016 | 130518 | 541 | 0.0041450221425397265 |
| 2017 | 89700 | 399 | 0.004448160535117057 |
| 2018 | 44343 | 162 | 0.0036533387456870307 |


Chart: reports/dq_target_by_snapshot.png

## Step 4: WoE binning and variable selection (train split only, 410,447 loan-snapshots, 1,881 defaults)

Binning: optbinning, max 8 bins, min 4 bins where feasible, monotonic trend for numerics, missing in its own bin.

### IV table, all 20 candidates (with outcome of every selection rule)

| variable | type | n_bins | iv | trend_of_risk | min_snap_rho | iv_min_snap | iv_max_snap | outcome | note |
|---|---|---|---|---|---|---|---|---|---|
| months_since_dq | numerical | 8 | 2.4353 | -1.0 | 0.9048 | 1.8877 | 3.1939 | KEEP | IV 2.44 > 0.5: leakage check |
| max_dq_12m | numerical | 3 | 2.3406 | 1.0 | 1.0 | 1.8357 | 3.0405 | DROP: abs(r) = 0.89 with months_since_dq (higher IV 2.44 vs 2.34) | IV 2.34 > 0.5: leakage check |
| cur_dq | numerical | 3 | 2.048 | 1.0 | 1.0 | 1.4902 | 2.8195 | KEEP | IV 2.05 > 0.5: leakage check |
| fico | numerical | 9 | 1.0002 | -1.0 | 1.0 | 0.9769 | 1.0281 | KEEP | IV 1.00 > 0.5: leakage check |
| orig_rate | numerical | 8 | 0.4231 | 1.0 | 1.0 | 0.4019 | 0.5334 | KEEP |  |
| dti | numerical | 9 | 0.3821 | 1.0 | 0.95 | 0.3101 | 0.6198 | KEEP |  |
| cltv | numerical | 9 | 0.2763 | 1.0 | 0.9 | 0.1993 | 0.5507 | KEEP |  |
| state | categorical | 9 | 0.2357 | 1.0 | 0.6 | 0.0714 | 0.5476 | DROP: unstable bin ordering across snapshots (min Spearman 0.60 < 0.8) |  |
| ltv | numerical | 9 | 0.2346 | 1.0 | 0.9667 | 0.1804 | 0.3953 | DROP: abs(r) = 0.94 with cltv (higher IV 0.28 vs 0.23) |  |
| ever_modified | numerical | 2 | 0.221 | 1.0 | 1.0 | 0.2136 | 0.2342 | KEEP |  |
| n_borrowers | numerical | 2 | 0.1705 | -1.0 | 1.0 | 0.1555 | 0.2232 | KEEP |  |
| upb_ratio | numerical | 6 | 0.1371 | 1.0 | 0.9429 | 0.1156 | 0.2144 | KEEP |  |
| orig_term | numerical | 3 | 0.0813 | 1.0 | 1.0 | 0.0483 | 0.1177 | KEEP |  |
| purpose | categorical | 3 | 0.016 | 1.0 | 0.5 | 0.0057 | 0.0971 | DROP: IV 0.0160 < 0.02 |  |
| loan_age | numerical | 3 | 0.0133 | 1.0 | 1.0 | 0.0052 | 0.0359 | DROP: IV 0.0133 < 0.02 |  |
| occupancy | categorical | 2 | 0.0116 | 1.0 | 1.0 | 0.0053 | 0.0207 | DROP: IV 0.0116 < 0.02 |  |
| channel | categorical | 3 | 0.0094 | 1.0 | 0.5 | 0.0033 | 0.0677 | DROP: IV 0.0094 < 0.02 |  |
| first_time_buyer | categorical | 3 | 0.0072 | 1.0 | 0.5 | 0.0036 | 0.0139 | DROP: IV 0.0072 < 0.02 |  |
| prop_type | categorical | 4 | 0.0011 | 1.0 | -0.2 | 0.0005 | 0.0201 | DROP: IV 0.0011 < 0.02 |  |
| units | numerical | 2 | 0.0 | nan | 1.0 | 0.0 | 0.0 | DROP: IV 0.0000 < 0.02 |  |

Final variables (10): months_since_dq, cur_dq, fico, orig_rate, dti, cltv, ever_modified, n_borrowers, upb_ratio, orig_term

### Drop reasons

- **purpose**: IV 0.0160 < 0.02
- **loan_age**: IV 0.0133 < 0.02
- **occupancy**: IV 0.0116 < 0.02
- **channel**: IV 0.0094 < 0.02
- **first_time_buyer**: IV 0.0072 < 0.02
- **prop_type**: IV 0.0011 < 0.02
- **units**: IV 0.0000 < 0.02
- **state**: unstable bin ordering across snapshots (min Spearman 0.60 < 0.8)
- **max_dq_12m**: abs(r) = 0.89 with months_since_dq (higher IV 2.44 vs 2.34)
- **ltv**: abs(r) = 0.94 with cltv (higher IV 0.28 vs 0.23)

### WoE correlation matrix, final variables

| index | months_since_dq | cur_dq | fico | orig_rate | dti | cltv | ever_modified | n_borrowers | upb_ratio | orig_term |
|---|---|---|---|---|---|---|---|---|---|---|
| months_since_dq | 1.0 | 0.7 | 0.16 | 0.08 | 0.07 | 0.06 | 0.16 | 0.04 | 0.02 | 0.03 |
| cur_dq | 0.7 | 1.0 | 0.09 | 0.05 | 0.04 | 0.03 | 0.1 | 0.02 | 0.02 | 0.02 |
| fico | 0.16 | 0.09 | 1.0 | 0.24 | 0.22 | 0.17 | 0.04 | -0.02 | 0.12 | 0.05 |
| orig_rate | 0.08 | 0.05 | 0.24 | 1.0 | 0.22 | 0.27 | 0.03 | 0.1 | 0.43 | 0.59 |
| dti | 0.07 | 0.04 | 0.22 | 0.22 | 1.0 | 0.28 | 0.03 | 0.09 | 0.01 | 0.04 |
| cltv | 0.06 | 0.03 | 0.17 | 0.27 | 0.28 | 1.0 | 0.04 | 0.05 | 0.14 | 0.24 |
| ever_modified | 0.16 | 0.1 | 0.04 | 0.03 | 0.03 | 0.04 | 1.0 | 0.01 | 0.02 | 0.01 |
| n_borrowers | 0.04 | 0.02 | -0.02 | 0.1 | 0.09 | 0.05 | 0.01 | 1.0 | 0.08 | 0.07 |
| upb_ratio | 0.02 | 0.02 | 0.12 | 0.43 | 0.01 | 0.14 | 0.02 | 0.08 | 1.0 | 0.58 |
| orig_term | 0.03 | 0.02 | 0.05 | 0.59 | 0.04 | 0.24 | 0.01 | 0.07 | 0.58 | 1.0 |

### VIF, final variables

| index | VIF |
|---|---|
| months_since_dq | 2.01 |
| cur_dq | 1.94 |
| fico | 1.15 |
| orig_rate | 1.76 |
| dti | 1.17 |
| cltv | 1.18 |
| ever_modified | 1.02 |
| n_borrowers | 1.02 |
| upb_ratio | 1.56 |
| orig_term | 2.02 |

### Bin tables, final variables

**months_since_dq**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 0.50) | 2910 | 0.0071 | 0.2536 | -4.301390061184782 | 1.6648 |
| 1 | [0.50, 1.50) | 1224 | 0.003 | 0.0792 | -2.928246382101742 | 0.1429 |
| 2 | [1.50, 4.50) | 2270 | 0.0055 | 0.0493 | -2.4224116431151503 | 0.1314 |
| 3 | [4.50, 8.50) | 2190 | 0.0053 | 0.0205 | -1.5166175760463378 | 0.0283 |
| 4 | [8.50, 12.50) | 1817 | 0.0044 | 0.0171 | -1.3271033606868379 | 0.0161 |
| 5 | [12.50, 18.50) | 1946 | 0.0047 | 0.0134 | -1.07886599063779 | 0.0098 |
| 6 | [18.50, 51.50) | 4677 | 0.0114 | 0.0073 | -0.4640944542343757 | 0.0031 |
| 7 | [51.50, inf) | 393413 | 0.9585 | 0.002 | 0.8176262518329773 | 0.4388 |
| 8 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 9 | Missing | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 2.4353 |

**cur_dq**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 0.50) | 407537 | 0.9929 | 0.0028 | 0.4928168299249496 | 0.1907 |
| 1 | [0.50, 1.50) | 2382 | 0.0058 | 0.1671 | -3.774431635077755 | 0.7803 |
| 2 | [1.50, inf) | 528 | 0.0013 | 0.6439 | -5.973353572418393 | 1.077 |
| 3 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 4 | Missing | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 2.048 |

**fico**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 662.50) | 22514 | 0.0549 | 0.0215 | -1.5648837624558989 | 0.3191 |
| 1 | [662.50, 692.50) | 32958 | 0.0803 | 0.0127 | -1.0285261886669623 | 0.1472 |
| 2 | [692.50, 710.50) | 31698 | 0.0772 | 0.0074 | -0.488145975281296 | 0.0237 |
| 3 | [710.50, 729.50) | 38880 | 0.0947 | 0.0051 | -0.116089334618942 | 0.0014 |
| 4 | [729.50, 756.50) | 65732 | 0.1601 | 0.0038 | 0.18721973919569912 | 0.0051 |
| 5 | [756.50, 763.50) | 20727 | 0.0505 | 0.0024 | 0.6641553963978764 | 0.0163 |
| 6 | [763.50, 789.50) | 95447 | 0.2325 | 0.0015 | 1.10722241572224 | 0.1729 |
| 7 | [789.50, inf) | 102455 | 0.2496 | 0.0009 | 1.5806708550589441 | 0.3145 |
| 8 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 9 | Missing | 36 | 0.0001 | 0.0 | 0.0 | 0.0 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 1.0002 |

**orig_rate**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 3.12) | 28962 | 0.0706 | 0.0011 | 1.4578318414437632 | 0.0792 |
| 1 | [3.12, 3.60) | 65298 | 0.1591 | 0.0018 | 0.9418994419674593 | 0.0917 |
| 2 | [3.60, 3.85) | 65121 | 0.1587 | 0.0025 | 0.6007644460407819 | 0.0431 |
| 3 | [3.85, 3.99) | 38112 | 0.0929 | 0.0033 | 0.31990961984059485 | 0.0081 |
| 4 | [3.99, 4.25) | 71129 | 0.1733 | 0.0041 | 0.10026911642095904 | 0.0017 |
| 5 | [4.25, 4.26) | 35477 | 0.0864 | 0.0053 | -0.14060361782340625 | 0.0018 |
| 6 | [4.26, 4.74) | 72088 | 0.1756 | 0.0074 | -0.4811496780590039 | 0.0521 |
| 7 | [4.74, inf) | 34260 | 0.0835 | 0.0125 | -1.0084419943361729 | 0.1454 |
| 8 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 9 | Missing | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 0.4231 |

**dti**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 20.50) | 38058 | 0.0927 | 0.001 | 1.5541260873605136 | 0.1141 |
| 1 | [20.50, 23.50) | 22492 | 0.0548 | 0.0012 | 1.3066148790121979 | 0.0524 |
| 2 | [23.50, 28.50) | 45514 | 0.1109 | 0.002 | 0.8431362907443516 | 0.0534 |
| 3 | [28.50, 31.50) | 31423 | 0.0766 | 0.0032 | 0.3660878273319925 | 0.0086 |
| 4 | [31.50, 37.50) | 68319 | 0.1665 | 0.0037 | 0.2219596277214082 | 0.0074 |
| 5 | [37.50, 39.50) | 24656 | 0.0601 | 0.0044 | 0.036147126295740684 | 0.0001 |
| 6 | [39.50, 43.50) | 51984 | 0.1267 | 0.0049 | -0.0683396311283202 | 0.0006 |
| 7 | [43.50, inf) | 52166 | 0.1271 | 0.0058 | -0.24153599532702774 | 0.0084 |
| 8 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 9 | Missing | 75835 | 0.1848 | 0.0093 | -0.7149319790502586 | 0.1373 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 0.3821 |

**cltv**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 44.50) | 31990 | 0.0779 | 0.0017 | 1.00165521490878 | 0.0495 |
| 1 | [44.50, 52.50) | 21293 | 0.0519 | 0.0023 | 0.6911595657935274 | 0.0179 |
| 2 | [52.50, 75.50) | 127617 | 0.3109 | 0.0033 | 0.3419886135930774 | 0.0308 |
| 3 | [75.50, 79.50) | 24584 | 0.0599 | 0.0035 | 0.28288640524482656 | 0.0042 |
| 4 | [79.50, 80.50) | 75654 | 0.1843 | 0.0039 | 0.15879599765774444 | 0.0043 |
| 5 | [80.50, 93.50) | 55398 | 0.135 | 0.0054 | -0.17110963518924827 | 0.0043 |
| 6 | [93.50, 95.50) | 42115 | 0.1026 | 0.0064 | -0.33380980534077853 | 0.0135 |
| 7 | [95.50, inf) | 30988 | 0.0755 | 0.0126 | -1.0157066535886665 | 0.134 |
| 8 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 9 | Missing | 808 | 0.002 | 0.0272 | -1.8049355785672452 | 0.0176 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 0.2763 |

**ever_modified**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 0.50) | 409974 | 0.9988 | 0.0043 | 0.054843675081293064 | 0.0029 |
| 1 | [0.50, inf) | 473 | 0.0012 | 0.2156 | -4.089620668314971 | 0.2181 |
| 2 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | Missing | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 0.221 |

**n_borrowers**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 1.50) | 194345 | 0.4735 | 0.0065 | -0.35678810365330804 | 0.0723 |
| 1 | [1.50, inf) | 216102 | 0.5265 | 0.0028 | 0.48472754750302904 | 0.0982 |
| 2 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | Missing | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 0.1705 |

**upb_ratio**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 0.78) | 37890 | 0.0923 | 0.0013 | 1.2482491000995326 | 0.0824 |
| 1 | [0.78, 0.84) | 25002 | 0.0609 | 0.0027 | 0.5384851809511213 | 0.0137 |
| 2 | [0.84, 0.91) | 59675 | 0.1454 | 0.0033 | 0.3293081288848283 | 0.0135 |
| 3 | [0.91, 0.94) | 56088 | 0.1367 | 0.0046 | -0.011500734025128523 | 0.0 |
| 4 | [0.94, 1.00) | 210613 | 0.5131 | 0.0056 | -0.20110788526490175 | 0.023 |
| 5 | [1.00, inf) | 21179 | 0.0516 | 0.006 | -0.27817683763735435 | 0.0046 |
| 6 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 7 | Missing | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 0.1371 |

**orig_term**

| index | Bin | Count | Count (%) | Event rate | WoE | IV |
|---|---|---|---|---|---|---|
| 0 | (-inf, 198.00) | 99710 | 0.2429 | 0.0026 | 0.5775144472062994 | 0.0617 |
| 1 | [198.00, 262.50) | 25063 | 0.0611 | 0.0039 | 0.1697092820694781 | 0.0016 |
| 2 | [262.50, inf) | 285674 | 0.696 | 0.0053 | -0.1546632339912115 | 0.018 |
| 3 | Special | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 4 | Missing | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Totals |  | 410447 | 1.0 | 0.0046 |  | 0.0813 |

## Step 5: champion WoE logistic scorecard

Fitted on train (410,447 loan-snapshots, 1,881 defaults). Fitted on risk-WoE (= -WoE); all coefficients positive. Variables dropped for a wrong sign: [('orig_term', -0.22825342792367198)]. Final model variables (9): months_since_dq, cur_dq, fico, orig_rate, dti, cltv, ever_modified, n_borrowers, upb_ratio.

### Coefficients and p-values

| variable | coefficient | std_err | z | p_value |
|---|---|---|---|---|
| const | -5.3289 | 0.0302 | -176.65 | 0.00e+00 |
| months_since_dq | 0.6012 | 0.0248 | 24.21 | 1.77e-129 |
| cur_dq | 0.2894 | 0.0246 | 11.77 | 5.41e-32 |
| fico | 0.4919 | 0.028 | 17.54 | 7.43e-69 |
| orig_rate | 0.2305 | 0.0456 | 5.06 | 4.21e-07 |
| dti | 0.308 | 0.0499 | 6.17 | 6.83e-10 |
| cltv | 0.402 | 0.0493 | 8.15 | 3.55e-16 |
| ever_modified | 0.2775 | 0.0353 | 7.86 | 3.75e-15 |
| n_borrowers | 0.8195 | 0.0637 | 12.86 | 7.65e-38 |
| upb_ratio | 0.5045 | 0.092 | 5.49 | 4.10e-08 |

Gini train: 0.8029

Gini in-time holdout: 0.7837

Gini gap (train - holdout): 0.0192

KS train / holdout: 0.6320 / 0.6040

Mean PD on train 0.00458 vs observed 0.00458

Scaling: Factor = 20/ln2 = 28.8539, Offset = 600 - Factor*ln(50) = 487.1229 (600 points at 50:1 good:bad, 20 points to double the odds).

Charts: reports/champion_ks_score.png

### Points table (fitted intercept; Step 7 shifts the constant after calibration)

| variable | bin | count | event_rate | woe | points |
|---|---|---|---|---|---|
| months_since_dq | (-inf, 0.50) | 2910 | 25.3608% | -4.301 | -3.4 |
| months_since_dq | [0.50, 1.50) | 1224 | 7.9248% | -2.928 | 20.4 |
| months_since_dq | [1.50, 4.50) | 2270 | 4.9339% | -2.422 | 29.2 |
| months_since_dq | [4.50, 8.50) | 2190 | 2.0548% | -1.517 | 44.9 |
| months_since_dq | [8.50, 12.50) | 1817 | 1.7061% | -1.327 | 48.2 |
| months_since_dq | [12.50, 18.50) | 1946 | 1.3361% | -1.079 | 52.5 |
| months_since_dq | [18.50, 51.50) | 4677 | 0.7270% | -0.464 | 63.2 |
| months_since_dq | [51.50, inf) | 393413 | 0.2028% | 0.818 | 85.4 |
| cur_dq | (-inf, 0.50) | 407537 | 0.2805% | 0.493 | 75.3 |
| cur_dq | [0.50, 1.50) | 2382 | 16.7086% | -3.774 | 39.7 |
| cur_dq | [1.50, inf) | 528 | 64.3939% | -5.973 | 21.3 |
| fico | (-inf, 662.50) | 22514 | 2.1542% | -1.565 | 49.0 |
| fico | [662.50, 692.50) | 32958 | 1.2713% | -1.029 | 56.6 |
| fico | [692.50, 710.50) | 31698 | 0.7445% | -0.488 | 64.3 |
| fico | [710.50, 729.50) | 38880 | 0.5144% | -0.116 | 69.6 |
| fico | [729.50, 756.50) | 65732 | 0.3803% | 0.187 | 73.9 |
| fico | [756.50, 763.50) | 20727 | 0.2364% | 0.664 | 80.6 |
| fico | [763.50, 789.50) | 95447 | 0.1519% | 1.107 | 86.9 |
| fico | [789.50, inf) | 102455 | 0.0947% | 1.581 | 93.6 |
| fico | Missing | 36 | 0.0000% | 0.0 | 71.2 |
| orig_rate | (-inf, 3.12) | 28962 | 0.1070% | 1.458 | 80.9 |
| orig_rate | [3.12, 3.60) | 65298 | 0.1792% | 0.942 | 77.5 |
| orig_rate | [3.60, 3.85) | 65121 | 0.2518% | 0.601 | 75.2 |
| orig_rate | [3.85, 3.99) | 38112 | 0.3332% | 0.32 | 73.3 |
| orig_rate | [3.99, 4.25) | 71129 | 0.4147% | 0.1 | 71.9 |
| orig_rate | [4.25, 4.26) | 35477 | 0.5271% | -0.141 | 70.3 |
| orig_rate | [4.26, 4.74) | 72088 | 0.7394% | -0.481 | 68.0 |
| orig_rate | [4.74, inf) | 34260 | 1.2464% | -1.008 | 64.5 |
| dti | (-inf, 20.50) | 38058 | 0.0972% | 1.554 | 85.0 |
| dti | [20.50, 23.50) | 22492 | 0.1245% | 1.307 | 82.8 |
| dti | [23.50, 28.50) | 45514 | 0.1977% | 0.843 | 78.7 |
| dti | [28.50, 31.50) | 31423 | 0.3182% | 0.366 | 74.5 |
| dti | [31.50, 37.50) | 68319 | 0.3674% | 0.222 | 73.2 |
| dti | [37.50, 39.50) | 24656 | 0.4421% | 0.036 | 71.5 |
| dti | [39.50, 43.50) | 51984 | 0.4905% | -0.068 | 70.6 |
| dti | [43.50, inf) | 52166 | 0.5828% | -0.242 | 69.1 |
| dti | Missing | 75835 | 0.9323% | -0.715 | 64.9 |
| cltv | (-inf, 44.50) | 31990 | 0.1688% | 1.002 | 82.8 |
| cltv | [44.50, 52.50) | 21293 | 0.2301% | 0.691 | 79.2 |
| cltv | [52.50, 75.50) | 127617 | 0.3260% | 0.342 | 75.2 |
| cltv | [75.50, 79.50) | 24584 | 0.3458% | 0.283 | 74.5 |
| cltv | [79.50, 80.50) | 75654 | 0.3913% | 0.159 | 73.1 |
| cltv | [80.50, 93.50) | 55398 | 0.5433% | -0.171 | 69.2 |
| cltv | [93.50, 95.50) | 42115 | 0.6387% | -0.334 | 67.3 |
| cltv | [95.50, inf) | 30988 | 1.2553% | -1.016 | 59.4 |
| cltv | Missing | 808 | 2.7228% | -1.805 | 50.3 |
| ever_modified | (-inf, 0.50) | 409974 | 0.4339% | 0.055 | 71.6 |
| ever_modified | [0.50, inf) | 473 | 21.5645% | -4.09 | 38.5 |
| n_borrowers | (-inf, 1.50) | 194345 | 0.6535% | -0.357 | 62.8 |
| n_borrowers | [1.50, inf) | 216102 | 0.2827% | 0.485 | 82.7 |
| upb_ratio | (-inf, 0.78) | 37890 | 0.1320% | 1.248 | 89.4 |
| upb_ratio | [0.78, 0.84) | 25002 | 0.2680% | 0.538 | 79.0 |
| upb_ratio | [0.84, 0.91) | 59675 | 0.3301% | 0.329 | 76.0 |
| upb_ratio | [0.91, 0.94) | 56088 | 0.4636% | -0.012 | 71.0 |
| upb_ratio | [0.94, 1.00) | 210613 | 0.5598% | -0.201 | 68.3 |
| upb_ratio | [1.00, inf) | 21179 | 0.6044% | -0.278 | 67.2 |

## Step 6: challenger (LightGBM) and champion-challenger comparison, development data only

LightGBM, 400 trees, learning rate 0.03, 31 leaves, min_child_samples 200, monotone constraints on 12 of 14 numeric features (units and loan_age free; categoricals unconstrained), fitted on train, no early stopping.

| Criterion | Champion (scorecard) | Challenger (GBM) |
|---|---|---|
| Gini, train | 0.8029 | 0.8635 |
| Gini, in-time holdout | 0.7837 | 0.7803 |
| Gini, out-of-time (after Step 7) | 0.8430 | 0.8312 |
| Gini drop, holdout to OOT | -0.0593 | -0.0508 |
| Calibration by grade (Step 7) | see Step 7 back-test | not calibrated; mean PD OOT 0.5332% vs 0.4490% actual |
| Explainability | Points per bin | SHAP values (below) |
| Stability (PSI, Step 8) | grade-mix PSI 0.0018 (OOT), 0.0077 (Dec 2019) | not computed |
| Ease of monitoring and governance | Simple: points table, PSI per bin, clear SS1/23 route | Needs SHAP; its ranking differs from the scorecard; harder to monitor |

Gini difference on holdout (challenger - champion): -0.0034

**Decision (made on train and holdout only, before the OOT sample was opened): the scorecard stays champion and the GBM is kept as a benchmark.** Reasons: the challenger does not beat the scorecard on holdout Gini (0.7803 vs 0.7837), overfits more (train-to-holdout gap 0.083 vs 0.019), is harder to monitor and explain, and its SHAP ranking differs from the scorecard's. The OOT figures in the table, filled after Step 7, confirm it.

### Challenger mean |SHAP| (50,000 holdout loans, log-odds)

| feature | mean abs SHAP |
|---|---|
| fico | 0.6799 |
| n_borrowers | 0.4154 |
| dti | 0.3411 |
| upb_ratio | 0.2633 |
| loan_age | 0.2533 |
| cltv | 0.2094 |
| months_since_dq | 0.1811 |
| occupancy | 0.1776 |
| orig_rate | 0.1184 |
| purpose | 0.0774 |
| ltv | 0.0698 |
| prop_type | 0.0486 |
| channel | 0.0287 |
| first_time_buyer | 0.0228 |
| cur_dq | 0.0139 |
| units | 0.0088 |
| max_dq_12m | 0.0051 |
| orig_term | 0.0013 |
| ever_modified | 0.0005 |

## Step 7: calibration, master scale and OOT back-test

**7a Calibration.** Intercept shift -0.015222 (log-odds) so that mean PD on the development sample (585,900 loan-snapshots, Dec 2015-2017) = observed default rate 0.4542% (mean PD before shift 0.4600%). Scores rise by +0.44 points in total after the shift (constant spread over the 9 variables). Final points table: reports/scorecard_points.csv. Limitation: the anchor covers 2016 to 2018 only, a benign period with no downturn; a regulatory long-run average would need a full cycle.

**7b Master scale** (edges fixed on development; grade 1 = lowest risk)

| grade | PD range | grade PD | loans | share | defaults | observed rate |
|---|---|---|---|---|---|---|
| 1 | 0.0122% to 0.0350% | 0.0261% | 29,890 | 5.10% | 2 | 0.0067% |
| 2 | 0.0350% to 0.0500% | 0.0428% | 34,689 | 5.92% | 5 | 0.0144% |
| 3 | 0.0500% to 0.1000% | 0.0745% | 125,824 | 21.48% | 40 | 0.0318% |
| 4 | 0.1000% to 0.2000% | 0.1453% | 173,961 | 29.69% | 170 | 0.0977% |
| 5 | 0.2000% to 0.4000% | 0.2804% | 131,937 | 22.52% | 393 | 0.2979% |
| 6 | 0.4000% to 0.7999% | 0.5395% | 61,273 | 10.46% | 438 | 0.7148% |
| 7 | 0.8001% to 1.6000% | 1.0639% | 16,341 | 2.79% | 226 | 1.3830% |
| 8 | 1.6003% to 3.9988% | 2.3758% | 6,022 | 1.03% | 195 | 3.2381% |
| 9 | 4.0009% to 88.1702% | 21.3713% | 5,963 | 1.02% | 1192 | 19.9899% |

**7c Out-of-time (Dec 2018, 254,118 loans, 1,141 defaults)**

- OOT Gini: 0.8430 (in-time holdout 0.7837; drop -0.0593); OOT KS: 0.6992
- OOT actual default rate 0.4490% vs expected (mean PD) 0.4818%; actual/expected = 0.932

**7d Back-test by grade** (traffic light on the one-sided binomial p-value: Green >= 0.05, Amber 0.01 to 0.05, Red < 0.01)

| grade | loans | defaults | expected defaults | grade PD | observed | obs/PD | p binomial | p Jeffreys | light | p (too few) | too conservative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 13,961 | 3 | 3.6 | 0.0261% | 0.0215% | 0.82 | 0.7041 | 0.5995 | Green | 0.5070 |  |
| 2 | 14,377 | 1 | 6.2 | 0.0428% | 0.0070% | 0.16 | 0.9979 | 0.9936 | Green | 0.0152 | Yes |
| 3 | 52,604 | 19 | 39.2 | 0.0745% | 0.0361% | 0.48 | 0.9999 | 0.9998 | Green | 0.0003 | Yes |
| 4 | 73,418 | 52 | 106.7 | 0.1453% | 0.0708% | 0.49 | 1.0000 | 1.0000 | Green | 0.0000 | Yes |
| 5 | 58,843 | 94 | 165.0 | 0.2804% | 0.1597% | 0.57 | 1.0000 | 1.0000 | Green | 0.0000 | Yes |
| 6 | 28,056 | 155 | 151.4 | 0.5395% | 0.5525% | 1.02 | 0.3941 | 0.3787 | Green | 0.6366 |  |
| 7 | 7,045 | 93 | 75.0 | 1.0639% | 1.3201% | 1.24 | 0.0236 | 0.0208 | Amber | 0.9818 |  |
| 8 | 2,702 | 109 | 64.2 | 2.3758% | 4.0340% | 1.7 | 0.0000 | 0.0000 | Red | 1.0000 |  |
| 9 | 3,112 | 615 | 665.1 | 21.3713% | 19.7622% | 0.92 | 0.9871 | 0.9863 | Green | 0.0145 | Yes |

Chart: reports/backtest_by_grade.png

**Gini by sample (champion, calibrated)**

| Sample | Loans | Gini | Gini, cur_dq = 0 only |
|---|---|---|---|
| Dev 2015-12 | 167,815 | 0.8212 | 0.7263 |
| Dev 2016-12 | 193,953 | 0.7576 | 0.6610 |
| Dev 2017-12 | 224,132 | 0.8291 | 0.6743 |
| Dev pooled | 585,900 | 0.7973 | 0.6787 |
| OOT 2018-12 | 254,118 | 0.8430 | 0.7186 |

Share of loans with cur_dq >= 1: dev 0.7179%, OOT 0.8303%.


## Step 8: monitoring MI pack (reports/MI_pack.pdf)

Thresholds are my choice (industry rules of thumb). Baseline = development sample.

### RAG table

| Test | Thresholds (G/A/R) | OOT Dec 2018 | RAG | Dec 2019 | RAG  |
|---|---|---|---|---|---|
| PSI, grade mix | <0.10 / .10-.25 / >.25 | 0.0018 | Green | 0.0077 | Green |
| PSI, score deciles | <0.10 / .10-.25 / >.25 | 0.0012 | Green | 0.0064 | Green |
| Gini (development 0.7973) | drop <10% / 10-20% / >20% | 0.8430 (drop -5.7%) | Green | 0.4759 (drop +40.3%) | Red |
| Grade back-test (worst grade) | p >=.05 / .01-.05 / <.01 | G7 A1 R1 | Red | G0 A0 R9 | Red |
| Overall actual / expected | 0.8-1.2 / 0.7-1.5 / outside | 0.932 (0.449% vs 0.482%) | Green | 7.535 (3.988% vs 0.529%) | Red |
| CSI, worst variable | <0.10 / .10-.25 / >.25 | 0.0667 (orig_rate) | Green | 0.4881 (upb_ratio) | Red |

Gini by snapshot: Dev 2015-12 0.8212; Dev 2016-12 0.7576; Dev 2017-12 0.8291; OOT 2018-12 0.8430; Dec 2019 0.4759

### CSI per variable

| variable | CSI OOT Dec 2018 | CSI Dec 2019 |
|---|---|---|
| upb_ratio | 0.0576 | 0.4881 |
| orig_rate | 0.0667 | 0.0639 |
| dti | 0.0339 | 0.0335 |
| months_since_dq | 0.0043 | 0.0209 |
| cltv | 0.0058 | 0.0054 |
| ever_modified | 0.0021 | 0.0035 |
| n_borrowers | 0.0016 | 0.0017 |
| fico | 0.0014 | 0.0011 |
| cur_dq | 0.0002 | 0.001 |

### Grade back-test, all snapshots

| snapshot | grade | loans | defaults | expected | observed | obs/PD | p binomial | light |
|---|---|---|---|---|---|---|---|---|
| OOT Dec 2018 | 1 | 13961 | 3 | 3.6 | 0.0215% | 0.82 | 0.7041 | Green |
| OOT Dec 2018 | 2 | 14377 | 1 | 6.2 | 0.0070% | 0.16 | 0.9979 | Green |
| OOT Dec 2018 | 3 | 52604 | 19 | 39.2 | 0.0361% | 0.48 | 0.9999 | Green |
| OOT Dec 2018 | 4 | 73418 | 52 | 106.7 | 0.0708% | 0.49 | 1.0000 | Green |
| OOT Dec 2018 | 5 | 58843 | 94 | 165.0 | 0.1597% | 0.57 | 1.0000 | Green |
| OOT Dec 2018 | 6 | 28056 | 155 | 151.4 | 0.5525% | 1.02 | 0.3941 | Green |
| OOT Dec 2018 | 7 | 7045 | 93 | 75.0 | 1.3201% | 1.24 | 0.0236 | Amber |
| OOT Dec 2018 | 8 | 2702 | 109 | 64.2 | 4.0340% | 1.7 | 0.0000 | Red |
| OOT Dec 2018 | 9 | 3112 | 615 | 665.1 | 19.7622% | 0.92 | 0.9871 | Green |
| Dec 2019 | 1 | 14832 | 78 | 3.9 | 0.5259% | 20.18 | 0.0000 | Red |
| Dec 2019 | 2 | 15072 | 136 | 6.4 | 0.9023% | 21.09 | 0.0000 | Red |
| Dec 2019 | 3 | 49777 | 741 | 37.1 | 1.4886% | 19.97 | 0.0000 | Red |
| Dec 2019 | 4 | 63720 | 1790 | 92.6 | 2.8092% | 19.33 | 0.0000 | Red |
| Dec 2019 | 5 | 48486 | 2424 | 136.0 | 4.9994% | 17.83 | 0.0000 | Red |
| Dec 2019 | 6 | 22904 | 1647 | 123.6 | 7.1909% | 13.33 | 0.0000 | Red |
| Dec 2019 | 7 | 6380 | 655 | 67.9 | 10.2665% | 9.65 | 0.0000 | Red |
| Dec 2019 | 8 | 2810 | 504 | 66.8 | 17.9359% | 7.55 | 0.0000 | Red |
| Dec 2019 | 9 | 3184 | 1084 | 680.5 | 34.0452% | 1.59 | 0.0000 | Red |
| Dec 2019 adjusted | 1 | 14832 | 3 | 3.9 | 0.0202% | 0.78 | 0.7416 | Green |
| Dec 2019 adjusted | 2 | 15072 | 4 | 6.4 | 0.0265% | 0.62 | 0.8845 | Green |
| Dec 2019 adjusted | 3 | 49777 | 24 | 37.1 | 0.0482% | 0.65 | 0.9910 | Green |
| Dec 2019 adjusted | 4 | 63720 | 57 | 92.6 | 0.0895% | 0.62 | 1.0000 | Green |
| Dec 2019 adjusted | 5 | 48486 | 129 | 136.0 | 0.2661% | 0.95 | 0.7363 | Green |
| Dec 2019 adjusted | 6 | 22904 | 117 | 123.6 | 0.5108% | 0.95 | 0.7351 | Green |
| Dec 2019 adjusted | 7 | 6380 | 88 | 67.9 | 1.3793% | 1.3 | 0.0104 | Amber |
| Dec 2019 adjusted | 8 | 2810 | 101 | 66.8 | 3.5943% | 1.51 | 0.0000 | Red |
| Dec 2019 adjusted | 9 | 3184 | 507 | 680.5 | 15.9234% | 0.75 | 1.0000 | Green |

### COVID check: delinquency by assistance flag, loan-months Apr-Dec 2020

| flag | rows | 30+ DPD | 90+ DPD |
|---|---|---|---|
| Forbearance plan (F) | 72608 | 86.1% | 49.7% |
| No flag | 1665823 | 0.9% | 0.2% |
| Other assistance / deferral / disaster | 15407 | 32.6% | 18.2% |

### What happened to the OOT Dec 2018 defaults (data to Mar 2026)

| outcome | Forborne | Not forborne | All |
|---|---|---|---|
| Credit-event termination (REO/short sale/charge-off/note sale) | 9 | 151 | 160 |
| Cured to current or paid off | 164 | 667 | 831 |
| Still delinquent / modified at end of data | 36 | 114 | 150 |
| All | 209 | 932 | 1141 |

### What happened to the Dec 2019 defaults (data to Mar 2026)

| outcome | Forborne | Not forborne | All |
|---|---|---|---|
| Credit-event termination (REO/short sale/charge-off/note sale) | 61 | 75 | 136 |
| Cured to current or paid off | 8029 | 448 | 8477 |
| Still delinquent / modified at end of data | 376 | 70 | 446 |
| All | 8466 | 593 | 9059 |

Dec 2019 defaults treated as non-defaults (forborne and later cured or paid off, no credit-event termination): 8,029 of 9,059. For OOT Dec 2018 the same rule would exclude 164 of 1,141.

Dec 2019 adjusted: default rate 0.4534%, actual/expected 0.857, Gini 0.8049 (all loans) / 0.6862 (cur_dq = 0 only); grade back-test lights {'Green': 7, 'Amber': 1, 'Red': 1}.

## Step 9: IFRS 9 extension, Dec 2019 portfolio (227,165 loans, $42,263.23m balance)

Illustrative assumptions (mine): scenarios and weights below; LGD 10%; discount at original rate; no prepayment; Stage 3 out of scope (loans already in default are excluded from the snapshots).

### 9a Staging (Stage 2 if 30+ days past due, or PD >= 2.5x PD at first observation and PD > 0.3%)

| stage | loans | balance | share of balance | mean 12m PD |
|---|---|---|---|---|
| 1 | 219,026 | $40,695.27m | 96.29% | 0.219% |
| 2 | 8,139 | $1,567.96m | 3.71% | 8.885% |

Stage 2 triggers: 2,283 loans 30+ DPD; 5,856 on the relative-PD test only. Stage 3: out of scope.

### 9b Lifetime PD

Mean 12-month PD 0.529%; mean lifetime PD 5.696% (Stage 1 4.145%, Stage 2 47.435%). Base hazard by loan age (years): 0: 0.359%, 1: 0.460%, 2: 0.474%, 3: 0.480%, 4: 0.455%, 5: 0.345%, 6: 0.344%; held flat beyond year 6 (no data).

### 9d Unemployment regression: logit(default rate) = alpha + beta x dU (84 month-end snapshots, Jan 2013 - Dec 2019; dU = change in UNRATE over the 12-month outcome window)

- Specification as set (2013-2019, n = 84): alpha = -5.3647, **beta = 0.1817**, standard error 0.0244 (OLS) / 0.0623 (Newey-West, 12 lags), R-squared 0.404
- Sensitivity, pre-COVID snapshots only (2013-2018, n = 72): beta = 0.3049, standard error 0.0650 (OLS) / 0.0947 (Newey-West), R-squared 0.239
- Scenario weights (assumed): Upside -0.5 pp 20%, Base 0.0 pp 50% (the table's base value was garbled; assumed 0), Downside +3.0 pp 30%. Beta used for ECL: 0.1817.

### 9c ECL by scenario (LGD 10%)

| scenario | dU (pp) | weight | ECL | of which Stage 1 | of which Stage 2 | coverage (ECL / balance) |
|---|---|---|---|---|---|---|
| Upside | -0.5 | 20% | $53.14m | $7.87m | $45.27m | 0.1257% |
| Base | +0.0 | 50% | $55.69m | $8.61m | $47.08m | 0.1318% |
| Downside | +3.0 | 30% | $73.54m | $14.77m | $58.77m | 0.1740% |
| Probability-weighted |  | 100% | $60.54m | $10.31m | $50.22m | 0.1432% |

Weighted minus base: $4.85m. Of this, $4.31m comes from the weights being skewed to the downside (expected dU = +0.8 pp, ECL at that dU $60.00m) and $0.54m from convexity.
Share of weighted ECL: Stage 2 83.0%; loans 30+ DPD 52.2%.

With the pre-COVID beta (0.3049): weighted ECL $64.78m, Base $55.69m, Downside $88.81m.

### LGD sensitivity

| LGD | weighted ECL | coverage | Base ECL | Downside ECL |
|---|---|---|---|---|
| 5% | $30.27m | 0.0716% | $27.84m | $36.77m |
| 10% | $60.54m | 0.1432% | $55.69m | $73.54m |
| 20% | $121.07m | 0.2865% | $111.38m | $147.09m |

### 9e Post model adjustment sizing (illustrative; base-scenario ECL = $55.69m)

| Reference point | ECL | Difference to model ECL |
|---|---|---|
| Model as is (base scenario) | $55.69m | |
| PDs scaled by OOT actual/expected by grade (known at Dec 2019) | $57.02m | $1.33m (+2.4%) |
| PDs scaled to reported 2020 experience (A/E 7.53) | $165.98m | $110.30m (+198%) |
| PDs scaled to forbearance-adjusted 2020 experience (A/E 0.86) | $51.01m | $-4.67m (-8.4%) |

PMA needed to undo a mechanical recalibration to reported 2020 defaults: $-114.97m (-69% of that figure).

### Hurricane check on the development default spike: share of each snapshot's defaults with a disaster flag during the window

| snapshot | defaults | with disaster flag | share |
|---|---|---|---|
| 2015-12 | 592 | 15 | 2.5% |
| 2016-12 | 1,084 | 340 | 31.4% |
| 2017-12 | 985 | 146 | 14.8% |
| 2018-12 | 1,141 | 40 | 3.5% |

## Step 10: validation findings (reports/Validation_Report.pdf)

Overall rating: Fit for use with conditions. Findings: 10 in total, 2 High, 7 Medium, 1 Low.

Top three:
- F1 (High): Calibration anchor excludes a downturn
- F2 (High): Model and monitoring misread COVID forbearance
- F3 (Medium): Grades 7 and 8 under-predicted; grades 3-5 and 9 over-predicted


## Azure phase: local rehearsal of the monthly batch-and-monitor loop (no Azure resources used)

36 month-end snapshots (Jan 2018 - Dec 2020) scored with the packaged champion pyfunc (reproduces Step 7 PD and grade exactly), monitoring signals computed per month, performance lagged 12 months. The Azure run (model versions, endpoint type, batch runs, alert timing in the workspace, monthly cost) is to be filled in after deployment.

### First month each alert fires (local rehearsal)

| signal | first month |
|---|---|
| CSI > 0.25 (any variable) | 2019-06 |
| CSI > 0.25 excluding upb_ratio | never |
| Grade PSI amber (> 0.10) | never |
| Delinquent share > 1.5x development (amber) | 2020-03 |
| Delinquent share > 2x development (red) | 2020-04 |
| Grade PSI red (> 0.25) | never |
| Data quality flag | never |
| Gini or A/E not Green (performance) | 2020-05 |
| Performance first red, scored month (A/E red) | 2019-06 |

### Monthly signals

| month | loans | psi_grade | dq_ratio | csi_max | csi_max_var | n_dq_flags | status |
|---|---|---|---|---|---|---|---|
| 2018-01 | 226322 | 0.0007 | 1.0488 | 0.028 | upb_ratio | 0 | Green |
| 2018-02 | 228459 | 0.0008 | 1.1353 | 0.0316 | upb_ratio | 0 | Green |
| 2018-03 | 230541 | 0.0007 | 0.9239 | 0.0344 | upb_ratio | 0 | Green |
| 2018-04 | 233877 | 0.0007 | 0.8887 | 0.0355 | upb_ratio | 0 | Green |
| 2018-05 | 236116 | 0.0005 | 0.9027 | 0.039 | upb_ratio | 0 | Green |
| 2018-06 | 238351 | 0.0006 | 0.9801 | 0.0427 | upb_ratio | 0 | Green |
| 2018-07 | 240597 | 0.0007 | 0.9327 | 0.0458 | upb_ratio | 0 | Green |
| 2018-08 | 242837 | 0.0007 | 0.8903 | 0.0493 | upb_ratio | 0 | Green |
| 2018-09 | 245552 | 0.0012 | 1.1925 | 0.0519 | upb_ratio | 0 | Green |
| 2018-10 | 247854 | 0.0012 | 0.983 | 0.0541 | upb_ratio | 0 | Green |
| 2018-11 | 251499 | 0.0014 | 1.0773 | 0.0558 | upb_ratio | 0 | Green |
| 2018-12 | 254118 | 0.0018 | 1.1566 | 0.0667 | orig_rate | 0 | Green |
| 2019-01 | 256538 | 0.002 | 1.0844 | 0.0807 | orig_rate | 0 | Green |
| 2019-02 | 255474 | 0.0022 | 1.1952 | 0.082 | orig_rate | 0 | Green |
| 2019-03 | 255199 | 0.002 | 1.0917 | 0.0999 | upb_ratio | 0 | Green |
| 2019-04 | 253282 | 0.0019 | 1.0373 | 0.1395 | upb_ratio | 0 | Green |
| 2019-05 | 250572 | 0.0026 | 1.292 | 0.2036 | upb_ratio | 0 | Green |
| 2019-06 | 247964 | 0.0034 | 1.4994 | 0.2546 | upb_ratio | 0 | Green |
| 2019-07 | 245317 | 0.003 | 1.1544 | 0.3119 | upb_ratio | 0 | Green |
| 2019-08 | 241786 | 0.0034 | 1.1995 | 0.3682 | upb_ratio | 0 | Green |
| 2019-09 | 237994 | 0.0043 | 1.2731 | 0.3958 | upb_ratio | 0 | Green |
| 2019-10 | 233831 | 0.0052 | 1.2046 | 0.4261 | upb_ratio | 0 | Green |
| 2019-11 | 230485 | 0.0063 | 1.3846 | 0.4578 | upb_ratio | 0 | Green |
| 2019-12 | 227165 | 0.0077 | 1.4 | 0.4881 | upb_ratio | 0 | Green |
| 2020-01 | 224380 | 0.0087 | 1.2255 | 0.519 | upb_ratio | 0 | Green |
| 2020-02 | 221535 | 0.0103 | 1.3544 | 0.5456 | upb_ratio | 0 | Green |
| 2020-03 | 217432 | 0.013 | 1.5318 | 0.5799 | upb_ratio | 0 | Amber |
| 2020-04 | 212059 | 0.0573 | 5.1711 | 0.6141 | upb_ratio | 0 | Red |
| 2020-05 | 206197 | 0.0899 | 6.7665 | 0.6505 | upb_ratio | 0 | Red |
| 2020-06 | 195962 | 0.0528 | 4.0682 | 0.7001 | upb_ratio | 0 | Red |
| 2020-07 | 188514 | 0.0397 | 2.4023 | 0.7409 | upb_ratio | 0 | Red |
| 2020-08 | 182489 | 0.0408 | 2.1297 | 0.7726 | upb_ratio | 0 | Red |
| 2020-09 | 176657 | 0.0444 | 2.0384 | 0.8074 | upb_ratio | 0 | Red |
| 2020-10 | 170602 | 0.0487 | 1.8453 | 0.8469 | upb_ratio | 0 | Amber |
| 2020-11 | 165305 | 0.0517 | 1.977 | 0.8944 | upb_ratio | 0 | Amber |
| 2020-12 | 159667 | 0.0563 | 1.8583 | 0.9603 | upb_ratio | 0 | Amber |

### Lagged performance (month scored, run in which it becomes available)

| scored_month | evaluated_at_run | defaults | actual | expected | ae | gini | gini_drop | rag_gini | rag_ae | n_red_grades |
|---|---|---|---|---|---|---|---|---|---|---|
| 2018-01 | 2019-01 | 958 | 0.0042 | 0.0046 | 0.9185 | 0.8143 | -0.0213 | Green | Green | 0 |
| 2018-02 | 2019-02 | 975 | 0.0043 | 0.0047 | 0.9128 | 0.8034 | -0.0076 | Green | Green | 1 |
| 2018-03 | 2019-03 | 980 | 0.0043 | 0.0044 | 0.9732 | 0.8103 | -0.0163 | Green | Green | 2 |
| 2018-04 | 2019-04 | 994 | 0.0043 | 0.0043 | 0.9929 | 0.816 | -0.0234 | Green | Green | 2 |
| 2018-05 | 2019-05 | 1015 | 0.0043 | 0.0043 | 0.9917 | 0.8239 | -0.0334 | Green | Green | 2 |
| 2018-06 | 2019-06 | 1051 | 0.0044 | 0.0045 | 0.9799 | 0.8228 | -0.032 | Green | Green | 1 |
| 2018-07 | 2019-07 | 1045 | 0.0043 | 0.0044 | 0.9833 | 0.8115 | -0.0178 | Green | Green | 2 |
| 2018-08 | 2019-08 | 1082 | 0.0045 | 0.0044 | 1.0126 | 0.8175 | -0.0254 | Green | Green | 1 |
| 2018-09 | 2019-09 | 1087 | 0.0044 | 0.0049 | 0.9125 | 0.8217 | -0.0306 | Green | Green | 1 |
| 2018-10 | 2019-10 | 1085 | 0.0044 | 0.0046 | 0.9601 | 0.8266 | -0.0367 | Green | Green | 1 |
| 2018-11 | 2019-11 | 1109 | 0.0044 | 0.0047 | 0.9449 | 0.843 | -0.0573 | Green | Green | 1 |
| 2018-12 | 2019-12 | 1141 | 0.0045 | 0.0048 | 0.9318 | 0.843 | -0.0574 | Green | Green | 1 |
| 2019-01 | 2020-01 | 1149 | 0.0045 | 0.0048 | 0.9405 | 0.8379 | -0.0509 | Green | Green | 1 |
| 2019-02 | 2020-02 | 1133 | 0.0044 | 0.0049 | 0.9045 | 0.825 | -0.0347 | Green | Green | 2 |
| 2019-03 | 2020-03 | 1159 | 0.0045 | 0.0048 | 0.95 | 0.8181 | -0.026 | Green | Green | 2 |
| 2019-04 | 2020-04 | 1219 | 0.0048 | 0.0047 | 1.0276 | 0.8244 | -0.034 | Green | Green | 2 |
| 2019-05 | 2020-05 | 1530 | 0.0061 | 0.0048 | 1.2669 | 0.8117 | -0.0181 | Green | Amber | 3 |
| 2019-06 | 2020-06 | 5435 | 0.0219 | 0.0053 | 4.172 | 0.526 | 0.3402 | Red | Red | 9 |
| 2019-07 | 2020-07 | 7443 | 0.0303 | 0.0049 | 6.1844 | 0.4708 | 0.4095 | Red | Red | 9 |
| 2019-08 | 2020-08 | 8109 | 0.0335 | 0.005 | 6.7572 | 0.4674 | 0.4137 | Red | Red | 9 |
| 2019-09 | 2020-09 | 8454 | 0.0355 | 0.0051 | 6.9901 | 0.4698 | 0.4108 | Red | Red | 9 |
| 2019-10 | 2020-10 | 8755 | 0.0374 | 0.005 | 7.465 | 0.4743 | 0.4051 | Red | Red | 9 |
| 2019-11 | 2020-11 | 8926 | 0.0387 | 0.0052 | 7.4159 | 0.4749 | 0.4044 | Red | Red | 9 |
| 2019-12 | 2020-12 | 9059 | 0.0399 | 0.0053 | 7.5349 | 0.4759 | 0.4031 | Red | Red | 9 |

Chart: reports/azure_sim_exhibit.png

## Azure run (workspace)

Workspace `mlw-pd-mortgage`, region UK South, private workspace storage. Batch endpoint scoring and monitoring pipeline run for all 36 month-ends, Jan 2018 to Dec 2020.

| Item | Result |
|---|---|
| Models registered | `mortgage_pd_champion` versions 1 (MLflow pyfunc) and 2 (custom model, the deployed one); `mortgage_pd_challenger` version 1 (MLflow pyfunc, registered, not deployed) |
| Endpoint type | Batch endpoint `pd-batch`, deployment `champion` (default); no online endpoint created |
| Batch runs | 36 completed (one per month) plus 1 failed (image build, D59) = 37 jobs |
| Monitoring runs | 36 monthly results: 27 completed, 9 failed on purpose (alerts, April to December 2020); 38 pipeline jobs in total including one failed run from the scores-file format bug (D61) and one repeat of 2018-01 |
| Azure vs local rehearsal | All 36 statuses identical; every signal equal to 1e-15; 2018-01 scores equal to the local model to 3e-16 with 0 grade mismatches |
| Running cost | GBP 1.42 in total for the whole build and the 36-month run (Cost Management, resource group rg-pd-mortgage, 11-12 Oct 2026, two days of use). Not a steady monthly figure: the cluster scales to zero, so what would remain when idle is storage, the container registry and Key Vault, which I have not measured |

### First month each alert fired (Azure run)

| Signal | First month |
|---|---|
| Grade-mix PSI amber (above 0.10) | never (peak 0.0899, May 2020) |
| Any CSI above 0.25, excluding upb_ratio | never (upb_ratio excluded as a sample artefact, D62) |
| Data-quality flag | never |
| Delinquent share above 1.5x development (amber) | 2020-03 (1.53x) |
| Delinquent share above 2x development (red) | 2020-04 (5.2x) |
| Performance alert: 3 or more red grades | 2020-05 run (month scored 2019-05: Gini 0.812, actual/expected 1.27, 3 red grades) |
| Performance Gini and actual/expected red | 2020-06 run (month scored 2019-06: Gini 0.526, actual/expected 4.17, 9 red grades) |

Sequence: delinquent-share red in April 2020 (the day the month is scored), first performance alert in May, first Red Gini and actual/expected in June. Grade PSI and CSI as specified (thresholds 0.10 and 0.25) never alerted. From October 2020 the input status falls back to Amber (delinquent share 1.8x to 2.0x) while the performance alerts continue (Gini 0.47 to 0.48, actual/expected 7.4 to 7.5).
