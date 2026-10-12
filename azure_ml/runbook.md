# Runbook: mortgage 12-month PD model monitoring

**Scope.** Monthly batch scoring (`pd-batch`, champion) and monitoring pipeline (`pd-monitoring`). Alert channel: Teams webhook (`ALERT_WEBHOOK_URL`).

**Who is alerted.** Model owner (first response, 1 working day), credit risk analytics lead (informed), Finance ECL team (informed on any Red that persists two months).
In this project one person plays every role.

| Signal | Amber | Red |
|---|---|---|
| Grade-mix PSI | > 0.10 | > 0.25 |
| CSI per variable | n/a | > 0.25 |
| Data quality (missing or out-of-range per field) | n/a | > 2x development rate and above 0.5 percentage points |
| Delinquent share (30+ DPD) vs development | > 1.5x | > 2x |
| Gini relative drop vs 0.797 (12 months later) | 10-20% | > 20% |
| Actual / expected (12 months later) | 0.7-0.8 or 1.2-1.5 | outside |
| Grade back-test (12 months later) | binomial p 0.01-0.05 | p < 0.01 in 3 or more grades (grades 7-8 fail every month: standing finding F3) |

**First 30 minutes after a Red.**
1. Did the batch run complete with the expected loan count (within about 5% of last month)? If not, it is a pipeline fault, not model drift.
2. Check data quality fields first (sentinel codes, missing, out-of-range). A feed change shows here before anything else.
3. Read which variable drives the CSI. `upb_ratio` is a known artefact in this sample (no new originations); it is excluded from alerting (`--csi_exclude`) but still logged, do not escalate.
4. Check delinquency incidence and the forbearance, deferral and disaster flags (Borrower Assistance Plan, Payment Deferral Flag, Delinquency Due to Disaster). A forbearance wave inflates delinquency without credit losses (see validation report finding F2).
5. Compare the grade mix and the share of loans in grades 7-9 with last month.

**Escalation.**
- One Red input signal: log, investigate, no model change.
- Red input signal for two consecutive months, or any Red performance signal: raise to the model risk forum; run the validation tests on the affected months; consider a PMA (see decisions.md D53).
- Recalibration (intercept and slope on development data): two consecutive Amber-or-worse grade back-tests, or A/E outside 0.7-1.5 once the cause is not forbearance or a data fault.
- Replace the champion by the challenger only if the champion fails the performance tests and the challenger passes the same tests on the same months, with the model risk forum approving.
- Do not recalibrate on defaults that are mostly forbearance cures (D46).

**Record.** For every alert: date, signal, cause, decision, owner, expiry of any overlay.
