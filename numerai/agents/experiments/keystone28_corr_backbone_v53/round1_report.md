# Keystone Round-1 report (KW33)

Generated 2026-08-18T00:13:36+00:00. Terminal state: **STOPPED_AT_KW33_PIPELINE_PARITY_FAILURE**.

Primary comparison: CANDIDATE-V vs CONTROL-T on the Ender20 seed-mean
weighted model score (corr x 0.75 + mmc x 2.25) over the 87-era zone
1133-1219 (six walk-forward blocks, eight-era embargo).

| quantity | CONTROL-T | CANDIDATE-V |
| --- | --- | --- |
| seed-mean weighted mean | -0.003828 | 0.009970 |
| worst-seed weighted mean | -0.004485 | 0.009591 |
| worst-seed sharpe | -0.1198 | 0.2610 |
| worst-seed zero-baseline drawdown | 1.009714 | 0.629038 |
| seed dispersion of means | 0.000468 | 0.000491 |

Benchmark v53_lgbm_ender20 mean CORR 0.020948, mean weighted 0.031586 on the same eras.

## Decision reconstruction

Pipeline parity gate FAILED; no scientific decision was taken.

## Ender60 auxiliary (cutover readiness, non-selecting)

Raw CORR60/MMC60 are reported per vector in round1_result.json. Any
combined number is the HYPOTHETICAL_CUTOVER_WEIGHTED_DIAGNOSTIC and never
selects a winner. Current 60D round score configs are non-payout.

## Non-actions

No GAP/HOLDOUT access; no training beyond the frozen 21-fit cohort; no upload, model creation, submission, staking, deployment, or Numerai account action.
