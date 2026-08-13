# Predeclared Ender-20 hybrid stability gate

Frozen on 2026-08-03 before computing any benchmark/TabM hybrid result.

## Purpose

The existing consecutive-era TabM residual has positive full and recent BMC,
plausible target correlation, acceptable BMC Sharpe, low benchmark similarity,
and a deployment-compatible inference artifact. It is not promotion-approved
because its full-period BMC maximum drawdown is `0.3009819625`, above the
predeclared `0.15` ceiling.

This experiment tests whether a direct prediction that combines the official
`v53_lgbm_ender20` benchmark with that fixed TabM residual can retain meaningful
unique signal while reducing BMC drawdown. It does not retrain the TabM or tune
its architecture.

## Frozen inputs

- TabM OOF predictions:
  `../ender20_nn_architecture_v53/predictions/scale_disk_tabm_k64_train500k.parquet`
- Feature-store manifest containing exact targets and benchmark predictions:
  `../../../v5.3/target_ender_20_feature_store/manifest-*.parquet`
- Target: `target_ender_20`
- Benchmark: `v53_lgbm_ender20`
- OOF cohort: the exact 5,112,039-row, 855-era consecutive cohort already
  validated by the original deployment gate.

All IDs, eras, targets, folds, and prediction-semantics metadata must match the
existing validated artifacts exactly. Any mismatch fails closed.

## Frozen transform and variants

Within each era:

1. Percentile-rank the official benchmark using average tie ranks.
2. Percentile-rank the raw TabM residual using average tie ranks.
3. Form `score = (1 - w) * benchmark_rank + w * residual_rank`.
4. Percentile-rank `score` again to produce the submission prediction.

The only swept parameter is residual weight `w`:

| candidate | benchmark weight | residual weight |
| --- | ---: | ---: |
| `hybrid_w35` | 0.65 | 0.35 |
| `hybrid_w45` | 0.55 | 0.45 |
| `hybrid_w55` | 0.45 | 0.55 |
| `hybrid_w65` | 0.35 | 0.65 |
| `hybrid_w75` | 0.25 | 0.75 |

Benchmark-only and residual-only signals are reference rows, not selectable
candidates.

## Frozen calibration and holdout

- Sort the 855 OOF eras chronologically.
- Calibration: the first 655 eras.
- Holdout: the final 200 eras.
- Do not change the boundary after reading results.

The holdout is formula-new but not data-new: prior work reported the residual's
aggregate last-200 BMC. It is therefore a useful robustness gate, not a claim of
fully independent live validation.

## Calibration selection rule

A candidate is calibration-eligible only if all checks pass:

- BMC mean is positive.
- BMC mean retains at least 40% of residual-only calibration BMC.
- BMC Sharpe is above 0.45.
- BMC maximum drawdown is below 0.15.
- Target Corr mean is at least 90% of benchmark-only calibration Corr.
- Average per-era correlation with the benchmark is below 0.95.

Select the eligible candidate with the highest calibration BMC mean. Break an
exact tie by lower BMC maximum drawdown, then lower residual weight. If no
candidate is eligible, stop without a winner.

## Holdout and full-period promotion gate

The selected candidate is promotion-eligible only if every check passes:

- Exact finite coverage and one prediction per expected ID.
- Holdout BMC mean is positive and retains at least 35% of residual-only
  holdout BMC.
- Holdout BMC Sharpe is above 0.35.
- Holdout BMC maximum drawdown is below 0.15.
- Holdout target Corr is at least 90% of benchmark-only holdout Corr.
- Full-period BMC mean and last-200-era BMC mean are positive.
- Full-period BMC Sharpe is above 0.45.
- Full-period BMC maximum drawdown is below 0.15.
- Full-period target Corr is in the plausible non-leakage range `0.005` to
  `0.04`.
- Full-period average per-era correlation with the benchmark is below 0.95.

These checks are strict inequalities where stated as above/below. There is no
post-result weight interpolation, threshold revision, or best-looking holdout
substitution.

## Deployment boundary

Passing this offline gate permits packaging and local runtime validation only.
It does not authorize upload or staking. The user must separately approve a new
Numerai model slot/upload after seeing the evidence and caveats.
