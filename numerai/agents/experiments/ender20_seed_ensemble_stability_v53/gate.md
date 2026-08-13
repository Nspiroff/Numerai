# Predeclared Ender-20 two-seed stability gate

Frozen on 2026-08-03 after the single-seed hybrid experiment completed and
before launching the full-consecutive model-seed-2027 run.

## Purpose

The single-seed benchmark/TabM hybrid sweep stopped without a calibration
winner. `hybrid_w35` was close: its calibration drawdown was `0.143763` versus
the `< 0.15` ceiling, while BMC Sharpe was `0.442194` versus the strict `> 0.45`
minimum. Its locked-holdout Sharpe was also close (`0.334145` versus `> 0.35`).

This final stability iteration tests whether rank averaging two independently
initialized versions of the same frozen TabM architecture reduces variance
enough to clear the unchanged gate. It does not tune architecture, learning
rate, epoch selection, row sampling, blend weights, or thresholds.

## Frozen model inputs

- Seed 1337 OOF predictions (already complete):
  `../ender20_nn_architecture_v53/predictions/scale_disk_tabm_k64_train500k.parquet`
- Seed 2027 config (not yet run when this gate was frozen):
  `../ender20_nn_architecture_v53/configs/scale_disk_tabm_k64_train500k_seed2027.py`
- Expected seed-2027 OOF predictions:
  `../ender20_nn_architecture_v53/predictions/scale_disk_tabm_k64_train500k_seed2027.parquet`
- Exact feature-store manifest:
  `../../../v5.3/target_ender_20_feature_store/manifest-*.parquet`
- Target: `target_ender_20`
- Benchmark: `v53_lgbm_ender20`
- Expected cohort: exactly 5,112,039 one-to-one OOF IDs across 855 consecutive
  eras, `0371` through `1225`, with exact era, target, fold, and residual
  prediction-semantics agreement.

The seed-2027 run must use the existing frozen configuration without edits. Any
coverage, semantics, alignment, or configuration mismatch fails closed.

## Frozen ensemble and hybrid transforms

Within each era:

1. Percentile-rank each seed's raw residual prediction with average tie ranks.
2. Average the two ranked residual predictions with equal weight.
3. Percentile-rank that average again to form `two_seed_residual`.
4. Percentile-rank the official benchmark with average tie ranks.
5. For each frozen residual weight `w`, compute
   `(1 - w) * benchmark_rank + w * two_seed_residual` and percentile-rank the
   result again.

The only hybrid weights are the same five tested in the prior round:

| candidate | benchmark weight | two-seed residual weight |
| --- | ---: | ---: |
| `two_seed_hybrid_w35` | 0.65 | 0.35 |
| `two_seed_hybrid_w45` | 0.55 | 0.45 |
| `two_seed_hybrid_w55` | 0.45 | 0.55 |
| `two_seed_hybrid_w65` | 0.35 | 0.65 |
| `two_seed_hybrid_w75` | 0.25 | 0.75 |

Benchmark-only, each single seed, and `two_seed_residual` are reference rows,
not selectable candidates. Benchmark similarity is symmetric per-era Spearman
correlation (average tie ranks on both sides); benchmark self-similarity must be
exactly one within numerical tolerance.

## Frozen split and calibration selection

- Calibration: first 655 chronological OOF eras (`0371`-`1025`).
- Locked holdout: final 200 eras (`1026`-`1225`).
- The holdout may accept or reject the calibration winner; it may not select a
  different weight.

A candidate is calibration-eligible only if all checks pass:

- BMC mean is positive.
- BMC mean retains at least 40% of `two_seed_residual` calibration BMC.
- BMC Sharpe is strictly above 0.45.
- BMC maximum drawdown is strictly below 0.15.
- target Corr is at least 90% of benchmark-only calibration Corr.
- symmetric average per-era Spearman correlation with the benchmark is strictly
  below 0.95.

Select the eligible candidate with highest calibration BMC; exact ties use
lower drawdown, then lower residual weight. If none is eligible, stop without a
winner.

## Locked-holdout and full-period promotion gate

The calibration winner is promotion-eligible only if every check passes:

- exact finite one-to-one coverage;
- holdout BMC is positive and retains at least 35% of `two_seed_residual`
  holdout BMC;
- holdout BMC Sharpe is strictly above 0.35;
- holdout BMC maximum drawdown is strictly below 0.15;
- holdout target Corr is at least 90% of benchmark-only holdout Corr;
- full and last-200 BMC means are positive;
- full BMC Sharpe is strictly above 0.45;
- full BMC maximum drawdown is strictly below 0.15;
- full target Corr is in the inclusive range `0.005` to `0.04`;
- full symmetric benchmark similarity is strictly below 0.95.

There is no post-result interpolation, seed weighting, threshold change, or
holdout substitution. If this final stability iteration fails, stop the current
benchmark-blend approach rather than adding more seed or weight searches.

## Deployment boundary

Passing permits local packaging and runtime validation only. It does not
authorize a Numerai upload or staking; those require separate user approval.
