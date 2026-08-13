# Xerxes-20 LightGBM Challenger Gate (v5.3)

Frozen on 2026-08-03 before any Xerxes BMC scout result was computed.

## Research question

Can a direct `target_xerxes_20` LightGBM using the medium feature set produce a
more stable and sufficiently orthogonal `target_ender_20` signal than the
benchmark-residual TabM line?

This changes both the training target and inductive bias. It does not tune the
previous TabM, add another TabM seed, or search benchmark-blend weights.

## Immutable data and scoring definitions

- Data version: Numerai `v5.3`.
- Training target: `target_xerxes_20`, without residualization.
- Scoring target: `target_ender_20`.
- Benchmark: `v53_lgbm_ender20`.
- Features: the 780 columns in `features.json` feature set `medium`, in the
  declared order.
- Scout sources: `v5.3/downsampled_full.parquet` and
  `v5.3/downsampled_full_benchmark_models.parquet`.
- Confirmation sources: `v5.3/train.parquet`, `v5.3/validation.parquet`, and
  their matching benchmark-model files, restricted to exact one-to-one
  benchmark coverage and finite Xerxes/Ender targets.
- Predictions are ranked within era before scoring.
- Corr and BMC use the repository's Numerai scoring utilities. Similarity to
  the benchmark and prior TabM is symmetric Spearman correlation within era,
  averaged equally across eras.
- Additive maximum drawdown is computed from chronological per-era BMC.

## Scout protocol

All four profiles use LightGBM GPU training, `colsample_bytree=0.1`,
`min_data_in_leaf=10000`, `max_train_samples=500000`, model/sample seed 1337,
and fixed-tree training without outer-fold early stopping.

| config | trees | learning rate | max depth | leaves |
| --- | ---: | ---: | ---: | ---: |
| `r1_base_d6_t6000` | 6,000 | 0.003 | 6 | 63 |
| `r1_trees2k` | 2,000 | 0.003 | 6 | 63 |
| `r1_depth5` | 6,000 | 0.003 | 5 | 31 |
| `r1_depth8` | 6,000 | 0.003 | 8 | 255 |

The scout uses five expanding era folds with a 13-retained-era embargo. Fold 0
has no earlier training data and is skipped; folds 1-4 produce exactly
1,279,658 OOF rows over 214 retained eras, `0373` through `1225`. This matches
the existing every-fourth-era research cohort and corresponds to an effective
52-original-era embargo.

The first 164 retained OOF eras, through `1025`, are scout calibration. The
final 50, `1029` through `1225`, are locked robustness context. A scout artifact
is calibration-eligible only if it has exact unique ID/era/fold coverage,
finite predictions and targets, calibration Ender BMC mean `> 0.0010`, BMC
Sharpe `> 0.20`, BMC maximum drawdown `< 0.15`, Ender Corr `> 0.010`, and
symmetric benchmark Spearman `< 0.85`.

Among calibration-eligible artifacts, select the highest calibration BMC mean.
Break an exact tie by lower calibration BMC drawdown, then shallower depth, then
lexicographically by config name. If none are eligible, stop without opening
the locked 50-era metrics or running a confirmation.

The sole selection must then pass the locked 50 eras with BMC mean `> 0`, BMC
Sharpe `> 0.20`, BMC maximum drawdown `< 0.10`, and Ender Corr `> 0.008`. A
failure stops the family; the holdout cannot substitute another scout.

## Deterministic confirmation

The sole scout winner is copied without changing its LightGBM parameters,
500,000-row cap, or seeds. Only these data-path changes are permitted:

- use the committed medium-feature `target_xerxes_20` disk store;
- change the outer embargo from 13 retained eras to 52 consecutive eras;
- stream validation prediction in bounded batches;
- use a confirmation-specific result name.

The expected confirmation OOF cohort is 5,112,039 unique rows over 855 eras,
`0371` through `1225`. The first 655 OOF eras (`0371`-`1025`) are calibration;
the final 200 (`1026`-`1225`) are locked confirmation context. No alternate
scout may replace the frozen winner after that slice is opened.

## Promotion gate

Every comparison below is strict as written. All metrics and artifact fields
must be finite and exact coverage/provenance checks must pass.

Calibration (`0371`-`1025`):

- Ender BMC mean `>= 0.0015`.
- Ender BMC Sharpe `> 0.35`.
- Ender BMC maximum drawdown `< 0.15`.
- Ender Corr mean `>= 0.012`.
- average benchmark Spearman `< 0.75`.
- average Spearman with the frozen two-seed TabM residual `< 0.75`.

Locked final 200 eras:

- Ender BMC mean `> 0`.
- Ender BMC maximum drawdown `< 0.15`.
- Ender Corr mean `> 0`.

Full 855 eras:

- Ender BMC mean `>= 0.0015`.
- Ender BMC Sharpe `> 0.35`.
- Ender BMC maximum drawdown `< 0.15`.
- Ender Corr mean `>= 0.012`.

## Packaging and hosted-runtime boundary

Passing every offline check permits one local final fit using the exact winning
profile and deterministic 500,000-row sample over all finite, benchmark-covered
historical rows. The resulting cloudpickle must then pass the official Numerai
Predict Docker image with one CPU, at most 4 GiB peak memory, under ten minutes,
no internet, exact input-index preservation, one finite `prediction` column,
and values in `[0, 1]`.

Packaging or Docker failure means `NOT_PROMOTION_ELIGIBLE`. Passing both permits
only a local `PROMOTION_ELIGIBLE_NOT_UPLOADED` artifact. This gate does not
authorize model upload, assignment to a model slot, API submission, or staking.

## Stop rule

This is one frozen four-profile scout and at most one deterministic consecutive
confirmation. If no candidate passes, stop this line without adding targets,
changing thresholds, tuning another capacity profile, blending with TabM, or
searching weights after seeing results.
