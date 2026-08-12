# Ender21 Residual Stability Search (v5.3)

**Protocol frozen:** 2026-08-11, before any Ender21 prediction was scored

## Research question

Can a benchmark-residual neural model retain the unique BMC of the proven K64
TabM while materially reducing its temporal drawdown?

This is a new experiment family. `ender20_nn_architecture_v53` and
`ender20_aux_target_rank_ensemble_v53` are immutable evidence and are never
rewritten, renamed, deleted, or used to reopen their frozen gates.

## Why this experiment exists

The earlier neural search already answered the broad architecture question:
training on the era-wise residual to `v53_lgbm_ender20` was much stronger for
BMC than training directly on Ender20, and K64 TabM was the best scout. Its
consecutive-era confirmation retained useful signal but failed stability:

| historical control | BMC | last-200 BMC | Corr | BMC Sharpe | max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| K64 residual TabM | 0.005798 | 0.003723 | 0.009358 | 0.5087 | 0.300982 |

The next hypothesis is therefore regularization across time, not more width,
more ensemble members, or another post-hoc blend-weight sweep.

## Evidence boundary

All v5.3 historical eras through `1225` were previously scored by the older
neural experiment. They are development/backtest evidence, not an untouched
holdout. Ender21 uses nested expanding-era validation to compare predeclared
variants honestly, but historical success can authorize only a **shadow-live
candidate**. It cannot by itself authorize upload, assignment, staking, or a
claim of production performance. Genuine confirmation must come from future
live rounds that were unavailable at protocol freeze.

The Ender20 auxiliary ensemble's locked slice remains closed under its own
protocol. Ender21 does not invoke that evaluator or inspect its locked receipt.

## Fixed data and validation

- Data version: v5.3.
- Target: `target_ender_20`.
- Benchmark: `v53_lgbm_ender20`.
- Inputs: all 3,555 Numerai features; era and benchmark are metadata only.
- Objective label: era-wise linear residual of Ender20 to the benchmark, with
  intercept and proportion 1.0.
- Scout data: `v5.3/downsampled_full.parquet` plus its benchmark file, restricted
  to benchmark-covered rows and the exact committed era allowlist ending at
  `1025`. Eras `1026`-`1225` remain unavailable to Ender21 because they are the
  protected Ender20 confirmation/locked period.
- Outer validation: five expanding splits; 13 retained-era embargo. The first
  empty-training fold is skipped exactly as in the shared pipeline.
- Inner early stopping: most recent 10% of training eras with a separate
  13-retained-era embargo.
- Training cap: 500,000 rows per outer fold; validation rows are never sampled.
- Seeds: model 1337 and row sample 1337 unless a replication config explicitly
  changes exactly one of them.
- Prediction semantics: raw neural residual, rank-normalized per era only by the
  scoring code.

## Round 1: frozen candidates

All candidates retain K64, 3x512 ReLU blocks, dropout 0.1, AdamW, and the same
residual target. Only the named factors change.

| config | architecture | training loss | purpose |
| --- | --- | --- | --- |
| `r1_control_tabm_k64` | TabM | pooled row MSE | current-source control |
| `r1_tabm_mini_k64` | TabM-mini | pooled row MSE | stronger parameter sharing |
| `r1_tabm_k64_era_balanced` | TabM | inverse-era-count MSE | equal era influence |
| `r1_tabm_k64_block_dro` | TabM | chronological block DRO | emphasize weak time blocks |
| `r1_tabm_mini_k64_block_dro` | TabM-mini | chronological block DRO | combine both regularizers |

Chronological block DRO is frozen as follows. After the inner split and embargo,
the sorted training eras are divided into eight contiguous, near-equal blocks.
For each minibatch, member MSE is averaged to a per-row loss, then to one loss
per represented block. Detached weights are
`softmax(2 * (block_loss / mean_block_loss - 1))`; the differentiable objective
is their weighted sum of block losses. Validation and early stopping always use
ordinary MSE of the averaged ensemble prediction, keeping model selection
comparable. Era-balanced MSE gives every training era equal total weight.

## Selection and stopping rules

Exact metrics use the repository Numerai implementation. Primary selection is
`bmc_last_200_eras.mean`; tie-breaker is full `bmc.mean`. Corr is a guardrail,
not the optimizer.

A Round-1 challenger is eligible only if all are true:

1. full BMC and retained-last-200 BMC are positive;
2. Corr is at least 0.005 and below 0.04;
3. full and retained-last-200 BMC each retain at least 90% of the freshly run
   Round-1 control;
4. full BMC max drawdown improves by at least 15% relative to the control;
5. BMC Sharpe is no worse than 0.05 below the control;
6. every outer fold has positive mean BMC; and
7. predictions, IDs, eras, folds, targets, and benchmark joins validate exactly.

Eligible challengers are ranked by retained-last-200 BMC, then full BMC, then
lower drawdown. If none is eligible, the round is a terminal negative and no
full-data confirmation opens.

Round 2 changes randomness only: selected and control each run once with model
seed 2027 and once with row-sample seed 2027. The selected family advances only
if at least two of its three realizations satisfy the Round-1 retention and
drawdown-improvement rules against the matched control evidence. Two consecutive
rounds without an eligible improvement end the search.

## Deferred historical confirmation gate

This gate is frozen now but **may not run while the Ender20 locked period remains
protected**. A later explicit authorization would allow the selected seed-1337
and model-seed-2027 variants to use the existing consecutive 6,195,697-row disk
store with a 52-era outer/inner embargo.
The final candidate is the frozen 50/50 mean of the two predictions after
within-era percentile ranking. No blend-weight search is allowed.

The historical candidate is `SHADOW_READY` only if all are true:

- full BMC >= 0.0055;
- last-200 BMC >= 0.0035;
- Corr >= 0.0075 and < 0.04;
- BMC Sharpe >= 0.50;
- max drawdown <= 0.225 (at least about 25% below the historical control);
- average correlation with the benchmark <= 0.15;
- each chronological quartile has positive BMC; and
- both individual seeds have positive full and last-200 BMC.

Failure is recorded without trying another blend on the confirmation eras.
Until that authorization exists, a passing Round 2 stops at `SCOUT_WINNER` and
the only honest out-of-sample confirmation is future live data. Passing the
deferred gate would create a shadow-live research artifact only; Numerai upload
and account changes require a separate explicit user decision.

## Research basis

- Numerai BMC measures the unique contribution remaining after neutralization
  to benchmark predictions: https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc
- TabM uses parameter-efficient ensembling; its official implementation exposes
  the more strongly shared `tabm-mini` variant: https://github.com/yandex-research/tabm
- Time-related distribution shift can materially reorder tabular model results,
  motivating chronological robustness rather than random-row validation:
  https://arxiv.org/abs/2406.19380
- Group distributionally robust optimization motivates emphasizing the weakest
  predefined groups: https://arxiv.org/abs/1911.08731

## Status

`PROTOCOL_FROZEN_IMPLEMENTATION_PENDING`

No Ender21 model has been scored at this point.
