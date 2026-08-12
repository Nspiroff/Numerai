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
- Round-1 data: physically isolated
  `v5.3/ender21_discovery_full_through_0861.parquet` and
  `v5.3/ender21_discovery_benchmark_models_through_0861.parquet`, plus the exact
  committed discovery allowlist. No later target row is opened by Round 1.
- A separate custody extract through `1021` exists for a future authorized
  Ender21 confirmation, but Round-1 configs do not reference or read it. The
  source row group containing era `1025` also contains protected era `1029`, so
  that entire group remains excluded.
- Discovery eras: exact retained eras `0161`-`0861` from the committed discovery
  allowlist. Ender21 confirmation eras `0865`-`1021` are excluded from discovery.
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
the sorted training eras are divided into four contiguous, near-equal blocks.
For each minibatch, member MSE is averaged to a per-row loss, then to one loss
per represented block. Detached weights are
`softmax(2 * (block_loss / mean_block_loss - 1))`; the differentiable objective
is their weighted sum of block losses. Validation and early stopping always use
ordinary MSE of the averaged ensemble prediction, keeping model selection
comparable. Era-balanced MSE gives every training era equal total weight.

## Selection and stopping rules

Exact metrics use the repository Numerai implementation. Primary selection is
full `bmc.mean`; tie-breaker is mean BMC in the most recent outer fold, then
lower drawdown. `bmc_last_200_eras` is recorded but is not a separate selection
window because discovery contains fewer than 200 retained OOF eras. Corr is a
guardrail, not the optimizer.

A Round-1 challenger is eligible only if all are true:

1. full BMC and most-recent-fold BMC are positive;
2. Corr is at least 0.005 and below 0.04;
3. full and most-recent-fold BMC each retain at least 90% of the freshly run
   Round-1 control;
4. full BMC max drawdown improves by at least 15% relative to the control;
5. BMC Sharpe is no worse than 0.05 below the control;
6. every outer fold has positive mean BMC; and
7. predictions, IDs, eras, folds, targets, and benchmark joins validate exactly.

Eligible challengers are ranked by full BMC, then most-recent-fold BMC, then
lower drawdown. If none is eligible, the round is a terminal negative and no
Ender21 confirmation opens.

Round 2 changes randomness only: selected and control each run once with model
seed 2027 and once with row-sample seed 2027. The selected family advances only
if at least two of its three realizations satisfy the Round-1 retention and
drawdown-improvement rules against the matched control evidence. Two consecutive
rounds without an eligible improvement end the search.

The four mechanically derived Round-2 configs are:

- `r2_control_tabm_k64_model_seed2027` and
  `r2_selected_tabm_k64_block_dro_model_seed2027`, both with model seed 2027
  and row-sample seed 1337;
- `r2_control_tabm_k64_sample_seed2027` and
  `r2_selected_tabm_k64_block_dro_sample_seed2027`, both with model seed 1337
  and row-sample seed 2027.

All other fields remain byte-bound to their Round-1 family. Round 2 uses a new
source manifest committed before any replication score is read.

## Ender21 family-locked confirmation

If Round 2 passes, the selected family is refit using discovery eras only, with
the 13-retained-era embargo preserved, and scores the exact 40 retained eras
`0865`-`1021` once. Those eras are locked within Ender21 but are not described as
globally unseen. The candidate passes this research gate only if BMC >=0.0020,
BMC Sharpe >0.25, drawdown <0.10, Corr >=0.008, benchmark correlation <0.25,
three of four chronological 10-era blocks have positive BMC, the worst block is
above -0.001, and confirmation BMC retains at least 60% of discovery BMC.

Passing ends at `HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED`. True
promotion requires an unchanged predictor to collect at least 52 newly resolved
future eras.

## Deferred full-consecutive confirmation gate

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
Until that authorization exists, Ender21 cannot claim a full-consecutive
historical pass. Numerai upload and account changes require a separate explicit
user decision regardless of any research result.

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

`ROUND1_SCOUT_WINNER_ROUND2_FROZEN`

Round 1 selected `r1_tabm_k64_block_dro`. Against the fresh matched control,
full BMC improved from 0.005390 to 0.006877, BMC Sharpe from 0.4512 to 0.6444,
and max drawdown fell from 0.090385 to 0.043614. Recent-fold BMC was 0.007541
versus 0.008177 for control, Corr was 0.011474, and every frozen eligibility
check passed. No Ender21 confirmation era has been opened.

An initial control process was stopped before producing predictions, results, or
metrics when review found that runtime allowlist filtering still materialized
later confirmation rows from a shared extract. The input contract was tightened
to discovery-only physical Parquets before any Ender21 score became visible.
Pre-score review also showed that eight DRO blocks exceeded the earliest inner
training split's seven eras. The block count was therefore reduced to four
before any scored output existed.

The final launcher reserves both canonical outputs with create-new semantics,
then verifies the committed source, runtime, config, allowlist, and physical
input hashes before evaluating config code or opening modeling data. The
Round-1 evaluator independently requires exact OOF row/fold coverage, expanding
CV geometry, sampling settings, prediction semantics, and stored config/data
bindings before it computes the predeclared metrics.
