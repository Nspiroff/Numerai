# Ender22 temporal-retention experiment

Status: **frozen scaffold; no training or scoring has been authorized by this
document alone**.

## Hypothesis

Ender21's fixed TabM-k64 chronological-block-DRO signal may retain more of its
historical full-span BMC in recent eras if training emphasizes recent eras. Two
mechanisms are tested once: exponential loss weighting with a 52-era half-life,
and an expanding CV whose training side is capped at the most recent 78 eras.
Architecture, target, residual transform, data, features, outer folds, embargo,
row cap, and base seeds remain fixed.

This is historical discovery research, not a tournament acceptance result. The
consumed Ender21 confirmation cohort, eras 0865-1021, is forbidden. The only
confirmation Ender22 may ever use is a prospective stream of resolved eras
1231-1282 collected after this protocol is frozen.

## Exact data and geometry

- Physical inputs: the existing Ender21 discovery-only Parquet pair ending at
  0861, identified byte-for-byte in `protocol/discovery_data_authority.json`.
- Era authority and feature order: the exact committed Ender21 discovery
  allowlist (176 every-fourth eras 0161-0861) and exact 3,555-feature list. The
  files are reused by path; they are not copied or widened.
- Target: `target_ender_20`; benchmark: `v53_lgbm_ender20`; per-era linear
  residual target with intercept and proportion 1.0.
- Outer CV: five expanding folds, 13-era embargo, 500,000-row training cap.
  Fold zero has no training set and is unused. Exact OOF scoring is therefore
  141 eras 0301-0861. The recent window is exactly 40 eras 0705-0861.
- Recent blocks: 0705-0741, 0745-0781, 0785-0821, and 0825-0861 (ten
  every-fourth eras each).
- Inner validation: latest 10% of each training fold, with a separate 13-era
  internal embargo. Model seed and row-sample seed are 1337 in Round 1.

## Round 1: one predeclared comparison

| Label | Config | Only procedure change |
|---|---|---|
| A | `r1_control_block_dro` | none; fixed Ender21 block-DRO control |
| B | `r1_recent_half_life52` | `recency_half_life_eras=52.0` retained-era observations |
| C | `r1_recent_window78` | `cv.max_train_eras=78` |

The evaluator first proves identical OOF IDs, eras, target values, fold
assignments, prediction semantics, physical data authority, configs, stored
results, and CV geometry. Each challenger must pass every gate against A:

1. recent-40 BMC mean is at least A plus 0.00030;
2. full OOF BMC is at least 90% of A;
3. recent-40 BMC is at least 80% of the challenger's full OOF BMC;
4. full OOF and every used fold have strictly positive BMC;
5. at least three of four recent blocks have positive BMC and the worst block
   is greater than -0.001;
6. full OOF BMC Sharpe is at least A minus 0.05 and max drawdown is no greater
   than A;
7. Corr mean is at least 0.005 and strictly below 0.04; average correlation
   with the benchmark is strictly below 0.25.

Eligible challengers are ranked, in order, by higher recent-40 BMC, higher
worst-block BMC, higher full BMC, lower max drawdown, then lexical name. If
neither is eligible the terminal state is
`NEGATIVE_NO_TEMPORAL_RETENTION_GAIN` and no Round 2 run is authorized.

For B, age is the ordinal distance among retained training eras after the
inner split and embargo, so a 52-era half-life means 52 retained observations
(about 208 consecutive Numerai-era labels in this every-fourth-era extract).
Each chronological block uses its recency-weighted row MSE. The detached
Block-DRO softmax is then multiplied by that block's share of total recency
weight and renormalized, so recent blocks receive greater total gradient mass
instead of recency cancelling inside each block. Validation loss remains the
ordinary unweighted ensemble-mean MSE.

All BMC Sharpe values use population standard deviation (`ddof=0`). Max
drawdown uses the repository's frozen `score_summary` convention: cumulative
BMC's running peak begins at the first scored era rather than an injected zero
baseline. The same convention is applied to control and challengers.

## Round 2: selected-family seed replication

Only the mechanically derived pair for the single Round-1-selected family is
authorized: model seed 2027 with sample seed 1337, and model seed 1337 with
sample seed 2027. Files for both possible families are predeclared solely so
the Round-1 receipt can select one exact pair; the unselected pair is forbidden.

The selected base realization plus those two replications form three
realizations. Each is checked against A's base-seed metrics and must have:

- full BMC at least 90% of A's base full BMC;
- recent-40 BMC at least A's base recent-40 BMC;
- positive BMC in every used fold;
- at least three positive recent blocks and worst block greater than -0.001;
- Sharpe at least A minus 0.05 and drawdown no greater than A;
- Corr in [0.005, 0.04) and benchmark correlation below 0.25.

At least two of three individual realizations must pass. The evaluator then
forms exactly one ensemble: within each era and realization it computes
average-tie percentile ranks, then takes the equal arithmetic mean of the three
ranks. It does not rerank that mean. The ensemble must pass all Round-1
challenger gates against A, including the +0.00030 and 80% retention gates.

If both requirements pass, the state is
`HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED`. Otherwise it is
`NEGATIVE_SEED_INSTABILITY`. Neither state authorizes upload, staking, model
creation, or any account/API mutation.

## Forward validation

The historical pass, if reached, only authorizes construction of a separately
reviewed shadow forward-validation procedure for resolved eras 1231-1282.
Those 52 consecutive prospective eras must be accumulated without changing the
family, weights, rank-ensemble rule, or gates. No historical local range may be
substituted, and no interim prospective score may influence the procedure.
