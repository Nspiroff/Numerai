# Ender24 EMA model-seed stability experiment

Status: **frozen source-only scaffold; no manifest, data access, training, or
scoring is authorized by this document**.

## Hypothesis and boundary

A fixed exponential moving average (EMA) of the optimizer trajectory may reduce
TabM model-initialization variance while preserving the recent-window signal.
Ender24 tests this single weight-space stabilization procedure. It is not an
Ender23 rescue, another temporal-retention sweep, or authority to select a seed.

The baseline is Ender23's fixed K64 TabM chronological-Block-DRO procedure with
the 78-retained-era training window. The experiment retains the exact Ender23
discovery-only physical inputs through era 0861, 3,555-feature order,
`target_ender_20`, `v53_lgbm_ender20`, residual-to-benchmark target, optimizer,
inner validation, five expanding outer folds, and both 13-era embargoes.
The 500,000-row cap remains configured but is inactive for these window-78
folds. Sample seed is fixed at 1337 and is not replication evidence.

Exact OOF scoring remains eras 0301-0861 and recent-40 scoring remains eras
0705-0861. The recent blocks remain 0705-0741, 0745-0781, 0785-0821, and
0825-0861. All Ender24 predictions, results, completions, and decisions must use
new paths; artifacts from earlier experiment families are evidence only.

## Single procedure change

The control omits `ema_decay`, leaving existing non-EMA behavior unchanged.
The challenger sets exactly `ema_decay=0.995`:

1. Shadow weights initialize from the first completed optimizer step.
2. After every later optimizer step, each shadow weight is updated as
   `shadow = 0.995 * shadow + 0.005 * live`.
3. Epoch validation, early stopping, best-checkpoint selection, final
   restoration, and prediction use the EMA shadow.
4. Optimizer updates continue on live weights, and every validation swap must
   restore the live weights exactly.

No alternate decay, warmup, averaging window, checkpoint blend, or post-score
variant is permitted.

## Mechanical-activity precondition

Before any source manifest is created, deterministic synthetic tests must prove:

1. model seeds 1337, 2027, and 7331 produce different initial-parameter and
   shuffled-batch-order hashes;
2. matched control and EMA runs at one seed start from identical parameters and
   batch order;
3. every EMA fold records at least one update and its final EMA-state hash
   differs from its live-state hash;
4. sample seed is excluded from replication because the 500,000-row cap is
   inactive for window-78 folds; and
5. eager/disk behavior, prediction semantics, and existing non-EMA behavior
   remain unchanged.

Failure of any proof ends the family before data access.

## Round 1: two matched seed pairs

Exactly four one-shot runs are predeclared:

| Run | Procedure | Model seed | Sample seed |
|---|---|---:|---:|
| `r1_control_seed1337` | control window-78 | 1337 | 1337 |
| `r1_ema995_seed1337` | EMA window-78 | 1337 | 1337 |
| `r1_control_seed2027` | control window-78 | 2027 | 1337 |
| `r1_ema995_seed2027` | EMA window-78 | 2027 | 1337 |

No result may be read until all four completion receipts validate under one
exact Round-1 manifest. Within each matched pair, OOF IDs, eras, folds, targets,
benchmark coverage, and provenance must be identical.

EMA advances only if all of the following hold:

1. mean recent-40 BMC across EMA seeds is at least the control mean;
2. worst-seed recent-40 EMA BMC is at least the control worst seed;
3. mean full BMC is at least 95% of the control mean, and worst-seed full BMC
   is at least 95% of the control worst seed;
4. the absolute two-seed gap is at most 75% of the control gap for both full
   and recent-40 BMC;
5. each EMA run has positive full BMC and positive BMC in every used fold;
6. each EMA run has at least three of four positive recent blocks and a worst
   recent block greater than -0.001;
7. each EMA run has Corr in `[0.005, 0.04)` and benchmark correlation below
   `0.25`; and
8. each EMA run's full-BMC Sharpe is at least its matched control minus `0.05`,
   and its max drawdown is no more than its matched control plus `0.01`.

Failure is terminal `NEGATIVE_NO_EMA_STABILITY_GAIN`. Passing authorizes only
Round 2.

## Round 2: predeclared third seed

Exactly two new runs are permitted only after a Round-1 pass:
`r2_control_seed7331` and `r2_ema995_seed7331`, both with model seed 7331 and
sample seed 1337. Round-1 artifacts remain immutable evidence and are not rerun.

Across exact model seeds `{1337, 2027, 7331}`, EMA passes only if:

1. population standard deviation (`ddof=0`) of both full and recent-40 BMC is
   at most 75% of the matched control standard deviation;
2. EMA mean recent-40 BMC is at least the control mean;
3. EMA worst-seed recent-40 BMC is at least the control worst seed;
4. EMA mean and worst-seed full BMC are each at least 95% of control;
5. at least two of three EMA seeds meet every per-run fold, block, Corr,
   benchmark-correlation, Sharpe, and drawdown guard above; and
6. the seed-7331 pair meets the exact matched provenance and cohort checks.

Pass state: `HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED`.
Failure state: `NEGATIVE_SEED_INSTABILITY`. Round 2 is terminal either way. No
seed ensemble is constructed or scored because this experiment tests whether
the training procedure itself is stable.

## Stop and data boundaries

A failed or partial one-shot run invalidates its whole matched round and cannot
be retried under this family. Do not add another seed, tune EMA decay, alter the
78-era window, blend checkpoints, relax gates, or reuse an output path after
scores are known.

Eras 0865-1021 are consumed and forbidden for selection. Eras 1022-1230 are
not an alternate confirmation cohort. A historical pass could authorize only
a separately reviewed prospective 1231-1282 shadow protocol; it never directly
authorizes packaging, upload, assignment, submission, staking, or account
mutation.
