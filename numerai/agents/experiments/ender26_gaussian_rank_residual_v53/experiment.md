# Ender26 Gaussian-rank benchmark-residual experiment

Status: **frozen source-only Round-1 scaffold; no manifest, data access,
training, scoring, or evaluation is authorized by this document**.

## Research question

Can target-side alignment with Numerai's benchmark-contribution geometry
increase stable benchmark-unique BMC without changing the proven Ender23/24
window-78 Block-DRO TabM procedure?

Ender26 changes only the representation of the benchmark used to construct the
training residual. The control retains the existing per-era linear residual to
the raw `v53_lgbm_ender20` values. The challenger first transforms that
benchmark independently within each era using tie-kept percentile ranks and a
Gaussian inverse-CDF. It then centers the target and subtracts the target's
projection onto that uncentered Gaussian benchmark, with proportion `1.0`.

This is a new experiment family, not an Ender24 or Ender25 retry. It does not
reuse their predictions, results, completions, or decisions, and it does not
change EMA decay, add seeds, relax a threshold, or reopen Round 2.

## Scientific rationale

For one era, let `y_c` be the centered target and let
`m = gaussian(tie_kept_rank(benchmark))`. Numerai BMC Gaussian-ranks both the
prediction and benchmark, orthogonalizes the transformed prediction to `m`,
and takes its covariance with `y_c`. By symmetry of the one-column orthogonal
projection, that covariance has the exact target-side direction
`y_c - m * (m.T @ y_c) / (m.T @ m)`. The Gaussian benchmark is deliberately
not recentered before that projection because the scorer uses it uncentered.

The challenger therefore aligns the residual label with BMC's benchmark space
while leaving the established training loss and model unchanged. This is
target-side alignment only: persisted predictions remain the direct raw
`model.predict` output, and the evaluator alone applies the canonical BMC
transform when scoring.

The normative per-era benchmark transform is:

1. retain ties with average ranks;
2. map each finite value to `(rank - 0.5) / n` within its era; and
3. apply the standard-normal inverse CDF.

For validated finite, nonconstant per-era groups, the implementation must
exactly match the frozen `numerai-tools==0.6.0`
`gaussian(tie_kept_rank(...))` behavior, restore original row order, isolate
eras, and fail closed before fitting on malformed, non-finite, or constant
groups. No first-tie breaking, cross-era ranking, Gaussian-benchmark
recentering, clipping, power transform, or prediction transform is permitted.

## Fixed procedure and data

- Data version: v5.3.
- Inputs: the exact Ender21 discovery-only full and benchmark Parquets through
  era `0861`, identified in `protocol/discovery_data_authority.json`.
- Discovery allowlist: 176 every-fourth eras `0161`-`0861`.
- Features: the exact ordered 3,555-feature Ender21 authority list.
- Target: `target_ender_20`.
- Benchmark: `v53_lgbm_ender20`.
- Architecture: K64 TabM, three 512-wide ReLU blocks, dropout `0.1`.
- Loss: existing four-group chronological Block-DRO over row MSE.
- Optimizer: AdamW, learning rate `0.002`, weight decay `0.0003`.
- Training: at most 30 epochs, patience 4, batch size 1,024, AMP enabled.
- Outer CV: five expanding splits with a 13-retained-era embargo; the first
  empty-training fold is skipped.
- Training window: most recent 78 retained eras per usable outer fold.
- Inner validation: latest 10% of the training eras with its own 13-era
  embargo; validation and early stopping retain ordinary MSE.
- Configured row cap: 500,000. It is mechanically inactive under the frozen
  window-78 folds, so sample seed 1337 is fixed and is not replication
  evidence.
- EMA: disabled and absent from all configs.

Exact OOF evaluation contains 768,362 matched rows across 141 eras
`0301`-`0861`. The recent window is the final 40 retained eras `0705`-`0861`.
Its four fixed ten-era blocks are `0705`-`0741`, `0745`-`0781`,
`0785`-`0821`, and `0825`-`0861`.

Eras `0865`-`1021` are consumed Ender21 confirmation evidence and are forbidden
for Ender26 selection, tuning, or reporting. Eras `1022`-`1230` are not a
substitute confirmation cohort. A historical pass could only support a
separately frozen prospective protocol over newly resolved eras `1231`-`1282`.

## Round 1: exact matched two-seed cohort

| Run | Procedure | Model seed | Sample seed |
| --- | --- | ---: | ---: |
| `r1_control_rawresid_seed1337` | raw-benchmark residual control | 1337 | 1337 |
| `r1_grank_resid_seed1337` | Gaussian-rank-benchmark residual | 1337 | 1337 |
| `r1_control_rawresid_seed2027` | raw-benchmark residual control | 2027 | 1337 |
| `r1_grank_resid_seed2027` | Gaussian-rank-benchmark residual | 2027 | 1337 |

Within each seed pair, the challenger's only scientific config delta is
`model.target_transform.benchmark_transform="tie_kept_rank_gaussian"`.
The control omits `benchmark_transform`, preserving the established identity
behavior. `output.results_name` changes only to give every run a new canonical
identity.

The four runs form one indivisible future cohort. No prediction, result, or
metric may be read until all four create-new completions validate under one
future Ender26 manifest. Model seeds 1337 and 2027 are fixed matched
replications, not candidates from which a favorable seed may be selected.

The bootstrap enforces the table's fixed order mechanically. Before the first
run reserves anything, all twelve prediction/result/completion destinations
and the decision destination must be absent. Each later run requires every
predecessor triple to exist as unique, nonempty plain files, requires its own
and every later triple plus the decision to remain absent, then leases and
opaque-validates all predecessor completions and output identities against the
current manifest before entering the modeling pipeline. Those predecessor
leases remain held until the current completion is durable.

Evaluation is also one-shot: its decision path is reserved CREATE_NEW before
the manifest or any target-bearing artifact is opened. Any uncommitted exit
preserves that reservation, even if it remains zero bytes, as terminal failure
evidence. No failed evaluator invocation may be retried.

## Frozen Round-1 decision law

The Gaussian-rank procedure is eligible only if every condition below holds:

1. its two-seed mean full BMC is at least the control mean plus `0.00020`;
2. its two-seed mean recent-40 BMC is at least the control mean plus `0.00030`;
3. at each matched seed, challenger recent-40 BMC is at least control;
4. at each matched seed, challenger full BMC retains at least 95% of control;
5. every used outer fold has strictly positive challenger BMC;
6. each challenger has at least three of four positive recent blocks and its
   worst recent block is strictly greater than `-0.001`;
7. each challenger BMC Sharpe is at least its matched control minus `0.05`;
8. each challenger max drawdown is no greater than matched control plus
   `0.01`;
9. each challenger Corr is in `[0.008, 0.04)`; and
10. each challenger's average correlation with the benchmark is below `0.25`.

A completed scientific evaluation has exactly one of two states:

- `ENDER26_ROUND2_SOURCE_GATE_ELIGIBLE` when every check passes; or
- `ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN` otherwise.

The eligible state permits only a proposal for independently reviewed Round-2
source. It does not authorize a Round-2 scaffold, manifest, data read, run,
score, prospective validation, deployment, upload, model creation, staking,
submission, or account action.

## Current negative authority

This directory intentionally contains specifications, four configs, the
discovery authority, and a source-only runner/bootstrap/evaluator candidate.
It contains no manifest, prediction, result, completion, decision, output
directory, receipt directory, or Round-2 source. The source candidate must pass
its synthetic source-gate suite, receive independent review, and be committed
before a separate manifest-only seal may be proposed. A later manifest-only
seal and explicit user launch authorization would still be required.
