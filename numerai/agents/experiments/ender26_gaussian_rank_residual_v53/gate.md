# Ender26 source and execution gate

## Current authority: source gate only

- This directory freezes the Ender26 hypothesis, exact Round-1 configs,
  discovery-data identity, metric law, runner/bootstrap/evaluator source, and
  synthetic source-gate tests.
- It does not authorize a manifest, governed artifact access, output
  reservation, training, prediction read, target read, scoring, evaluation,
  confirmation, Round-2 source, deployment, upload, model creation, staking,
  submission, or account/API mutation.
- No Ender21-25 prediction, result, completion, or decision is an Ender26
  input. Prior records are immutable documentary evidence only.
- Every future Ender26 output must use a new create-only identity. An existing,
  partial, missing, changed, or aliased artifact is a stop, never retry
  authority.

## Frozen scientific delta

Round 1 is exactly two matched model-seed pairs. All four configs retain the
Ender24 non-EMA window-78 control procedure. Within a matched pair, the only
scientific challenger delta is:

```text
model.target_transform.benchmark_transform = tie_kept_rank_gaussian
```

The control omits `benchmark_transform`, which means identity/raw benchmark
behavior. Output names differ only for create-new identity. Sample seed is
fixed at 1337 because the 500,000-row cap is inactive in every frozen
window-78 fold.

For validated finite, nonconstant per-era groups, the Gaussian-rank transform
must operate independently per era using average tie-kept percentile rank
`(rank - 0.5) / n`, followed by the standard-normal inverse CDF. Malformed,
non-finite, missing, or constant groups must stop before fitting. The challenger
centers the target and subtracts its projection onto the uncentered Gaussian
benchmark exactly as
`y_c - m * (m.T @ y_c) / (m.T @ m)`, with proportion `1.0`. Predictions remain
direct raw model outputs; no prediction-side postprocessing is permitted.

## Before a manifest may be proposed

1. Implement and independently review the exact target transform without
   changing control behavior, Block-DRO, validation MSE, optimizer, model, CV,
   prediction semantics, or data loading.
2. Pass deterministic synthetic tests for exact agreement with frozen
   `numerai-tools==0.6.0`, ties, rejection of constant or non-finite groups,
   finite endpoints, row-order restoration, per-era isolation, and fail-closed
   invalid inputs.

Constant and singleton Gaussian benchmark groups are rejected because the
frozen scorer's orthogonalization denominator is zero and its contribution is
undefined. This fit-path rule is stricter than the standalone rank helper,
which maps an all-tied group to zero before the norm guard rejects it.
3. Prove the two model seeds are mechanically active and each matched pair
   begins with identical model initialization and batch order.
4. Independently review the new Ender26-only runner, bootstrap, evaluator,
   completion, and custody source. Generic modeling CLI invocation must not be
   an authorized execution path.
5. Prove all twelve future prediction/result/completion destinations and the
   one decision destination are absent before any governed read.
6. Create a separate manifest-only seal that binds the exact reviewed source,
   runtime, four configs, authority lists, and two discovery inputs.
7. Obtain explicit user authorization after the seal. Neither this scaffold
   nor a future manifest grants launch authority.

## Future Round-1 execution law

If separately authorized, run exactly once and under one manifest, in this
fixed order:

1. `r1_control_rawresid_seed1337`
2. `r1_grank_resid_seed1337`
3. `r1_control_rawresid_seed2027`
4. `r1_grank_resid_seed2027`

All four create-new completion envelopes must validate before a future
evaluator may parse any result or prediction. A failed, partial, missing, or
mismatched run invalidates the indivisible cohort without producing a
scientific decision and without permitting a retry.

The first training invocation must prove all twelve run destinations and the
decision destination absent before reservation or governed reads. Every later
invocation must prove all predecessor triples unique, plain, and nonempty;
prove its current and all future triples plus the decision absent; and, after
source custody but before pipeline/data entry, hold and opaque-validate every
predecessor completion and output identity against the same manifest. An
evaluation invocation reserves its decision CREATE_NEW before governed reads;
any uncommitted exit preserves the reservation, including a zero-byte one, as
terminal no-retry evidence.

The Gaussian-rank procedure advances only if all frozen checks pass:

- mean full BMC is at least control mean plus `0.00020`;
- mean recent-40 BMC is at least control mean plus `0.00030`;
- at both matched seeds, recent-40 BMC is at least matched control;
- at both matched seeds, full BMC is at least 95% of matched control;
- every used fold has positive BMC;
- each challenger has at least three positive recent blocks and worst block
  strictly above `-0.001`;
- each challenger Sharpe is at least matched control minus `0.05`;
- each challenger drawdown is no greater than matched control plus `0.01`;
- each challenger Corr is in `[0.008, 0.04)`; and
- each challenger benchmark correlation is below `0.25`.

The future evaluator's only scientific terminal states are
`ENDER26_ROUND2_SOURCE_GATE_ELIGIBLE` and
`ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN`. The first means only that new Round-2
source may be proposed for review. No Round-2 source exists or is authorized by
this scaffold.

## Data and stopping boundaries

- Selection data ends at era `0861`; OOF scoring is exactly 768,362 rows over
  141 eras `0301`-`0861`, and recent-40 is `0705`-`0861`.
- Eras `0865`-`1021` are consumed and forbidden. Eras `1022`-`1230` are not an
  alternate confirmation set.
- Only a separately frozen prospective stream of resolved eras `1231`-`1282`
  could confirm a historical research pass.
- Stop on source/runtime/config/manifest/input drift; link, reparse, alias, or
  overwrite risk; wrong cohort or fold geometry; non-finite values; unexpected
  transform semantics; mismatched seed pairs; premature metric access; a retry;
  or either terminal negative state.
