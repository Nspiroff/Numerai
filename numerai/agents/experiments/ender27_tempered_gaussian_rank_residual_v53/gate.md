# Ender27 source and execution gate

## Current authority: source gate only

- This directory freezes the Ender27 hypothesis, exact Round-1 configs,
  documentary discovery-data identity, metric law, and source-only
  runner/bootstrap/evaluator/custody candidates.
- Those source candidates do not authorize a manifest, governed artifact
  access, output reservation, training, prediction read, target read, scoring,
  evaluation, confirmation, Round-2 source, deployment, upload, model
  creation, staking, submission, or account/API mutation.
- No Ender21-26 prediction, result, completion, or decision is an Ender27
  experiment input. Prior records are immutable documentary evidence only.
- Every possible future Ender27 output must use a new create-only identity. An
  existing, partial, missing, changed, or aliased artifact would be a stop,
  never retry authority.

## Frozen scientific delta

Round 1 is exactly two matched model-seed pairs. All four configs retain the
Ender26 non-EMA window-78 control procedure. Within a matched pair, the only
scientific challenger additions are:

```text
model.target_transform.benchmark_transform = tie_kept_rank_gaussian
model.target_transform.benchmark_transform_strength = 0.5
```

The control omits both keys, preserving the legacy raw-benchmark residual.
The existing residual `proportion` remains `1.0`. Output names differ only for
new Ender27 identities. Sample seed is fixed at 1337 because the 500,000-row
cap is inactive in every frozen window-78 fold.

For each era, let `r_identity` be the established intercept-fitted target
residual to the raw benchmark. Let `r_grank` be the Ender26 residual of the
centered target projected off the uncentered, tie-kept Gaussian-ranked
benchmark. The challenger target is exactly:

```text
r_0.5 = 0.5 * r_identity + 0.5 * r_grank
```

The implementation must blend the two completed target residuals. It must not
blend benchmark representations. It must not reinterpret strength as the
existing original-target-to-residual `proportion`. No alternate strength,
sweep, adaptive choice, or post-hoc selection is permitted.

The Gaussian branch must operate independently per era using average
tie-kept percentile rank `(rank - 0.5) / n`, followed by the standard-normal
inverse CDF. It must project the centered target against the uncentered
Gaussian benchmark, restore original row order, and stop before fitting for
malformed, missing, non-finite, constant, or otherwise undefined groups. The
identity branch must remain byte-semantically equivalent to legacy control
behavior. Predictions remain direct raw model outputs; no prediction-side
postprocessing is permitted.

## Before a manifest may be proposed

1. Implement and independently review
   `benchmark_transform_strength=0.5` as the exact target-residual blend above,
   while preserving omitted-strength Ender26 full-transform behavior and
   omitted-transform control behavior.
2. Pass deterministic synthetic tests for endpoint equivalence at strengths
   `0.0` and `1.0`, the exact midpoint, validation of finite numeric strengths
   in `[0, 1]`, ties, row-order restoration, per-era isolation, and fail-closed
   malformed or undefined Gaussian inputs. No Numerai data may be used by
   those tests.
3. Prove the two model seeds are mechanically active and each future matched
   pair would begin with identical model initialization and batch order.
4. Independently review the new Ender27-only runner, bootstrap, evaluator,
   completion, and custody source. Generic modeling CLI invocation must not
   become an authorized execution path.
5. Prove every future prediction/result/completion destination and the future
   decision destination absent before any governed read or reservation.
6. Create a separate manifest-only seal binding the exact reviewed source,
   runtime, four configs, authority lists, and two discovery inputs.
7. Obtain explicit user launch authorization after that seal. Neither this
   scaffold nor a future manifest grants launch authority by itself.

## Future Round-1 execution and decision law

If this source candidate is independently reviewed, a separate manifest is
later sealed, and execution is explicitly authorized, the indivisible cohort
order must be:

1. `r1_control_rawresid_seed1337`
2. `r1_tempered_grank_resid_seed1337`
3. `r1_control_rawresid_seed2027`
4. `r1_tempered_grank_resid_seed2027`

All four create-new completion envelopes would have to validate under one
future Ender27 manifest before an evaluator could parse any result or
prediction. A failed, partial, missing, or mismatched run would invalidate the
cohort without producing a scientific decision and without permitting a
retry. Any future evaluator must be one-shot and reserve its decision path
CREATE_NEW before governed reads; an uncommitted exit must preserve that
reservation as terminal no-retry evidence.

The tempered procedure advances only if all frozen Ender26 checks pass:

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
`ENDER27_ROUND2_SOURCE_GATE_ELIGIBLE` and
`ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN`.

The eligible state means only that independently reviewed Round-2 source may
be proposed. It does not authorize creating that source, any Round-2 action,
data access, training, scoring, prospective validation, deployment, upload,
model creation, staking, submission, or account/API mutation.

## Data and stopping boundaries

- Selection geometry ends at era `0861`; historical OOF scoring would remain
  exactly 768,362 rows over 141 eras `0301`-`0861`, with recent-40
  `0705`-`0861`.
- Eras `0301`-`0861` are consumed discovery evidence. Any future historical
  pass is a source gate, not independent confirmation.
- Eras `0865`-`1021` are consumed and forbidden. Eras `1022`-`1230` are not an
  alternate confirmation set.
- Only a separately frozen prospective stream of newly resolved eras
  `1231`-`1282` could confirm a historical research pass.
- Stop on source/runtime/config/manifest/input drift; link, reparse, alias, or
  overwrite risk; wrong cohort or fold geometry; non-finite values; unexpected
  transform semantics; mismatched seed pairs; premature metric access; a
  retry; or `ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN`.

All execution, deployment, and account authorities are false now.
