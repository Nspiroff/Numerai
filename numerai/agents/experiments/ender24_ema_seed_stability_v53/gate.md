# Ender24 execution gate

## Current authority: source only

- This scaffold authorizes only review of the EMA implementation, deterministic
  synthetic tests, these six exact configs, the Round-1 execution/evaluation
  source, and these specifications.
- The presence of launcher, bootstrap, evaluator, and discovery-authority
  source grants no execution authority. It does not authorize a source
  manifest, mechanical-activity receipt, output reservation,
  data/target/prediction read, training, scoring, evaluation, confirmation,
  deployment, upload, model creation, staking, or account/API mutation.
- Ender24 paths must be create-new. Earlier predictions, results, completions,
  scores, and decisions may not be copied, overwritten, or treated as Ender24
  outputs.
- Model seeds are the only replication axis. Sample seed remains fixed at 1337
  because the configured row cap is inactive for window-78 folds.

## Before a source manifest may be proposed

1. Independently review EMA initialization, update order, validation swapping,
   checkpoint restoration, inference behavior, and non-EMA compatibility.
2. Run deterministic synthetic tests proving every mechanical-activity
   precondition in `experiment.md`, without accessing Numerai data.
3. Review the exact config deltas: within each seed pair, the EMA config may
   differ from control only by `output.results_name` and
   `model.params.ema_decay=0.995`.
4. Review schemas, source custody, one-shot output naming, the all-four-
   completions-before-results rule, and fail-closed behavior in the Round-1
   bootstrap/evaluator source.
5. Stop the family before data access if any proof or review fails.

## Before Round 1 could run

Round 1 requires a separate, reviewed manifest-only seal commit binding exactly
31 source/evidence paths: the 12 shared modeling/metric sources; two reused
Ender21 authority lists; Ender24 experiment, gate, discovery authority, and a
new `protocol/mechanical_activity_receipt.json`; base plus six configs; the
Round-1 bootstrap, launcher, evaluator common/wrapper/implementation; and the
Ender24 EMA synthetic-test source. The external set is exactly the two
discovery-only physical inputs through 0861. The seal may not add another path.
The mechanical-activity receipt does not exist during this source-only step.
It also requires proof that every new Round-1 prediction, result, completion,
and the exact decision path `receipts/round1_ema_stability.json` is absent.
Neither this scaffold nor a future manifest grants launch authority: explicit
user authorization is required afterward.

If authorized, exactly the four Round-1 configs named in `experiment.md` must
run once under one manifest. No result may be read until all four create-new
completion receipts validate. Any failed, partial, missing, overwritten, or
mismatched run terminates the family as an invalid matched round.
Each component receipt must have stage
`ender24-round1-training-completion`; no other completion stage is accepted.

The only Round-1 decision stage is
`ender24-round1-ema-seed-stability`; its only terminal states are
`ROUND2_AUTHORIZED` and `NEGATIVE_NO_EMA_STABILITY_GAIN`. A pass advances the
fixed EMA procedure and does not select or authorize reuse of one seed.

## Before Round 2 could run

Round 1 must produce exact state `ROUND2_AUTHORIZED`. No Round-2 launcher,
evaluator, bootstrap branch, or manifest is present in this scaffold. Those
sources require separate review; a later Round-2 manifest must bind the
immutable Round-1 receipt and exactly the two seed-7331 configs. Explicit user
authorization is again required before launch. The Round-1 runs are not
repeated.

## Immediate stop conditions

Stop on any changed or missing governed artifact; dirty or uncommitted
manifest source; source/runtime/config/provenance mismatch; overwrite attempt;
non-finite output; duplicate or mismatched IDs; unexpected fold/cohort geometry;
an inactive replication axis; validation that fails to restore live weights;
EMA with zero updates or identical final live/shadow hashes; any selection read
from eras 0865 or later; a retry under the same family; an extra seed or EMA
variant; or either terminal negative state.

The metric rules in `experiment.md` are normative. Even a historical pass
requires separate prospective validation over resolved eras 1231-1282 and has
no production or Numerai-account authority.
