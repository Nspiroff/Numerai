# Ender24 execution gate

## Current authority: source only

- This scaffold authorizes only review of the EMA implementation, deterministic
  synthetic tests, these six exact configs, and these Markdown specifications.
- It does not authorize a source manifest, protocol JSON, evaluator, launcher,
  output reservation, data/target/prediction read, training, scoring,
  confirmation, deployment, upload, model creation, staking, or account/API
  mutation.
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
4. Review schemas, source custody, one-shot output naming, and fail-closed
   behavior for a future manifest/runner/evaluator proposal.
5. Stop the family before data access if any proof or review fails.

## Before Round 1 could run

Round 1 requires a separate, reviewed manifest-only seal commit binding the
exact committed source, runtime, synthetic-test evidence, six configs, reused
Ender21 allowlist and feature list, and exact discovery-only physical inputs.
It also requires proof that every new Round-1 prediction, result, completion,
and decision path is absent. Neither this scaffold nor a future manifest grants
launch authority: explicit user authorization is required afterward.

If authorized, exactly the four Round-1 configs named in `experiment.md` must
run once under one manifest. No result may be read until all four create-new
completion receipts validate. Any failed, partial, missing, overwritten, or
mismatched run terminates the family as an invalid matched round.

## Before Round 2 could run

Round 1 must produce the exact state authorizing Round 2. A separately reviewed
Round-2 manifest must bind the immutable Round-1 receipt and exactly the two
seed-7331 configs. Explicit user authorization is again required before launch.
The Round-1 runs are not repeated.

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
