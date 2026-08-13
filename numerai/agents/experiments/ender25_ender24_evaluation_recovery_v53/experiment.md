# Ender25: Ender24 Round-1 evaluation recovery

## Current status

`SOURCE_ONLY_NOT_SEALED_NOT_AUTHORIZED`

This family exists to recover the scientific decision from the four immutable
Ender24 Round-1 runs after the Ender24 evaluator stopped before scoring on a
line-ending-sensitive authority check. Ender25 does not retrain, replace, or
extend that cohort. It defines a new evaluator namespace, a new `CREATE_NEW`
decision path, and a byte-and-semantics authority contract that treats uniform
LF and uniform CRLF serializations as equivalent only after each physical input
has independently matched its future sealed manifest entry.

Nothing in this source commit authorizes creation of
`source_manifest_evaluation_recovery.json`, access to the preserved prediction
Parquets or truth data, evaluation, scoring, training, deployment, Numerai
account activity, or any Round 2 work.

## Identity

- Family: `ender25_ender24_evaluation_recovery_v53`
- Stage: `ender25-ender24-round1-evaluation-recovery`
- Frozen predecessor: `ender24_ema_seed_stability_v53`
- Ender24 launch head: `7adc6724bd41689e34e8d21effa088b0ff606022`
- Ender24 terminal evidence commit: `d12f75552d76ebceb7c73fa3ff0ef9c608105599`
- Ender24 terminal state:
  `STOPPED_AT_ENDER24_ROUND1_EVALUATOR_PRECONDITION_FAILURE`
- New decision path:
  `numerai/agents/experiments/ender25_ender24_evaluation_recovery_v53/receipts/ender24_round1_recovery_decision.json`
- Superseded Ender24 decision path, required to remain absent:
  `numerai/agents/experiments/ender24_ema_seed_stability_v53/receipts/round1_ema_stability.json`

The complete machine-readable input authority is frozen in
`protocol/ender24_input_authority.json`.

## Scientific question

Using exactly the completed Ender24 Round-1 cohort—control and EMA 0.995 at
seeds 1337 and 2027—does the EMA candidate satisfy the already-frozen Ender24
matched-pair stability decision law?

Ender25 changes no model, seed, feature, era, target, benchmark, metric,
aggregation, tolerance, or scientific threshold. Its only permitted semantic
change is the evaluator-authority repair described below. The recovery evaluator
must use the frozen Ender24 scoring implementation and decision law under new
Ender25 bootstrap and output custody.

## Immutable cohort

The cohort has exactly four members:

1. control, seed 1337
2. EMA 0.995, seed 1337
3. control, seed 2027
4. EMA 0.995, seed 2027

For each member, the completion receipt, result JSON, and prediction Parquet
path, size, and SHA-256 are fixed in `protocol/ender24_input_authority.json`.
Every completion receipt must be leased and preflighted before any prediction,
result, or truth payload is parsed. A missing, extra, duplicate, mismatched, or
non-success member is a fail-closed evaluator precondition failure, not a
scientific negative.

The Ender24 source manifest and execution postmortem are also immutable inputs.
The old manifest fixes the governed scoring sources and external truth inputs;
the postmortem fixes the preserved terminal history. Ender25 must not rewrite
either document or reinterpret the Ender24 stop as a completed decision.

## Authority repair

Ender24 compared Windows CRLF physical bytes directly with canonical LF hashes.
Ender25 separates two identities:

1. **Physical custody.** The exact raw bytes opened by the recovery evaluator
   must match the size and SHA-256 recorded by the later sealed Ender25 source
   manifest. That lease remains open through decision-file `fsync`.
2. **Canonical text authority.** Only after physical custody succeeds, the raw
   bytes are decoded as strict UTF-8, rejected for BOM, NUL, bare CR, or mixed
   newline styles, and normalized from uniform CRLF to LF when necessary. The
   normalized bytes must match the frozen canonical size and SHA-256.
3. **Semantic authority.** The normalized JSON must have the exact list shape,
   ordering, uniqueness, values, and counts frozen in the authority JSON.

This is not newline-insensitive hashing. Raw-byte identity and canonical
identity are independent required checks. Same-length mutations, reordered
values, duplicate values, BOM-prefixed text, mixed newlines, and an incorrect
raw manifest receipt must all fail closed.

## One-shot recovery execution

If a later reviewed and explicitly authorized launch occurs, the recovery
evaluator may be invoked exactly once. It must:

1. reserve the new decision path with `CREATE_NEW` before opening governed
   sources or evidence;
2. establish and retain sealed source custody;
3. establish Ender24 manifest, postmortem, completion, result, prediction, and
   external-input custody;
4. preflight all four completion envelopes before parsing predictions, results,
   or truth;
5. apply the physical, canonical, and semantic authority checks;
6. load the frozen scoring dependencies and execute the unchanged Ender24
   matched-pair decision law; and
7. publish exactly one Ender25 decision receipt through its original
   create-new handle and retain every required lease through file `fsync`.

There is no automatic retry and no permission to rerun the four training jobs.
An evaluator infrastructure failure stops this gate and preserves its first
truthful failure. Any further attempt requires a new, separately reviewed
recovery identity and explicit authorization.

## Decision meaning

Only these scientific terminal states may be emitted after a complete scoring
pass:

- `ENDER25_ROUND2_SOURCE_GATE_AUTHORIZED`
- `ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN`

The positive state authorizes only preparation and review of a separate Round 2
source gate. It does not authorize a Round 2 manifest, Round 2 launch, training,
submission, deployment, or Numerai account action. A separate explicit launch
authorization remains mandatory.

## Seal and launch sequence

Ender25 uses a two-commit source seal:

1. **Source commit A:** implementation, tests, these documents, the authority
   JSON, and the empty `receipts/.gitkeep` directory anchor file are reviewed and
   frozen.
2. **Manifest-only child B:** a later authorized change creates
   `source_manifest_evaluation_recovery.json`, binding commit A and the exact
   reviewed source set. B may contain only that manifest.
3. **Explicit launch authorization:** only after B is independently reviewed
   may the one-shot evaluator be launched.

The exact future manifest source set is these 19 reviewed paths:

- `experiment.md`
- `gate.md`
- `protocol/ender24_input_authority.json`
- `receipts/.gitkeep`
- `evaluate_recovery.py`
- `evaluation_bootstrap.py`
- `evaluation_common.py`
- `evaluate_recovery_impl.py`
- `numerai/agents/tests/test_ender25_ema_evaluation_recovery.py`
- `numerai/agents/experiments/ender24_ema_seed_stability_v53/evaluation_common.py`
- `numerai/agents/experiments/ender24_ema_seed_stability_v53/configs/base_r1.py`
- the four exact Ender24 Round-1 config wrappers
- `numerai/agents/code/metrics/numerai_metrics.py`
- `numerai/agents/code/modeling/utils/constants.py`
- `numerai/agents/experiments/ender21_residual_stability_v53/protocol/discovery_eras_through_0861.json`
- `numerai/agents/experiments/ender21_residual_stability_v53/protocol/feature_columns_all_v53.json`

The later manifest must match this exact set. Adding or removing a runtime
source requires review before the source seal; it may not be silently repaired
in manifest-only commit B.

The frozen package runtime must also pin the decision-active scoring closure,
including `numerai-tools==0.6.0`, `scipy==1.18.0`, and
`scikit-learn==1.9.0`, in addition to the direct NumPy, pandas, PyArrow,
NumerAPI, and Python versions.

## Capability boundary

All current capabilities are false: manifest creation, artifact reads, data
reads, evaluation, scoring, output publication, training, Round 2 execution,
submission, deployment, and Numerai account actions. Synthetic unit tests may
exercise temporary fixtures only; they must not discover, open, hash, parse, or
score any preserved Ender24 artifact or Numerai dataset.
