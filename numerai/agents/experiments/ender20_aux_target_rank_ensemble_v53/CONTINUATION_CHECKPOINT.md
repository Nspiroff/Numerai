# Ender20 auxiliary-target ensemble continuation checkpoint

Date: 2026-08-03

Branch: `agent/optimized-ender20-model`

Frozen pre-scoring protocol commit:
`ef4ee304d6088f10d27e4d49a80d67ec925dbbf3`

This checkpoint was created because the Codex usage window was nearly
exhausted. It is a durable work-in-progress handoff, not a training-ready or
promotion-ready result.

## Safety state

- No new Jasper, Teager2b, Victor, or Tyler GPU Scout training was started.
- No Scout calibration or locked Ender20 metrics were computed.
- No consecutive confirmation data was scored.
- No final model was fit or packaged.
- Nothing was uploaded, assigned, submitted, staked, or changed in the
  Numerai account.
- All eight new Scout result/prediction destinations, all four deployment
  destinations, and the receipt directory were absent at the final pre-save
  check.
- The local `.gpu-lgbm-source-build/` directory is unrelated untracked build
  residue and is intentionally excluded from Git.

## Frozen experiment

The protocol is in `gate.md`, with source and artifact anchors in
`source_manifest.json`. Its fixed Scout is a five-component within-era rank
ensemble over direct-target LightGBM models for Jasper20, Teager2b20,
Victor20, Xerxes20, and Tyler20. The only candidates vary Tyler weight across
0.0, 0.1, 0.2, 0.3, and 0.4. The other four components share the remaining
weight equally.

Protocol hashes:

- `gate.md`: `c851e3e0637e26bff5b2c26eda5752a46a9d72fce2621678bd39ffa320983ffe`
- `source_manifest.json`:
  `3cc96dce9938306cc1f2e7d4ef6b6628f24494f5c30a1ca87d791b64ace662a8`

The preceding Xerxes20 challenger closed with
`STOP_NO_SCOUT_CALIBRATION_WINNER`; its best depth-8 component is reused only
as the frozen Xerxes component in this new family.

## Implemented evaluator work

`agents/code/analysis/evaluate_ender20_aux_target_rank_ensemble.py` currently
contains:

- frozen Git/source/GPU/config validation;
- exact Scout and confirmation cohort derivation;
- exact ID, era, stored-target, fold, prediction-semantics, GPU-diagnostics,
  and artifact-hash checks;
- legacy downsampled and full-consecutive two-seed TabM contracts;
- within-era average-tie component ranking, weighted blending, and reranking;
- Corr, BMC, population-standard-deviation Sharpe, additive drawdown, and
  symmetric per-era Spearman similarity calculations;
- anchored calibration selection and strict locked/confirmation thresholds;
- content-addressed receipt primitives with one-prefix refusal;
- non-scoring Scout component sealing;
- Scout calibration and selected-only Scout locked stage bodies;
- a strict confirmation-pretraining receipt validator.

Real frozen-artifact smoke validation passed before this checkpoint:

- Scout expected cohort: 1,279,658 rows, 214 eras;
- frozen two-seed TabM residual: exact artifacts validated, finite rank range
  `0.0001384083044982699` through `1.0`;
- reused Xerxes depth-8 component: exact result, GPU folds, semantics, target,
  IDs, eras, and folds validated for all 1,279,658 rows; finite prediction
  range `0.43853421346712984` through `0.5564638511519919`.

## Validation at save time

- Python syntax compilation: pass.
- Focused unit suite:
  `python -m unittest -q agents.tests.test_ender20_aux_target_rank_ensemble`
  -> 29/29 pass.
- `git diff --check`: pass.

The tests cover checkpoint/imported-dependency dirtiness, manifest and path
hashes, exact Scout/confirmation config shape, joins and producer folds,
fractional-fold rejection, GPU/result/semantics contracts, legacy/full TabM,
ranks/ties/metrics/similarities, thresholds and selection, receipt hashing and
same-prefix refusal, standalone confirmation calibration checks, and
selected-only blend construction.

## Mandatory blockers before any training

Do not use this progress commit as the pretraining checkpoint and do not start
the four GPU runs until every item below is resolved and independently tested:

1. Split the receipt prefix claim from final receipt writing. Claim the unique
   canonical stage/component prefix before any scoring or holdout access, so a
   rerun refuses before recomputing metrics. An incomplete early claim must
   remain fail-closed.
2. Constrain bound receipts to the canonical receipt directory, exact stage
   prefix, and finalized claim/hash relationship.
3. Before opening Scout locked eras, deterministically recompute and exactly
   validate the calibration candidates, checks, anchored tie set, and selected
   formula from the sealed inputs. A fabricated content-addressed calibration
   JSON must not be able to choose a candidate.
4. Add a durable per-component pre-run absence receipt/claim for each Scout
   result and prediction destination. Scout sealing must bind that proof.
5. Add the non-scoring confirmation-component seal stage for all five
   confirmation outputs. Confirmation calibration must require exactly five
   unique seal path/hash bindings.
6. Finish `run_confirmation_calibrate` and `run_confirmation_locked`. The CLI
   currently references these undefined functions and therefore confirmation
   is intentionally not runnable.
7. Make confirmation pretraining path/hash mandatory in the CLI and require
   its exact bindings for configs, loader implementation, chosen stores or
   sidecar, canonical Xerxes store, and ten absent output destinations.
8. In confirmation locked, score only the selected locked-200 formula first.
   Compute full-period metrics only after the locked-200 checks pass.
9. Add mocked stage-body tests proving calibration slices before all-candidate
   construction, locked stages never call the all-candidate builder, seal
   receipts exclude automatic model metrics, early claims block reruns before
   access, forged selections fail, and confirmation seals/pretraining bindings
   are mandatory.

## Safe resume order

1. Pull this branch and read `gate.md`, `source_manifest.json`, this checkpoint,
   the evaluator, and its focused tests.
2. Confirm the twelve Scout/deployment paths and receipt directory are still
   absent. Existing or partial paths are a stop, never a cleanup/rerun signal.
3. Complete the blockers above using `apply_patch`; run syntax, focused unit,
   diff, and an independent static audit.
4. Commit the completed evaluator/tests as a new pretraining checkpoint and
   use that full 40-character commit for evaluator CLI calls.
5. Recheck each component's exact result and prediction destinations
   immediately before its one sequential GPU run.
6. Train Jasper, Teager2b, Victor, and Tyler sequentially in the frozen
   `numerai-lgbm-gpu312` environment; seal each successful run immediately.
7. Run calibration. Open locked Scout eras only for a passing, recomputed,
   fixed selection. Continue to confirmation and packaging only if every
   frozen gate passes.
8. Even after a full offline pass, stop at
   `PROMOTION_ELIGIBLE_NOT_UPLOADED`; uploading or assigning still requires a
   separate explicit user request.
