# Ender20 auxiliary-target ensemble continuation checkpoint

Date: 2026-08-08

Branch: `agent/optimized-ender20-model`

State: `TRAINING_READY_NOT_RUN`

Frozen pre-scoring protocol commit:
`ef4ee304d6088f10d27e4d49a80d67ec925dbbf3`

The pretraining checkpoint is the full 40-character commit containing this
document and the implementation/test changes described below. After checking
out the pushed branch, obtain it with `git rev-parse HEAD`; every authorized
training and evaluator command must bind that exact commit.

## Safety state

- No Jasper, Teager2b, Victor, or Tyler Scout training has started.
- No Scout calibration or locked Ender20 metric has been computed.
- No confirmation component has been trained or scored.
- No final model has been fit or packaged.
- Nothing has been uploaded, assigned, submitted, staked, or changed in the
  Numerai account.
- The experiment's `receipts/`, `results/`, and `predictions/`
  directories are absent.
- All eight new Scout result/prediction destinations and all ten confirmation
  result/prediction destinations are therefore absent.
- The unrelated untracked `.gpu-lgbm-source-build/` directory remains
  protected and is intentionally excluded from Git.

An existing receipt, claim, completion marker, result, prediction, partial
file, dangling link, or output reservation is a hard stop. It is never
authority to delete, rename, overwrite, or rerun.

## Frozen experiment

The protocol is defined by `gate.md`, with frozen sources and artifact
anchors in `source_manifest.json`.

Protocol hashes:

- `gate.md`:
  `c851e3e0637e26bff5b2c26eda5752a46a9d72fce2621678bd39ffa320983ffe`
- `source_manifest.json`:
  `3cc96dce9938306cc1f2e7d4ef6b6628f24494f5c30a1ca87d791b64ace662a8`

The Scout is a five-component within-era rank ensemble over direct-target
LightGBM models for Jasper20, Teager2b20, Victor20, frozen Xerxes20, and
Tyler20. The only candidate dimension is Tyler weight: 0.0, 0.1, 0.2, 0.3,
or 0.4. The remaining weight is split equally across the other four
components. Scout run order is fixed:

1. Jasper
2. Teager2b
3. Victor
4. Tyler

The preceding Xerxes20 depth-8 artifact is reused exactly; it is not retrained.

## Training-ready implementation

The evaluator and modeling pipeline now enforce:

- exact Git checkpoint, source-manifest, gate, imported evaluator, config,
  helper, loader, GPU runtime, raw-source, and historical-artifact bindings;
- SHA/JSON parsing from the same no-write/no-delete leased manifest bytes;
- exact Windows file identity, regular-file, link, reparse, directory-chain,
  size, and SHA checks at every protected read boundary;
- frozen Scout configs cached only after joint wrapper/base leases, with those
  leases held through downstream artifact receipts;
- exact confirmation wrapper source plus manifest-pinned base-helper leases
  held through result/prediction validation and artifact receipts;
- exact ID, era, stored target, producer fold, prediction semantics, GPU
  diagnostics, result, Parquet, feature-store, and target-label validation;
- immutable external confirmation store inventory and metadata, feature,
  manifest, order, source, label, and checkpoint-blob bindings;
- exact legacy/downsampled and full-consecutive two-seed TabM references and
  frozen Xerxes reads under manifest-bound leases;
- protected generic-output guards for every new destination and all ten
  historical Xerxes/TabM result/prediction paths, including hardlink and
  reparse aliases;
- within-era average-tie ranking, weighted blending, reranking, Corr, BMC,
  population-standard-deviation Sharpe, additive drawdown, and symmetric
  per-era Spearman similarity;
- content-addressed receipts with canonical directories, exact prefixes,
  exclusive early claims, finalized-claim bindings, closed schemas, and
  same-prefix rerun refusal;
- deterministic Scout calibration rederivation before locked access and
  selected-only locked scoring;
- five mandatory non-scoring confirmation component seals and selected-only
  655/200/855 confirmation ordering;
- one-shot trainer authorization from the exact finalized pre-run receipt;
- `CREATE_NEW` canonical output reservations, a durable consumed-run marker,
  and a completion prefix claimed before model code;
- completion receipts finalized from the same still-open output handles,
  binding path, device, inode, size, and SHA;
- completion, marker, output, config, and historical leases held through seal
  construction and finalization;
- terminal orphan claims/partial reservations on failure, intentionally
  preventing a retry.

The trainer also requires a fresh Python launch with both:

- `-B`; and
- `-X pycache_prefix=<unique empty absolute directory outside the repo>`.

The prefix must be a plain, empty, non-reparse directory from process start.
Adjacent ignored `__pycache__` files are not trusted.

## Validation evidence

No real Numerai data, protected metrics, GPU training, or production scoring
was accessed during this hardening checkpoint. All executed validation used
synthetic temporary fixtures.

Primary focused validation:

- evaluator and disk/modeling suites: 116/116 pass;
- Python syntax compilation: pass;
- `git diff --check`: pass (line-ending warnings only).

Independent settled-byte validation:

- evaluator suite: 82/82 pass;
- disk/modeling suite: 34/34 pass;
- adjacent disk feature-store suite: 12/12 pass;
- adjacent scoring/prediction-semantics suite: 13/13 pass;
- total: 141/141 pass;
- six changed Python modules compile;
- `git diff --check`: exit 0.

Two independent final static reviews reported no remaining usable false-pass
or pre-fit training-authority blocker under the declared Windows execution and
trusted fresh-launch model.

Covered negative cases include checkpoint/source drift, hash-to-parse swaps,
config/helper swaps, dirty or untracked executable dependencies, receipt
aliasing and nested leakage, forged selection and per-era metrics, fractional
folds, wrong GPU/result/semantics/store identity, early reruns, duplicate or
disconnected seal chains, output hardlinks/reparse points, poisoned timestamp
bytecode, changed completion/consumption markers, concurrent/reused
authorization, same-inode post-training output mutation, and post-verification
historical artifact swaps.

Prior real frozen-artifact smoke evidence from the 2026-08-03 checkpoint was
not rerun here:

- Scout expected cohort: 1,279,658 rows, 214 eras;
- frozen two-seed TabM residual: exact artifacts validated;
- reused Xerxes depth-8 result/predictions: exact 1,279,658-row validation.

## Remaining order of operations

There is no known implementation blocker before Scout training. Continue only
in this order:

1. Confirm the pushed branch is clean and `HEAD` is the intended full
   pretraining checkpoint. Keep `.gpu-lgbm-source-build/` untouched.
2. Create one unique empty external pycache directory and launch every
   authorized Python process with `-B -X pycache_prefix=<that-directory>`.
3. For Jasper, create its canonical pre-run absence claim/receipt.
4. Run Jasper once with its exact Scout component and pre-run receipt
   path/SHA. Capture the emitted completion receipt path/SHA.
5. Seal Jasper immediately with the exact pre-run and completion bindings.
6. Repeat steps 3-5 sequentially for Teager2b, Victor, and Tyler, each binding
   the immediately preceding finalized seal.
7. Run Scout calibration from exactly the four seals.
8. Only if calibration passes, run the locked Scout stage. It must rederive
   calibration and score only the selected formula.
9. Only if Scout locked passes, create the committed confirmation store
   inventory/config checkpoint, then the dynamic confirmation-pretraining
   receipt.
10. Run and seal the five confirmation components sequentially, then run
    confirmation calibration and locked/full gates.
11. Only if every frozen gate passes, create the offline model package and
    stop at `PROMOTION_ELIGIBLE_NOT_UPLOADED`.

Uploading, assigning the model to a Numerai slot, submitting predictions, or
staking requires a separate explicit user request.
