# Ender20 auxiliary-target ensemble continuation checkpoint

Date: 2026-08-08

Branch: `agent/optimized-ender20-model`

State: `STOP_NO_ELIGIBLE_CANDIDATE`

Frozen pre-scoring protocol commit:
`ef4ee304d6088f10d27e4d49a80d67ec925dbbf3`

Frozen pretraining implementation commit:
`b020661f1c7cf74b975a95e8ceb45d3f7c13b704`

All four one-shot Scout runs and seals were bound to that exact pretraining
commit. Later checkpoint-only documentation commits do not change that frozen
training authority.

## Safety state

- Jasper, Teager2b, Victor, and Tyler each completed exactly one authorized
  GPU Scout run and each has a finalized non-scoring seal.
- Scout calibration completed exactly once over the authorized 164-era slice.
  None of the five frozen Tyler-weight candidates passed every calibration
  check, so the experiment stopped at `STOP_NO_ELIGIBLE_CANDIDATE`.
- No Scout locked Ender20 metric has been computed; the 50 locked eras remain
  unopened.
- No confirmation component has been trained or scored.
- No final model has been fit or packaged.
- Nothing has been uploaded, assigned, submitted, staked, or changed in the
  Numerai account.
- The eight new Scout result/prediction files and their immutable receipt
  chains now exist locally and must not be deleted, renamed, overwritten, or
  rerun.
- The finalized calibration receipt now exists locally and must not be
  deleted, renamed, overwritten, substituted, or rerun.
- All ten confirmation result/prediction destinations remain absent.
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

## Sealed Scout run evidence

All four component seals have `state=SEALED` and `passed=true`. No Ender
blend candidate, calibration, or locked metric was computed while producing
these receipts.

- Jasper:
  - pre-run SHA:
    `71dedcf16b736b6b3291d03c8072f4b1beaf39c9192b09ef316dbc401542792e`
  - completion SHA:
    `7da3cdaa0e8bff88b93cf49aad2220c4fe3f84e1cdd99df1314879e20c15f7d3`
  - seal SHA:
    `dcbd661be85ff81707e516fe4f4078fc68966344ed601508ef1d40754909ffb8`
  - result SHA:
    `f9c5950d42dcb876c138d1cebe2238a8ab74fca76ca1096a31826c64b6a4ea7d`
  - prediction SHA:
    `5756b91c405a232a01679123bcb39a0e83577f8a9e86757fc5c8ea0c4c613890`
- Teager2b:
  - pre-run SHA:
    `3bb2ef6f0d46851b268fb2010a44fc479f081bb8aea38fc3e55d2eb8c49584bf`
  - completion SHA:
    `14770a461070a5a4e51d061f147f919b74f9ab2da9de414ca0773b932b5ac620`
  - seal SHA:
    `85b98e7a41071b79518802e53ec2905b491d2fcbc1ae5582c3e2fb599ab532d7`
  - result SHA:
    `2d1fac004726a707825110e008b23ae675168b999726d9f06ca7dfa862716cdd`
  - prediction SHA:
    `a414fc5f5815a89284825566d11e4859be00a7ced59253c2c4f6dcef26bd0954`
- Victor:
  - pre-run SHA:
    `fa08fd8541cab0b193150cd53552f8ff58926c03e885edaa677288f10ef5d321`
  - completion SHA:
    `c22f25af88a4b431415a540e4e54727cc1c233f88fa64dd05bab93790cc93f85`
  - seal SHA:
    `582a5dbd3fdcf9ffa3b7fde4b96f73e010e427fc1cfbe04f0d2fc84e3a3e2658`
  - result SHA:
    `a3bffc54934655aaf40578bd813f5e89d7f62d9380a3c8af938c324cd753f2af`
  - prediction SHA:
    `3a52d47cb82588cc53221a1c76a171fedf59dfb4cb27f7e18ba01fc35041eec9`
- Tyler:
  - pre-run SHA:
    `2a82c4824a269522971fde606e26048e6c543b8aab2a24f5d423f575438b421f`
  - completion SHA:
    `da7a09b1e5539c4a7f72d1c309eceb2a6c232be23bfa57854ba2af2ea194e5a5`
  - seal SHA:
    `c9ec868182ba8838dbae1e015869eca57c306612486ca28016f4f220ead0471b`
  - result SHA:
    `4831cb591d1b4fbca89b6d601ca49f0887a4d47e7da70f37e1e61b9b4ac243db`
  - prediction SHA:
    `3b4903d2c3e5b38bd0a2c6562ebaa168eaa431e671b97174b95cb822819b6c32`

## Scout calibration result

The frozen calibration stage revalidated the four exact seals above and
scored only the five predeclared Tyler-weight candidates over 164 calibration
eras (`0373` through `1025`, 957,371 rows). It finalized with:

- state: `STOP_NO_ELIGIBLE_CANDIDATE`;
- passed: `false`;
- selected formula: none;
- receipt:
  `receipts/calibrate-0544ff7beeb156266d859997636a4dc4af373af5dcb8bcea2d3a6f9a7fbd0e99.json`;
- receipt SHA:
  `0544ff7beeb156266d859997636a4dc4af373af5dcb8bcea2d3a6f9a7fbd0e99`.

The required BMC checks were mean at least `0.0020`, Sharpe greater than
`0.25`, and maximum drawdown less than `0.10`:

| Candidate | BMC mean | BMC Sharpe | Max drawdown | Failed checks |
| --- | ---: | ---: | ---: | --- |
| `tyler_w00` | 0.001237 | 0.120002 | 0.076781 | mean, Sharpe |
| `tyler_w10` | 0.001098 | 0.107232 | 0.082961 | mean, Sharpe |
| `tyler_w20_equal5` | 0.000956 | 0.093845 | 0.092577 | mean, Sharpe |
| `tyler_w30` | 0.000805 | 0.079438 | 0.100926 | mean, Sharpe, drawdown |
| `tyler_w40` | 0.000656 | 0.064958 | 0.108313 | mean, Sharpe, drawdown |

Every candidate passed the Corr-mean check and all three frozen similarity
checks. That does not override the failed BMC checks: no candidate is eligible,
the tie set is empty, and opening the locked slice is forbidden.

## Remaining order of operations

Stop here permanently for this frozen experiment:

1. Do not run the Scout locked stage.
2. Do not train confirmation components or run confirmation gates.
3. Do not package, upload, assign, submit, or stake this ensemble.
4. Preserve the sealed Scout artifacts and calibration receipt as immutable
   research evidence.
5. Any future attempt must be a separately predeclared experiment with a new
   frozen protocol and checkpoint; the candidates or thresholds in this
   experiment must not be changed after observing this calibration result.

Uploading, assigning the model to a Numerai slot, submitting predictions, or
staking requires a separate explicit user request.
