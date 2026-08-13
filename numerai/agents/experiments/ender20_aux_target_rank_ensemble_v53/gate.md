# Ender20 Auxiliary-Target Rank-Ensemble Gate (v5.3)

**Protocol date:** 2026-08-03

**Scoring status at definition:** no new Jasper, Teager2b, Victor, or Tyler
component has been trained, and no new ensemble metric has been computed. This
file, its component configs, and `source_manifest.json` must be committed before
component training or ensemble scoring begins.

## Research question

Can an equal-architecture ensemble of diverse 20-day auxiliary targets improve
the stability and uniqueness of the direct Xerxes20 LightGBM signal when scored
on `target_ender_20` against Numerai's official Ender benchmark models?

This is a new model family. The stopped Xerxes capacity sweep may contribute
only its sealed depth-8 artifact; its other profiles and its still-unopened
locked metrics are not selection inputs.

## Frozen component family

Every component is a direct-target LightGBM using the 780 medium features:

- 6,000 fixed trees, learning rate `0.003`;
- maximum depth 8 and 255 leaves;
- `colsample_bytree=0.1` and `min_data_in_leaf=10000`;
- GPU tree learner, 12 threads;
- model seed 1337;
- deterministic 500,000-row cap per training fold with sample seed 1337;
- no target residualization, prediction post-processing, or outer-fold early
  stopping.

The five component targets and their frozen roles are:

| key | target | role |
| --- | --- | --- |
| `jasper` | `target_jasper_20` | quality anchor; highest saved auxiliary-20 target correlation with Ender in the official tutorial |
| `teager2b` | `target_teager2b_20` | official tutorial ensemble candidate |
| `victor` | `target_victor_20` | official tutorial quality/diversity choice |
| `xerxes` | `target_xerxes_20` | prior positive-BMC component |
| `tyler` | `target_tyler_20` | diversity tail; lowest saved auxiliary-20 target correlation with Ender in the official tutorial |

The official notebook is hypothesis-generation evidence only. Its small-feature,
single-split scores are not promotion evidence for this experiment.

The sealed Xerxes component must be reused exactly from
`../xerxes20_lgbm_challenger_v53/predictions/r1_depth8.parquet`:

- prediction SHA-256:
  `5b2ac138624ca147177d9d8e09362f0edd983ff7fb87ae3a790d9995342a0b56`;
- result SHA-256:
  `d9ad3b14cdd5e79e62ac6e4920f6160f38b870dbf592db8e1847c44fa4bbcf2e`;
- config SHA-256:
  `ed186040dcd6899090302575c1a1503c892dba953365049ee5ceefd3467f6d69`.

No other historical Xerxes profile may replace it.

## New-artifact immutability

The four new target-qualified destinations are fixed:

| component | prediction | result |
| --- | --- | --- |
| Jasper | `predictions/r1_jasper_d8_t6000.parquet` | `results/r1_jasper_d8_t6000.json` |
| Teager2b | `predictions/r1_teager2b_d8_t6000.parquet` | `results/r1_teager2b_d8_t6000.json` |
| Victor | `predictions/r1_victor_d8_t6000.parquet` | `results/r1_victor_d8_t6000.json` |
| Tyler | `predictions/r1_tyler_d8_t6000.parquet` | `results/r1_tyler_d8_t6000.json` |

Because the current pipeline can overwrite a matching `results_name`, a
mandatory preflight immediately before each sequential run must prove both of
that component's destinations absent. An existing or partially created path
stops this family; it may not be deleted, renamed, overwritten, or reused to
permit a rerun. After each successful run, seal the config, result, and
prediction hashes in an immutable artifact receipt and require every fold to
report `effective_device_type == "gpu"` and `gpu_fallback_used == false`.

## Frozen ensemble transform and candidates

For each era, percentile-rank each raw component prediction using average tie
ranks. Form the weighted row sum, then percentile-rank that sum within era using
the same tie rule. No benchmark blend or feature neutralization is permitted.

The only sweep dimension is Tyler's diversity weight `lambda`. The remaining
weight is divided equally among Jasper, Teager2b, Victor, and Xerxes:

| candidate | Tyler weight | each core-component weight |
| --- | ---: | ---: |
| `tyler_w00` | 0.00 | 0.250 |
| `tyler_w10` | 0.10 | 0.225 |
| `tyler_w20_equal5` | 0.20 | 0.200 |
| `tyler_w30` | 0.30 | 0.175 |
| `tyler_w40` | 0.40 | 0.150 |

Individual component models are diagnostic references only and are never
selectable. No interpolation, alternate weighting, target substitution, or
post-result formula is allowed.

## Scout cohort and chronology

- Feature/target source: `numerai/v5.3/downsampled_full.parquet`.
- Benchmark source:
  `numerai/v5.3/downsampled_full_benchmark_models.parquet`.
- Restrict to IDs with complete benchmark coverage.
- Exact OOF cohort: 1,279,658 rows and 214 retained eras, `0373`-`1225`.
- Each new component uses the frozen `n_splits=5` expanding scheme, which has
  one initial train-only block and four OOF validation folds, with a
  13-retained-era embargo.
- The evaluator independently derives and validates each artifact's ID, era,
  stored target, fold, and prediction semantics.
- Calibration: first 164 OOF eras, `0373`-`1025`.
- Locked robustness: final 50 OOF eras, `1029`-`1225`.

All five selected targets, the Ender scoring target, and both benchmark columns
are null-free and finite on the exact benchmark-covered scout cohort, as pinned
in `source_manifest.json`. Missing or non-finite values, duplicate IDs, cohort
drift, fold drift relative to an artifact's own frozen producer, altered
semantics, CPU fallback, or a changed source/config/runtime receipt invalidate
the experiment before scoring. Cross-artifact blending aligns on exact
one-to-one `id` and `era`; `cv_fold` is a provenance field, not a join key.
Historical TabM and LightGBM fold labels must each match their independently
derived maps and are never required to equal one another.

## Calibration scoring and selection

Score on `target_ender_20`. BMC is computed against
`v53_lgbm_ender20`. Similarities are the average per-era symmetric Spearman
correlation using average tie ranks.

A candidate is calibration-eligible only if all checks pass:

- BMC mean `>= 0.0020`;
- BMC Sharpe `> 0.25`;
- BMC maximum drawdown `< 0.10`;
- Ender Corr mean `>= 0.012`;
- similarity to `v53_lgbm_ender20` `< 0.75`;
- similarity to `v53_lgbm_ender60` `< 0.75`;
- similarity to the frozen two-seed Ender TabM residual `< 0.75`.

The TabM reference is constructed only from the exact downsampled OOF artifacts
`r5_tabm_k64_train500k.parquet` and
`r6_tabm_k64_train500k_seed2027.parquet`: rank each seed within era, average
them, then rank the average within era. Their exact config/result/prediction
hashes are pinned by `source_manifest.json`.

Those two legacy TabM Parquets predate custom prediction-semantics metadata.
The evaluator must therefore validate their exact hashes, the complete inherited
config chain, each result's `residual_to_benchmark` target transform, and exact
ID/era/Ender-target alignment plus each file's own fold provenance. It must not
weaken the check, compare TabM folds to LightGBM folds, or apply a
metadata-presence requirement that these sealed files cannot satisfy.

## Frozen metric math

Order eras by their numeric value ascending and require every per-era score to
be finite. Per-era Corr is `numerai_tools.scoring.numerai_corr`; per-era BMC is
`numerai_tools.scoring.correlation_contribution` against
`v53_lgbm_ender20`.

For each chronologically ordered score series, use the arithmetic mean,
population standard deviation (`ddof=0`), and unannualized Sharpe
`mean / std`. A zero or non-finite standard deviation fails the corresponding
check. Additive maximum drawdown is exactly:

```python
cumsum = scores.cumsum()
running_max = cumsum.expanding(min_periods=1).max()
max_drawdown = (running_max - cumsum).max()
```

Do not prepend an initial zero. Each reported similarity is an equal-era-weight
arithmetic mean of per-era symmetric Spearman correlations, with both signals
converted to average tie ranks within the era.

Find the eligible candidate maximum calibration BMC mean `max_mean`, then form
one anchored tie set containing every eligible candidate for which
`max_mean - candidate_mean <= 0.0001`. Inside only that set, select by higher
BMC Sharpe, then lower BMC maximum drawdown, then lower Tyler weight, then
lexicographically smaller candidate name. This definition forbids pairwise
tie chaining. If no candidate is eligible, stop the family.

## Enforced evaluator stages

The evaluator must expose separate, fail-closed stages with immutable,
content-addressed receipts:

1. `calibrate` validates every source and component, scores only the 164-era
   calibration slice, and writes all five calibration candidates plus exactly
   one selected formula or null. It creates no locked output.
2. `locked` requires the exact hash of a passing, non-null calibration receipt,
   revalidates its protocol/evaluator/input bindings, and scores only the fixed
   selected formula on the final 50 eras. It refuses a substitute or any
   unselected candidate.
3. `confirmation-calibrate` requires the exact hash of a passing scout-locked
   receipt and scores only the fixed formula on the 655-era confirmation
   calibration slice.
4. `confirmation-locked` requires the exact hash of a passing confirmation
   calibration receipt and opens only the fixed formula's final 200 eras. If
   that slice passes, the stage may then compute the fixed formula's full-period
   metrics; if it fails, it persists only the locked result and stops.

Every stage must refuse an existing output path rather than overwrite it. The
first durable output is therefore calibration-only, and no locked component or
unselected-candidate **Ender20 ensemble-evaluation** metric may be computed or
persisted. The sealed scalar training pipeline's automatic full-OOF diagnostics
against each component's own auxiliary target are permitted implementation
receipts; the evaluator must ignore them, must not copy them into selection
outputs, and must never use them to select a blend or open an Ender20 holdout.

## Locked scout robustness

Only the selected calibration formula may open the final 50 retained eras. It
must pass every strict check:

- BMC mean `> 0`;
- BMC Sharpe `> 0.20`;
- BMC maximum drawdown `< 0.10`;
- Ender Corr mean `> 0.008`.

A failure stops the family. No alternate candidate may substitute.

## Consecutive confirmation

Only a locked-scout passage authorizes confirmation. Reproduce all five direct
target components with unchanged LightGBM parameters, feature order, row cap,
and seeds on the consecutive full cohort:

- exact OOF cohort: 5,112,039 rows and 855 eras, `0371`-`1225`;
- the same `n_splits=5` expanding scheme, yielding four OOF validation folds,
  with a 52-era embargo;
- confirmation calibration: first 655 eras, `0371`-`1025`;
- locked final 200 eras: `1026`-`1225`.

All five component targets, `target_ender_20`, and both benchmark columns have
zero null and non-finite values on this exact 5,112,039-row cohort, as pinned in
`source_manifest.json`.

The canonical feature bytes are the existing Xerxes medium store, generation
`e5f640acd0f84f178e6c5a1f8bb2f7ba`, with 6,195,697 rows, 780 features,
feature SHA-256
`088f9479ebc2bed51528ee9623079a185119cb4ce8342c1d60737b78c02bfc62`,
and manifest SHA-256
`effad46864d84cc277587abefa9c7b92d5150ff7259ff7644221f0e442c2a4a3`.
Its complete receipt is pinned in `source_manifest.json`.

The default, lowest-risk confirmation input reuses that Xerxes store and builds
four additional target-specific medium stores consumed by the existing proven
loader. A storage-saving sidecar remains permissible only if, before any
confirmation component training, a generic `data.label_sidecar_path` loader
extension and its tests are committed and frozen in a new confirmation
pre-training receipt. That immutable Parquet must contain exact consecutive
`row_offset`, unique `id`, `era`, all five component targets, the Ender scoring
target, and both Ender benchmark columns. The loader must verify the canonical
generation ID, row count, feature hash, feature-order hash, one-to-one
alignment, and finite labels.

The sidecar is storage-only. Each target still has its own scalar-target config,
model, and OOF artifact, and blending occurs only in the external evaluator; a
multi-output learner or training wrapper is outside this family. Neither input
layout may change feature bytes, row order, sampling, folds, or model behavior,
and filesystem links may not substitute for a validated layout.

Before the first confirmation component runs, the five exact target-qualified
configs and the chosen loader/store receipts must be committed in a separate
confirmation pre-training checkpoint. The only permitted `results_name` values
and corresponding destinations are:

| component | config | prediction | result |
| --- | --- | --- | --- |
| Jasper | `configs/confirmation_jasper_d8_t6000.py` | `predictions/confirmation_jasper_d8_t6000.parquet` | `results/confirmation_jasper_d8_t6000.json` |
| Teager2b | `configs/confirmation_teager2b_d8_t6000.py` | `predictions/confirmation_teager2b_d8_t6000.parquet` | `results/confirmation_teager2b_d8_t6000.json` |
| Victor | `configs/confirmation_victor_d8_t6000.py` | `predictions/confirmation_victor_d8_t6000.parquet` | `results/confirmation_victor_d8_t6000.json` |
| Xerxes | `configs/confirmation_xerxes_d8_t6000.py` | `predictions/confirmation_xerxes_d8_t6000.parquet` | `results/confirmation_xerxes_d8_t6000.json` |
| Tyler | `configs/confirmation_tyler_d8_t6000.py` | `predictions/confirmation_tyler_d8_t6000.parquet` | `results/confirmation_tyler_d8_t6000.json` |

The pre-training checkpoint must prove all ten result/prediction destinations
absent. Immediately before each sequential run, repeat the two-path absence
check; any existing or partial path stops the family. Seal each config, result,
and prediction hash after its one permitted run. The scout no-overwrite,
no-delete, no-rename, no-rerun, and GPU-proof rules apply unchanged.

Confirmation similarity to the TabM residual uses only the full-consecutive
`scale_disk_tabm_k64_train500k.parquet` and
`scale_disk_tabm_k64_train500k_seed2027.parquet` artifacts: average-tie rank
each seed within era, average equally, then rerank within era. Their exact
config/result/prediction receipts and required custom prediction-semantics
metadata are separately pinned in `source_manifest.json`; the downsampled TabM
pair is never used for confirmation.

Use the exact selected scout weights. Confirmation passes only if all checks
pass:

- calibration and full BMC mean `>= 0.0020`;
- calibration and full BMC Sharpe `> 0.35`;
- calibration and full BMC maximum drawdown `< 0.15`;
- calibration and full Ender Corr mean `>= 0.012`;
- all three calibration/full similarity checks `< 0.75`;
- locked-200 BMC mean `> 0`;
- locked-200 BMC Sharpe `> 0.20`;
- locked-200 BMC maximum drawdown `< 0.15`;
- locked-200 Ender Corr mean `> 0.008`.

## Stop rule and deployment boundary

A failed calibration, locked scout, or confirmation stops this family without
replacement, target swaps, extra weights, hyperparameter tuning, threshold
changes, or holdout reuse.

Passing every offline gate permits exactly one local final fit. Each of the five
components uses the unchanged frozen LightGBM profile and the same deterministic
500,000-row without-replacement sample from the canonical 6,195,697-row finite,
benchmark-covered historical cohort, in canonical manifest-row order. Sampling
is `numpy.random.default_rng(1337).choice(N, 500000, replace=False)`, preserving
the returned draw order. The sample-position digest is exactly
`sha256(np.asarray(positions, dtype="<i8", order="C").tobytes(order="C"))`;
no textual, sorted, or platform-native encoding may replace it.

The four fixed deployment destinations are:

- `artifacts/optimized_ender20_target_ensemble.pkl`;
- `artifacts/optimized_ender20_target_ensemble.final_fit.json`;
- `artifacts/optimized_ender20_target_ensemble.docker_predictions.parquet`;
- `artifacts/optimized_ender20_target_ensemble.docker.json`.

All four must be absent immediately before final fit. An existing or partial
path stops packaging and may not be deleted, renamed, overwritten, or rerun.
The final-fit receipt must bind the protocol, scout, and confirmation receipt
hashes; canonical store generation and manifest hash; all five config hashes;
selected weights; sample-position digest and its `little-endian int64 C-order,
RNG-draw-order` encoding; each component's SHA-256 over
`booster_.model_to_string().encode("utf-8")`; the cloudpickle SHA-256; exact
Python/package versions; and the fixed output path.

The predictor must preserve the selected Tyler weight and perform the frozen
per-live-cross-section component rank, weighted sum, and final rank transform.
No all-row refit, resampling, target drop, component compression, or retraining
variant is allowed.

The cloudpickle must then pass the official Numerai Predict Docker image on the
current live file with one CPU, no more than 4 GiB peak memory, under ten
minutes, and no internet. It must preserve the exact input index and emit
exactly one finite `prediction` column with every value in `[0, 1]`. Runtime
failure is `NOT_PROMOTION_ELIGIBLE`; it may not be repaired by dropping or
compressing a component.

Before that one Docker run, resolve the official image to an immutable OCI
`sha256:` digest and invoke the digest, never a mutable tag. The Docker receipt
must bind the image name/digest, exact argv, `--network none`, one-CPU and
4-GiB limits, Numerai round identifier, full live-input SHA-256, cloudpickle
SHA-256, exit code, wall-clock seconds, measured peak memory bytes, stdout and
stderr SHA-256, and the fixed Docker-prediction file SHA-256. Input and output
index digests use each string ID encoded as unsigned 64-bit little-endian byte
length followed by UTF-8 bytes, concatenated in row order; they must match
exactly. The receipt also records row count, column names, finite count, and
prediction minimum/maximum. It passes only if the enforced constraints and all
observed measurements satisfy this gate.

Passing final fit and Docker produces only a local
`PROMOTION_ELIGIBLE_NOT_UPLOADED` artifact. It does not authorize Numerai
upload, model-slot assignment, prediction submission, or staking. Those remain
separate user decisions.
