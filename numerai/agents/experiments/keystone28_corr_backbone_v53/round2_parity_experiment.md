# Keystone Round-2 — parity-calibration ladder (KP35)

Gate: `KP35_KEYSTONE_PARITY_LADDER_SOURCE_FREEZE`
Base commit: `f247bcc0ddff6352b8b933d6d1192df590ef2330` (merge of PR #33)
Predecessor terminal state: `STOPPED_AT_KW33_PIPELINE_PARITY_FAILURE`

**No fit has been run. No prediction exists. No evaluator has been invoked on real
artifacts.** This document, `round2_parity_protocol.json`, `round2_parity_lib.py`,
`round2_parity_train.py`, `round2_parity_evaluate.py` and
`agents/tests/test_keystone35_parity_ladder.py` are frozen *before* any result, so
that the design can be reviewed independently of the numbers it will produce.

---

## 1. The question

> Can either the documented benchmark information boundary or the documented v5
> deep LightGBM profile restore a static Keystone CONTROL backbone to
> benchmark-plausible CORR on eras 1133–1219?

Round 1 stopped because its static CONTROL-T backbone reached only **31.7%** of the
published `v53_lgbm_ender20` benchmark's mean CORR on the score zone — far below the
70% pipeline-parity floor. KP34 narrowed the cause to **two proven mismatches** between
CONTROL-T and the documented v5 benchmark procedure, plus two unbound factors. KP35
tests the two proven mismatches, one at a time, and nothing else.

### What this is not

This is **parity calibration only**. It does not test, and cannot promote:

Candidate-V · validation recency promotion · MMC specialist models · feature
ensembles · target ensembles · blending · deployment · live performance.

The ladder contains **P1 and P2 only**. If both fail, the experiment ends with a
finding. Feature-universe and row-budget expansion require a later, separately
reviewed gate — there is no P3 rescue inside this one.

---

## 2. Authority revalidation

Retrieved **2026-08-19T03:10:55+00:00**, unauthenticated, no credentials used.

| Item | Finding | Source |
| --- | --- | --- |
| Current public round | **1335** (opened 2026-08-18T13:12:17Z) | public GraphQL |
| Payout target | **`target_ender_20`** | public GraphQL |
| Payout CORR config | `v2_corr20`, `correlation` v6, multiplier **0.75**, 20D, lag 2, `isPayout: true` | public GraphQL |
| Payout MMC config | `mmc`, `meta_model_contribution` v5, multiplier **2.25**, 20D, lag 2, `isPayout: true` | public GraphQL |
| Ender60 cutover | **Not occurred.** `corr60` v3 and `mmc60` v3 are `isPayout: false`, multiplier `0.0` | public GraphQL |
| Recommended dataset version | **v5.3** — the latest version published by the data API | public GraphQL `listDatasets` |
| Benchmark walk-forward | 156-era chunks; purge **8** for 20D; window *n* trains `1 … 148+156(n−1)`, predicts `157+156(n−1) … 312+156(n−1)` | `numerai/docs` `models.md` |
| v5 deep LightGBM profile | 30,000 trees · lr 0.001 · depth 10 · 1,024 leaves · colsample 0.1 · min_data_in_leaf 10,000 | `numerai/docs` `models.md` |

Exact public query used for the round and payout configuration:

```
query { rounds(tournament: 8, number: 1335) { number openTime closeTime target
  v3Staking payoutFactor roundScoreConfigs { name displayName version isPayout
  defaultMultiplier minMultiplier maxMultiplier totalScoreDays returnsLagDays
  roundNumberStart } } }
```

Documentation identities:

* repository `numerai/docs`, branch `master`, commit **`5bf294adbac78d0cde497a7d1589694ee9951169`** (2026-07-01T19:07:43Z)
* `numerai-tournament/models.md` blob `676cfa7892efb9b4b0a01c224f3481ea34794335` (7,765 bytes)
* `numerai-tournament/data.md` blob `729a1215625679b6f881848cbcef6f70daa4e06c`
* URLs: `https://github.com/numerai/docs/blob/master/numerai-tournament/models.md` and `.../data.md`

**The active payout target and both multipliers are unchanged from the values KW33
froze.** The public round advanced 1334 → 1335, which is ordinary round progression
and not an authority change.

### Recorded ambiguities

Two, neither of which changes the frozen design, both recorded because they are real:

1. **Standard-large vs deep parameters.** `models.md` says the `{data_version}_LGBM_{target}`
   family is trained "with standard LGBM parameters" (20,000 · lr 0.001 · depth 6 ·
   2⁶ leaves), *and separately* says the deep block is "the higher performance 'deep'
   parameters we used to train the v5 benchmark models". The v5-specific statement is
   the more specific and more recent claim, so P2 freezes the **deep** profile — but
   the page does not settle the question beyond doubt. This is why the authority class
   is the pair `PUBLISHED_V5_BENCHMARK_WALKFORWARD_AND_DEEP_PROFILE_PROVEN` /
   `EXACT_V53_LGBM_ENDER20_FULL_RECIPE_NOT_PROVEN`.
2. **Documented example version lags the API.** `data.md` and `models.md` code samples
   pin `VERSION = "v5.2"`, while the data API publishes **v5.3** and the documented
   policy is to use the latest version. KP35 stays on v5.3, matching KW33 and the live
   API.

### Boundary derivation (not assumed)

`round2_parity_lib.derive_history_boundary()` searches the documented window table for
the chunk containing era 1133 and returns window **7**:

| quantity | value |
| --- | --- |
| train end | **1084** |
| purge | **1085–1092** (8 eras) |
| prediction chunk | **1093–1248** (156 eras, contains 1133–1219) |

A protected test asserts this derivation rather than the constant.

---

## 3. Data identities

Data root `D:/numerai-data/keystone28/v5.3`. All seven SHA-256 values were
**recomputed at source freeze** and every one matches the identity frozen by Round 1.
No file was redownloaded or overwritten.

| file | bytes | sha256 |
| --- | --- | --- |
| `features.json` | 387,149 | `27de6b59…da3631` |
| `train.parquet` | 3,296,841,026 | `bae773dd…c328f8` |
| `train_benchmark_models.parquet` | 65,045,676 | `5a872994…f80248` |
| `validation.parquet` | 5,611,121,884 | `8ed7859b…24fae6` |
| `validation_benchmark_models.parquet` | 126,278,375 | `801876e3…d6a5aaf5` |
| `meta_model.parquet` | 13,590,154 | `add54179…4b98d6d6` |
| `validation_example_preds.parquet` | 97,057,487 | `4bfd3ab7…83c5e3` |

Feature-list identity: `medium`, **780** features, list hash
`dd03cd099eb2c2283786eb123fe13a374460251b49f4717ec5ad34cabede80ba` — recomputed and
matching.

Every audit read projected only the required columns and applied era filters at the
Parquet scan boundary. **No GAP (1223–1230) or HOLDOUT (≥1231) target value was read.**

---

## 4. P1 — `P1_HISTORY_BOUNDARY_1084`

Isolates the **documented training-history boundary**.

| field | value |
| --- | --- |
| procedure | static CONTROL, one fit |
| model seed | **42** |
| target | `target_ender_20` (bare `target` rejected in source) |
| features | `medium`, exactly **780**, frozen list hash |
| max sampled rows | **1,000,000** |
| sampling seed | **20260817** |
| sampling law | frozen KW33 deterministic era-balanced law |
| profile | exact KW33 **FALLBACK**: 6,000 trees · lr 0.005 · depth 8 · 255 leaves · min_leaf 10,000 · feature_fraction 0.1 · deterministic CPU · force row-wise · 8 threads · no early stopping · no eval set |

Eligible training data: `train.parquet` eras **0001–0574** plus resolved validation
history **0575–1084**.
Explicitly excluded: purge **1085–1092**, benchmark chunk **1093–1248**, GAP
**1223–1230**, HOLDOUT **≥1231**.

Scoring: exactly eras **1133–1219**, exactly the complete authoritative
**575,597-row** scoring universe, against published `v53_lgbm_ender20` on identical
rows. CORR is the screening authority; MMC and the weighted score are diagnostic only.

**The sole configured change from KW33 CONTROL-T is the eligible training-history
endpoint (0574 → 1084).**

### Honest composition caveat

Because the one-million-row cap **remains fixed**, this configuration mechanically
changes the sampled era composition and **replaces** some original train rows with
history rows. **P1 is not a pure additive-data experiment.** A keys-only audit at
source freeze (no features loaded, no training, no prediction) computed the exact
composition:

| quantity | value |
| --- | --- |
| eligible eras | 1,084 |
| rows before sampling | 5,890,287 |
| per-era quota | 922 (take 922–1,474) |
| sampled from `train.parquet` | 529,780 (52.978%) |
| sampled from validation history | 470,220 (47.022%) |
| selected rows | 1,000,000 |

This reproduces the KP34 audit-3 composition exactly.

---

## 5. P2 — `P2_DEEP_PROFILE`

Isolates the **documented v5 deep LightGBM profile**.

P2 becomes executable **only** when a valid P1 screen artifact exists and records
`KP35_P1_SCREEN_FAILED_P2_AUTHORIZABLE`. A P1 *pass* goes to confirmation and can
never reach P2.

P2 holds identical: target · feature list · feature count · training-era boundary ·
purge · score zone · row cap · sampling seed · **the exact sampled `(era,id)`
universe** · model seed 42 · scoring universe · benchmark reference · evaluator ·
output semantics.

P2 must **reuse** P1's sample selection, proven by comparing the canonical sample
SHA-256 and the parameter-independent sample identity. Regenerating an allegedly
equivalent sample without comparing identities is refused in source.

### The only permitted difference

| field | P1 (FALLBACK) | P2 (documented v5 deep) |
| --- | --- | --- |
| `num_trees` | 6,000 | **30,000** |
| `learning_rate` | 0.005 | **0.001** |
| `max_depth` | 8 | **10** |
| `num_leaves` | 255 | **1,024** |

Identical in both: `objective` regression · `min_data_in_leaf` 10,000 ·
`feature_fraction` 0.1 · deterministic CPU · force row-wise · 8 threads · no early
stopping · no evaluation set. A protected test asserts that P2 differs from P1 in
**no other field**.

P2 uses the **same screen threshold** as P1. A pass is
`KP35_P2_SCREEN_PASSED_AWAITING_CONFIRMATION`; a failure is
`KP35_PARITY_NOT_RESTORED_BY_PROVEN_MISMATCHES`, and the experiment ends there.

---

## 6. Screening law

Frozen benchmark mean CORR from KW33:

```
0.02094843151562169
```

Before any future scientific evaluation this is **recomputed** from the published
benchmark column on the identical rows and must equal the frozen value within a
strict declared tolerance of `1e-12`. A drifting benchmark would silently move every
threshold, so identity is enforced rather than assumed.

Screen factor **0.6755**, derived exactly as `0.70 × (1 − 0.0350)`, where 0.0350 is
the worst single-seed deviation below the three-seed mean ever observed in the KW33
CONTROL-T cohort (seed 2024, −3.50%). The screen therefore errs low by exactly the
observed dispersion and by no more.

```
seed42_corr >= 0.6755 * benchmark_mean_corr
screen threshold = 0.014150665488802451
```

A **pass means only** `KP35_P1_SCREEN_PASSED_AWAITING_CONFIRMATION`.
A **failure means only** `KP35_P1_SCREEN_FAILED_P2_AUTHORIZABLE`.

A screen pass is **not** final parity, **not** model promotion, **not** recency
promotion, and **not** deployment authority. **Confirmation and P2 are never begun
automatically after evaluating P1.**

---

## 7. Final confirmation law

For whichever recipe first passes the seed-42 screen: reuse its completed seed-42
artifact, run **only** seeds **1337** and **2024** on the identical recipe and exact
sampled-row identity, and **never rerun a valid seed-42 fit**.

Final parity requires **both**:

1. three-seed mean — `mean(corr_42, corr_1337, corr_2024) >= 0.70 × benchmark_mean_corr`
2. untouched pair — `mean(corr_1337, corr_2024) >= 0.70 × benchmark_mean_corr`

```
both thresholds = 0.014663902060935183
```

The **untouched-pair gate** exists because seed 42 is the seed selection was performed
*on*. Requiring the two seeds that took no part in selection to clear the same bar
independently is what stops a lucky screening draw from carrying a recipe through
confirmation. Either gate failing prevents confirmation.

Terminal: `KP35_PARITY_BACKBONE_CONFIRMED` or `KP35_SCREEN_PASS_CONFIRMATION_FAILED`.

**No weighted-score, MMC, Ender60, bootstrap, recent-window, or risk statistic may
change the parity decision.** Those are reported diagnostically; parity selection uses
CORR only. The decision functions accept CORR floats and have no input path for
anything else — a protected test asserts their exact signatures.

**No return to Candidate-V is authorized by source freeze or even by a future parity
pass.** A separately reviewed recency experiment is required.

---

## 8. Exact-row contract (prospective repair)

Round 1's frozen sources checked era-set *coverage* and delegated row identity to a
join, so a strict subset of the scoring universe would have scored without complaint.
KP34 proved the KW33 artifacts were factually complete — that proof is the custody
record, and **no merged Round-1 file is altered here.** The repair lives only in the
KP35 packet.

`assert_exact_row_universe` requires equality of the complete canonical `(era,id)`
universe among the prediction vector, the scoring target frame, the Meta Model frame,
the published Ender20 benchmark frame, and the published Ender60 benchmark frame when
auxiliary diagnostics are loaded.

It compares **total rows · era set · per-era row counts · complete sorted `(era,id)`
pairs · canonical SHA-256**, and rejects any strict subset, strict superset, missing
row, extra row, duplicate ID, era disagreement, row with an unexpected era, or
non-finite prediction.

The canonical universe identity was **recomputed independently at source freeze** from
the current data files — not copied:

| frame | rows | canonical SHA-256 |
| --- | --- | --- |
| `validation.parquet` | 575,597 | `91e519aff5c656c9acf7cc6fe74daebfc034650bae47ee4e3889a98ec8fac033` |
| `meta_model.parquet` | 575,597 | identical |
| `validation_benchmark_models.parquet` | 575,597 | identical |

Canonicalisation: stable sort by `(era,id)`; join `era,id` lines with `\n`; UTF-8;
SHA-256. This reproduces the KP34 scratch evidence exactly.

---

## 9. Sample custody

A deterministic sample-identity artifact binds: data identities · eligible era range ·
feature-list identity · sampling-law version · sampling seed · row cap · exact selected
row count · selected rows per era · train-versus-validation source split · canonical
ordered `(era,id)` hash · **parameter-independent sample identity**.

The identity is a function of the data, era range, feature list, sampling law, seed,
cap and resulting key universe — and of *nothing else*. Model seed and model profile
are structurally excluded, and the manifest validator **refuses** a manifest that
carries them. That exclusion is what lets P2 *prove* it reused P1's exact sampled rows.

P1 and P2 reference the same sample identity; confirmation seeds reference the same
recipe-specific identity. The artifact lives outside Git under
`D:/numerai-data/keystone28/round2-parity`.

At source freeze a **keys-only, explicitly non-scientific** dry audit was generated
outside Git (`source-freeze-preflight/kp35_universe_and_sample_audit.json`). It loaded
only `id` and `era` columns, trained nothing and produced no prediction or metric. Its
P1 sample key hash is `e555e848770f4acd276020aca833541e8b0702a2f1b7c3ebc8068b657d101350`.

---

## 10. One-shot artifact law

Every future scientific artifact path is a pure function of kind, stage and seed, so
two fits can never collide and one fit can never be written twice under two names:
prediction · fit log · failure record · screen result · confirmation result · sample
identity · final report.

Never overwrite a completed prediction. Never rerun a completed fit. Never overwrite a
failed-attempt record. An infrastructural retry is permitted **at most once**, only
when no valid prediction artifact exists, preserving the original failure, with seed,
stage, data, rows, parameters and sample identity unchanged. The evaluator refuses a
second invocation against an existing result path — before doing any work. Atomic
temp-to-final replacement is used for **first-time creation only**; a final path is
never replaced. There is no automatic stage chaining anywhere.

---

## 11. Expected compute

| fit | shape | profile | est. runtime | est. peak RAM |
| --- | --- | --- | --- | --- |
| P1 seed 42 | 1,000,000 × 780 (cohort 5,890,287) | 6,000 FALLBACK | **15–23 min** | ≈ 16.5 GB |
| P2 seed 42 | 1,000,000 × 780 | 30,000 deep | **≈ 2.7 h** + 10–20 min predict | ≈ 17–19 GB |
| confirmation (1337, 2024) | as winning stage | as winning stage | **2 × stage time** (0.5–5.4 h) | as winning stage |

Basis: 18 KW33 fits of identical shape ran 907–1,380 s at 16.53 GB peak; the KW33
throughput probe measured 0.3212 s/tree at depth 10 / 1,024 leaves on this exact
shape. Host: 28 GB RAM, 4 cores / 8 logical, CPU-only — the GPU is present but unused
because no GPU-enabled LightGBM is available here.

**These are projections. No fit has run.**

---

## 12. Terminal states

| state | meaning |
| --- | --- |
| `KP35_P1_SCREEN_PASSED_AWAITING_CONFIRMATION` | P1 cleared the screen; confirmation is *authorisable* |
| `KP35_P1_SCREEN_FAILED_P2_AUTHORIZABLE` | P1 failed; P2 is *authorisable* |
| `KP35_P2_SCREEN_PASSED_AWAITING_CONFIRMATION` | P2 cleared the screen; confirmation is *authorisable* |
| `KP35_PARITY_NOT_RESTORED_BY_PROVEN_MISMATCHES` | both proven mismatches failed; **absorbing** |
| `KP35_PARITY_BACKBONE_CONFIRMED` | both final gates passed; **absorbing** |
| `KP35_SCREEN_PASS_CONFIRMATION_FAILED` | a gate failed at confirmation; **absorbing** |

Absorbing states admit no further transition, and no transition may run backward — a
protected test asserts the complete forward transition law.

---

## 13. Non-actions at source freeze

No model training · no prediction generated · no scientific evaluator run on real
artifacts · no GAP or HOLDOUT access · no Numerai account action, model creation,
submission, upload, staking or deployment · no prior artifact mutated or deleted · no
protected-checkout synchronisation · no branch, tag or worktree deleted · no merged
Round-1 file altered · no PR merged.

Training and evaluation await independent review of this frozen source and a separate
execution authority.
