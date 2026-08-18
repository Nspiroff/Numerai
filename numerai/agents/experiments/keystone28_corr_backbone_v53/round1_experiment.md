# Keystone Round 1 — walk-forward recency-augmentation test (KW33)

Family: `keystone28_corr_backbone_v53`. Builds directly on the merged Round-0
foundation (PR #32, merge commit `dcacb0a3b78d718f059891138ea36f5c065a8a79`):
the authority-explicit scoring harness, the frozen partition, and the
revalidated Ender20 score authority.

## Frozen primary question

> Does adding only chronologically prior, resolved validation history to an
> otherwise identical LightGBM backbone improve genuinely out-of-sample
> Ender20 weighted model score relative to a train-parquet-only control?

This is a recency/update hypothesis, not a hyperparameter sweep. Round 1 is
not expected to produce the final best Numerai model; it establishes the
first honest backbone and determines whether recent resolved history helps.
Feature-set, target-ensemble, and architecture expansion belong to later
rounds. The published `v53_lgbm_ender20` prediction column is the external
reference line and is scored on the identical rows and eras.

## Design in one page

- **Score zone:** eras 1133–1219 (87 weekly eras; the full MMC-capable
  development zone). Six contiguous walk-forward blocks: 1133–1147,
  1148–1162, 1163–1177, 1178–1192, 1193–1207, 1208–1219. Every selection
  score is out of sample; the eight eras before each scored block are
  embargoed (latest eligible validation training era = scored_start − 9).
- **CONTROL-T:** LightGBM on `train.parquet` only (eras 0001–0574), one
  static fit per model seed (3 fits), predicting all 87 scored eras.
- **CANDIDATE-V:** identical parameters and feature set; per block and seed,
  trains on `train.parquet` plus validation eras strictly before that
  block's embargo (18 fits); concatenated predictions cover the zone exactly
  once.
- **Cohort:** exactly 21 fits. Feature set `medium` (780 features, list hash
  frozen in the protocol). Target `target_ender_20` only; the bare `target`
  alias is never used. Seeds 42/1337/2024; deterministic era-balanced
  sampling with fixed seed 20260817 and a 1,000,000-row cap, identical law
  for both procedures and invariant across model seeds.
- **Scoring:** `keystone_round0.score_round0` under
  `ScoreAuthority.from_json(round0_score_authority.json)` (revalidated
  against the live round configuration immediately before the freeze),
  `expected_eras` = the 87-era zone, meta-model policy
  `fail_on_missing_meta_model_era`.
- **Decision:** the pre-registered law in `round1_protocol.json` /
  `round1_lib.decide_round1` — Ender20 seed-mean weighted model score only,
  with worst-seed mean/Sharpe/zero-baseline-drawdown guards and a
  four-of-six positive-block requirement. A fixed-seed moving-block
  bootstrap (10,000 resamples, block 8, seed 20260817) is report-only.
- **Pipeline sanity gate:** if CONTROL-T's mean CORR is below 70% of the
  published benchmark's mean CORR on the same eras, the round stops as an
  implementation failure (`STOPPED_AT_KW33_PIPELINE_PARITY_FAILURE`), not a
  negative result.
- **Ender60 auxiliary:** raw CORR60/MMC60 on the identical vectors are
  pre-registered cutover-readiness diagnostics. Any combined number is the
  `HYPOTHETICAL_CUTOVER_WEIGHTED_DIAGNOSTIC` and can never select the
  winner (the decision law has no Ender60 input path — tested).

## Hard boundaries

No GAP (1223–1230) or HOLDOUT (≥1231) row is loaded for training, scoring,
metric calculation, or model selection. No early stopping and no eval set
touch scored folds. Parameters, folds, sampling, and thresholds are frozen
before any scientific fit and never changed after a score is seen. No
upload, model creation, submission, staking, deployment, or any Numerai
account action is part of this round. Large artifacts live outside Git under
`D:\numerai-data\keystone28\round1\`.
